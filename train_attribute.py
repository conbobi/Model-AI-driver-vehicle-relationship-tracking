#!/usr/bin/env python3

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from attribute_model import MultiHeadAttributeModel

LOGGER = logging.getLogger("train_attribute")

ATTRIBUTE_COLUMNS = ["shirt", "helmet_color", "bike_color", "bike_type", "action"]


class AttributeDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_path = Path(row["image_path"])
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        labels = {
            "shirt": torch.tensor(int(row["shirt"]), dtype=torch.long),
            "helmet_present": torch.tensor(float(row["helmet_present"]), dtype=torch.float32),
            "helmet_color": torch.tensor(int(row["helmet_color"]), dtype=torch.long),
            "bike_color": torch.tensor(int(row["bike_color"]), dtype=torch.long),
            "bike_type": torch.tensor(int(row["bike_type"]), dtype=torch.long),
            "action": torch.tensor(int(row["action"]), dtype=torch.long),
        }
        return image, labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multi-head attribute classifier.")
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path("data/yolo_dataset_verify/attribute_labels.csv"),
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=Path("models/attribute/attribute_model.pth"),
    )
    parser.add_argument(
        "--output-encoder",
        type=Path,
        default=Path("models/attribute/label_encoders.pkl"),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def prepare_dataframe(csv_path: Path) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "class_name" in df.columns:
        df = df[df["class_name"] == "motorcyclist"].copy()

    if df.empty:
        raise ValueError("No motorcyclist samples found in attribute CSV.")

    if "helmet_present" in df.columns:
        helmet_present = pd.to_numeric(df["helmet_present"], errors="coerce").fillna(0)
        df["helmet_present"] = helmet_present.astype(int)
    elif "helmet" in df.columns:
        df["helmet_present"] = df["helmet"].fillna("").astype(str).str.strip().ne("").astype(int)
    else:
        df["helmet_present"] = 0

    for col in ATTRIBUTE_COLUMNS:
        if col not in df.columns:
            df[col] = "unknown"

    df["shirt"] = df["shirt"].fillna("unknown").astype(str)
    df["bike_color"] = df["bike_color"].fillna("unknown").astype(str)
    df["bike_type"] = df["bike_type"].fillna("unknown").astype(str)
    df["action"] = df["action"].fillna("unknown").astype(str)
    df["helmet_color"] = df["helmet_color"].fillna("none").astype(str)
    df.loc[df["helmet_present"] == 0, "helmet_color"] = "none"

    le_dict: dict[str, LabelEncoder] = {}
    for col in ATTRIBUTE_COLUMNS:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col].astype(str))
        le_dict[col] = encoder

    return df, le_dict


def build_dataloaders(args: argparse.Namespace, df: pd.DataFrame):
    if "split" not in df.columns:
        raise KeyError("CSV is missing required `split` column.")

    train_df = df[df["split"].astype(str).str.lower() == "train"].copy()
    val_df = df[df["split"].astype(str).str.lower().isin({"val", "valid", "validation"})].copy()
    if train_df.empty or val_df.empty:
        raise ValueError("Attribute CSV must contain both train and val samples.")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            normalize,
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ]
    )

    train_loader = DataLoader(
        AttributeDataset(train_df, train_tf),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        AttributeDataset(val_df, val_tf),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


def compute_loss(outputs: dict[str, torch.Tensor], labels: dict[str, torch.Tensor]) -> torch.Tensor:
    ce = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss()
    return (
        ce(outputs["shirt"], labels["shirt"])
        + bce(outputs["helmet_present"], labels["helmet_present"])
        + ce(outputs["helmet_color"], labels["helmet_color"])
        + ce(outputs["bike_color"], labels["bike_color"])
        + ce(outputs["bike_type"], labels["bike_type"])
        + ce(outputs["action"], labels["action"])
    )


def run_epoch(model, loader, device, optimizer=None) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = {key: value.to(device) for key, value in labels.items()}

        with torch.set_grad_enabled(training):
            outputs = model(images)
            loss = compute_loss(outputs, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    args.output_encoder.parent.mkdir(parents=True, exist_ok=True)

    df, le_dict = prepare_dataframe(args.csv_path)
    with args.output_encoder.open("wb") as handle:
        pickle.dump(le_dict, handle)
    LOGGER.info("Saved label encoders to %s", args.output_encoder)

    num_classes = {key: len(encoder.classes_) for key, encoder in le_dict.items()}
    train_loader, val_loader = build_dataloaders(args, df)

    device = torch.device(args.device)
    model = MultiHeadAttributeModel(num_classes_dict=num_classes, pretrained_backbone=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    best_state_dict = None

    for epoch in range(args.epochs):
        train_loss = run_epoch(model, train_loader, device, optimizer=optimizer)
        val_loss = run_epoch(model, val_loader, device, optimizer=None)
        LOGGER.info(
            "Epoch %d/%d - train_loss=%.4f - val_loss=%.4f",
            epoch + 1,
            args.epochs,
            train_loss,
            val_loss,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = model.state_dict()
            torch.save(best_state_dict, args.output_model)
            LOGGER.info("Saved improved checkpoint to %s", args.output_model)

    if best_state_dict is None:
        torch.save(model.state_dict(), args.output_model)
        LOGGER.warning("Validation loop produced no best checkpoint; saved final weights instead.")


if __name__ == "__main__":
    main()
