from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torchvision import models

ATTRIBUTE_HEADS = ("shirt", "helmet_color", "bike_color", "bike_type", "action")


def _efficientnet_weights(pretrained: bool):
    if pretrained:
        return models.EfficientNet_B0_Weights.DEFAULT
    return None


class MultiHeadAttributeModel(nn.Module):
    def __init__(self, num_classes_dict: dict[str, int], pretrained_backbone: bool = True):
        super().__init__()
        backbone = models.efficientnet_b0(weights=_efficientnet_weights(pretrained_backbone))
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.2)

        in_features = backbone.classifier[1].in_features
        self.fc_shirt = nn.Linear(in_features, num_classes_dict["shirt"])
        self.fc_helmet_present = nn.Linear(in_features, 1)
        self.fc_helmet_color = nn.Linear(in_features, num_classes_dict["helmet_color"])
        self.fc_bike_color = nn.Linear(in_features, num_classes_dict["bike_color"])
        self.fc_bike_type = nn.Linear(in_features, num_classes_dict["bike_type"])
        self.fc_action = nn.Linear(in_features, num_classes_dict["action"])

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)

        return {
            "shirt": self.fc_shirt(x),
            "helmet_present": self.fc_helmet_present(x).squeeze(1),
            "helmet_color": self.fc_helmet_color(x),
            "bike_color": self.fc_bike_color(x),
            "bike_type": self.fc_bike_type(x),
            "action": self.fc_action(x),
        }


def build_num_classes_from_encoders(le_dict: dict[str, Any]) -> dict[str, int]:
    missing = [key for key in ATTRIBUTE_HEADS if key not in le_dict]
    if missing:
        raise KeyError(f"Missing label encoders for: {missing}")
    return {key: len(le_dict[key].classes_) for key in ATTRIBUTE_HEADS}


def load_attribute_model(
    weights_path: str | Path,
    label_encoder_path: str | Path,
    device: str | torch.device = "cpu",
    num_classes: dict[str, int] | None = None,
) -> tuple[MultiHeadAttributeModel, dict[str, Any]]:
    weights_path = Path(weights_path)
    label_encoder_path = Path(label_encoder_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Attribute weights not found: {weights_path}")
    if not label_encoder_path.exists():
        raise FileNotFoundError(f"Label encoder file not found: {label_encoder_path}")

    with label_encoder_path.open("rb") as handle:
        le_dict = pickle.load(handle)

    resolved_num_classes = num_classes or build_num_classes_from_encoders(le_dict)
    model = MultiHeadAttributeModel(
        num_classes_dict=resolved_num_classes,
        pretrained_backbone=False,
    )

    state_dict = torch.load(weights_path, map_location=device)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, le_dict
