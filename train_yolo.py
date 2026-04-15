#!/usr/bin/env python3
"""
train_yolo.py
=============

Fine-tune YOLO detection model từ checkpoint có sẵn.

Hỗ trợ:
- Ultralytics backend (`pip install ultralytics`)
- Local YOLOv5 repo backend (`--backend yolov5 --yolov5_repo /path/to/yolov5`)

Ví dụ:
    venv/bin/python train_yolo.py
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger("train_yolo")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a YOLO detector.")
    parser.add_argument(
        "--dataset_yaml",
        type=Path,
        default=Path("data/yolo_dataset_verify/dataset.yaml"),
        help="Input dataset.yaml",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("data/yolo_dataset_verify/best.pt"),
        help="Initial YOLO weights.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("models/yolo"),
        help="Directory where the fine-tuned model will be stored.",
    )
    parser.add_argument(
        "--best_name",
        type=str,
        default="best_yolo_v2.pt",
        help="Filename for the copied best checkpoint inside output_dir.",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Training device: auto/cpu/0/0,1/...",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "ultralytics", "yolov5"],
        default="auto",
        help="YOLO backend preference.",
    )
    parser.add_argument(
        "--yolov5_repo",
        type=Path,
        default=Path("src/model_train/TPHYolov5"),
        help="Local ultralytics/yolov5 repo path for YOLOv5 backend.",
    )
    return parser.parse_args()


def ensure_exists(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")


def resolve_device_arg(device_arg: str) -> str:
    if device_arg == "auto":
        return "0" if torch.cuda.is_available() else "cpu"
    return device_arg


def write_resolved_dataset_yaml(dataset_yaml: Path, output_dir: Path) -> Path:
    text = dataset_yaml.read_text(encoding="utf-8")
    dataset_root = dataset_yaml.parent.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_yaml = output_dir / "dataset_resolved.yaml"

    lines = []
    has_path = False
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("path:"):
            lines.append(f"path: {dataset_root.as_posix()}")
            has_path = True
        else:
            lines.append(raw_line)
    if not has_path:
        lines.insert(0, f"path: {dataset_root.as_posix()}")

    resolved_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOGGER.info("Resolved dataset YAML written to %s", resolved_yaml)
    return resolved_yaml


def save_training_metadata(output_dir: Path, metadata: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "train_config.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    LOGGER.info("Saved training metadata to %s", metadata_path)


def copy_best_weight(weights_dir: Path, output_dir: Path, best_name: str) -> Path:
    source_best = weights_dir / "best.pt"
    if not source_best.exists():
        raise FileNotFoundError(f"Trained best.pt not found at {source_best}")
    target_best = output_dir / best_name
    shutil.copy2(source_best, target_best)
    LOGGER.info("Copied fine-tuned best.pt to %s", target_best)
    return target_best


def train_with_ultralytics(
    dataset_yaml: Path,
    weights: Path,
    output_dir: Path,
    args: argparse.Namespace,
    device_str: str,
) -> Path:
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - dependency specific.
        raise ImportError(
            "ultralytics is not installed. Install it with `pip install ultralytics`."
        ) from exc

    LOGGER.info("Starting training with ultralytics backend")
    try:
        model = YOLO(str(weights))
    except Exception as exc:  # pragma: no cover - depends on weight compatibility.
        raise RuntimeError(
            "Could not load initial checkpoint with ultralytics. "
            "If this is a YOLOv5 checkpoint, consider using `--backend yolov5` "
            "with a local yolov5 repo."
        ) from exc

    model.train(
        data=str(dataset_yaml),
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.imgsz,
        device=device_str,
        workers=args.workers,
        project=str(output_dir.parent),
        name=output_dir.name,
        exist_ok=True,
    )

    save_dir = Path(model.trainer.save_dir)
    LOGGER.info("Ultralytics run directory: %s", save_dir)
    return copy_best_weight(save_dir / "weights", output_dir, args.best_name)


def train_with_yolov5_repo(
    dataset_yaml: Path,
    weights: Path,
    output_dir: Path,
    args: argparse.Namespace,
    device_str: str,
) -> Path:
    repo_path = args.yolov5_repo.resolve()
    train_script = repo_path / "train.py"
    ensure_exists(repo_path, "YOLOv5 repo")
    ensure_exists(train_script, "YOLOv5 train.py")

    cmd = [
        sys.executable,
        str(train_script),
        "--img",
        str(args.imgsz),
        "--batch",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
        "--data",
        str(dataset_yaml),
        "--weights",
        str(weights),
        "--project",
        str(output_dir.parent),
        "--name",
        output_dir.name,
        "--exist-ok",
        "--workers",
        str(args.workers),
        "--device",
        device_str,
    ]

    LOGGER.info("Starting training with YOLOv5 backend")
    LOGGER.info("Command: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(repo_path))
    return copy_best_weight(output_dir / "weights", output_dir, args.best_name)


def select_backend(args: argparse.Namespace) -> str:
    if args.backend != "auto":
        return args.backend

    try:
        import ultralytics  # noqa: F401
        return "ultralytics"
    except Exception:
        if args.yolov5_repo.exists():
            return "yolov5"
    raise RuntimeError(
        "No supported YOLO backend available. Install `ultralytics` or provide "
        "a local YOLOv5 repo via `--yolov5_repo`."
    )


def main() -> None:
    args = parse_args()
    dataset_yaml = args.dataset_yaml.resolve()
    weights = args.weights.resolve()
    output_dir = args.output_dir.resolve()
    ensure_exists(dataset_yaml, "dataset.yaml")
    ensure_exists(weights, "initial weights")

    device_str = resolve_device_arg(args.device)
    resolved_dataset_yaml = write_resolved_dataset_yaml(dataset_yaml, output_dir)
    backend = select_backend(args)
    LOGGER.info("Selected backend: %s", backend)

    metadata = {
        "dataset_yaml": str(dataset_yaml),
        "resolved_dataset_yaml": str(resolved_dataset_yaml),
        "weights": str(weights),
        "output_dir": str(output_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "imgsz": args.imgsz,
        "workers": args.workers,
        "device": device_str,
        "backend": backend,
        "best_name": args.best_name,
    }
    save_training_metadata(output_dir, metadata)

    if backend == "ultralytics":
        try:
            best_path = train_with_ultralytics(
                dataset_yaml=resolved_dataset_yaml,
                weights=weights,
                output_dir=output_dir,
                args=args,
                device_str=device_str,
            )
        except Exception as exc:
            if args.backend == "auto" and args.yolov5_repo.exists():
                LOGGER.warning(
                    "Ultralytics backend failed (%s). Falling back to local YOLOv5 repo.",
                    exc,
                )
                best_path = train_with_yolov5_repo(
                    dataset_yaml=resolved_dataset_yaml,
                    weights=weights,
                    output_dir=output_dir,
                    args=args,
                    device_str=device_str,
                )
            else:
                raise
    else:
        best_path = train_with_yolov5_repo(
            dataset_yaml=resolved_dataset_yaml,
            weights=weights,
            output_dir=output_dir,
            args=args,
            device_str=device_str,
        )

    LOGGER.info("Training completed successfully. Fine-tuned model: %s", best_path)


if __name__ == "__main__":
    main()
