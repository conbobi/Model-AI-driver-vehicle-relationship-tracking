#!/usr/bin/env python3
"""
prepare_yolo_dataset.py
=======================

Chuẩn bị YOLO detection dataset từ annotations.jsonl được tạo bởi
src/label_tool_v26.py.

Tính năng chính:
- Đọc nhiều file annotations.jsonl dưới ann_root và gắn _subfolder.
- Lọc class theo danh sách cho YOLO: motorcyclist, car, truck.
- Gom annotations theo frame (subfolder, clip_id, timestamp) để hỗ trợ
  nhiều object trong cùng một ảnh.
- Chia train/val theo clip, ưu tiên stratified split bằng primary class.
- Trích xuất frame chính xác bằng OpenCV + CAP_PROP_POS_MSEC.
- Ghi YOLO labels, metadata.json, attribute_labels.csv và dataset.yaml.

Ví dụ:
    venv/bin/python src/model_train/prepare_yolo_dataset.py \
        --ann_root data/ann/633_NguyenChiThanh/Ngay11-03-2026/cam01 \
        --event_root data/event_clips/633_NguyenChiThanh/Ngay11-03-2026/cam01 \
        --output_dir data/yolo_dataset_verify \
        --clean
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import cv2

try:
    from sklearn.model_selection import train_test_split
except ImportError:  # pragma: no cover - fallback is only for missing dependency.
    train_test_split = None

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is optional.
    tqdm = None


CLASS_NAMES = ["motorcyclist", "car", "truck"]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}
TARGET_VAL_RATIO = 0.20
VAL_RATIO_TOLERANCE = 0.02

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger("prepare_yolo_dataset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare YOLO detection dataset from traffic annotations."
    )
    parser.add_argument(
        "--ann_root",
        type=Path,
        default=Path("data/ann/633_NguyenChiThanh/Ngay11-03-2026/cam01"),
        help="Root folder containing subfolders like 174/216/231 with annotations.jsonl.",
    )
    parser.add_argument(
        "--event_root",
        type=Path,
        default=Path("data/event_clips/633_NguyenChiThanh/Ngay11-03-2026/cam01"),
        help="Root folder containing event clips with the same subfolder layout.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/yolo_dataset_verify"),   # ← ĐÃ SỬA: mặc định vào trong project
        help="Output YOLO dataset directory.",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Validation split ratio. Default: 0.2",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for train/val split. Default: 42",
    )
    parser.add_argument(
        "--jpeg_quality",
        type=int,
        default=95,
        help="JPEG quality for extracted frames. Default: 95",
    )
    parser.add_argument(
        "--dataset_yaml_path",
        type=str,
        default=None,
        help=(
            "Value written to `path:` inside dataset.yaml. "
            "Default: ./<output_dir_name> to match the requested YOLO layout."
        ),
    )
    parser.add_argument(
        "--project_root",
        type=Path,
        default=Path("."),
        help="Base path used to make image_path relative in CSV/metadata.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing output_dir before generating a new dataset.",
    )
    return parser.parse_args()


def ensure_output_dirs(output_dir: Path, clean: bool) -> dict[str, Path]:
    if clean and output_dir.exists():
        LOGGER.info("Removing existing output directory: %s", output_dir)
        shutil.rmtree(output_dir)

    paths = {
        "train_images": output_dir / "train" / "images",
        "train_labels": output_dir / "train" / "labels",
        "val_images": output_dir / "val" / "images",
        "val_labels": output_dir / "val" / "labels",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def discover_annotation_files(ann_root: Path) -> list[Path]:
    ann_files = sorted(ann_root.rglob("annotations.jsonl"))
    if not ann_files:
        raise FileNotFoundError(f"No annotations.jsonl found under {ann_root}")
    LOGGER.info("Found %d annotation files under %s", len(ann_files), ann_root)
    for ann_file in ann_files:
        LOGGER.info("  - %s", ann_file)
    return ann_files


def normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def parse_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_annotations(ann_files: Iterable[Path]) -> tuple[list[dict], list[str], Counter]:
    records: list[dict] = []
    attribute_keys: set[str] = set()
    skipped = Counter()

    for ann_file in ann_files:
        subfolder = ann_file.parent.name
        with ann_file.open("r", encoding="utf-8") as handle:
            for line_idx, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    ann = json.loads(line)
                except json.JSONDecodeError:
                    skipped["malformed_json"] += 1
                    LOGGER.warning("Malformed JSON skipped: %s:%d", ann_file, line_idx)
                    continue

                class_name = ann.get("class_name")
                if class_name not in CLASS_TO_ID:
                    skipped["unsupported_class"] += 1
                    continue

                clip_id = normalize_whitespace(str(ann.get("clip_id", "")))
                timestamp = parse_float(ann.get("timestamp"))
                bbox = ann.get("bbox")
                attributes = ann.get("attributes") or {}

                if not clip_id:
                    skipped["missing_clip_id"] += 1
                    LOGGER.warning("Missing clip_id skipped: %s:%d", ann_file, line_idx)
                    continue
                if timestamp is None or timestamp < 0:
                    skipped["invalid_timestamp"] += 1
                    LOGGER.warning(
                        "Invalid timestamp skipped: %s:%d (%s)",
                        ann_file,
                        line_idx,
                        ann.get("timestamp"),
                    )
                    continue
                if not isinstance(bbox, list) or len(bbox) != 4:
                    skipped["invalid_bbox"] += 1
                    LOGGER.warning("Invalid bbox skipped: %s:%d (%s)", ann_file, line_idx, bbox)
                    continue
                bbox_floats = [parse_float(v) for v in bbox]
                if any(v is None for v in bbox_floats):
                    skipped["invalid_bbox"] += 1
                    LOGGER.warning(
                        "Non-numeric bbox skipped: %s:%d (%s)", ann_file, line_idx, bbox
                    )
                    continue

                duration = parse_float(ann.get("duration"))
                record = {
                    "clip_id": clip_id,
                    "timestamp": round(timestamp, 6),
                    "duration": duration,
                    "bbox": bbox_floats,
                    "class_name": class_name,
                    "class_id": CLASS_TO_ID[class_name],
                    "attributes": attributes if isinstance(attributes, dict) else {},
                    "query_vi": normalize_whitespace(str(ann.get("query_vi", ""))),
                    "query_en": normalize_whitespace(str(ann.get("query_en", ""))),
                    "_subfolder": subfolder,
                    "_source_file": str(ann_file),
                    "_line_number": line_idx,
                }
                attribute_keys.update(record["attributes"].keys())
                records.append(record)

    LOGGER.info("Loaded %d filtered annotations", len(records))
    if skipped:
        LOGGER.info("Skipped annotations summary: %s", dict(skipped))
    return records, sorted(attribute_keys), skipped


def build_clip_groups(records: Iterable[dict]) -> dict[tuple[str, str], list[dict]]:
    clip_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in records:
        key = (record["_subfolder"], record["clip_id"])
        clip_groups[key].append(record)
    return clip_groups


def choose_primary_class(labels: Iterable[str]) -> str:
    counts = Counter(labels)
    return counts.most_common(1)[0][0]


def fallback_train_test_split(
    clip_keys: list[tuple[str, str]],
    primary_labels: list[str],
    test_size: float,
    random_state: int,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for clip_key, label in zip(clip_keys, primary_labels):
        grouped[label].append(clip_key)

    rng_seed = int(random_state)
    train_keys: list[tuple[str, str]] = []
    val_keys: list[tuple[str, str]] = []

    for class_name in CLASS_NAMES:
        clips = sorted(grouped.get(class_name, []))
        if not clips:
            continue
        rng = math.floor((rng_seed + 1) * (CLASS_TO_ID[class_name] + 11))
        local_random = random.Random(rng)
        local_random.shuffle(clips)

        n_val = int(round(len(clips) * test_size))
        if len(clips) > 1:
            n_val = min(max(1, n_val), len(clips) - 1)
        else:
            n_val = 0

        val_keys.extend(clips[:n_val])
        train_keys.extend(clips[n_val:])

    return sorted(train_keys), sorted(val_keys)


def split_clips_stratified(
    clip_groups: dict[tuple[str, str], list[dict]],
    test_size: float,
    random_state: int,
) -> tuple[set[tuple[str, str]], set[tuple[str, str]], dict[tuple[str, str], str]]:
    clip_keys = sorted(clip_groups.keys())
    primary_by_clip = {
        clip_key: choose_primary_class(record["class_name"] for record in records)
        for clip_key, records in clip_groups.items()
    }
    primary_labels = [primary_by_clip[key] for key in clip_keys]
    label_counts = Counter(primary_labels)
    n_total = len(clip_keys)
    n_val = math.ceil(n_total * test_size)
    n_train = n_total - n_val

    LOGGER.info("Total clips: %d", len(clip_keys))
    LOGGER.info("Primary labels by clip: %s", dict(label_counts))

    can_stratify = (
        len(label_counts) > 1
        and all(count >= 2 for count in label_counts.values())
        and n_val >= len(label_counts)
        and n_train >= len(label_counts)
    )

    if train_test_split is not None and can_stratify:
        train_keys, val_keys = train_test_split(
            clip_keys,
            test_size=test_size,
            random_state=random_state,
            stratify=primary_labels,
        )
    elif train_test_split is not None:
        LOGGER.warning(
            "Cannot stratify because at least one class has <2 clips. Falling back to non-stratified split."
        )
        train_keys, val_keys = train_test_split(
            clip_keys,
            test_size=test_size,
            random_state=random_state,
            stratify=None,
        )
    else:
        LOGGER.warning(
            "scikit-learn is not installed. Using deterministic fallback split instead of train_test_split."
        )
        train_keys, val_keys = fallback_train_test_split(
            clip_keys=clip_keys,
            primary_labels=primary_labels,
            test_size=test_size,
            random_state=random_state,
        )

    train_set = set(train_keys)
    val_set = set(val_keys)

    if train_set & val_set:
        raise RuntimeError("Clip leakage detected between train and val splits.")
    if train_set | val_set != set(clip_keys):
        raise RuntimeError("Split does not cover all clips.")

    LOGGER.info("Split result: %d train clips | %d val clips", len(train_set), len(val_set))
    return train_set, val_set, primary_by_clip


def group_frames(records: Iterable[dict]) -> dict[tuple[str, str], dict[float, list[dict]]]:
    grouped: dict[tuple[str, str], dict[float, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        clip_key = (record["_subfolder"], record["clip_id"])
        grouped[clip_key][record["timestamp"]].append(record)
    return grouped


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def convert_bbox_to_yolo(
    bbox_xywh: list[float],
    image_width: int,
    image_height: int,
) -> tuple[tuple[float, float, float, float], list[float]] | None:
    x, y, w, h = bbox_xywh
    if w <= 0 or h <= 0:
        return None

    x1 = clamp(x, 0.0, float(image_width))
    y1 = clamp(y, 0.0, float(image_height))
    x2 = clamp(x + w, 0.0, float(image_width))
    y2 = clamp(y + h, 0.0, float(image_height))
    clipped_w = x2 - x1
    clipped_h = y2 - y1

    if clipped_w <= 0 or clipped_h <= 0:
        return None

    x_center = ((x1 + x2) / 2.0) / float(image_width)
    y_center = ((y1 + y2) / 2.0) / float(image_height)
    norm_w = clipped_w / float(image_width)
    norm_h = clipped_h / float(image_height)

    yolo_bbox = (
        clamp(x_center, 0.0, 1.0),
        clamp(y_center, 0.0, 1.0),
        clamp(norm_w, 0.0, 1.0),
        clamp(norm_h, 0.0, 1.0),
    )
    if yolo_bbox[2] <= 0 or yolo_bbox[3] <= 0:
        return None

    return yolo_bbox, [x1, y1, clipped_w, clipped_h]


def timestamp_to_token(timestamp: float) -> str:
    return f"{timestamp:.6f}".rstrip("0").rstrip(".")


def format_relative_path(path: Path, base: Path) -> str:
    return os.path.relpath(path.resolve(), start=base.resolve())


def infer_video_duration(cap: cv2.VideoCapture) -> float | None:
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if fps and fps > 0 and frame_count and frame_count > 0:
        return frame_count / fps
    return None


def normalize_attribute_value(value: object) -> str:
    if value is None:
        return ""
    return normalize_whitespace(str(value))


def split_helmet_attribute(raw_value: object) -> tuple[int, str]:
    helmet = normalize_attribute_value(raw_value)
    if not helmet:
        return 0, ""

    lowered = helmet.lower()
    negative_patterns = [
        "không đội nón bảo hiểm",
        "không đội mũ bảo hiểm",
        "không đội nón",
        "không đội mũ",
        "không đội helmet",
        "no helmet",
    ]
    if any(pattern in lowered for pattern in negative_patterns):
        return 0, ""

    color = re.sub(
        r"^(đội\s+)?(nón bảo hiểm|mũ bảo hiểm)\s*",
        "",
        helmet,
        flags=re.IGNORECASE,
    ).strip()
    return 1, color


def build_attribute_fieldnames(attribute_keys: list[str]) -> list[str]:
    preferred = ["shirt", "helmet", "helmet_present", "helmet_color", "bike_color", "bike_type"]
    extras = [key for key in attribute_keys if key not in {"helmet_present", "helmet_color"}]
    ordered_extras = [key for key in preferred if key in extras or key in {"helmet_present", "helmet_color"}]
    remaining = [key for key in attribute_keys if key not in {"shirt", "helmet", "bike_color", "bike_type"}]
    return ordered_extras + remaining


def iter_with_progress(items: list, desc: str) -> Iterable:
    if tqdm is None:
        return items
    return tqdm(items, desc=desc)


def process_split(
    split_name: str,
    clip_keys: set[tuple[str, str]],
    frames_by_clip: dict[tuple[str, str], dict[float, list[dict]]],
    event_root: Path,
    image_dir: Path,
    label_dir: Path,
    project_root: Path,
    jpeg_quality: int,
) -> tuple[dict[str, dict], list[dict], dict[str, Counter], Counter]:
    metadata: dict[str, dict] = {}
    attribute_rows: list[dict] = []
    per_class_stats = {
        "image_presence": Counter(),
        "object_count": Counter(),
    }
    skipped = Counter()

    for clip_key in iter_with_progress(sorted(clip_keys), desc=f"Extract {split_name}"):
        subfolder, clip_id = clip_key
        video_path = event_root / subfolder / f"{clip_id}.mp4"
        frame_groups = frames_by_clip.get(clip_key, {})

        if not video_path.exists():
            skipped["missing_video"] += len(frame_groups)
            LOGGER.warning("Missing video for clip %s/%s: %s", subfolder, clip_id, video_path)
            continue

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            skipped["cannot_open_video"] += len(frame_groups)
            LOGGER.warning("Cannot open video: %s", video_path)
            continue

        video_duration = infer_video_duration(cap)

        try:
            for timestamp in sorted(frame_groups.keys()):
                annotations = frame_groups[timestamp]
                expected_duration = next(
                    (record["duration"] for record in annotations if record["duration"] is not None),
                    None,
                )
                duration_limit = video_duration if video_duration is not None else expected_duration
                if duration_limit is not None and timestamp > duration_limit:
                    skipped["timestamp_out_of_range"] += 1
                    LOGGER.warning(
                        "Timestamp %.3f exceeds duration %.3f for %s",
                        timestamp,
                        duration_limit,
                        video_path,
                    )
                    continue

                cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
                ok, frame = cap.read()
                if not ok or frame is None:
                    skipped["frame_read_failed"] += 1
                    LOGGER.warning("Cannot read frame %.3f from %s", timestamp, video_path)
                    continue

                image_height, image_width = frame.shape[:2]
                stem = f"{subfolder}_{clip_id}_{timestamp_to_token(timestamp)}"
                image_path = image_dir / f"{stem}.jpg"
                label_path = label_dir / f"{stem}.txt"
                rel_image_path = format_relative_path(image_path, project_root)
                rel_video_path = format_relative_path(video_path, project_root)

                label_lines: list[str] = []
                object_entries: list[dict] = []
                image_classes: set[str] = set()
                pending_attribute_rows: list[dict] = []
                pending_object_counts = Counter()

                for instance_index, record in enumerate(annotations):
                    converted = convert_bbox_to_yolo(record["bbox"], image_width, image_height)
                    if converted is None:
                        skipped["invalid_normalized_bbox"] += 1
                        LOGGER.warning(
                            "Invalid bbox after normalization skipped: %s:%d (%s)",
                            record["_source_file"],
                            record["_line_number"],
                            record["bbox"],
                        )
                        continue

                    yolo_bbox, clipped_bbox = converted
                    class_name = record["class_name"]
                    class_id = record["class_id"]
                    label_lines.append(
                        f"{class_id} "
                        f"{yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} "
                        f"{yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}"
                    )

                    helmet_present, helmet_color = split_helmet_attribute(
                        record["attributes"].get("helmet")
                    )
                    object_entries.append(
                        {
                            "instance_index": instance_index,
                            "class_name": class_name,
                            "class_id": class_id,
                            "bbox_xywh": [round(v, 3) for v in record["bbox"]],
                            "bbox_xywh_clipped": [round(v, 3) for v in clipped_bbox],
                            "bbox_yolo": [round(v, 6) for v in yolo_bbox],
                            "query_vi": record["query_vi"],
                            "query_en": record["query_en"],
                            "attributes": record["attributes"],
                            "helmet_present": helmet_present,
                            "helmet_color": helmet_color,
                            "source_file": record["_source_file"],
                            "source_line": record["_line_number"],
                        }
                    )

                    image_classes.add(class_name)
                    pending_object_counts[class_name] += 1

                    attribute_row = {
                        "image_path": rel_image_path,
                        "split": split_name,
                        "subfolder": subfolder,
                        "clip_id": clip_id,
                        "timestamp": f"{timestamp:.6f}",
                        "class_name": class_name,
                        "class_id": class_id,
                        "instance_index": instance_index,
                        "bbox_x": round(clipped_bbox[0], 3),
                        "bbox_y": round(clipped_bbox[1], 3),
                        "bbox_w": round(clipped_bbox[2], 3),
                        "bbox_h": round(clipped_bbox[3], 3),
                        "query_vi": record["query_vi"],
                        "query_en": record["query_en"],
                        "shirt": normalize_attribute_value(record["attributes"].get("shirt")),
                        "helmet": normalize_attribute_value(record["attributes"].get("helmet")),
                        "helmet_present": helmet_present,
                        "helmet_color": helmet_color,
                        "bike_color": normalize_attribute_value(
                            record["attributes"].get("bike_color")
                        ),
                        "bike_type": normalize_attribute_value(record["attributes"].get("bike_type")),
                    }
                    for key, value in record["attributes"].items():
                        attribute_row[key] = normalize_attribute_value(value)
                    pending_attribute_rows.append(attribute_row)

                if not label_lines:
                    skipped["frame_without_valid_labels"] += 1
                    continue

                success = cv2.imwrite(
                    str(image_path),
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
                )
                if not success:
                    skipped["image_write_failed"] += 1
                    LOGGER.warning("Failed to write image: %s", image_path)
                    continue

                with label_path.open("w", encoding="utf-8") as label_handle:
                    label_handle.write("\n".join(label_lines) + "\n")

                attribute_rows.extend(pending_attribute_rows)
                for class_name, count in pending_object_counts.items():
                    per_class_stats["object_count"][class_name] += count

                metadata[rel_image_path] = {
                    "split": split_name,
                    "image_path": rel_image_path,
                    "video_path": rel_video_path,
                    "subfolder": subfolder,
                    "clip_id": clip_id,
                    "timestamp": timestamp,
                    "image_width": image_width,
                    "image_height": image_height,
                    "num_objects": len(object_entries),
                    "queries_vi": [entry["query_vi"] for entry in object_entries],
                    "queries_en": [entry["query_en"] for entry in object_entries],
                    "objects": object_entries,
                }

                for class_name in image_classes:
                    per_class_stats["image_presence"][class_name] += 1
        finally:
            cap.release()

    return metadata, attribute_rows, per_class_stats, skipped


def write_metadata(output_dir: Path, metadata: dict[str, dict]) -> None:
    metadata_path = output_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    LOGGER.info("Wrote metadata: %s", metadata_path)


def write_attribute_csv(
    output_dir: Path,
    rows: list[dict],
    attribute_keys: list[str],
) -> None:
    csv_path = output_dir / "attribute_labels.csv"
    base_columns = [
        "image_path",
        "split",
        "subfolder",
        "clip_id",
        "timestamp",
        "class_name",
        "class_id",
        "instance_index",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
        "query_vi",
        "query_en",
    ]
    dynamic_attr_columns = build_attribute_fieldnames(attribute_keys)
    fieldnames = base_columns + dynamic_attr_columns

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            for field in dynamic_attr_columns:
                row.setdefault(field, "")
            writer.writerow(row)

    LOGGER.info("Wrote attribute labels CSV: %s", csv_path)


def write_dataset_yaml(output_dir: Path, dataset_yaml_path: str | None) -> None:
    yaml_path = output_dir / "dataset.yaml"
    yaml_root = dataset_yaml_path or f"./{output_dir.name}"
    yaml_text = (
        f"path: {yaml_root}\n"
        "train: train/images\n"
        "val: val/images\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: {CLASS_NAMES}\n"
    )
    with yaml_path.open("w", encoding="utf-8") as handle:
        handle.write(yaml_text)
    LOGGER.info("Wrote dataset config: %s", yaml_path)


def print_class_statistics(
    train_stats: dict[str, Counter],
    val_stats: dict[str, Counter],
) -> None:
    print("\n=== Train/Val Statistics (image-level presence) ===")
    print(f"{'Class':<15}{'Train':>10}{'Val':>10}{'Total':>10}{'Val/Total':>12}")
    warnings: list[str] = []

    for class_name in CLASS_NAMES:
        train_count = train_stats["image_presence"][class_name]
        val_count = val_stats["image_presence"][class_name]
        total_count = train_count + val_count
        ratio = (val_count / total_count) if total_count else 0.0
        print(
            f"{class_name:<15}{train_count:>10}{val_count:>10}{total_count:>10}{ratio:>12.2%}"
        )
        if total_count and abs(ratio - TARGET_VAL_RATIO) > VAL_RATIO_TOLERANCE:
            warnings.append(
                f"Class '{class_name}' has val ratio {ratio:.2%}, "
                f"which deviates more than {VAL_RATIO_TOLERANCE:.0%} from {TARGET_VAL_RATIO:.0%}."
            )

    print("\n=== Train/Val Statistics (object count) ===")
    print(f"{'Class':<15}{'Train':>10}{'Val':>10}{'Total':>10}{'Val/Total':>12}")
    for class_name in CLASS_NAMES:
        train_count = train_stats["object_count"][class_name]
        val_count = val_stats["object_count"][class_name]
        total_count = train_count + val_count
        ratio = (val_count / total_count) if total_count else 0.0
        print(
            f"{class_name:<15}{train_count:>10}{val_count:>10}{total_count:>10}{ratio:>12.2%}"
        )

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"- {warning}")


def main() -> None:
    args = parse_args()

    ann_root = args.ann_root.resolve()
    event_root = args.event_root.resolve()
    output_dir = args.output_dir.resolve()
    project_root = args.project_root.resolve()

    output_paths = ensure_output_dirs(output_dir, clean=args.clean)
    ann_files = discover_annotation_files(ann_root)
    records, attribute_keys, skipped_load = load_annotations(ann_files)
    if not records:
        raise RuntimeError("No valid annotations found after filtering target classes.")

    clip_groups = build_clip_groups(records)
    train_clips, val_clips, _ = split_clips_stratified(
        clip_groups=clip_groups,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    frames_by_clip = group_frames(records)

    LOGGER.info("Train clips: %d | Val clips: %d", len(train_clips), len(val_clips))

    train_metadata, train_attr_rows, train_stats, train_skipped = process_split(
        split_name="train",
        clip_keys=train_clips,
        frames_by_clip=frames_by_clip,
        event_root=event_root,
        image_dir=output_paths["train_images"],
        label_dir=output_paths["train_labels"],
        project_root=project_root,
        jpeg_quality=args.jpeg_quality,
    )
    val_metadata, val_attr_rows, val_stats, val_skipped = process_split(
        split_name="val",
        clip_keys=val_clips,
        frames_by_clip=frames_by_clip,
        event_root=event_root,
        image_dir=output_paths["val_images"],
        label_dir=output_paths["val_labels"],
        project_root=project_root,
        jpeg_quality=args.jpeg_quality,
    )

    all_metadata = {}
    all_metadata.update(train_metadata)
    all_metadata.update(val_metadata)
    write_metadata(output_dir, all_metadata)
    write_attribute_csv(output_dir, train_attr_rows + val_attr_rows, attribute_keys)
    write_dataset_yaml(output_dir, args.dataset_yaml_path)

    train_image_count = len(train_metadata)
    val_image_count = len(val_metadata)
    LOGGER.info("Saved %d train images and %d val images", train_image_count, val_image_count)
    if skipped_load:
        LOGGER.info("Load-time skipped summary: %s", dict(skipped_load))
    if train_skipped:
        LOGGER.info("Train extraction skipped summary: %s", dict(train_skipped))
    if val_skipped:
        LOGGER.info("Val extraction skipped summary: %s", dict(val_skipped))

    print(f"\nTrain clips: {len(train_clips)}")
    print(f"Val clips:   {len(val_clips)}")
    print(f"Train images: {train_image_count}")
    print(f"Val images:   {val_image_count}")
    print_class_statistics(train_stats, val_stats)


if __name__ == "__main__":
    main()