from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms

from attribute_model import load_attribute_model
from src.model_train.TPHYolov5.models.common import DetectMultiBackend
from src.model_train.TPHYolov5.utils.augmentations import letterbox
from src.model_train.TPHYolov5.utils.general import non_max_suppression, scale_coords
from src.model_train.TPHYolov5.utils.torch_utils import select_device
from tracker import SimpleTracker

LOGGER = logging.getLogger("inference_pipeline")


class InferencePipeline:
    def __init__(
        self,
        yolo_weights,
        attribute_weights=None,
        label_encoder_path=None,
        device="cpu",
        attribute_every_n_frames=5,
        track_vote_window=10,
        roi_path=None,
        camera_id=None,
    ):
        self.device = select_device(device)
        self.attribute_every_n_frames = max(1, int(attribute_every_n_frames))
        self.track_vote_window = max(1, int(track_vote_window))

        yolo_path = self._resolve_existing_path(
            [
                yolo_weights,
                "models/yolo/best_yolo_v2.pt",
                "data/yolo_dataset_verify/best.pt",
            ],
            description="YOLO weights",
        )
        self.yolo = DetectMultiBackend(str(yolo_path), device=self.device)
        self.stride = int(self.yolo.stride.max()) if hasattr(self.yolo.stride, "max") else int(self.yolo.stride)
        self.names = self.yolo.names

        self.attr_model = None
        self.le_dict = None
        self.attribute_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        if attribute_weights and label_encoder_path:
            weights_path = Path(attribute_weights)
            encoder_path = Path(label_encoder_path)
            if weights_path.exists() and encoder_path.exists():
                self.attr_model, self.le_dict = load_attribute_model(weights_path, encoder_path, self.device)
            else:
                LOGGER.warning(
                    "Attribute model files not found yet. weights=%s encoders=%s. Attribute inference will be skipped.",
                    weights_path,
                    encoder_path,
                )

        self.tracker = SimpleTracker()
        self.track_history = defaultdict(lambda: deque(maxlen=30))
        self.track_attribute_votes = {}
        self.last_attributes = {}
        self.roi_polygon = self._load_roi_polygon(roi_path, camera_id)

    @staticmethod
    def _resolve_existing_path(candidates, description: str) -> Path:
        for candidate in candidates:
            if not candidate:
                continue
            path = Path(candidate)
            if path.exists():
                return path
        raise FileNotFoundError(f"{description} not found in candidates: {candidates}")

    @staticmethod
    def _load_roi_polygon(roi_path, camera_id):
        if not roi_path:
            return None
        roi_file = Path(roi_path)
        if not roi_file.exists():
            LOGGER.warning("ROI file not found: %s", roi_file)
            return None

        with roi_file.open("r", encoding="utf-8") as handle:
            roi_data = json.load(handle)

        if camera_id and camera_id in roi_data:
            polygon = roi_data[camera_id].get("roi_polygon")
            return np.array(polygon, dtype=np.int32) if polygon else None

        if len(roi_data) == 1:
            only_camera = next(iter(roi_data.values()))
            polygon = only_camera.get("roi_polygon")
            return np.array(polygon, dtype=np.int32) if polygon else None

        LOGGER.warning("ROI file has multiple cameras; pass camera_id to enable ROI filtering.")
        return None

    def preprocess(self, img0, img_size=640):
        img = letterbox(img0, img_size, stride=self.stride, auto=True)[0]
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img

    def detect(self, img0):
        img = self.preprocess(img0)
        with torch.no_grad():
            raw_pred = self.yolo(img, augment=False)
        pred = raw_pred[0] if isinstance(raw_pred, (list, tuple)) else raw_pred
        pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)

        detections = []
        if pred and len(pred[0]):
            pred[0][:, :4] = scale_coords(img.shape[2:], pred[0][:, :4], img0.shape).round()
            for *xyxy, conf, cls in reversed(pred[0]):
                detections.append(
                    [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]), float(conf), int(cls)]
                )
        return detections

    @staticmethod
    def _clip_box(box, frame_shape):
        frame_h, frame_w = frame_shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1 = max(0, min(x1, frame_w - 1))
        x2 = max(0, min(x2, frame_w))
        y1 = max(0, min(y1, frame_h - 1))
        y2 = max(0, min(y2, frame_h))
        return x1, y1, x2, y2

    def _default_attributes(self):
        return {
            "shirt": "unknown",
            "helmet_present": False,
            "helmet_color": "không có mũ",
            "bike_color": "unknown",
            "bike_type": "unknown",
            "action": "unknown",
        }

    def get_attributes(self, crop_img):
        if self.attr_model is None or self.le_dict is None:
            return self._default_attributes()
        if crop_img is None or crop_img.size == 0 or crop_img.shape[0] < 2 or crop_img.shape[1] < 2:
            return None

        try:
            tensor = self.attribute_transform(crop_img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                out = self.attr_model(tensor)
        except Exception as exc:
            LOGGER.warning("Attribute inference failed: %s", exc)
            return None

        shirt = self.le_dict["shirt"].inverse_transform([out["shirt"].argmax().item()])[0]
        helmet_present = torch.sigmoid(out["helmet_present"]).item() > 0.5
        if helmet_present:
            helmet_color = self.le_dict["helmet_color"].inverse_transform(
                [out["helmet_color"].argmax().item()]
            )[0]
        else:
            helmet_color = "không có mũ"
        bike_color = self.le_dict["bike_color"].inverse_transform([out["bike_color"].argmax().item()])[0]
        bike_type = self.le_dict["bike_type"].inverse_transform([out["bike_type"].argmax().item()])[0]
        action = self.le_dict["action"].inverse_transform([out["action"].argmax().item()])[0]
        return {
            "shirt": shirt,
            "helmet_present": helmet_present,
            "helmet_color": helmet_color,
            "bike_color": bike_color,
            "bike_type": bike_type,
            "action": action,
        }

    def _update_attribute_votes(self, track_id: int, attrs: dict):
        if track_id not in self.track_attribute_votes:
            self.track_attribute_votes[track_id] = {
                key: deque(maxlen=self.track_vote_window) for key in attrs.keys()
            }

        for key, value in attrs.items():
            self.track_attribute_votes[track_id][key].append(value)

        aggregated = {}
        for key, history in self.track_attribute_votes[track_id].items():
            if not history:
                continue
            aggregated[key] = Counter(history).most_common(1)[0][0]

        if not aggregated.get("helmet_present", False):
            aggregated["helmet_color"] = "không có mũ"

        self.last_attributes[track_id] = aggregated
        return aggregated

    def compute_direction(self, history, window=5):
        if len(history) < window:
            return "unknown"

        cx_prev, cy_prev, _, h_prev = history[-window]
        cx_curr, cy_curr, _, h_curr = history[-1]
        height_ratio = h_curr / max(h_prev, 1e-6)
        delta_y = cy_curr - cy_prev
        delta_x = abs(cx_curr - cx_prev)

        if height_ratio > 1.1 or delta_y > 12:
            return "đi từ xa về gần camera"
        if height_ratio < 0.9 or delta_y < -12:
            return "đi ra xa camera"
        if delta_x > 10:
            return "đi ngang qua"
        return "khó xác định"

    def _is_in_roi(self, box):
        if self.roi_polygon is None:
            return True
        x1, y1, x2, y2 = box
        bottom_center = (int((x1 + x2) / 2), int(y2))
        return cv2.pointPolygonTest(self.roi_polygon, bottom_center, False) >= 0

    def run_on_video(self, video_path):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Unable to open video: {video_path}")

        frame_id = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detections = self.detect(frame)
            tracks = self.tracker.update(detections, frame)
            if not tracks:
                frame_id += 1
                continue

            for track in tracks:
                x1, y1, x2, y2, tid, conf, cls_id = track
                cls_id = int(cls_id)
                if cls_id != 0:
                    continue

                clipped_box = self._clip_box([x1, y1, x2, y2], frame.shape)
                if clipped_box[2] <= clipped_box[0] or clipped_box[3] <= clipped_box[1]:
                    continue
                if not self._is_in_roi(clipped_box):
                    continue

                x1_i, y1_i, x2_i, y2_i = clipped_box
                crop = frame[y1_i:y2_i, x1_i:x2_i]
                if crop.size == 0:
                    continue

                cx = (x1_i + x2_i) / 2
                cy = (y1_i + y2_i) / 2
                w = x2_i - x1_i
                h = y2_i - y1_i
                self.track_history[int(tid)].append((cx, cy, w, h))

                attrs = self.last_attributes.get(int(tid), self._default_attributes())
                should_refresh_attributes = (
                    self.attr_model is not None
                    and (frame_id % self.attribute_every_n_frames == 0 or int(tid) not in self.last_attributes)
                )
                if should_refresh_attributes:
                    fresh_attrs = self.get_attributes(crop)
                    if fresh_attrs is not None:
                        attrs = self._update_attribute_votes(int(tid), fresh_attrs)

                direction = self.compute_direction(self.track_history[int(tid)])
                class_name = self.names[cls_id] if isinstance(self.names, (list, tuple)) else self.names.get(cls_id, str(cls_id))
                result = {
                    "track_id": int(tid),
                    "bbox": [x1_i, y1_i, x2_i, y2_i],
                    "class": class_name,
                    "confidence": float(conf),
                    **attrs,
                    "direction": direction,
                }
                print(result)

            frame_id += 1

        cap.release()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    pipeline = InferencePipeline(
        yolo_weights="models/yolo/best_yolo_v2.pt",
        attribute_weights="models/attribute/attribute_model.pth",
        label_encoder_path="models/attribute/label_encoders.pkl",
        device="cuda" if torch.cuda.is_available() else "cpu",
        roi_path="data/meta/roi.json",
    )
    pipeline.run_on_video("path/to/video.mp4")
