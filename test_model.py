#!/usr/bin/env python3
"""
Inference pipeline:
YOLOv5 (custom C3STR) + crop object

- Load best.pt (custom)
- Detect objects
- Crop objects
"""

from pathlib import Path
import torch
import cv2
import numpy as np

from src.model_train.TPHYolov5.models.common import DetectMultiBackend
from src.model_train.TPHYolov5.utils.general import non_max_suppression, scale_coords
from src.model_train.TPHYolov5.utils.torch_utils import select_device
from src.model_train.TPHYolov5.utils.augmentations import letterbox


class YOLODetector:
    def __init__(self, weights_path, device="cpu", imgsz=640):
        self.device = select_device(device)
        self.model = DetectMultiBackend(weights_path, device=self.device)

        self.stride = self.model.stride
        self.names = self.model.names
        self.imgsz = imgsz

        self.model.eval()

    def preprocess(self, img):
        img0 = img.copy()

        img = letterbox(img, self.imgsz, stride=self.stride, auto=True)[0]
        img = img.transpose((2, 0, 1))[::-1]  # BGR → RGB
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.float() / 255.0

        if len(img.shape) == 3:
            img = img[None]  # add batch dim

        return img, img0

    def infer(self, img):
        img_tensor, img0 = self.preprocess(img)

        with torch.no_grad():
            pred = self.model(img_tensor)

        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

        results = []

        for det in pred:
            if len(det):
                det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], img0.shape).round()

                for *xyxy, conf, cls in det:
                    x1, y1, x2, y2 = map(int, xyxy)
                    results.append({
                        "bbox": [x1, y1, x2, y2],
                        "conf": float(conf),
                        "class_id": int(cls),
                        "class_name": self.names[int(cls)]
                    })

        return results


def crop_objects(image, detections):
    crops = []

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        crops.append({
            "crop": crop,
            "meta": det
        })

    return crops


# 🔥 MAIN TEST
if __name__ == "__main__":
    weights = "data/yolo_dataset_verify/best.pt"
    image_path = "test.jpg"

    detector = YOLODetector(weights, device="0")

    img = cv2.imread(image_path)
    detections = detector.infer(img)

    print("\nDetections:")
    for d in detections:
        print(d)

    crops = crop_objects(img, detections)

    # save crops
    out_dir = Path("outputs/crops")
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, item in enumerate(crops):
        crop = item["crop"]
        meta = item["meta"]

        save_path = out_dir / f"{i}_{meta['class_name']}.jpg"
        cv2.imwrite(str(save_path), crop)

    print(f"\nSaved {len(crops)} crops to {out_dir}")
