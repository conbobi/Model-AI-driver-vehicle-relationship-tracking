# src/extract_features.py
import os
import torch
import cv2
import numpy as np

from src.model_train.TPHYolov5.models.experimental import attempt_load
from src.model_train.TPHYolov5.utils.datasets import letterbox
from src.model_train.TPHYolov5.utils.general import non_max_suppression, scale_coords
from src.model_train.TPHYolov5.utils.torch_utils import select_device

# ================= CONFIG =================
weights = "data/yolo_dataset_verify/best.pt"
image_dir = "data/yolo_dataset_verify/train/images"
save_dir = "data/crops"

img_size = 640
conf_thres = 0.3
iou_thres = 0.45

os.makedirs(save_dir, exist_ok=True)

# ================= LOAD MODEL =================
device = select_device('cpu')
model = attempt_load(weights, map_location=device)
model.eval()

stride = int(model.stride.max())

# ================= PROCESS =================
for img_name in os.listdir(image_dir):
    img_path = os.path.join(image_dir, img_name)
    img0 = cv2.imread(img_path)

    if img0 is None:
        continue

    img = letterbox(img0, img_size, stride=stride)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device).float() / 255.0
    img = img.unsqueeze(0)

    with torch.no_grad():
        pred = model(img)[0]

    pred = non_max_suppression(pred, conf_thres, iou_thres)

    for i, det in enumerate(pred):
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

            for j, (*xyxy, conf, cls) in enumerate(det):
                x1, y1, x2, y2 = map(int, xyxy)

                crop = img0[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                save_path = os.path.join(save_dir, f"{img_name}_{j}.jpg")
                cv2.imwrite(save_path, crop)

print("Done extracting crops!")