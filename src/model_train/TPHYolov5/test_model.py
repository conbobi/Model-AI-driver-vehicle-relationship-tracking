import torch
import cv2
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TPHYolov5.models.experimental import attempt_load
from TPHYolov5.utils.datasets import letterbox
from TPHYolov5.utils.general import non_max_suppression, scale_coords
from TPHYolov5.utils.torch_utils import select_device

# ======================
# CONFIG
# ======================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
weights = os.path.join(BASE_DIR, 'data/yolo_dataset_verify/best.pt')
img_size = 640
conf_thres = 0.25
iou_thres = 0.45

# ======================
# LOAD MODEL
# ======================
device = select_device('0' if torch.cuda.is_available() else 'cpu')  # 'cpu' nếu không có GPU
model = attempt_load(weights, map_location=device)
model.eval()

stride = int(model.stride.max())
names = model.names

# ======================
# LOAD IMAGE
# ======================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))

img_path = os.path.join(BASE_DIR, 'data/yolo_dataset_verify/train/images/174_e000004_1.44.jpg')
img0 = cv2.imread(img_path)
img = letterbox(img0, img_size, stride=stride)[0]

# Convert
img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR -> RGB
img = np.ascontiguousarray(img)

img = torch.from_numpy(img).to(device)
img = img.float() / 255.0
if img.ndimension() == 3:
    img = img.unsqueeze(0)

# ======================
# INFERENCE
# ======================
with torch.no_grad():
    pred = model(img)[0]

# ======================
# NMS
# ======================
pred = non_max_suppression(pred, conf_thres, iou_thres)

# ======================
# PROCESS RESULT
# ======================
for det in pred:
    if len(det):
        # scale về ảnh gốc
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

        for *xyxy, conf, cls in det:
            label = f"{names[int(cls)]} {conf:.2f}"
            print(label, xyxy)

            # vẽ bbox
            x1, y1, x2, y2 = map(int, xyxy)
            cv2.rectangle(img0, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(img0, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

# ======================
# SHOW
# ======================
cv2.imshow("result", img0)
cv2.waitKey(0)
cv2.destroyAllWindows()