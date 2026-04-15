import json
import os
from pathlib import Path
import torch
from torch.utils.data import Dataset
import cv2

class MultiSplitTrafficDataset(Dataset):
    def __init__(self, root_ann, root_event, splits):
        """
        root_ann:   thư mục gốc chứa annotation (vd: data/ann/.../cam01)
        root_event: thư mục gốc chứa video (vd: data/event_clips/.../cam01)
        splits:     danh sách các split_id (vd: ["174", "216", "231"])
        """
        self.samples = []
        for split in splits:
            ann_file = Path(root_ann) / split / "annotations.jsonl"
            if not ann_file.exists():
                print(f"⚠️ Bỏ qua split {split} vì không tìm thấy {ann_file}")
                continue
            video_dir = Path(root_event) / split
            with open(ann_file, 'r', encoding='utf-8') as f:
                for line in f:
                    ann = json.loads(line)
                    vid_name = ann["image_path"]
                    vid_path = video_dir / vid_name
                    if not vid_path.exists():
                        print(f"⚠️ Bỏ qua mẫu do thiếu video: {vid_path}")
                        continue
                    # Lưu đường dẫn tuyệt đối để tránh lỗi working directory
                    ann["_video_path"] = str(vid_path.resolve())
                    self.samples.append(ann)

        print(f"✅ Loaded {len(self.samples)} samples from {len(splits)} splits.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ann = self.samples[idx]
        video_path = ann["_video_path"]
        # Đọc video bằng OpenCV (hoặc torchvision)
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)  # BGR
        cap.release()
        # Chuyển đổi sang tensor, xử lý segment, bbox, class...
        # ...
        return frames, ann["class_name"], ann["bbox"], ann["query_en"]

# Sử dụng
dataset = MultiSplitTrafficDataset(
    root_ann="data/ann/633_NguyenChiThanh/Ngay11-03-2026/cam01",
    root_event="data/event_clips/633_NguyenChiThanh/Ngay11-03-2026/cam01",
    splits=["174", "216", "231"]
)