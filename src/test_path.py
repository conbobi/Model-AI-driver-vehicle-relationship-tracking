import json
import os
from pathlib import Path

base_event = Path("data/event_clips/633_NguyenChiThanh/Ngay11-03-2026/cam01")
base_ann   = Path("data/ann/633_NguyenChiThanh/Ngay11-03-2026/cam01")

splits = ["174", "216", "231"]

missing = []
for split in splits:
    ann_file = base_ann / split / "annotations.jsonl"
    if not ann_file.exists():
        print(f"❌ Không tìm thấy annotation file: {ann_file}")
        continue
    with open(ann_file) as f:
        for line in f:
            ann = json.loads(line)
            vid_name = ann["image_path"]
            vid_path = base_event / split / vid_name
            if not vid_path.exists():
                missing.append(str(vid_path))
                print(f"❌ Thiếu video: {vid_path}")
            else:
                print(f"✅ {vid_path}")

if missing:
    print(f"\n⚠️ Có {len(missing)} video bị thiếu. Cần bổ sung trước khi train.")
else:
    print("\n🎉 Tất cả video đều tồn tại. Bạn có thể viết script train ngay!")