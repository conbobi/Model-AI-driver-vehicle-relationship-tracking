# Moment-DETR Training Pipeline — CCTV Video Moment Retrieval

Pipeline để train Moment-DETR trên dataset CCTV camera với queries tiếng Việt và tiếng Anh.

## Yêu cầu

```bash
# Core dependencies
pip install torch torchvision numpy opencv-python Pillow scipy tqdm

# CLIP (OpenAI)
pip install git+https://github.com/openai/CLIP.git

# Tùy chọn: TensorBoard
pip install tensorboard
```

## Quy trình sử dụng

### Bước 1: Chuẩn bị dữ liệu

Đọc tất cả `annotations.jsonl` dưới `--ann_root`, chia train/val 80/20 theo `clip_id`, chuyển sang QVHighlights format.

```bash
python src/model_train/prepare_data.py \
    --ann_root   data/ann/633_NguyenChiThanh/Ngay11-03-2026/cam01 \
    --output_dir data/moment_detr \
    --seed 42
```

**Output:**
- `data/moment_detr/train.jsonl`
- `data/moment_detr/val.jsonl`
- `data/moment_detr/vid_split_map.json`

### Bước 2: Trích xuất features

Dùng CLIP ViT-B/32 trích xuất video features (1 fps) và text features cho cả tiếng Anh và tiếng Việt.

```bash
python src/model_train/extract_features.py \
    --data_dir   data/moment_detr \
    --event_root data/event_clips/633_NguyenChiThanh/Ngay11-03-2026/cam01 \
    --output_dir data/features \
    --sample_fps 1.0 \
    --device cuda
```

**Output structure:**
```
data/features/
├── train/
│   ├── video/          # {clip_id}.npy — shape (T, 512)
│   ├── text_en/        # {qid}.npy — shape (512,)
│   └── text_vi/        # {qid}.npy — shape (512,)
└── val/
    ├── video/
    ├── text_en/
    └── text_vi/
```

### Bước 3: Training

#### Train với English queries:
```bash
python src/model_train/train.py \
    --data_root    data/moment_detr \
    --feature_root data/features \
    --lang en \
    --epochs 50 \
    --batch_size 32 \
    --lr 1e-4 \
    --output_dir checkpoints/moment_detr_en
```

#### Train với Vietnamese queries:
```bash
python src/model_train/train.py \
    --data_root    data/moment_detr \
    --feature_root data/features \
    --lang vi \
    --epochs 50 \
    --batch_size 32 \
    --lr 1e-4 \
    --output_dir checkpoints/moment_detr_vi
```

## Tham số quan trọng

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--lang` | `en` | Ngôn ngữ query: `en` hoặc `vi` |
| `--epochs` | `50` | Số epoch training |
| `--batch_size` | `32` | Batch size (tự giảm nếu dataset nhỏ) |
| `--lr` | `1e-4` | Learning rate |
| `--dropout` | `0.2` | Dropout rate (cao hơn vì dataset nhỏ) |
| `--label_smoothing` | `0.1` | Label smoothing cho class loss |
| `--patience` | `10` | Early stopping patience (0 = tắt) |
| `--num_queries` | `10` | Số moment queries |
| `--d_model` | `256` | Transformer hidden dim |
| `--max_v_len` | `75` | Max video sequence length |

## Metrics

- **R1@0.5**: Recall@1 với IoU ≥ 0.5
- **R1@0.7**: Recall@1 với IoU ≥ 0.7
- **R5@0.5**: Recall@5 với IoU ≥ 0.5
- **R5@0.7**: Recall@5 với IoU ≥ 0.7
- **mIoU**: Mean IoU trên toàn bộ queries

## Ghi chú

- Dataset nhỏ (~1500 annotations): code đã tích hợp dropout cao, label smoothing, early stopping để hạn chế overfitting.
- Video features được extract offline (1 fps) và cache trong `.npy` files.
- Training script **không** extract features on-the-fly.
- Saliency scores tính dựa trên overlap giữa segment và các clip 2-giây (score=5 nếu overlap, score=1 nếu không).
- Train/val split ở mức `clip_id` — không có data leakage giữa 2 tập.
