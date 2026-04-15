#!/usr/bin/env python3
"""
train.py — Moment-DETR: Video Moment Retrieval with Transformers
=================================================================
Triển khai đầy đủ kiến trúc Moment-DETR cho hệ thống CCTV_VMR,
bao gồm: Model, Dataset, Training loop, Evaluation.

Hỗ trợ training với queries Tiếng Anh hoặc Tiếng Việt qua flag --lang.

Kiến trúc:
    Text Query (CLIP) ──┐
                         ├── Transformer Encoder ── Decoder ── Predictions
    Video Feat (CLIP) ──┘        (cross-attn)      (queries)    ├── moments [center, width]
                                                                ├── class (fg/bg)
                                                                └── saliency scores

Cách chạy:
    # Train với English queries
    python src/model_train/train.py \
        --data_root data/moment_detr \
        --feature_root data/features \
        --lang en \
        --epochs 50 --batch_size 32 --lr 1e-4 \
        --output_dir checkpoints/moment_detr_en

    # Train với Vietnamese queries
    python src/model_train/train.py \
        --data_root data/moment_detr \
        --feature_root data/features \
        --lang vi \
        --epochs 50 --batch_size 32 --lr 1e-4 \
        --output_dir checkpoints/moment_detr_vi
"""

import argparse
import json
import logging
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


# ══════════════════════════════════════════════════════════════════════
#  1. REPRODUCIBILITY
# ══════════════════════════════════════════════════════════════════════


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")


# ══════════════════════════════════════════════════════════════════════
#  2. DATASET
# ══════════════════════════════════════════════════════════════════════


class QVHighlightsDataset(Dataset):
    """
    Dataset cho Moment-DETR, đọc pre-extracted CLIP features.

    Parameters
    ----------
    data_file : str — path tới train.jsonl hoặc val.jsonl
    feature_root : str — thư mục gốc features (chứa {split}/video/, {split}/text_en/, etc.)
    split : str — 'train' hoặc 'val'
    lang : str — 'en' hoặc 'vi' (chọn text features)
    max_v_len : int — max video length (pad/truncate)
    feat_dim : int — CLIP feature dimension (512)

    Mỗi sample gồm:
      - video_feat  : (max_v_len, feat_dim) — CLIP video features, padded
      - text_feat   : (feat_dim,) — CLIP text feature
      - moment      : (2,) — [center, width] normalized [0, 1]
      - saliency    : (max_v_len,) — saliency scores, padded
      - video_mask  : (max_v_len,) — 1 for real, 0 for padding
      - duration    : float
    """

    def __init__(
        self,
        data_file: str,
        feature_root: str,
        split: str,
        lang: str = "en",
        max_v_len: int = 75,
        feat_dim: int = 512,
    ):
        self.feature_root = Path(feature_root)
        self.split = split
        self.lang = lang
        self.max_v_len = max_v_len
        self.feat_dim = feat_dim

        # Text feature subdirectory
        self.text_subdir = f"text_{lang}"  # "text_en" or "text_vi"

        # Load entries from JSONL
        self.entries = []
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.entries.append(json.loads(line))

        # Filter: only keep entries with both video + text features
        valid_entries = []
        n_missing_video = 0
        n_missing_text = 0

        for e in self.entries:
            vid = e["vid"]
            qid = e["qid"]

            video_path = self.feature_root / split / "video" / f"{vid}.npy"
            text_path = self.feature_root / split / self.text_subdir / f"{qid}.npy"

            if not video_path.exists():
                n_missing_video += 1
                continue
            if not text_path.exists():
                n_missing_text += 1
                continue

            valid_entries.append(e)

        if n_missing_video > 0:
            logger.warning(
                f"[{split}] Dropped {n_missing_video} entries: missing video features"
            )
        if n_missing_text > 0:
            logger.warning(
                f"[{split}] Dropped {n_missing_text} entries: missing {self.text_subdir} features"
            )

        self.entries = valid_entries
        logger.info(
            f"[{split}] Dataset: {len(self.entries)} valid entries "
            f"(lang={lang})"
        )

        # Cache video features (keyed by vid, shared across queries)
        self._video_cache = {}

    def __len__(self):
        return len(self.entries)

    def _load_video_feat(self, vid: str) -> np.ndarray:
        """Load video features with caching."""
        if vid not in self._video_cache:
            path = self.feature_root / self.split / "video" / f"{vid}.npy"
            self._video_cache[vid] = np.load(str(path))
        return self._video_cache[vid]

    def _load_text_feat(self, qid: int) -> np.ndarray:
        """Load text feature for given qid."""
        path = self.feature_root / self.split / self.text_subdir / f"{qid}.npy"
        return np.load(str(path))

    def __getitem__(self, idx):
        entry = self.entries[idx]
        vid = entry["vid"]
        qid = entry["qid"]
        duration = entry["duration"]

        # ── Video features ──
        v_feat = self._load_video_feat(vid)  # (T, feat_dim)
        T = v_feat.shape[0]

        video_feat = np.zeros((self.max_v_len, self.feat_dim), dtype=np.float32)
        video_mask = np.zeros(self.max_v_len, dtype=np.float32)
        actual_len = min(T, self.max_v_len)
        video_feat[:actual_len] = v_feat[:actual_len]
        video_mask[:actual_len] = 1.0

        # ── Text features ──
        text_feat = self._load_text_feat(qid)  # (feat_dim,)

        # ── Moment target ──
        windows = entry.get("relevant_windows", [[0, duration]])
        start, end = windows[0][0], windows[0][1]

        if duration > 0:
            center = (start + end) / 2.0 / duration
            width = (end - start) / duration
        else:
            center, width = 0.5, 1.0

        center = np.clip(center, 0, 1)
        width = np.clip(width, 0, 1)
        moment = np.array([center, width], dtype=np.float32)

        # ── Saliency scores ──
        saliency_raw = entry.get("saliency_scores", [[]])
        sal = saliency_raw[0] if saliency_raw else []

        saliency = np.zeros(self.max_v_len, dtype=np.float32)
        for i, s in enumerate(sal):
            if i < self.max_v_len:
                saliency[i] = s / 5.0  # Normalize to 0-1

        return {
            "video_feat": torch.from_numpy(video_feat),
            "video_mask": torch.from_numpy(video_mask),
            "text_feat": torch.from_numpy(text_feat),
            "moment": torch.from_numpy(moment),
            "saliency": torch.from_numpy(saliency),
            "duration": torch.tensor(duration, dtype=torch.float32),
            "qid": qid,
        }


# ══════════════════════════════════════════════════════════════════════
#  3. MODEL ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 200, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """x: (B, T, D)"""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class MLP(nn.Module):
    """Simple multi-layer perceptron."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.1):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_d = input_dim if i == 0 else hidden_dim
            out_d = output_dim if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_d, out_d))
            if i < num_layers - 1:
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MomentDETR(nn.Module):
    """
    Moment-DETR: DEtection TRansformer for Video Moment Retrieval.

    Architecture:
      1) Project video + text features to d_model
      2) Concatenate [text; video] → Transformer Encoder
      3) Learnable queries → Transformer Decoder (cross-attend to encoder)
      4) Prediction heads: moment [center, width], class, saliency
    """

    def __init__(
        self,
        feat_dim: int = 512,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        num_queries: int = 10,
        max_v_len: int = 75,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_queries = num_queries
        self.max_v_len = max_v_len

        # ── Input projections ──
        self.video_proj = nn.Sequential(
            nn.Linear(feat_dim, d_model),
            nn.ReLU(inplace=True),
            nn.LayerNorm(d_model),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(feat_dim, d_model),
            nn.ReLU(inplace=True),
            nn.LayerNorm(d_model),
        )

        # ── Positional encoding ──
        self.video_pos = PositionalEncoding(
            d_model, max_len=max_v_len + 10, dropout=dropout
        )
        self.text_pos = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # ── Type embeddings (video vs text) ──
        self.type_embed = nn.Embedding(2, d_model)  # 0=text, 1=video

        # ── Transformer Encoder ──
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        # ── Transformer Decoder ──
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers
        )

        # ── Learnable query embeddings ──
        self.query_embed = nn.Embedding(num_queries, d_model)

        # ── Prediction heads ──
        self.moment_head = MLP(d_model, d_model, 2, num_layers=3, dropout=dropout)
        self.class_head = nn.Linear(d_model, 2)  # [bg, fg]
        self.saliency_head = MLP(
            d_model, d_model // 2, 1, num_layers=2, dropout=dropout
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, video_feat, video_mask, text_feat):
        """
        Parameters
        ----------
        video_feat : (B, T, feat_dim)
        video_mask : (B, T) — 1 for real, 0 for padding
        text_feat  : (B, feat_dim)

        Returns
        -------
        dict with pred_moments, pred_logits, pred_saliency
        """
        B, T, _ = video_feat.shape

        # Project
        v = self.video_proj(video_feat)  # (B, T, d_model)
        t = self.text_proj(text_feat).unsqueeze(1)  # (B, 1, d_model)

        # Positional encoding
        v = self.video_pos(v)
        t = t + self.text_pos

        # Type embeddings
        t = t + self.type_embed(
            torch.zeros(B, 1, dtype=torch.long, device=v.device)
        )
        v = v + self.type_embed(
            torch.ones(B, T, dtype=torch.long, device=v.device)
        )

        # Concatenate [text; video]
        src = torch.cat([t, v], dim=1)  # (B, 1+T, d_model)

        # Padding mask (True = ignore)
        text_mask = torch.ones(B, 1, device=v.device)
        full_mask = torch.cat([text_mask, video_mask], dim=1)
        padding_mask = (full_mask == 0)

        # Encoder
        memory = self.transformer_encoder(
            src, src_key_padding_mask=padding_mask
        )

        # Decoder
        query_embed = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        tgt = torch.zeros_like(query_embed)
        decoder_out = self.transformer_decoder(
            tgt=tgt,
            memory=memory,
            memory_key_padding_mask=padding_mask,
        )
        decoder_out = decoder_out + query_embed

        # Predictions
        pred_moments = self.moment_head(decoder_out).sigmoid()  # (B, Q, 2)
        pred_logits = self.class_head(decoder_out)  # (B, Q, 2)
        video_memory = memory[:, 1:, :]
        pred_saliency = self.saliency_head(video_memory).squeeze(-1)  # (B, T)

        return {
            "pred_moments": pred_moments,
            "pred_logits": pred_logits,
            "pred_saliency": pred_saliency,
        }


# ══════════════════════════════════════════════════════════════════════
#  4. HUNGARIAN MATCHER
# ══════════════════════════════════════════════════════════════════════


def generalized_temporal_iou(pred, target):
    """
    Generalized Temporal IoU.

    pred   : (N, 2) — [center, width]
    target : (M, 2) — [center, width]

    Returns: iou (N, M), giou (N, M)
    """
    pred_s = pred[:, 0] - pred[:, 1] / 2
    pred_e = pred[:, 0] + pred[:, 1] / 2
    tgt_s = target[:, 0] - target[:, 1] / 2
    tgt_e = target[:, 0] + target[:, 1] / 2

    inter_s = torch.max(pred_s.unsqueeze(1), tgt_s.unsqueeze(0))
    inter_e = torch.min(pred_e.unsqueeze(1), tgt_e.unsqueeze(0))
    inter = (inter_e - inter_s).clamp(min=0)

    pred_len = (pred_e - pred_s).clamp(min=1e-6)
    tgt_len = (tgt_e - tgt_s).clamp(min=1e-6)
    union = pred_len.unsqueeze(1) + tgt_len.unsqueeze(0) - inter

    iou = inter / union.clamp(min=1e-6)

    enclose_s = torch.min(pred_s.unsqueeze(1), tgt_s.unsqueeze(0))
    enclose_e = torch.max(pred_e.unsqueeze(1), tgt_e.unsqueeze(0))
    enclose = (enclose_e - enclose_s).clamp(min=1e-6)

    giou = iou - (enclose - union) / enclose
    return iou, giou


class HungarianMatcher(nn.Module):
    """Bipartite matching using SciPy's linear_sum_assignment."""

    def __init__(self, cost_class=1.0, cost_span=5.0, cost_giou=2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_span = cost_span
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, outputs, targets):
        from scipy.optimize import linear_sum_assignment

        B, Q, _ = outputs["pred_moments"].shape
        indices = []

        for b in range(B):
            pred_mom = outputs["pred_moments"][b]
            pred_cls = outputs["pred_logits"][b].softmax(-1)
            tgt_mom = targets[b]["moments"]
            K = tgt_mom.shape[0]

            if K == 0:
                indices.append(([], []))
                continue

            cost_span = torch.cdist(pred_mom, tgt_mom, p=1)
            _, giou = generalized_temporal_iou(pred_mom, tgt_mom)
            cost_giou = -giou
            cost_class = -pred_cls[:, 1].unsqueeze(1).expand(-1, K)

            C = (
                self.cost_span * cost_span
                + self.cost_giou * cost_giou
                + self.cost_class * cost_class
            )

            row_ind, col_ind = linear_sum_assignment(C.cpu().numpy())
            indices.append(
                (
                    torch.tensor(row_ind, dtype=torch.long),
                    torch.tensor(col_ind, dtype=torch.long),
                )
            )

        return indices


# ══════════════════════════════════════════════════════════════════════
#  5. LOSS FUNCTIONS
# ══════════════════════════════════════════════════════════════════════


class MomentDETRLoss(nn.Module):
    """
    Combined loss for Moment-DETR:
      - Span loss (L1 + GIoU) on matched pairs
      - Classification loss (foreground vs background)
      - Saliency loss (per-clip saliency prediction)
    """

    def __init__(
        self,
        matcher: HungarianMatcher,
        weight_span: float = 5.0,
        weight_giou: float = 2.0,
        weight_class: float = 4.0,
        weight_saliency: float = 1.0,
        eos_coef: float = 0.1,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.matcher = matcher
        self.weight_span = weight_span
        self.weight_giou = weight_giou
        self.weight_class = weight_class
        self.weight_saliency = weight_saliency
        self.eos_coef = eos_coef
        self.label_smoothing = label_smoothing

    def forward(self, outputs, video_mask, moments_gt, saliency_gt):
        B = moments_gt.shape[0]
        device = moments_gt.device

        targets = [{"moments": moments_gt[b].unsqueeze(0)} for b in range(B)]
        indices = self.matcher(outputs, targets)

        # ── Span Loss (L1 + GIoU) ──
        loss_span = torch.tensor(0.0, device=device)
        loss_giou = torch.tensor(0.0, device=device)
        n_matches = 0

        for b, (pred_idx, tgt_idx) in enumerate(indices):
            if len(pred_idx) == 0:
                continue
            pred_mom = outputs["pred_moments"][b][pred_idx]
            tgt_mom = targets[b]["moments"][tgt_idx]

            loss_span = loss_span + F.l1_loss(pred_mom, tgt_mom, reduction="sum")
            _, giou = generalized_temporal_iou(pred_mom, tgt_mom)
            loss_giou = loss_giou + (1 - giou.diag()).sum()
            n_matches += len(pred_idx)

        if n_matches > 0:
            loss_span = loss_span / n_matches
            loss_giou = loss_giou / n_matches

        # ── Classification Loss ──
        pred_logits = outputs["pred_logits"]
        Q = pred_logits.shape[1]

        target_classes = torch.zeros(B, Q, dtype=torch.long, device=device)
        for b, (pred_idx, _) in enumerate(indices):
            if len(pred_idx) > 0:
                target_classes[b, pred_idx] = 1

        weight = torch.tensor([self.eos_coef, 1.0], device=device)
        loss_class = F.cross_entropy(
            pred_logits.reshape(-1, 2),
            target_classes.reshape(-1),
            weight=weight,
            label_smoothing=self.label_smoothing,
        )

        # ── Saliency Loss ──
        pred_sal = outputs["pred_saliency"]
        sal_mask = video_mask.bool()
        if sal_mask.any():
            loss_saliency = F.mse_loss(pred_sal[sal_mask], saliency_gt[sal_mask])
        else:
            loss_saliency = torch.tensor(0.0, device=device)

        # ── Total ──
        total_loss = (
            self.weight_span * loss_span
            + self.weight_giou * loss_giou
            + self.weight_class * loss_class
            + self.weight_saliency * loss_saliency
        )

        return {
            "loss_total": total_loss,
            "loss_span": loss_span,
            "loss_giou": loss_giou,
            "loss_class": loss_class,
            "loss_saliency": loss_saliency,
        }


# ══════════════════════════════════════════════════════════════════════
#  6. EVALUATION METRICS
# ══════════════════════════════════════════════════════════════════════


def compute_temporal_iou(pred_start, pred_end, gt_start, gt_end):
    """IoU between two temporal segments."""
    inter_start = max(pred_start, gt_start)
    inter_end = min(pred_end, gt_end)
    inter = max(0, inter_end - inter_start)

    pred_len = max(0, pred_end - pred_start)
    gt_len = max(0, gt_end - gt_start)
    union = pred_len + gt_len - inter

    return inter / union if union > 0 else 0.0


@torch.no_grad()
def evaluate(model, dataloader, device, verbose=True):
    """
    Evaluate Moment-DETR model.

    Metrics: Recall@1 (IoU≥0.5), Recall@1 (IoU≥0.7),
             Recall@5 (IoU≥0.5), Recall@5 (IoU≥0.7), Mean IoU
    """
    model.eval()

    all_ious = []
    all_ious_top5 = []
    n_samples = 0

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        video_feat = batch["video_feat"].to(device)
        video_mask = batch["video_mask"].to(device)
        text_feat = batch["text_feat"].to(device)
        moment_gt = batch["moment"].to(device)
        duration = batch["duration"]

        outputs = model(video_feat, video_mask, text_feat)

        pred_moments = outputs["pred_moments"]
        pred_logits = outputs["pred_logits"]
        B, Q, _ = pred_moments.shape

        for b in range(B):
            dur = duration[b].item()

            gt_center = moment_gt[b, 0].item() * dur
            gt_width = moment_gt[b, 1].item() * dur
            gt_start = gt_center - gt_width / 2
            gt_end = gt_center + gt_width / 2

            fg_scores = pred_logits[b].softmax(-1)[:, 1]
            sorted_idx = fg_scores.argsort(descending=True)

            ious_topk = []
            for k in range(min(5, Q)):
                qi = sorted_idx[k].item()
                p_center = pred_moments[b, qi, 0].item() * dur
                p_width = pred_moments[b, qi, 1].item() * dur
                p_start = p_center - p_width / 2
                p_end = p_center + p_width / 2
                iou = compute_temporal_iou(p_start, p_end, gt_start, gt_end)
                ious_topk.append(iou)

            best_iou = max(ious_topk) if ious_topk else 0.0
            all_ious.append(best_iou)
            all_ious_top5.append(ious_topk)
            n_samples += 1

    all_ious = np.array(all_ious)

    metrics = {
        "R1@0.5": (all_ious >= 0.5).mean() * 100,
        "R1@0.7": (all_ious >= 0.7).mean() * 100,
        "mIoU": all_ious.mean() * 100,
        "n_samples": n_samples,
    }

    r5_05 = sum(1 for ious in all_ious_top5 if any(iou >= 0.5 for iou in ious))
    r5_07 = sum(1 for ious in all_ious_top5 if any(iou >= 0.7 for iou in ious))
    metrics["R5@0.5"] = r5_05 / max(1, n_samples) * 100
    metrics["R5@0.7"] = r5_07 / max(1, n_samples) * 100

    if verbose:
        logger.info(
            f"  R1@0.5: {metrics['R1@0.5']:.1f}%  |  "
            f"R1@0.7: {metrics['R1@0.7']:.1f}%"
        )
        logger.info(
            f"  R5@0.5: {metrics['R5@0.5']:.1f}%  |  "
            f"R5@0.7: {metrics['R5@0.7']:.1f}%"
        )
        logger.info(
            f"  mIoU:   {metrics['mIoU']:.1f}%    |  N={n_samples}"
        )

    model.train()
    return metrics


# ══════════════════════════════════════════════════════════════════════
#  7. TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════


def train(args):
    """Main training function."""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Language: {args.lang}")

    set_seed(args.seed)

    # ── Paths ──
    data_root = Path(args.data_root)
    feature_root = Path(args.feature_root)
    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # ── Dataset ──
    logger.info("Loading datasets...")

    train_dataset = QVHighlightsDataset(
        data_file=str(data_root / "train.jsonl"),
        feature_root=str(feature_root),
        split="train",
        lang=args.lang,
        max_v_len=args.max_v_len,
        feat_dim=args.feat_dim,
    )
    val_dataset = QVHighlightsDataset(
        data_file=str(data_root / "val.jsonl"),
        feature_root=str(feature_root),
        split="val",
        lang=args.lang,
        max_v_len=args.max_v_len,
        feat_dim=args.feat_dim,
    )

    # Adjust batch size for small datasets
    effective_batch_size = min(args.batch_size, len(train_dataset))
    if effective_batch_size < args.batch_size:
        logger.warning(
            f"Reducing batch_size from {args.batch_size} to {effective_batch_size} "
            f"(dataset has only {len(train_dataset)} samples)"
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=len(train_dataset) > effective_batch_size,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=min(args.batch_size, max(1, len(val_dataset))),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    logger.info(
        f"Train: {len(train_dataset)} samples, {len(train_loader)} batches"
    )
    logger.info(
        f"Val:   {len(val_dataset)} samples, {len(val_loader)} batches"
    )

    # ── Model ──
    logger.info("Building Moment-DETR model...")
    model = MomentDETR(
        feat_dim=args.feat_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        num_queries=args.num_queries,
        max_v_len=args.max_v_len,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parameters: {n_params:,} ({n_params / 1e6:.1f}M)")

    # ── Matcher & Loss ──
    matcher = HungarianMatcher(
        cost_class=args.cost_class,
        cost_span=args.cost_span,
        cost_giou=args.cost_giou,
    )
    criterion = MomentDETRLoss(
        matcher=matcher,
        weight_span=args.weight_span,
        weight_giou=args.weight_giou,
        weight_class=args.weight_class,
        weight_saliency=args.weight_saliency,
        label_smoothing=args.label_smoothing,
    )

    # ── Optimizer & Scheduler ──
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # ── TensorBoard ──
    writer = None
    if HAS_TENSORBOARD:
        writer = SummaryWriter(str(log_dir))
        logger.info(f"TensorBoard: tensorboard --logdir {log_dir}")

    # ── Resume ──
    start_epoch = 0
    best_metric = 0.0
    if args.resume and os.path.exists(args.resume):
        logger.info(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_metric = ckpt.get("best_metric", 0.0)
        logger.info(
            f"Resumed from epoch {start_epoch}, best R1@0.5={best_metric:.1f}%"
        )

    # ── Early stopping ──
    patience = args.patience
    patience_counter = 0

    # ══════════════════════════════════════════════════════════════
    #  Training Loop
    # ══════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info(f"Starting Training (lang={args.lang})")
    logger.info("=" * 60)

    model.train()
    global_step = start_epoch * len(train_loader)

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        loss_accum = defaultdict(float)
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for batch in pbar:
            video_feat = batch["video_feat"].to(device)
            video_mask = batch["video_mask"].to(device)
            text_feat = batch["text_feat"].to(device)
            moment_gt = batch["moment"].to(device)
            saliency_gt = batch["saliency"].to(device)

            # Forward + Loss
            outputs = model(video_feat, video_mask, text_feat)
            loss_dict = criterion(outputs, video_mask, moment_gt, saliency_gt)
            loss = loss_dict["loss_total"]

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.clip_grad
                )

            optimizer.step()

            # Accumulate
            for k, v in loss_dict.items():
                loss_accum[k] += v.item()
            n_batches += 1
            global_step += 1

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                span=f"{loss_dict['loss_span'].item():.3f}",
                giou=f"{loss_dict['loss_giou'].item():.3f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

            # TensorBoard
            if writer and global_step % 10 == 0:
                for k, v in loss_dict.items():
                    writer.add_scalar(f"train/{k}", v.item(), global_step)
                writer.add_scalar(
                    "train/lr", optimizer.param_groups[0]["lr"], global_step
                )

        # ── End of epoch ──
        scheduler.step()
        epoch_time = time.time() - epoch_start

        avg_losses = {k: v / max(1, n_batches) for k, v in loss_accum.items()}
        logger.info(
            f"Epoch {epoch + 1}/{args.epochs} ({epoch_time:.0f}s) | "
            f"Loss: {avg_losses.get('loss_total', 0):.4f} | "
            f"Span: {avg_losses.get('loss_span', 0):.3f} | "
            f"GIoU: {avg_losses.get('loss_giou', 0):.3f} | "
            f"Class: {avg_losses.get('loss_class', 0):.3f} | "
            f"Sal: {avg_losses.get('loss_saliency', 0):.3f}"
        )

        # ── Validation ──
        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            logger.info(f"Evaluating at epoch {epoch + 1}...")
            metrics = evaluate(model, val_loader, device)

            if writer:
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        writer.add_scalar(f"val/{k}", v, epoch + 1)

            # Best model
            current_metric = metrics["R1@0.5"]
            if current_metric > best_metric:
                best_metric = current_metric
                patience_counter = 0
                best_path = ckpt_dir / "best_model.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "best_metric": best_metric,
                        "metrics": metrics,
                        "args": vars(args),
                    },
                    str(best_path),
                )
                logger.info(
                    f"New best! R1@0.5={best_metric:.1f}% → {best_path}"
                )
            else:
                patience_counter += 1
                if patience > 0 and patience_counter >= patience:
                    logger.info(
                        f"Early stopping: no improvement for {patience} eval cycles"
                    )
                    break

        # ── Periodic checkpoint ──
        if (epoch + 1) % args.save_every == 0:
            ckpt_path = ckpt_dir / f"epoch_{epoch + 1:04d}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_metric": best_metric,
                    "args": vars(args),
                },
                str(ckpt_path),
            )
            logger.info(f"Checkpoint saved: {ckpt_path}")

    # ── Final ──
    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info(f"  Language    : {args.lang}")
    logger.info(f"  Best R1@0.5 : {best_metric:.1f}%")
    logger.info(f"  Checkpoints : {ckpt_dir}")
    if HAS_TENSORBOARD:
        logger.info(f"  TensorBoard : tensorboard --logdir {log_dir}")
    logger.info("=" * 60)

    if writer:
        writer.close()


# ══════════════════════════════════════════════════════════════════════
#  8. CLI
# ══════════════════════════════════════════════════════════════════════


def parse_args():
    parser = argparse.ArgumentParser(
        description="Moment-DETR Training cho CCTV Video Moment Retrieval",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Data ──
    data = parser.add_argument_group("Data")
    data.add_argument(
        "--data_root", type=str, default="data/moment_detr",
        help="Thư mục chứa train.jsonl, val.jsonl",
    )
    data.add_argument(
        "--feature_root", type=str, default="data/features",
        help="Thư mục chứa pre-extracted features",
    )
    data.add_argument(
        "--output_dir", type=str, default="checkpoints/moment_detr",
        help="Thư mục output (checkpoints, logs)",
    )
    data.add_argument(
        "--lang", type=str, default="en", choices=["en", "vi"],
        help="Ngôn ngữ query: 'en' (English) hoặc 'vi' (Vietnamese)",
    )
    data.add_argument(
        "--max_v_len", type=int, default=75,
        help="Max video feature length (pad/truncate)",
    )

    # ── Model ──
    model_g = parser.add_argument_group("Model")
    model_g.add_argument("--feat_dim", type=int, default=512)
    model_g.add_argument("--d_model", type=int, default=256)
    model_g.add_argument("--nhead", type=int, default=8)
    model_g.add_argument("--enc_layers", type=int, default=2)
    model_g.add_argument("--dec_layers", type=int, default=2)
    model_g.add_argument("--dim_feedforward", type=int, default=1024)
    model_g.add_argument("--dropout", type=float, default=0.2,
                          help="Dropout (0.2 for small datasets)")
    model_g.add_argument("--num_queries", type=int, default=10)

    # ── Training ──
    train_g = parser.add_argument_group("Training")
    train_g.add_argument("--epochs", type=int, default=50)
    train_g.add_argument("--batch_size", type=int, default=32)
    train_g.add_argument("--lr", type=float, default=1e-4)
    train_g.add_argument("--weight_decay", type=float, default=1e-4)
    train_g.add_argument("--clip_grad", type=float, default=0.1,
                          help="Gradient clipping max norm (0 = disabled)")
    train_g.add_argument("--num_workers", type=int, default=4)
    train_g.add_argument("--seed", type=int, default=42)
    train_g.add_argument("--device", type=str, default="cuda")
    train_g.add_argument("--label_smoothing", type=float, default=0.1,
                          help="Label smoothing for class loss (helps small datasets)")
    train_g.add_argument("--patience", type=int, default=10,
                          help="Early stopping patience (0 = disabled)")

    # ── Loss weights ──
    loss_g = parser.add_argument_group("Loss")
    loss_g.add_argument("--cost_class", type=float, default=1.0)
    loss_g.add_argument("--cost_span", type=float, default=5.0)
    loss_g.add_argument("--cost_giou", type=float, default=2.0)
    loss_g.add_argument("--weight_span", type=float, default=5.0)
    loss_g.add_argument("--weight_giou", type=float, default=2.0)
    loss_g.add_argument("--weight_class", type=float, default=4.0)
    loss_g.add_argument("--weight_saliency", type=float, default=1.0)

    # ── Checkpointing ──
    ckpt_g = parser.add_argument_group("Checkpointing")
    ckpt_g.add_argument("--eval_every", type=int, default=5)
    ckpt_g.add_argument("--save_every", type=int, default=10)
    ckpt_g.add_argument("--resume", type=str, default="",
                         help="Path to checkpoint to resume from")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
