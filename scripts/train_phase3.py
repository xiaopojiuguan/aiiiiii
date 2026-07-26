#!/usr/bin/env python3
"""SteelGuard-YOLO 阶段3 —— 从 baseline 微调 + 真 QFL + 消融实验

用法:
  # 完整 Phase 3 (QFL + HN + CP)
  python scripts/train_phase3.py --qfl --hn --cp

  # 消融: 仅 QFL
  python scripts/train_phase3.py --qfl

  # 消融: 仅 HN (需先生成 HN 数据)
  python scripts/train_phase3.py --hn

  # 消融: 仅 CP (需先生成 CP 数据)
  python scripts/train_phase3.py --cp

  # 消融: 仅从 baseline 继续训练 (无改动)
  python scripts/train_phase3.py
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import yaml, json, logging, argparse, platform, shutil
from datetime import datetime
import torch
import numpy as np
from ultralytics import YOLO

from src.paths import (
    PROJECT_ROOT, CONFIG_DIR, CHECKPOINT_DIR, LOG_DIR,
    CLASS_NAMES, TILE_SIZE, DATA_ROOT, SPLIT_DIR,
)

logger = logging.getLogger(__name__)

BASELINE_WEIGHTS = CHECKPOINT_DIR / "baseline_p2_20260725_153654" / "weights" / "best.pt"


def print_env():
    info = {"Python": platform.python_version(), "PyTorch": torch.__version__}
    if torch.cuda.is_available():
        info["GPU"] = torch.cuda.get_device_name(0)
        info["VRAM"] = f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB"
    for k, v in info.items():
        logger.info(f"  {k}: {v}")


def check_data_consistency(img_dir: Path, lbl_dir: Path):
    """训练前检查数据一致性。"""
    # 先清理 stale cache，防止 CP/HN 数据删除后缓存未更新
    for cache_file in [img_dir.parent / "labels.cache", lbl_dir.parent / "labels.cache"]:
        cache_path = lbl_dir.parent / "labels.cache"
        if cache_path.exists():
            cache_path.unlink()
            logger.info(f"Cleaned stale cache: {cache_path}")

    imgs = set(p.stem for p in img_dir.glob("*.jpg"))
    lbls = set(p.stem for p in lbl_dir.glob("*.txt"))

    orphan_img = imgs - lbls
    orphan_lbl = lbls - imgs

    if orphan_img:
        logger.warning(f"Orphan images (no label): {len(orphan_img)}")
        for s in sorted(orphan_img)[:5]:
            logger.warning(f"  {s}.jpg")
    if orphan_lbl:
        logger.warning(f"Orphan labels (no image): {len(orphan_lbl)}")
        for s in sorted(orphan_lbl)[:5]:
            logger.warning(f"  {s}.txt")

    # 检查 CP/HN 文件
    cp_count = sum(1 for s in imgs if s.startswith("cp_"))
    hn_count = sum(1 for s in imgs if s.startswith("hn_"))
    logger.info(f"Images: {len(imgs)} (base + {cp_count} CP + {hn_count} HN)")
    logger.info(f"Labels: {len(lbls)}")

    return len(orphan_img) == 0 and len(orphan_lbl) == 0


def compute_class_weights() -> list:
    """从训练集计算逆频率类别权重。"""
    counts = {}
    lbl_dir = DATA_ROOT.parent / "tiles_train" / "labels"
    for lbl_path in lbl_dir.glob("*.txt"):
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    cid = int(parts[0])
                    counts[cid] = counts.get(cid, 0) + 1

    nc = len(CLASS_NAMES)
    total = sum(counts.values())
    alpha = 0.5
    weights = np.ones(nc, dtype=np.float32)
    for cid in range(nc):
        n_c = counts.get(cid, 1)
        w = (total / (nc * n_c)) ** alpha
        weights[cid] = w

    # 归一化 mean=1
    weights /= weights.mean()
    weights = np.clip(weights, 0.3, 3.0)

    logger.info("Class weights (inverse freq, alpha=0.5):")
    for cid, name in enumerate(CLASS_NAMES):
        logger.info(f"  {name:<15s}: {weights[cid]:.2f} (n={counts.get(cid,0)})")

    return weights.tolist()


def train_phase3(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 实验名
    parts = ["phase3"]
    if args.qfl: parts.append("qfl")
    if args.hn: parts.append("hn")
    if args.cp: parts.append("cp")
    if len(parts) == 1: parts.append("continue")
    exp_name = "_".join(parts) + f"_{timestamp}"

    exp_dir = CHECKPOINT_DIR / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_DIR / f"{exp_name}.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    print_env()

    # ---- 数据检查 ----
    tileset_img = DATA_ROOT.parent / "tiles_train" / "images"
    tileset_lbl = DATA_ROOT.parent / "tiles_train" / "labels"
    ok = check_data_consistency(tileset_img, tileset_lbl)
    if not ok:
        logger.warning("Data consistency issues found, but continuing training")

    # ---- QFL 类别权重 ----
    cls_weights = None
    if args.qfl:
        cls_weights = compute_class_weights()

    # ---- 从 Baseline 加载模型 ----
    if BASELINE_WEIGHTS.exists():
        logger.info(f"Loading baseline weights: {BASELINE_WEIGHTS}")
        model = YOLO(str(CONFIG_DIR / "yolo11m-p2.yaml")).load(str(BASELINE_WEIGHTS))
        logger.info("Model loaded from baseline checkpoint")
    else:
        logger.error(f"Baseline weights not found: {BASELINE_WEIGHTS}")
        return

    # ---- QFL 类别权重接入 ----
    # ultralytics 的 set_class_weights() 在 cls_pw>0 时自动计算 (1/counts)^cls_pw
    # 我们设 cls_pw=0 跳过自动计算，然后手动设 class_weights 由 v8DetectionLoss 读取
    # 原理: utils/loss.py:360 getattr(model, "class_weights") → bce_loss *= class_weights
    if cls_weights is not None:
        try:
            weights_tensor = torch.tensor(cls_weights, dtype=torch.float32)
            # 关键: cls_pw=0 → ultralytics 的 set_class_weights() 跳过自动计算
            model.model.args["cls_pw"] = 0.0
            # 手动设置 class_weights → v8DetectionLoss.__init__ 读取并应用于 BCE loss
            model.model.class_weights = weights_tensor
            logger.info(f"QFL class_weights (via BCE weighting): {[f'{w:.2f}' for w in cls_weights]}")
        except Exception as e:
            logger.warning(f"Failed to set class_weights: {e}")

    # ---- 训练参数 ----
    epochs = 40  # 从 baseline 微调，不需要太多 epoch
    batch = 2
    close_mosaic = int(epochs * 0.7)

    train_args = {
        "data": str(PROJECT_ROOT / "tile_dataset.yaml"),
        "epochs": epochs,
        "imgsz": TILE_SIZE,
        "batch": batch,
        "optimizer": "AdamW",
        "lr0": 5e-5,               # 微调用更低 LR
        "weight_decay": 5e-4,
        "warmup_epochs": 1,
        "cos_lr": True,
        "amp": True,
        "patience": 12,
        "project": str(CHECKPOINT_DIR),
        "name": exp_dir.name,
        "exist_ok": True,
        "pretrained": True,
        "verbose": True,
        "seed": 42,
        "device": args.device or (0 if torch.cuda.is_available() else "cpu"),
        "workers": 0,
        "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.3,
        "degrees": 0.0, "translate": 0.1, "scale": 0.2,
        "shear": 0.0, "perspective": 0.0,
        "flipud": 0.0, "fliplr": 0.5,
        "mosaic": 0.3,
        "mixup": 0.0,
        "close_mosaic": close_mosaic,
    }

    logger.info(f"Experiment: {exp_name}")
    logger.info(f"Modules: QFL={args.qfl}, HN={args.hn}, CP={args.cp}")
    logger.info(f"Training: {epochs} epochs, batch={batch}, lr={train_args['lr0']}")
    logger.info(f"Baseline: {BASELINE_WEIGHTS}")

    results = model.train(**train_args)

    best_pt = exp_dir / "weights" / "best.pt"
    logger.info(f"Complete. Best: {best_pt}")

    # Final validation
    val_results = model.val(data=str(PROJECT_ROOT / "tile_dataset.yaml"), split="val", imgsz=TILE_SIZE)
    logger.info(f"Validation: {val_results}")

    return model, results


def main():
    parser = argparse.ArgumentParser(description="SteelGuard-YOLO Phase 3")
    parser.add_argument("--qfl", action="store_true", help="Enable QFL class re-weighting")
    parser.add_argument("--hn", action="store_true", help="Use Hard Negative tiles")
    parser.add_argument("--cp", action="store_true", help="Use Copy-Paste tiles")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if not any([args.qfl, args.hn, args.cp]):
        logger.warning("No modules enabled — training from baseline with no changes (control experiment)")

    train_phase3(args)


if __name__ == "__main__":
    main()
