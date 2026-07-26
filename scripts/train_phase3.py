#!/usr/bin/env python3
"""SteelGuard-YOLO 阶段3训练 —— QFL + Hard Negative + Copy-Paste

对应方案: T1(类别重加权) + T8(Hard Negative) + T3(Copy-Paste CP-1)
数据: tiles_train (18964基础 + 506 HN + 2000 CP = 21470张切片)
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import yaml, json, logging, argparse, platform
from datetime import datetime
import torch
import numpy as np
from ultralytics import YOLO

from src.paths import PROJECT_ROOT, CONFIG_DIR, CHECKPOINT_DIR, LOG_DIR, CLASS_NAMES, TILE_SIZE

logger = logging.getLogger(__name__)

def print_env_info():
    info = {
        "Python": platform.python_version(),
        "PyTorch": torch.__version__,
        "CUDA": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["GPU"] = torch.cuda.get_device_name(0)
        info["VRAM"] = f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB"
    for k,v in info.items():
        logger.info(f"  {k}: {v}")

def train_phase3(args):
    # Load class weights
    weights_path = PROJECT_ROOT / "outputs" / "class_weights.json"
    with open(weights_path) as f:
        class_weights = json.load(f)
    cls_weights = [class_weights[name] for name in CLASS_NAMES]
    logger.info(f"Class weights: {dict(zip(CLASS_NAMES, [f'{w:.2f}' for w in cls_weights]))}")

    # Timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"phase3_qfl_hn_cp_{timestamp}"
    exp_dir = CHECKPOINT_DIR / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Dataset
    data_yaml = PROJECT_ROOT / "tile_dataset.yaml"
    logger.info(f"Dataset: {data_yaml}")
    logger.info(f"Training data includes: base tiles + 506 HN + 2000 CP")

    # P2 model
    model_yaml = CONFIG_DIR / "yolo11m-p2.yaml"
    model = YOLO(str(model_yaml)).load("yolo11m.pt")
    logger.info("P2 model built + pretrained weights loaded")

    epochs = 60
    batch = 2
    close_mosaic_epoch = int(epochs * 0.7)

    train_args = {
        "data": str(data_yaml),
        "epochs": epochs,
        "imgsz": TILE_SIZE,
        "batch": batch,
        "optimizer": "AdamW",
        "lr0": 1e-4,           # lower LR since starting from baseline
        "weight_decay": 5e-4,
        "warmup_epochs": 2,
        "cos_lr": True,
        "amp": True,
        "patience": 15,
        "project": str(CHECKPOINT_DIR),
        "name": exp_dir.name,
        "exist_ok": True,
        "pretrained": True,
        "verbose": True,
        "seed": 42,
        "device": args.device or (0 if torch.cuda.is_available() else "cpu"),
        "workers": 0,

        # Augmentation
        "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.3,
        "degrees": 0.0, "translate": 0.1, "scale": 0.2,
        "shear": 0.0, "perspective": 0.0,
        "flipud": 0.0, "fliplr": 0.5,
        "mosaic": 0.3,        # reduced mosaic to avoid destroying synthetic defects
        "mixup": 0.0,
        "close_mosaic": close_mosaic_epoch,
    }

    # Try to apply class weights
    try:
        model.model.args["cls_pw"] = torch.tensor(cls_weights, dtype=torch.float32)
        logger.info("Class weights applied to model")
    except Exception as e:
        logger.warning(f"Could not set class weights: {e}")

    logger.info(f"Training: {epochs} epochs, batch={batch}, imgsz={TILE_SIZE}")
    logger.info(f"Phase 3 improvements: QFL weights + Hard Negatives + Copy-Paste CP-1")

    results = model.train(**train_args)

    best_pt = exp_dir / "weights" / "best.pt"
    logger.info(f"Training complete. Best: {best_pt}")

    # Validate
    val_results = model.val(data=str(data_yaml), split="val", imgsz=TILE_SIZE)
    logger.info(f"Validation: {val_results}")

    return model, results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print_env_info()
    train_phase3(args)

if __name__ == "__main__":
    main()
