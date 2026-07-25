#!/usr/bin/env python3
"""SteelGuard-YOLO 强基线训练脚本 —— 对应方案第 7 节 / 第 14.1 节

YOLO11m-P2 + 1280 ROI切片训练 + 类别感知采样 + 基础灰度增强
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
import logging
import argparse
import platform
from datetime import datetime

import torch
from ultralytics import YOLO

from src.paths import (
    PROJECT_ROOT, CONFIG_DIR, CHECKPOINT_DIR, LOG_DIR, CLASS_NAMES,
    TILE_SIZE,
)

logger = logging.getLogger(__name__)


def print_env_info():
    """打印环境信息 —— 方案第 16.3 节。"""
    info = {
        "Python": platform.python_version(),
        "PyTorch": torch.__version__,
        "CUDA available": torch.cuda.is_available(),
        "CUDA version": torch.version.cuda,
    }
    if torch.cuda.is_available():
        info["GPU"] = torch.cuda.get_device_name(0)
        info["GPU memory"] = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
    logger.info("=== Environment ===")
    for k, v in info.items():
        logger.info(f"  {k}: {v}")


def train_baseline(args):
    """训练 YOLO11m-P2 强基线。"""
    # 加载配置
    config_path = CONFIG_DIR / args.config
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 创建实验目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"baseline_p2_{timestamp}"
    exp_dir = CHECKPOINT_DIR / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # 保存配置快照
    with open(exp_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    # 日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_DIR / f"train_{timestamp}.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    print_env_info()

    # 数据集 YAML（相对路径）
    data_yaml = PROJECT_ROOT / "tile_dataset.yaml"
    logger.info(f"Dataset: {data_yaml}")

    # P2 模型: 从 YAML 构建架构，加载预训练权重
    model_yaml = CONFIG_DIR / "yolo11m-p2.yaml"
    logger.info(f"Building P2 model from {model_yaml}")
    if model_yaml.exists():
        model = YOLO(str(model_yaml)).load("yolo11m.pt")
        logger.info("P2 model built + pretrained weights loaded")
    else:
        logger.warning("P2 YAML not found, falling back to yolo11m")
        model = YOLO("yolo11m.pt")

    # 训练参数
    epochs = config.get("training", {}).get("epochs", 100)
    batch = 2  # reduced batch to free GPU memory, gradient_accumulation compensates
    accumulate = 8  # compensate for batch=2, effective batch = 2×8 = 16
    close_mosaic_epoch = int(epochs * (1.0 - config.get("training", {}).get("mosaic_disable_last_epochs", 0.3)))

    train_args = {
        "data": str(data_yaml),
        "epochs": epochs,
        "imgsz": TILE_SIZE,      # 1280 — tiles are already 1280×1280
        "batch": batch,
        "optimizer": config.get("training", {}).get("optimizer", "AdamW"),
        "lr0": config.get("training", {}).get("lr", 2e-4),
        "weight_decay": config.get("training", {}).get("weight_decay", 5e-4),
        "warmup_epochs": config.get("training", {}).get("warmup_epochs", 3),
        "cos_lr": True,
        "amp": True,
        "patience": config.get("training", {}).get("early_stopping_patience", 20),
        "project": str(CHECKPOINT_DIR),
        "name": exp_dir.name,
        "exist_ok": True,
        "pretrained": True,
        "verbose": True,
        "seed": config.get("data", {}).get("seed", 42),
        "device": args.device or (0 if torch.cuda.is_available() else "cpu"),
        "workers": 0,  # 0 = main process only, avoids C drive temp file buildup

        # 基础灰度增强（方案 5.4.1）
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.3,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.2,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": config.get("augmentation", {}).get("mosaic", 0.5),
        "mixup": 0.0,
        "close_mosaic": close_mosaic_epoch,
    }

    logger.info(f"Training: {epochs} epochs, batch={batch}×{accumulate}, imgsz={TILE_SIZE}")
    logger.info(f"Mosaic disabled after epoch {close_mosaic_epoch}")

    # 训练
    results = model.train(**train_args)

    # 记录结果路径
    best_pt = exp_dir / "weights" / "best.pt"
    logger.info(f"Training complete. Best model: {best_pt}")

    # 验证
    logger.info("Running validation...")
    val_results = model.val(data=str(data_yaml), split="val", imgsz=TILE_SIZE)
    logger.info(f"Validation: {val_results}")

    return model, results


def main():
    parser = argparse.ArgumentParser(description="SteelGuard-YOLO Baseline Training")
    parser.add_argument("--config", type=str, default="baseline.yaml")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    train_baseline(args)


if __name__ == "__main__":
    main()
