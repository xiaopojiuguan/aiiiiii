#!/usr/bin/env python3
"""SteelGuard-YOLO 阶段3 —— 消融实验 + AAFM/MFFM 结构模块训练

用法:
  # 消融实验
  python scripts/train_phase3.py --qfl --hn --cp

  # 结构模块
  python scripts/train_phase3.py --aafm
  python scripts/train_phase3.py --mffm
  python scripts/train_phase3.py --aafm --mffm
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
YOLO11M_PT = PROJECT_ROOT / "yolo11m.pt"


def print_env():
    info = {"Python": platform.python_version(), "PyTorch": torch.__version__}
    if torch.cuda.is_available():
        info["GPU"] = torch.cuda.get_device_name(0)
        info["VRAM"] = f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB"
    for k, v in info.items():
        logger.info(f"  {k}: {v}")


def check_data_consistency(img_dir: Path, lbl_dir: Path):
    """训练前检查数据一致性。"""
    # 先清理 stale cache
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

    weights /= weights.mean()
    weights = np.clip(weights, 0.3, 3.0)

    logger.info("Class weights (inverse freq, alpha=0.5):")
    for cid, name in enumerate(CLASS_NAMES):
        logger.info(f"  {name:<15s}: {weights[cid]:.2f} (n={counts.get(cid,0)})")

    return weights.tolist()


def apply_aafm(model):
    """对已加载的 YOLO 模型应用 AAFM 结构修改。

    替换 DetectionModel._predict_once：backbone (layers 0-10) 完成后，
    截取 P2/P3 特征 → AAFM 门控融合 → 注入 y[2]/y[4] → neck 继续。
    """
    import types
    from src.models.aafm import AAFMBlock
    from src.models.steelguard import GaussianBlur

    detection = model.model
    width = detection.yaml.get("width_multiple", 0.50)
    p2_ch = int(128 * width)
    p3_ch = int(256 * width)

    aafm_block = AAFMBlock(in_channels=3, stem_channels=32,
                           p2_channels=p2_ch, p3_channels=p3_ch)
    blur = GaussianBlur(kernel_size=31, sigma=10.0, channels=3)

    detection.aafm = aafm_block
    detection.aafm_blur = blur
    detection.to(model.device if hasattr(model, 'device') else None)

    def aafm_predict_once(self, x, profile=False, visualize=False, embed=None):
        y, dt, embeddings = [], [], []
        embed_set = frozenset(embed) if embed else {-1}
        max_idx = max(embed_set)
        x_orig = x  # 保留原始输入用于残差计算

        # ---- Backbone + SPPF + C2PSA (layers 0-10) ----
        for i in range(11):
            m = self.model[i]
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)
            y.append(x if m.i in self.save else None)
            if visualize:
                from ultralytics.utils.plotting import feature_visualization
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if m.i in embed_set:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)

        # ---- AAFM 门控注入: 在 neck 读取前替换 y[2]/y[4] ----
        residual = x_orig - self.aafm_blur(x_orig)
        p2_aafm, p3_aafm = self.aafm(residual, y[2], y[4])
        if p2_aafm is not None:
            y[2] = p2_aafm
        if p3_aafm is not None:
            y[4] = p3_aafm

        # ---- Neck + Head (layers 11-end) ----
        for i in range(11, len(self.model)):
            m = self.model[i]
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)
            y.append(x if m.i in self.save else None)
            if visualize:
                from ultralytics.utils.plotting import feature_visualization
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if m.i in embed_set:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)

        return x

    detection._predict_once = types.MethodType(aafm_predict_once, detection)
    logger.info("AAFM applied: _predict_once interception (backbone→gate→neck)")
    return model


def apply_mffm(model):
    """对已加载的 YOLO 模型应用 MFFM 结构修改。"""
    from src.models.mffm import MFFMBlock

    detection = model.model
    width = detection.yaml.get("width_multiple", 0.50)

    p2_ch = int(128 * width)
    p3_ch = int(256 * width)

    # P2/P3 Neck 融合层索引
    replace_layers = {
        19: (p2_ch, p2_ch, False),    # P2 fusion, no DCN
        16: (p3_ch, p3_ch, True),     # P3 upsample fusion, DCN on
        22: (p3_ch, p3_ch, True),     # P3 downsample fusion, DCN on
    }

    replaced = 0
    for idx, (in_ch, out_ch, dcn) in replace_layers.items():
        if idx < len(detection.model):
            old = detection.model[idx]
            detection.model[idx] = MFFMBlock(in_ch, out_ch, use_dcn=dcn)
            logger.info(f"MFFM: layer {idx} {type(old).__name__}→MFFMBlock (dcn={dcn})")
            replaced += 1

    logger.info(f"MFFM applied: {replaced} layers replaced")
    return model


def train_phase3(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 实验名
    parts = ["phase3"]
    if args.aafm: parts.append("aafm")
    if args.mffm: parts.append("mffm")
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

    # ---- 加载模型 ----
    # AAFM/MFFM 从 yolo11m.pt 重新训练（结构改动大，不适合从 baseline 微调）
    # QFL/HN/CP 从 baseline 微调
    use_pretrained = args.aafm or args.mffm

    if use_pretrained:
        logger.info(f"Building model from YAML + COCO pretrained (structure modified)")
        model = YOLO(str(CONFIG_DIR / "yolo11m-p2.yaml")).load(str(YOLO11M_PT))
    elif BASELINE_WEIGHTS.exists():
        logger.info(f"Loading baseline weights: {BASELINE_WEIGHTS}")
        model = YOLO(str(CONFIG_DIR / "yolo11m-p2.yaml")).load(str(BASELINE_WEIGHTS))
    else:
        logger.error(f"Baseline weights not found: {BASELINE_WEIGHTS}")
        return

    # ---- 应用结构模块 ----
    if args.aafm:
        model = apply_aafm(model)
    if args.mffm:
        model = apply_mffm(model)

    # ---- QFL 类别权重 ----
    if cls_weights is not None:
        weights_tensor = torch.tensor(cls_weights, dtype=torch.float32)
        model.model.args["cls_pw"] = 0.0
        model.model.class_weights = weights_tensor
        logger.info(f"QFL class_weights set: {[f'{w:.2f}' for w in cls_weights]}")

    # ---- 训练参数 ----
    epochs = args.epochs or (100 if use_pretrained else 40)
    batch = 2
    lr = args.lr or (2e-4 if use_pretrained else 5e-5)
    close_mosaic = int(epochs * 0.7)

    train_args = {
        "data": str(PROJECT_ROOT / "tile_dataset.yaml"),
        "epochs": epochs,
        "imgsz": TILE_SIZE,
        "batch": batch,
        "optimizer": "AdamW",
        "lr0": lr,
        "weight_decay": 5e-4,
        "warmup_epochs": 3 if use_pretrained else 1,
        "cos_lr": True,
        "amp": True,
        "patience": 20 if use_pretrained else 12,
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
        "mosaic": 0.3 if use_pretrained else 0.3,
        "mixup": 0.0,
        "close_mosaic": close_mosaic,
    }

    logger.info(f"Experiment: {exp_name}")
    logger.info(f"Modules: AAFM={args.aafm}, MFFM={args.mffm}, QFL={args.qfl}, HN={args.hn}, CP={args.cp}")
    logger.info(f"Training: {epochs} epochs, batch={batch}, lr={lr}")
    logger.info(f"Pretrained from: {'COCO' if use_pretrained else 'baseline'}")

    results = model.train(**train_args)

    best_pt = exp_dir / "weights" / "best.pt"
    logger.info(f"Complete. Best: {best_pt}")

    val_results = model.val(data=str(PROJECT_ROOT / "tile_dataset.yaml"), split="val", imgsz=TILE_SIZE)
    logger.info(f"Validation: {val_results}")

    return model, results


def main():
    parser = argparse.ArgumentParser(description="SteelGuard-YOLO Phase 3 Training")
    parser.add_argument("--qfl", action="store_true", help="QFL class re-weighting")
    parser.add_argument("--hn", action="store_true", help="Hard Negative tiles")
    parser.add_argument("--cp", action="store_true", help="Copy-Paste tiles")
    parser.add_argument("--aafm", action="store_true", help="AAFM artifact-aware fusion")
    parser.add_argument("--mffm", action="store_true", help="MFFM multi-morphology fusion")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    train_phase3(args)


if __name__ == "__main__":
    main()
