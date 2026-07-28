#!/usr/bin/env python3
"""
YOLO txt → COCO JSON 格式转换
==============================
把 tiles_train / tiles_val 的 YOLO 归一化标签转为 COCO JSON，
供 D-FINE / Co-DINO 等需要 COCO 格式的检测器使用。

用法:
  python plans/dfine/convert_to_coco.py                    # 转换 train + val
  python plans/dfine/convert_to_coco.py --split train       # 只转 train
  python plans/dfine/convert_to_coco.py --output_dir ./coco # 自定义输出目录
"""

import json, argparse, logging, sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import yaml

# ==== 路径配置 ====
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_YAML = PROJECT_ROOT / "tile_dataset.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "coco_annotations"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("coco_convert")


def load_data_config() -> dict:
    """从 tile_dataset.yaml 加载数据配置。"""
    with open(DATA_YAML) as f:
        cfg = yaml.safe_load(f)
    return cfg


def build_coco_dataset(
    img_dir: Path,
    label_dir: Path,
    categories: list,
    output_json: Path,
    desc: str = "",
) -> dict:
    """将一个 split 的 YOLO txt 标签转为 COCO JSON。

    Args:
        img_dir: 图像目录 (e.g. train/tiles_train/images)
        label_dir: 标签目录 (e.g. train/tiles_train/labels)
        categories: COCO category 列表 [{"id":1, "name":"jieba"}, ...]
        output_json: 输出 JSON 路径
        desc: 描述文字

    Returns:
        COCO 格式 dict
    """
    images = []
    annotations = []
    img_id = 0
    ann_id = 0
    missing_labels = 0
    empty_images = 0

    jpg_files = sorted(img_dir.glob("*.jpg"))
    if not jpg_files:
        # 尝试 png
        jpg_files = sorted(img_dir.glob("*.png"))
    if not jpg_files:
        logger.error(f"No images found in {img_dir}")
        return None

    logger.info(f"Processing {len(jpg_files)} images from {desc}...")

    import cv2

    for jpg_path in jpg_files:
        stem = jpg_path.stem
        label_path = label_dir / f"{stem}.txt"

        # 获取图像尺寸
        img = cv2.imread(str(jpg_path))
        if img is None:
            logger.warning(f"Cannot read {jpg_path}, skipping")
            continue
        h, w = img.shape[:2]

        img_id += 1
        images.append({
            "id": img_id,
            "file_name": jpg_path.name,
            "width": w,
            "height": h,
        })

        # 读取标签
        if not label_path.exists():
            missing_labels += 1
            continue

        with open(label_path) as f:
            lines = f.readlines()

        if not lines:
            empty_images += 1
            continue

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            cls_id = int(parts[0])
            x_center = float(parts[1])  # 归一化
            y_center = float(parts[2])
            bbox_w = float(parts[3])
            bbox_h = float(parts[4])

            # YOLO 归一化 → COCO 绝对坐标 (x, y, w, h)
            abs_x = (x_center - bbox_w / 2.0) * w
            abs_y = (y_center - bbox_h / 2.0) * h
            abs_w = bbox_w * w
            abs_h = bbox_h * h

            # 边界检查
            abs_x = max(0, abs_x)
            abs_y = max(0, abs_y)
            abs_w = min(w - abs_x, abs_w)
            abs_h = min(h - abs_y, abs_h)

            if abs_w <= 1 or abs_h <= 1:
                continue

            ann_id += 1
            # COCO category_id 从 1 开始
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cls_id + 1,
                "bbox": [round(abs_x, 2), round(abs_y, 2),
                         round(abs_w, 2), round(abs_h, 2)],
                "area": round(abs_w * abs_h, 2),
                "iscrowd": 0,
            })

        # 进度
        if img_id % 2000 == 0:
            logger.info(f"  ... {img_id}/{len(jpg_files)} images processed")

    # 构建 COCO dict
    coco = {
        "info": {
            "description": f"SteelGuard COCO dataset - {desc}",
            "version": "1.0",
            "year": 2026,
            "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "licenses": [{"id": 1, "name": "internal"}],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    # 统计
    cats_with_anns = set(a["category_id"] for a in annotations)
    logger.info(f"  Images:       {len(images)}")
    logger.info(f"  Annotations:  {len(annotations)}")
    logger.info(f"  Empty images: {empty_images}")
    logger.info(f"  Missing lbls: {missing_labels}")
    logger.info(f"  Categories w/ annots: {len(cats_with_anns)}/9")

    # 保存
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False)
    logger.info(f"  Saved → {output_json} ({output_json.stat().st_size/1024/1024:.1f} MB)")

    return coco


def main():
    parser = argparse.ArgumentParser(description="YOLO → COCO conversion")
    parser.add_argument("--split", type=str, default="all",
                        choices=["train", "val", "all"])
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_data_config()
    data_root = PROJECT_ROOT / cfg["path"]
    class_names = cfg["names"]  # {0: "jieba", 1: "zonglie", ...}

    # 构建 COCO categories (id 从 1 开始，COCO 惯例)
    categories = [
        {"id": int(k) + 1, "name": v, "supercategory": "defect"}
        for k, v in sorted(class_names.items(), key=lambda x: int(x[0]))
    ]

    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("YOLO → COCO Conversion")
    logger.info(f"  Data root: {data_root}")
    logger.info(f"  Classes:   {[c['name'] for c in categories]}")
    logger.info("=" * 60)

    if args.split in ("train", "all"):
        train_img_dir = data_root / cfg["train"]
        train_lbl_dir = data_root / cfg["train"].replace("images", "labels")
        build_coco_dataset(
            train_img_dir, train_lbl_dir, categories,
            output_dir / "train.json",
            desc="train",
        )

    if args.split in ("val", "all"):
        val_img_dir = data_root / cfg["val"]
        val_lbl_dir = data_root / cfg["val"].replace("images", "labels")
        build_coco_dataset(
            val_img_dir, val_lbl_dir, categories,
            output_dir / "val.json",
            desc="val",
        )

    logger.info(f"\nDone! COCO JSONs in: {output_dir}/")


if __name__ == "__main__":
    main()
