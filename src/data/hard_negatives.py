"""Hard Negative Mining v2 —— 对应方案第 10.3 节 实验 T8

修复:
- 从训练集收集 FP（非验证集，无泄漏）
- HN 切片保留真实 GT 标签
"""

import json
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

import cv2
import numpy as np

from src.paths import CLASS_NAMES, SPLIT_DIR, DATA_ROOT, CLASS_NAME_TO_ID
from src.utils.metrics import match_per_image
from src.data.health_check import parse_voc_xml

logger = logging.getLogger(__name__)


def collect_train_fps_from_model(
    pred_dict: Dict[str, list],
    gt_dict: Dict[str, list],
    iou_threshold: float = 0.5,
    min_score: float = 0.3,
    max_per_class: int = 500,
) -> Dict[int, list]:
    """从训练集预测中收集 FP（非验证集，无数据泄漏）。

    Returns:
        {class_id: [(image_id, xmin, ymin, xmax, ymax, score), ...]}
    """
    fp_collection = defaultdict(list)

    for img_id in sorted(gt_dict.keys()):
        gts = gt_dict.get(img_id, [])
        preds = pred_dict.get(img_id, [])

        for cid in range(len(CLASS_NAMES)):
            if len(fp_collection[cid]) >= max_per_class:
                continue

            gt_c = np.array([g[1:5] for g in gts if g[0] == cid])
            preds_c = [p for p in preds if p[0] == cid]

            if not preds_c:
                continue

            pred_boxes = np.array([p[1:5] for p in preds_c])
            pred_scores = np.array([p[5] for p in preds_c])

            tp_c, fp_c = match_per_image(gt_c, pred_boxes, pred_scores, iou_threshold)

            for i, is_fp in enumerate(fp_c):
                if is_fp and pred_scores[i] >= min_score:
                    if len(fp_collection[cid]) >= max_per_class:
                        break
                    fp_collection[cid].append({
                        "image_id": img_id,
                        "xmin": float(pred_boxes[i][0]),
                        "ymin": float(pred_boxes[i][1]),
                        "xmax": float(pred_boxes[i][2]),
                        "ymax": float(pred_boxes[i][3]),
                        "score": float(pred_scores[i]),
                    })

    total = sum(len(v) for v in fp_collection.values())
    logger.info(f"Collected {total} training-set FPs")
    for cid in sorted(fp_collection.keys()):
        if fp_collection[cid]:
            logger.info(f"  {CLASS_NAMES[cid]:<15s}: {len(fp_collection[cid])} FPs")
    return dict(fp_collection)


def generate_hard_negative_tiles(
    fp_collection: Dict[int, list],
    src_dir: Path,
    output_img_dir: Path,
    output_lbl_dir: Path,
    tile_size: int = 1280,
    max_total: int = 3000,
) -> int:
    """从 FP 位置生成硬负样本切片，保留真实 GT 标签。

    Args:
        fp_collection: FP 列表
        src_dir: 原图目录（含 JPG+XML）
        output_img_dir, output_lbl_dir: 输出路径
        tile_size: 切片尺寸
        max_total: 最多生成切片数

    Returns:
        生成的切片数
    """
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_lbl_dir.mkdir(parents=True, exist_ok=True)

    # 去重：同一原图的相近 FP 只取一个切片
    seen_regions = set()
    count = 0

    for cid, fps in fp_collection.items():
        if count >= max_total:
            break

        for fp in fps:
            if count >= max_total:
                break

            img_id = fp["image_id"]
            cx = int((fp["xmin"] + fp["xmax"]) / 2)
            cy = int((fp["ymin"] + fp["ymax"]) / 2)

            # 网格化去重
            grid_key = (img_id, cx // (tile_size // 2), cy // (tile_size // 2))
            if grid_key in seen_regions:
                continue
            seen_regions.add(grid_key)

            jpg_path = src_dir / img_id
            xml_path = src_dir / img_id.replace(".jpg", ".xml")
            if not jpg_path.exists():
                continue

            image = cv2.imread(str(jpg_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            h, w = image.shape[:2]

            # 以 FP 为中心裁剪
            x_start = cx - tile_size // 2
            y_start = cy - tile_size // 2
            x_end = x_start + tile_size
            y_end = y_start + tile_size

            # Clamp to image
            x_start_c = max(0, x_start)
            y_start_c = max(0, y_start)
            x_end_c = min(w, x_end)
            y_end_c = min(h, y_end)

            crop = image[y_start_c:y_end_c, x_start_c:x_end_c]

            # Pad if needed
            pad_left = x_start_c - x_start
            pad_top = y_start_c - y_start
            pad_right = x_end - x_end_c
            pad_bottom = y_end - y_end_c
            if pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0:
                crop = cv2.copyMakeBorder(
                    crop, pad_top, pad_bottom, pad_left, pad_right,
                    cv2.BORDER_REFLECT_101
                )

            if crop.shape[0] != tile_size or crop.shape[1] != tile_size:
                crop = cv2.resize(crop, (tile_size, tile_size))

            # ROI check
            if (crop > 10).sum() / (tile_size ** 2) < 0.3:
                continue

            # ---- 关键修复：保留该切片内所有真实 GT 标签 ----
            labels = []
            if xml_path.exists():
                parsed = parse_voc_xml(xml_path)
                if parsed:
                    tile_x1, tile_y1 = x_start, y_start
                    for obj in parsed["objects"]:
                        name = obj["name"]
                        if name not in CLASS_NAME_TO_ID:
                            continue
                        obj_cid = CLASS_NAME_TO_ID[name]
                        bx1, by1, bx2, by2 = obj["bbox"]

                        # 检查 GT 框是否与切片有交集（保留面积 >= 40%）
                        ix1 = max(bx1, tile_x1); iy1 = max(by1, tile_y1)
                        ix2 = min(bx2, tile_x1 + tile_size); iy2 = min(by2, tile_y1 + tile_size)
                        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                        area = (bx2 - bx1) * (by2 - by1)
                        if area > 0 and inter / area >= 0.4:
                            # 映射到切片坐标
                            mx1 = max(0, bx1 - tile_x1 + pad_left)
                            my1 = max(0, by1 - tile_y1 + pad_top)
                            mx2 = min(tile_size, bx2 - tile_x1 + pad_left)
                            my2 = min(tile_size, by2 - tile_y1 + pad_top)
                            if mx2 > mx1 + 1 and my2 > my1 + 1:
                                xc = (mx1 + mx2) / 2.0 / tile_size
                                yc = (my1 + my2) / 2.0 / tile_size
                                nw = (mx2 - mx1) / tile_size
                                nh = (my2 - my1) / tile_size
                                labels.append((obj_cid, xc, yc, nw, nh))

            # 保存
            tile_name = f"hn_{img_id.replace('.jpg','')}_{count}"
            tile_rgb = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
            cv2.imwrite(str(output_img_dir / f"{tile_name}.jpg"), tile_rgb)

            with open(output_lbl_dir / f"{tile_name}.txt", "w") as f:
                for cid_l, xc, yc, nw, nh in labels:
                    f.write(f"{cid_l} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n")

            count += 1

    logger.info(f"Generated {count} hard negative tiles (with GT labels preserved)")
    return count


def merge_to_tileset(src_img: Path, src_lbl: Path, dst_img: Path, dst_lbl: Path):
    """合并到已有 tile 数据集。"""
    import shutil
    for f in src_img.glob("*.jpg"):
        if not (dst_img / f.name).exists():
            shutil.copy2(f, dst_img / f.name)
    for f in src_lbl.glob("*.txt"):
        if not (dst_lbl / f.name).exists():
            shutil.copy2(f, dst_lbl / f.name)
    logger.info(f"Merged into {dst_img}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    with open(SPLIT_DIR / "train_val_split.json", "r") as f:
        split = json.load(f)

    train_stems = set(split["train_stems"])

    # 需要先有训练集预测（由 baseline 模型生成）
    pred_path = Path("outputs/train_predictions.json")
    if not pred_path.exists():
        print("Run baseline inference on training set first!")
        print("  python scripts/predict.py --weights .../best.pt --test-dir train/train --output outputs/train_preds.json")
        exit(1)

    with open(pred_path) as f:
        data = json.load(f)

    # 硬过滤：只保留训练集图片，防止验证集泄漏
    gt_dict = {}
    pred_dict = {}
    val_leaked = 0
    for img_id in data["gt"]:
        stem = img_id.replace(".jpg", "")
        if stem not in train_stems:
            val_leaked += 1
            continue
        gt_dict[img_id] = [(int(x[0]), x[1], x[2], x[3], x[4]) for x in data["gt"][img_id]]
    for img_id in data["pred"]:
        stem = img_id.replace(".jpg", "")
        if stem not in train_stems:
            continue
        pred_dict[img_id] = [(int(x[0]), x[1], x[2], x[3], x[4], x[5]) for x in data["pred"][img_id]]

    if val_leaked > 0:
        logger.warning(f"Filtered out {val_leaked} val-set images from HN (prevents data leakage)")

    fps = collect_train_fps_from_model(pred_dict, gt_dict)

    hn_dir = DATA_ROOT.parent / "hard_negatives_v2"
    n = generate_hard_negative_tiles(
        fps, DATA_ROOT,
        hn_dir / "images", hn_dir / "labels",
        max_total=3000,
    )

    tileset = DATA_ROOT.parent / "tiles_train"
    merge_to_tileset(hn_dir / "images", hn_dir / "labels",
                     tileset / "images", tileset / "labels")
