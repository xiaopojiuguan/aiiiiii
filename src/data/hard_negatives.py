"""Hard Negative Mining —— 对应方案第 10.3 节 实验 T8

收集模型误检（FP）区域作为硬负样本加入训练集，
重点抑制水渍、油污、黑边等类缺陷伪影。
"""

import json
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set

import numpy as np

from src.paths import CLASS_NAMES, SPLIT_DIR, DATA_ROOT
from src.utils.metrics import match_per_image

logger = logging.getLogger(__name__)


def collect_false_positives(
    gt_dict: Dict[str, list],
    pred_dict: Dict[str, list],
    iou_threshold: float = 0.5,
    min_score: float = 0.3,
    max_fp_per_image: int = 10,
) -> Dict[str, list]:
    """从预测中收集假阳性样本。

    对每张图，运行贪心匹配后，未被匹配且分数 > min_score 的预测 = FP。

    Args:
        gt_dict: {image_id: [(class_id, xmin, ymin, xmax, ymax), ...]}
        pred_dict: {image_id: [(class_id, xmin, ymin, xmax, ymax, score), ...]}
        iou_threshold: 匹配 IoU 阈值
        min_score: FP 最低置信度（低于此值不算有效 FP）
        max_fp_per_image: 每图最多收集的 FP 数量

    Returns:
        {class_id: [(image_id, xmin, ymin, xmax, ymax, score), ...]}
    """
    fp_collection = defaultdict(list)

    for img_id in sorted(gt_dict.keys()):
        gts = gt_dict.get(img_id, [])
        preds = pred_dict.get(img_id, [])

        if not preds:
            continue

        # 按类别分别处理
        for cid in range(len(CLASS_NAMES)):
            gt_c = np.array([g[1:5] for g in gts if g[0] == cid])
            pred_c = [(p[1:5], p[5]) for p in preds if p[0] == cid]

            if not pred_c:
                continue

            pred_boxes = np.array([x[0] for x in pred_c])
            pred_scores = np.array([x[1] for x in pred_c])

            tp_c, fp_c = match_per_image(gt_c, pred_boxes, pred_scores, iou_threshold)

            # 收集 FP
            fp_count = 0
            for i, is_fp in enumerate(fp_c):
                if is_fp and pred_scores[i] >= min_score:
                    bbox = pred_boxes[i]
                    fp_collection[cid].append({
                        "image_id": img_id,
                        "xmin": float(bbox[0]),
                        "ymin": float(bbox[1]),
                        "xmax": float(bbox[2]),
                        "ymax": float(bbox[3]),
                        "score": float(pred_scores[i]),
                    })
                    fp_count += 1
                    if fp_count >= max_fp_per_image:
                        break

    total = sum(len(v) for v in fp_collection.values())
    logger.info(f"Collected {total} FPs across {len(CLASS_NAMES)} classes")
    for cid, fps in sorted(fp_collection.items()):
        logger.info(f"  {CLASS_NAMES[cid]:<15s}: {len(fps)} FPs")

    return dict(fp_collection)


def generate_hard_negative_tiles(
    fp_collection: Dict[int, list],
    src_dir: Path,
    output_dir: Path,
    tile_size: int = 1280,
    max_tiles: int = 5000,
) -> int:
    """从 FP 位置生成硬负样本切片（无标签）。

    Args:
        fp_collection: collect_false_positives() 的输出
        src_dir: 原图目录
        output_dir: 输出目录（直接放 images/ 和 labels/）
        tile_size: 切片尺寸
        max_tiles: 最多生成的负样本切片数

    Returns:
        生成的切片数
    """
    import cv2

    img_dir = output_dir / "images"
    lbl_dir = output_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    count = 0

    for cid, fps in fp_collection.items():
        if count >= max_tiles:
            break

        for fp in fps:
            if count >= max_tiles:
                break

            img_path = src_dir / fp["image_id"]
            if not img_path.exists():
                continue

            image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            h, w = image.shape[:2]

            # 以 FP 框为中心裁剪 tile_size × tile_size
            cx = (fp["xmin"] + fp["xmax"]) / 2
            cy = (fp["ymin"] + fp["ymax"]) / 2

            x_start = int(cx - tile_size / 2)
            y_start = int(cy - tile_size / 2)
            x_end = x_start + tile_size
            y_end = y_start + tile_size

            # 边界处理
            x_start = max(0, x_start)
            y_start = max(0, y_start)
            x_end = min(w, x_end)
            y_end = min(h, y_end)

            crop = image[y_start:y_end, x_start:x_end]

            # 补齐到 tile_size
            if crop.shape[0] != tile_size or crop.shape[1] != tile_size:
                pad_bottom = tile_size - crop.shape[0]
                pad_right = tile_size - crop.shape[1]
                crop = cv2.copyMakeBorder(
                    crop, 0, pad_bottom, 0, pad_right,
                    cv2.BORDER_REFLECT_101
                )
                crop = crop[:tile_size, :tile_size]

            # 检查有效内容（不能是纯黑边）
            if (crop > 10).sum() / (tile_size ** 2) < 0.3:
                continue

            # 保存
            tile_name = f"hn_{fp['image_id'].replace('.jpg','')}_{count}"
            tile_rgb = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
            cv2.imwrite(str(img_dir / f"{tile_name}.jpg"), tile_rgb)

            # 空标签（负样本）
            (lbl_dir / f"{tile_name}.txt").touch()

            count += 1

    logger.info(f"Generated {count} hard negative tiles")
    return count


def merge_hard_negatives_to_tileset(
    hn_dir: Path,
    tileset_img_dir: Path,
    tileset_lbl_dir: Path,
) -> None:
    """将硬负样本合并到已有的 tile 数据集（使用软链接/复制）。"""
    import shutil

    for img_path in hn_dir.glob("images/*.jpg"):
        stem = img_path.stem
        lbl_path = hn_dir / "labels" / f"{stem}.txt"

        dst_img = tileset_img_dir / img_path.name
        dst_lbl = tileset_lbl_dir / f"{stem}.txt"

        if not dst_img.exists():
            shutil.copy2(img_path, dst_img)
        if lbl_path.exists() and not dst_lbl.exists():
            shutil.copy2(lbl_path, dst_lbl)

    logger.info(f"Merged hard negatives into {tileset_img_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 加载验证集预测和 GT
    with open("outputs/val_predictions.json", "r") as f:
        data = json.load(f)

    # 还原数据格式
    gt_dict = {k: [(int(x[0]), x[1], x[2], x[3], x[4]) for x in v]
               for k, v in data["gt"].items()}
    pred_dict = {k: [(int(x[0]), x[1], x[2], x[3], x[4], x[5]) for x in v]
                 for k, v in data["pred"].items()}

    # 收集 FP
    fps = collect_false_positives(gt_dict, pred_dict)

    # 生成硬负样本
    hn_dir = DATA_ROOT.parent / "hard_negatives"
    generate_hard_negative_tiles(fps, DATA_ROOT, hn_dir)

    # 合并到训练集
    merge_hard_negatives_to_tileset(
        hn_dir,
        DATA_ROOT.parent / "tiles_train" / "images",
        DATA_ROOT.parent / "tiles_train" / "labels",
    )
