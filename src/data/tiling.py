"""Tile 数据集构建模块 —— 对应方案第 5.1 节

将 4096×3000 原图按 ROI 切分为 1280×1280 重叠切片，
同时映射标签坐标到切片坐标系，用于训练。

关键规则（方案 5.1.2）:
- tile_size=1280, overlap=256, stride=1024
- 正样本：目标中心落在切片内
- 截断规则：保留面积 < 原框 40% → 训练时忽略；否则裁剪框
- 边缘不足处 reflect padding（不是纯黑 padding）
- 有效 ROI 占比过低的切片跳过
- 同时生成全局视图（长边 1536）用于推理
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
from collections import defaultdict
from tqdm import tqdm

from src.paths import (
    DATA_ROOT, CLASS_NAMES, CLASS_NAME_TO_ID,
    TILE_SIZE, TILE_OVERLAP, TILE_STRIDE,
    GLOBAL_LONG_SIDE, ORIGINAL_WIDTH, ORIGINAL_HEIGHT,
    SPLIT_DIR,
)
from src.data.health_check import parse_voc_xml
from src.utils.roi import estimate_plate_roi

logger = logging.getLogger(__name__)


def compute_retention_ratio(
    box: Tuple[float, float, float, float],
    tile_bbox: Tuple[int, int, int, int],
) -> float:
    """计算目标框在切片裁剪后的保留面积比例。

    Args:
        box: (xmin, ymin, xmax, ymax) 绝对坐标
        tile_bbox: (x_start, y_start, x_end, y_end) 切片在原图中的区域

    Returns:
        保留比例 [0, 1]
    """
    tx1, ty1, tx2, ty2 = tile_bbox
    bx1, by1, bx2, by2 = box

    # 计算交集
    ix1 = max(bx1, tx1)
    iy1 = max(by1, ty1)
    ix2 = min(bx2, tx2)
    iy2 = min(by2, ty2)

    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    inter_area = inter_w * inter_h

    box_w = bx2 - bx1
    box_h = by2 - by1
    box_area = box_w * box_h

    if box_area <= 0:
        return 0.0

    return inter_area / box_area


def map_box_to_tile(
    box: Tuple[float, float, float, float],
    tile_offset: Tuple[int, int],
    tile_size: int,
    padding: Tuple[int, int, int, int] = None,
) -> Tuple[float, float, float, float]:
    """将绝对坐标框映射到切片坐标系。

    Args:
        box: (xmin, ymin, xmax, ymax) 绝对坐标
        tile_offset: (x_start, y_start) 切片左上角在原图中的位置
        tile_size: 切片尺寸
        padding: (pad_left, pad_top, pad_right, pad_bottom) reflect padding 量

    Returns:
        (xmin, ymin, xmax, ymax) 切片内坐标
    """
    xmin, ymin, xmax, ymax = box
    x_off, y_off = tile_offset

    if padding:
        pad_left, pad_top, pad_right, pad_bottom = padding
        # reflect padding 会扩展图像边界
        xmin = xmin + pad_left - x_off
        ymin = ymin + pad_top - y_off
        xmax = xmax + pad_left - x_off
        ymax = ymax + pad_top - y_off
    else:
        xmin = xmin - x_off
        ymin = ymin - y_off
        xmax = xmax - x_off
        ymax = ymax - y_off

    # 裁剪到切片范围内
    xmin = max(0.0, min(float(tile_size), xmin))
    ymin = max(0.0, min(float(tile_size), ymin))
    xmax = max(0.0, min(float(tile_size), xmax))
    ymax = max(0.0, min(float(tile_size), ymax))

    return (xmin, ymin, xmax, ymax)


def generate_tiles_with_labels(
    image: np.ndarray,
    xml_path: Path,
    tile_size: int = TILE_SIZE,
    overlap: int = TILE_OVERLAP,
    min_retention: float = 0.4,
    roi_min_ratio: float = 0.3,
) -> List[Dict]:
    """对单张图像生成切片及对应标签。

    Args:
        image: 4096×3000 灰度图
        xml_path: VOC XML 标注文件路径
        tile_size: 切片尺寸
        overlap: 重叠像素
        min_retention: 目标保留面积阈值（低于此值忽略）
        roi_min_ratio: 切片内有效 ROI 占比阈值

    Returns:
        List of {
            "tile": np.ndarray,       # 1280×1280 切片 (灰度)
            "x_offset": int,          # 切片在原图中 x 偏移
            "y_offset": int,          # 切片在原图中 y 偏移
            "labels": [(cls_id, xc, yc, w, h), ...],  # YOLO 归一化标签
            "source_image": str,      # 来源图像 stem
        }
    """
    h, w = image.shape[:2]
    stride = tile_size - overlap

    # 解析标签
    parsed = parse_voc_xml(xml_path)
    objects = parsed["objects"] if parsed else []

    # 估计 ROI
    try:
        _, roi_bbox = estimate_plate_roi(image)
    except Exception:
        roi_bbox = (0, 0, w, h)

    rxmin, rymin, rxmax, rymax = roi_bbox
    # 扩展 ROI 边界以包含贴边缺陷
    rxmin = max(0, rxmin - 48)
    rymin = max(0, rymin - 48)
    rxmax = min(w, rxmax + 48)
    rymax = min(h, rymax + 48)

    tile_results = []

    # 在 ROI 上滑动窗口
    for y_start in range(rymin, rymax, stride):
        for x_start in range(rxmin, rxmax, stride):
            x_end = x_start + tile_size
            y_end = y_start + tile_size

            # 计算 padding
            pad_left = max(0, -x_start)
            pad_top = max(0, -y_start)
            pad_right = max(0, x_end - w)
            pad_bottom = max(0, y_end - h)

            x_start_clipped = max(0, x_start)
            y_start_clipped = max(0, y_start)
            x_end_clipped = min(w, x_end)
            y_end_clipped = min(h, y_end)

            # 提取切片
            crop = image[y_start_clipped:y_end_clipped, x_start_clipped:x_end_clipped]

            # Reflect padding
            if pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0:
                crop = cv2.copyMakeBorder(
                    crop, pad_top, pad_bottom, pad_left, pad_right,
                    cv2.BORDER_REFLECT_101
                )

            # 确保尺寸一致
            if crop.shape[0] != tile_size or crop.shape[1] != tile_size:
                crop = cv2.resize(crop, (tile_size, tile_size), interpolation=cv2.INTER_LINEAR)

            # ROI 占比检查
            bright_pixels = (crop > 10).sum()
            if bright_pixels / (tile_size * tile_size) < roi_min_ratio:
                continue

            # 映射标签
            tile_bbox = (x_start, y_start, x_end, y_end)
            padding = (pad_left, pad_top, pad_right, pad_bottom)
            labels = []

            for obj in objects:
                obj_name = obj["name"]
                if obj_name not in CLASS_NAME_TO_ID:
                    continue

                cls_id = CLASS_NAME_TO_ID[obj_name]
                bbox = tuple(obj["bbox"])

                # 检查保留比例
                retention = compute_retention_ratio(bbox, tile_bbox)
                if retention < min_retention:
                    continue

                # 映射坐标
                mapped = map_box_to_tile(bbox, (x_start, y_start), tile_size, padding)

                # 转为 YOLO 归一化格式
                xmin, ymin, xmax, ymax = mapped
                box_w = xmax - xmin
                box_h = ymax - ymin

                if box_w <= 1 or box_h <= 1:
                    continue

                x_center = (xmin + xmax) / 2.0 / tile_size
                y_center = (ymin + ymax) / 2.0 / tile_size
                norm_w = box_w / tile_size
                norm_h = box_h / tile_size

                # 检查有效性
                if 0 < norm_w <= 1.0 and 0 < norm_h <= 1.0:
                    labels.append((cls_id, x_center, y_center, norm_w, norm_h))

            tile_results.append({
                "tile": crop,
                "x_offset": x_start,
                "y_offset": y_start,
                "labels": labels,
                "source_image": xml_path.stem,
            })

    return tile_results


def build_tile_dataset(
    stem_list: List[str],
    src_dir: Path,
    dst_img_dir: Path,
    dst_label_dir: Path,
    tile_size: int = TILE_SIZE,
    overlap: int = TILE_OVERLAP,
    min_retention: float = 0.4,
) -> int:
    """批量构建切片数据集。

    对训练集每张图切片，保存切片图像和 YOLO 标签。
    文件命名: {原图stem}_T{x偏移}_{y偏移}.jpg / .txt

    Returns:
        生成的切片总数
    """
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_label_dir.mkdir(parents=True, exist_ok=True)

    total_tiles = 0
    total_skipped = 0

    for stem in tqdm(stem_list, desc="Generating tiles"):
        jpg_path = src_dir / f"{stem}.jpg"
        xml_path = src_dir / f"{stem}.xml"

        if not jpg_path.exists() or not xml_path.exists():
            total_skipped += 1
            continue

        # 加载灰度图
        image = cv2.imread(str(jpg_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            logger.warning(f"Failed to load {jpg_path}")
            total_skipped += 1
            continue

        # 生成切片
        tiles = generate_tiles_with_labels(
            image, xml_path,
            tile_size=tile_size,
            overlap=overlap,
            min_retention=min_retention,
        )

        for i, td in enumerate(tiles):
            # 文件名
            tile_name = f"{stem}_T{td['x_offset']}_{td['y_offset']}"

            # 保存切片（转 RGB 三通道，适配 YOLO 预训练权重）
            tile_rgb = cv2.cvtColor(td["tile"], cv2.COLOR_GRAY2RGB)
            cv2.imwrite(str(dst_img_dir / f"{tile_name}.jpg"), tile_rgb)

            # 保存标签
            label_path = dst_label_dir / f"{tile_name}.txt"
            with open(label_path, "w") as f:
                for cls_id, xc, yc, w, h in td["labels"]:
                    f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

            total_tiles += 1

    logger.info(f"Built {total_tiles} tiles ({total_skipped} images skipped)")
    logger.info(f"Avg tiles per image: {total_tiles / max(1, len(stem_list) - total_skipped):.1f}")

    return total_tiles


def build_global_view_dataset(
    stem_list: List[str],
    src_dir: Path,
    dst_img_dir: Path,
    dst_label_dir: Path,
    long_side: int = GLOBAL_LONG_SIDE,
) -> int:
    """为训练集生成全局视图（缩略图）及缩放后的标签。

    全局视图保持比例、长边缩放至目标尺寸。

    Returns:
        生成的全局视图数量
    """
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_label_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for stem in tqdm(stem_list, desc="Generating global views"):
        jpg_path = src_dir / f"{stem}.jpg"
        xml_path = src_dir / f"{stem}.xml"

        if not jpg_path.exists() or not xml_path.exists():
            continue

        image = cv2.imread(str(jpg_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue

        h, w = image.shape[:2]

        # 计算缩放尺寸
        if h >= w:
            new_h = long_side
            new_w = int(w * long_side / h)
        else:
            new_w = long_side
            new_h = int(h * long_side / w)

        # 确保能被 32 整除
        new_w = (new_w // 32) * 32
        new_h = (new_h // 32) * 32

        scale_x = new_w / w
        scale_y = new_h / h

        # 缩放图像
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        tile_rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        cv2.imwrite(str(dst_img_dir / f"{stem}_global.jpg"), tile_rgb)

        # 缩放标签
        parsed = parse_voc_xml(xml_path)
        objects = parsed["objects"] if parsed else []

        labels = []
        for obj in objects:
            name = obj["name"]
            if name not in CLASS_NAME_TO_ID:
                continue
            cls_id = CLASS_NAME_TO_ID[name]
            xmin, ymin, xmax, ymax = obj["bbox"]

            # 缩放并归一化
            xmin_s = max(0, xmin * scale_x)
            ymin_s = max(0, ymin * scale_y)
            xmax_s = min(new_w, xmax * scale_x)
            ymax_s = min(new_h, ymax * scale_y)

            bw = xmax_s - xmin_s
            bh = ymax_s - ymin_s
            if bw <= 1 or bh <= 1:
                continue

            xc = (xmin_s + xmax_s) / 2.0 / new_w
            yc = (ymin_s + ymax_s) / 2.0 / new_h
            nw = bw / new_w
            nh = bh / new_h

            labels.append((cls_id, xc, yc, nw, nh))

        label_path = dst_label_dir / f"{stem}_global.txt"
        with open(label_path, "w") as f:
            for cls_id, xc, yc, w_, h_ in labels:
                f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w_:.6f} {h_:.6f}\n")

        count += 1

    logger.info(f"Built {count} global views")
    return count


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)

    # 加载划分
    with open(SPLIT_DIR / "train_val_split.json", "r") as f:
        split = json.load(f)

    train_stems = split["train_stems"]
    val_stems = split["val_stems"]

    # 构建训练集切片
    logger.info("=== Building TRAIN tile dataset ===")
    train_tiles = build_tile_dataset(
        train_stems, DATA_ROOT,
        dst_img_dir=DATA_ROOT.parent / "tiles_train" / "images",
        dst_label_dir=DATA_ROOT.parent / "tiles_train" / "labels",
    )
    logger.info(f"Train tiles: {train_tiles}")

    # 构建验证集切片
    logger.info("=== Building VAL tile dataset ===")
    val_tiles = build_tile_dataset(
        val_stems, DATA_ROOT,
        dst_img_dir=DATA_ROOT.parent / "tiles_val" / "images",
        dst_label_dir=DATA_ROOT.parent / "tiles_val" / "labels",
    )
    logger.info(f"Val tiles: {val_tiles}")

    # 构建全局视图（推理时用，训练也可选）
    logger.info("=== Building VAL global views ===")
    build_global_view_dataset(
        val_stems, DATA_ROOT,
        dst_img_dir=DATA_ROOT.parent / "global_val" / "images",
        dst_label_dir=DATA_ROOT.parent / "global_val" / "labels",
    )

    print(f"\n[Tile Dataset Summary]:")
    print(f"   Train tiles: {train_tiles}")
    print(f"   Val tiles:   {val_tiles}")
    print(f"   Avg ~{train_tiles / len(train_stems):.1f} tiles per train image")
