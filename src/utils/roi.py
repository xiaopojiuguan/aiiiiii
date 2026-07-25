"""ROI 感知预处理模块 —— 对应方案第 5.1.1 节

板材 ROI 掩膜估计：
1. 对原图下采样
2. 根据亮度、梯度和最大连通区域估计板材主体
3. 形态学闭运算补全孔洞
4. ROI 向外扩张 32~64 像素
5. 训练和推理时忽略有效板材占比过低的切片
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def estimate_plate_roi(
    image: np.ndarray,
    downsample_factor: int = 8,
    black_threshold: int = 10,
    dilate_kernel_size: int = 15,
    roi_expansion: int = 48,
    min_roi_area_ratio: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray]:
    """估计板材 ROI 掩膜。

    Args:
        image: 输入灰度图 (H, W)，uint8
        downsample_factor: 下采样倍率
        black_threshold: 黑边判定阈值
        dilate_kernel_size: 形态学膨胀核大小
        roi_expansion: ROI 向外扩展像素数
        min_roi_area_ratio: 最小 ROI 面积比例

    Returns:
        (roi_mask, roi_bbox) - ROI 掩膜（原图尺寸）和 ROI 边界框 [xmin, ymin, xmax, ymax]
    """
    h, w = image.shape[:2]

    # 1. 下采样加速
    small_h, small_w = h // downsample_factor, w // downsample_factor
    small = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_AREA)

    # 2. 亮度阈值：非纯黑区域
    bright_mask = small > black_threshold

    # 3. 梯度：板材通常有纹理，纯黑区域梯度为零
    grad_x = cv2.Sobel(small.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(small.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    grad_mask = grad_mag > 1.0

    # 4. 综合掩膜
    combined = (bright_mask.astype(np.uint8) * 255) & (grad_mask.astype(np.uint8) * 255)

    # 5. 形态学闭运算补全孔洞
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel_size, dilate_kernel_size))
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

    # 6. 找最大连通区域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=4)
    if num_labels <= 1:
        # 没有找到有效区域，返回全图
        logger.warning("No valid ROI found, using full image")
        return np.ones((h, w), dtype=np.uint8), (0, 0, w, h)

    # 跳过背景(0号标签)
    areas = stats[1:, cv2.CC_STAT_AREA]
    if len(areas) == 0:
        return np.ones((h, w), dtype=np.uint8), (0, 0, w, h)

    largest_label = np.argmax(areas) + 1

    # 7. 上采样回原图
    roi_mask_small = (labels == largest_label).astype(np.uint8) * 255
    roi_mask = cv2.resize(roi_mask_small, (w, h), interpolation=cv2.INTER_NEAREST)

    # 8. 计算 ROI 边界框
    ys, xs = np.where(roi_mask > 0)
    if len(xs) == 0:
        return roi_mask, (0, 0, w, h)

    xmin = max(0, xs.min() - roi_expansion)
    ymin = max(0, ys.min() - roi_expansion)
    xmax = min(w, xs.max() + roi_expansion)
    ymax = min(h, ys.max() + roi_expansion)

    roi_bbox = (int(xmin), int(ymin), int(xmax), int(ymax))

    # 检查 ROI 面积
    roi_area = (xmax - xmin) * (ymax - ymin)
    total_area = w * h
    if roi_area < total_area * min_roi_area_ratio:
        logger.warning(f"ROI too small ({roi_area}/{total_area}), using full image")
        return np.ones((h, w), dtype=np.uint8), (0, 0, w, h)

    return roi_mask, roi_bbox


def generate_tiles(
    image: np.ndarray,
    tile_size: int = 1280,
    overlap: int = 256,
    roi_bbox: Optional[Tuple[int, int, int, int]] = None,
    min_roi_ratio: float = 0.3,
) -> list:
    """在 ROI 上生成 1280×1280 重叠切片。

    Args:
        image: 输入图像 (H, W)
        tile_size: 切片尺寸
        overlap: 重叠像素
        roi_bbox: ROI 边界框，None 则使用全图
        min_roi_ratio: 切片中有效 ROI 占比阈值，低于此值跳过

    Returns:
        List of (tile, x_offset, y_offset, tile_width, tile_height)
    """
    h, w = image.shape[:2]
    stride = tile_size - overlap

    if roi_bbox is None:
        roi_bbox = (0, 0, w, h)

    rxmin, rymin, rxmax, rymax = roi_bbox

    tiles = []

    for y_start in range(rymin, rymax, stride):
        for x_start in range(rxmin, rxmax, stride):
            x_end = x_start + tile_size
            y_end = y_start + tile_size

            # 处理边界：reflect padding
            pad_left = max(0, -x_start)
            pad_top = max(0, -y_start)
            pad_right = max(0, x_end - w)
            pad_bottom = max(0, y_end - h)

            x_start_clipped = max(0, x_start)
            y_start_clipped = max(0, y_start)
            x_end_clipped = min(w, x_end)
            y_end_clipped = min(h, y_end)

            crop = image[y_start_clipped:y_end_clipped, x_start_clipped:x_end_clipped]

            if pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0:
                crop = cv2.copyMakeBorder(
                    crop, pad_top, pad_bottom, pad_left, pad_right,
                    cv2.BORDER_REFLECT_101
                )

            # 确保尺寸一致（边界最后一块可能不足）
            if crop.shape[0] != tile_size or crop.shape[1] != tile_size:
                crop = cv2.resize(crop, (tile_size, tile_size), interpolation=cv2.INTER_LINEAR)

            # ROI 占比检查
            if roi_bbox is not None:
                roi_in_tile = calculate_roi_ratio(crop, min_roi_ratio)
                if roi_in_tile < min_roi_ratio:
                    continue

            tiles.append({
                "tile": crop,
                "x_offset": x_start,
                "y_offset": y_start,
                "width": tile_size,
                "height": tile_size,
                "original_crop": (x_start_clipped, y_start_clipped, x_end_clipped, y_end_clipped),
            })

    return tiles


def calculate_roi_ratio(tile: np.ndarray, threshold: float = 0.3) -> float:
    """计算切片中有效区域的占比。"""
    if tile.size == 0:
        return 0.0
    bright_pixels = (tile > 10).sum()
    return bright_pixels / tile.size


def generate_global_view(
    image: np.ndarray,
    long_side: int = 1536,
) -> np.ndarray:
    """生成全局视图（保持比例，长边缩放至目标尺寸）。

    Args:
        image: 输入图像 (H, W)
        long_side: 长边目标尺寸

    Returns:
        缩放后的图像
    """
    h, w = image.shape[:2]

    if h >= w:
        new_h = long_side
        new_w = int(w * long_side / h)
    else:
        new_w = long_side
        new_h = int(h * long_side / w)

    # 确保能被 stride=32 整除（YOLO 要求）
    new_w = (new_w // 32) * 32
    new_h = (new_h // 32) * 32

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return resized


def process_image_for_training(
    image_path,
    tile_size: int = 1280,
    overlap: int = 256,
    global_long_side: int = 1536,
) -> dict:
    """训练预处理：ROI → 切片 + 全局视图。

    Returns:
        dict with 'tiles' and 'global_view'
    """
    if isinstance(image_path, str):
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    elif isinstance(image_path, np.ndarray):
        image = image_path
    else:
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # ROI 估计
    roi_mask, roi_bbox = estimate_plate_roi(image)

    # 生成切片
    tiles = generate_tiles(image, tile_size=tile_size, overlap=overlap, roi_bbox=roi_bbox)

    # 全局视图
    global_view = generate_global_view(image, long_side=global_long_side)

    return {
        "image": image,
        "roi_mask": roi_mask,
        "roi_bbox": roi_bbox,
        "tiles": tiles,
        "global_view": global_view,
    }
