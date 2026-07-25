#!/usr/bin/env python3
"""SteelGuard-YOLO 推理脚本 —— 对应方案第 8 节

全局 + 局部双尺度推理：
1. 原图 → ROI 估计 → 1280 重叠切片 + 1536 全局视图
2. 真 batch 推理
3. 坐标还原 → Soft-NMS (写回分数) → JSON
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import json
import yaml
import logging
import argparse
from collections import defaultdict
from tqdm import tqdm
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from src.paths import (
    TEST_ROOT, OUTPUT_ROOT, CONFIG_DIR,
    CLASS_NAMES,
    TILE_SIZE, TILE_OVERLAP, TILE_STRIDE, GLOBAL_LONG_SIDE,
)
from src.utils.roi import estimate_plate_roi, generate_tiles, generate_global_view

logger = logging.getLogger(__name__)


class SteelGuardPredictor:
    """SteelGuard-YOLO 推理器。"""

    def __init__(
        self,
        weights_path: str,
        device: str = "cuda",
        conf_threshold: float = 0.001,
        max_det: int = 300,
        nms_iou: float = 0.55,
        nms_sigma: float = 0.5,
        tile_size: int = TILE_SIZE,
        tile_overlap: int = TILE_OVERLAP,
        global_long_side: int = GLOBAL_LONG_SIDE,
        batch_size: int = 4,
    ):
        self.conf_threshold = conf_threshold
        self.max_det = max_det
        self.nms_iou = nms_iou
        self.nms_sigma = nms_sigma
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.global_long_side = global_long_side
        self.batch_size = batch_size
        self.imgsz = tile_size  # 推理尺寸 = 切片尺寸

        logger.info(f"Loading model: {weights_path}")
        self.model = YOLO(weights_path)
        self.device_str = device if torch.cuda.is_available() else "cpu"
        logger.info(f"Device: {self.device_str}, imgsz={self.imgsz}")

        self._warmup()

    def _warmup(self):
        dummy = np.random.randint(0, 255, (self.tile_size, self.tile_size, 3), dtype=np.uint8)
        _ = self.model(dummy, imgsz=self.imgsz, verbose=False)

    @staticmethod
    def _grayscale_to_rgb(image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return image

    def predict_single(self, image_path) -> list:
        """单图推理：切片 + 全局 → 坐标还原 → Soft-NMS。"""
        if isinstance(image_path, (str, Path)):
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        else:
            image = image_path

        if image is None:
            raise ValueError(f"Failed to load: {image_path}")

        h, w = image.shape[:2]

        # ROI 估计
        _, roi_bbox = estimate_plate_roi(image)

        # 生成切片
        tiles_data = generate_tiles(
            image, tile_size=self.tile_size,
            overlap=self.tile_overlap, roi_bbox=roi_bbox,
        )

        # 全局视图
        global_view = generate_global_view(image, long_side=self.global_long_side)
        gh, gw = global_view.shape[:2]

        all_predictions = []

        # === 真 batch 推理切片 ===
        tile_batch_rgb = []
        tile_offsets = []
        for td in tiles_data:
            tile_batch_rgb.append(self._grayscale_to_rgb(td["tile"]))
            tile_offsets.append((td["x_offset"], td["y_offset"]))

            if len(tile_batch_rgb) >= self.batch_size:
                batch_results = self.model(
                    tile_batch_rgb, imgsz=self.imgsz,
                    conf=self.conf_threshold, verbose=False,
                )
                for result, (x_off, y_off) in zip(batch_results, tile_offsets):
                    mapped = self._map_predictions(result, x_off, y_off, self.tile_size, self.tile_size, h, w)
                    all_predictions.extend(mapped)
                tile_batch_rgb = []
                tile_offsets = []

        # 剩余切片
        if tile_batch_rgb:
            batch_results = self.model(
                tile_batch_rgb, imgsz=self.imgsz,
                conf=self.conf_threshold, verbose=False,
            )
            for result, (x_off, y_off) in zip(batch_results, tile_offsets):
                mapped = self._map_predictions(result, x_off, y_off, self.tile_size, self.tile_size, h, w)
                all_predictions.extend(mapped)

        # === 推理全局视图 ===
        global_rgb = self._grayscale_to_rgb(global_view)
        global_result = self.model(
            global_rgb, imgsz=gw,  # 使用实际全局图尺寸
            conf=self.conf_threshold, verbose=False,
        )[0]

        scale_x = w / gw
        scale_y = h / gh
        if global_result.boxes is not None:
            for box in global_result.boxes:
                xmin, ymin, xmax, ymax = box.xyxy[0].tolist()
                cls_id = int(box.cls[0])
                score = float(box.conf[0])
                all_predictions.append({
                    "xmin": xmin * scale_x, "ymin": ymin * scale_y,
                    "xmax": xmax * scale_x, "ymax": ymax * scale_y,
                    "score": score, "class_id": cls_id,
                })

        # === Soft-NMS 去重（分数写回 dict）===
        all_predictions = self._soft_nms(all_predictions)

        # 按分数排序 → 限制 max_det → 裁剪坐标
        all_predictions.sort(key=lambda x: x["score"], reverse=True)
        all_predictions = all_predictions[:self.max_det]

        for pred in all_predictions:
            pred["xmin"] = max(0.0, min(float(w), pred["xmin"]))
            pred["ymin"] = max(0.0, min(float(h), pred["ymin"]))
            pred["xmax"] = max(0.0, min(float(w), pred["xmax"]))
            pred["ymax"] = max(0.0, min(float(h), pred["ymax"]))

        return all_predictions

    def _map_predictions(self, result, x_offset, y_offset, tile_w, tile_h, img_w, img_h) -> list:
        """切片预测 → 原图坐标。"""
        if result.boxes is None:
            return []
        mapped = []
        for box in result.boxes:
            xmin, ymin, xmax, ymax = box.xyxy[0].tolist()
            mapped.append({
                "xmin": xmin + x_offset,
                "ymin": ymin + y_offset,
                "xmax": xmax + x_offset,
                "ymax": ymax + y_offset,
                "score": float(box.conf[0]),
                "class_id": int(box.cls[0]),
            })
        return mapped

    def _soft_nms(self, predictions: list) -> list:
        """类别内 Soft-NMS，分数写回 dict。"""
        if not predictions:
            return predictions

        by_class = defaultdict(list)
        for i, pred in enumerate(predictions):
            pred["_idx"] = i
            by_class[pred["class_id"]].append(pred)

        for cls_id, preds in by_class.items():
            preds.sort(key=lambda x: x["score"], reverse=True)
            n = len(preds)
            boxes = np.array([[p["xmin"], p["ymin"], p["xmax"], p["ymax"]] for p in preds])

            for i in range(n):
                for j in range(i + 1, n):
                    iou = self._box_iou(boxes[i], boxes[j])
                    if iou > self.nms_iou:
                        # Soft-NMS: 对 preds[j] 的 score 做衰减
                        weight = np.exp(-(iou ** 2) / self.nms_sigma)
                        preds[j]["score"] *= weight  # 写回 dict

        # 按降级后的分数过滤
        kept = [p for p in predictions if p["score"] >= self.conf_threshold]
        for p in kept:
            p.pop("_idx", None)
        return kept

    @staticmethod
    def _box_iou(box1, box2) -> float:
        x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        return float(inter / union) if union > 0 else 0.0


def generate_submission(predictor, test_dir, output_path, per_class_conf=None):
    """生成提交 JSON。"""
    test_images = sorted(test_dir.glob("*.jpg"))
    if not test_images:
        logger.error(f"No images in {test_dir}")
        return []

    logger.info(f"Predicting {len(test_images)} test images...")
    all_submissions = []
    timings = []

    for img_path in tqdm(test_images):
        t0 = time.perf_counter()
        predictions = predictor.predict_single(img_path)
        timings.append(time.perf_counter() - t0)

        for pred in predictions:
            cls_name = CLASS_NAMES[pred["class_id"]]
            if per_class_conf and cls_name in per_class_conf:
                if pred["score"] < per_class_conf[cls_name]:
                    continue
            all_submissions.append({
                "image_id": img_path.name,
                "category_name": cls_name,
                "bbox": [int(pred["xmin"]), int(pred["ymin"]),
                         int(pred["xmax"]), int(pred["ymax"])],
                "score": round(pred["score"], 6),
            })

    logger.info(f"Avg time: {np.mean(timings):.2f}s, P95: {np.percentile(timings, 95):.2f}s")
    logger.info(f"Total predictions: {len(all_submissions)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_submissions, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved to {output_path}")

    return all_submissions


def validate_submission(submissions):
    """校验提交格式。"""
    errors = []
    valid = set(CLASS_NAMES)
    for i, s in enumerate(submissions):
        for f in ["image_id", "category_name", "bbox", "score"]:
            if f not in s:
                errors.append(f"[{i}] missing '{f}'")
        if errors: continue
        if s["category_name"] not in valid:
            errors.append(f"[{i}] bad category: {s['category_name']}")
        b = s["bbox"]
        if len(b) != 4 or not (0 <= b[0] < b[2]) or not (0 <= b[1] < b[3]):
            errors.append(f"[{i}] bad bbox: {b}")
        if not (0 <= s["score"] <= 1):
            errors.append(f"[{i}] bad score: {s['score']}")
    return len(errors) == 0, errors


def main():
    parser = argparse.ArgumentParser(description="SteelGuard-YOLO Inference")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--config", type=str, default="baseline.yaml")
    parser.add_argument("--test-dir", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()

    config = {}
    config_path = CONFIG_DIR / args.config
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

    test_dir = Path(args.test_dir) if args.test_dir else TEST_ROOT
    output_path = Path(args.output) if args.output else OUTPUT_ROOT / "submission.json"

    predictor = SteelGuardPredictor(
        weights_path=args.weights,
        device=args.device,
        conf_threshold=args.conf,
        max_det=config.get("inference", {}).get("max_det", 300),
        nms_iou=config.get("inference", {}).get("nms", {}).get("iou_threshold", 0.55),
        batch_size=args.batch,
    )

    per_class_conf = config.get("inference", {}).get("per_class_conf", {})
    subs = generate_submission(predictor, test_dir, output_path, per_class_conf)

    ok, errs = validate_submission(subs)
    print(f"\n[Inference Complete]")
    print(f"   Images: {len(list(test_dir.glob('*.jpg')))}")
    print(f"   Predictions: {len(subs)}")
    print(f"   Output: {output_path}")
    print(f"   Validation: {'PASS' if ok else 'FAIL'}")

    return subs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
