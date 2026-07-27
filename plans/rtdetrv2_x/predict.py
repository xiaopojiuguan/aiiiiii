#!/usr/bin/env python3
"""
RT-DETRv2-X 推理脚本
====================
- 加载 RT-DETRv2-X (ResNet101) 训练好的 checkpoint
- 1280×1280 ROI 切片推理 + 全局视图
- 坐标还原 → 后处理 → submission.json

用法:
  python plans/rtdetrv2_x/predict.py --weights outputs/checkpoints/rtdetrv2_x_XXX/best.pth
  python plans/rtdetrv2_x/predict.py --weights best.pth --conf 0.01 --batch 2
"""

import sys, os, argparse, logging, json, time
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import cv2
import numpy as np
import torch
import yaml

# ==== 路径配置 ====
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLAN_DIR = Path(__file__).resolve().parent
RTDETR_DIR = PROJECT_ROOT / "RT-DETR"
sys.path.insert(0, str(RTDETR_DIR / "src"))

from src.paths import (
    TEST_ROOT, OUTPUT_ROOT, CHECKPOINT_DIR,
    CLASS_NAMES, CLASS_NAME_TO_ID,
    TILE_SIZE, TILE_OVERLAP, TILE_STRIDE, GLOBAL_LONG_SIDE,
)
from src.utils.roi import estimate_plate_roi, generate_tiles, generate_global_view

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rtdetrv2_pred")


class RTDETRv2Predictor:
    """RT-DETRv2-X 推理器 —— DETR 架构，无 NMS。"""

    def __init__(
        self,
        weights_path: str,
        device: str = "cuda",
        conf_threshold: float = 0.01,
        max_det: int = 300,
        tile_size: int = TILE_SIZE,
        tile_overlap: int = TILE_OVERLAP,
        global_long_side: int = GLOBAL_LONG_SIDE,
        batch_size: int = 4,
    ):
        self.conf_threshold = conf_threshold
        self.max_det = max_det
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.global_long_side = global_long_side
        self.batch_size = batch_size
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading RT-DETRv2-X: {weights_path}")
        self.model, self.postprocessor = self._load_model(weights_path)
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Device: {self.device}, imgsz={self.tile_size}")

        self._warmup()

    def _load_model(self, weights_path: str):
        """加载 RT-DETR 模型和后处理器。"""
        # 延迟导入 RT-DETR 模块
        from src.core import YAMLConfig

        # 从 checkpoint 目录找训练时的 config
        ckpt_dir = Path(weights_path).parent
        config_path = ckpt_dir / "config.yml"
        if not config_path.exists():
            raise FileNotFoundError(
                f"Config not found: {config_path}. "
                f"请确保 checkpoint 同目录下有 config.yml"
            )

        cfg = YAMLConfig(str(config_path))
        model = cfg.model
        postprocessor = cfg.postprocessor

        # 加载权重
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
        if "ema" in checkpoint and checkpoint["ema"]:
            state = checkpoint["ema"]["module"]
        elif "model" in checkpoint:
            state = checkpoint["model"]
        else:
            state = checkpoint

        # 去掉 "module." 前缀
        state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
        logger.info(f"Loaded weights from {weights_path}")

        return model, postprocessor

    def _warmup(self):
        dummy = torch.randn(1, 3, self.tile_size, self.tile_size).to(self.device)
        with torch.no_grad():
            _ = self.model(dummy)

    @staticmethod
    def _grayscale_to_rgb(image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return image

    @staticmethod
    def _preprocess_tile(tile_rgb: np.ndarray) -> torch.Tensor:
        """归一化 + ToTensor。"""
        # ImageNet 标准化
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        tile = tile_rgb.astype(np.float32) / 255.0
        tile = (tile - mean) / std
        # HWC → CHW
        tile = np.transpose(tile, (2, 0, 1))
        return torch.from_numpy(tile)

    def _infer_batch(self, tiles_rgb: list) -> list:
        """对一批 RGB tile 做推理，返回 [{boxes, scores, labels}, ...]"""
        tensors = [self._preprocess_tile(t) for t in tiles_rgb]
        batch = torch.stack(tensors).to(self.device)

        # Pad 到 1280 的倍数（RT-DETR 可能需要）
        orig_h, orig_w = batch.shape[2], batch.shape[3]

        with torch.no_grad():
            outputs = self.model(batch)

        # 后处理：decode box + filter by confidence
        orig_sizes = torch.tensor([[orig_h, orig_w]] * len(tiles_rgb)).to(self.device)
        results = self.postprocessor(outputs, orig_sizes)

        return results

    def predict_single(self, image_path) -> list:
        """单图推理：切片 + 全局 → 坐标还原 → 合并。"""
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

        # === 批量推理切片 ===
        tile_batch_rgb = []
        tile_offsets = []
        for td in tiles_data:
            tile_batch_rgb.append(self._grayscale_to_rgb(td["tile"]))
            tile_offsets.append((td["x_offset"], td["y_offset"]))

            if len(tile_batch_rgb) >= self.batch_size:
                batch_results = self._infer_batch(tile_batch_rgb)
                for result, (x_off, y_off) in zip(batch_results, tile_offsets):
                    mapped = self._map_predictions(result, x_off, y_off, self.tile_size, self.tile_size, h, w)
                    all_predictions.extend(mapped)
                tile_batch_rgb = []
                tile_offsets = []

        # 剩余切片
        if tile_batch_rgb:
            batch_results = self._infer_batch(tile_batch_rgb)
            for result, (x_off, y_off) in zip(batch_results, tile_offsets):
                mapped = self._map_predictions(result, x_off, y_off, self.tile_size, self.tile_size, h, w)
                all_predictions.extend(mapped)

        # === 推理全局视图 ===
        global_rgb = self._grayscale_to_rgb(global_view)
        global_tensor = self._preprocess_tile(global_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            global_output = self.model(global_tensor)
        global_orig = torch.tensor([[gh, gw]]).to(self.device)
        global_results = self.postprocessor(global_output, global_orig)

        scale_x = w / gw
        scale_y = h / gh
        for result in global_results:
            boxes = result.get("boxes", [])
            scores = result.get("scores", [])
            labels = result.get("labels", [])
            for box, score, label in zip(boxes, scores, labels):
                score_val = float(score)
                if score_val < self.conf_threshold:
                    continue
                xmin, ymin, xmax, ymax = box.tolist() if isinstance(box, torch.Tensor) else box
                all_predictions.append({
                    "xmin": xmin * scale_x, "ymin": ymin * scale_y,
                    "xmax": xmax * scale_x, "ymax": ymax * scale_y,
                    "score": score_val,
                    "class_id": int(label) - 1,  # COCO id (1-based) → 0-based
                })

        # 裁剪坐标 + 滤除无效框
        valid_preds = []
        for pred in all_predictions:
            pred["xmin"] = max(0.0, min(float(w), pred["xmin"]))
            pred["ymin"] = max(0.0, min(float(h), pred["ymin"]))
            pred["xmax"] = max(0.0, min(float(w), pred["xmax"]))
            pred["ymax"] = max(0.0, min(float(h), pred["ymax"]))
            if pred["xmax"] > pred["xmin"] + 1 and pred["ymax"] > pred["ymin"] + 1:
                valid_preds.append(pred)
        all_predictions = valid_preds

        # DETR 无需 NMS — 每个 query 天然只输出一个目标
        # 按分数排序 → 限制 max_det
        all_predictions.sort(key=lambda x: x["score"], reverse=True)
        all_predictions = all_predictions[:self.max_det]

        return all_predictions

    def _map_predictions(self, result, x_offset, y_offset, tile_w, tile_h, img_w, img_h) -> list:
        """切片预测 → 原图坐标。"""
        boxes = result.get("boxes", [])
        scores = result.get("scores", [])
        labels = result.get("labels", [])

        mapped = []
        for box, score, label in zip(boxes, scores, labels):
            score_val = float(score)
            if score_val < self.conf_threshold:
                continue
            xmin, ymin, xmax, ymax = box.tolist() if isinstance(box, torch.Tensor) else box
            mapped.append({
                "xmin": xmin + x_offset,
                "ymin": ymin + y_offset,
                "xmax": xmax + x_offset,
                "ymax": ymax + y_offset,
                "score": score_val,
                "class_id": int(label) - 1,  # COCO id → 0-based class
            })
        return mapped


def generate_submission(predictor, test_dir: Path, output_path: Path, per_class_conf: dict = None):
    """生成 submission.json。"""
    test_images = sorted(test_dir.glob("*.jpg"))
    if not test_images:
        logger.error(f"No images in {test_dir}")
        return []

    logger.info(f"Predicting {len(test_images)} test images...")
    all_submissions = []
    timings = []

    from tqdm import tqdm

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

    avg_time = np.mean(timings)
    p95_time = np.percentile(timings, 95)
    logger.info(f"Avg time: {avg_time:.2f}s, P95: {p95_time:.2f}s")
    logger.info(f"Total predictions: {len(all_submissions)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_submissions, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved → {output_path}")

    return all_submissions


def validate_submission(submissions: list) -> tuple:
    """校验提交格式。"""
    errors = []
    valid = set(CLASS_NAMES)
    for i, s in enumerate(submissions):
        for f in ["image_id", "category_name", "bbox", "score"]:
            if f not in s:
                errors.append(f"[{i}] missing '{f}'")
        if errors:
            continue
        if s["category_name"] not in valid:
            errors.append(f"[{i}] bad category: {s['category_name']}")
        b = s["bbox"]
        if len(b) != 4 or not (0 <= b[0] < b[2]) or not (0 <= b[1] < b[3]):
            errors.append(f"[{i}] bad bbox: {b}")
        if not (0 <= s["score"] <= 1):
            errors.append(f"[{i}] bad score: {s['score']}")
    return len(errors) == 0, errors


def main():
    parser = argparse.ArgumentParser(description="RT-DETRv2-X Inference")
    parser.add_argument("--weights", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--test-dir", type=str, default=None, help="Test images directory")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conf", type=float, default=0.01)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--per-class-conf", type=str, default=None,
                        help="JSON文件: {\"qilie\": 0.05, ...}")
    args = parser.parse_args()

    test_dir = Path(args.test_dir) if args.test_dir else TEST_ROOT
    output_path = Path(args.output) if args.output else OUTPUT_ROOT / f"submission_rtdetrv2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    # 加载 per-class confidence
    per_class_conf = None
    if args.per_class_conf:
        with open(args.per_class_conf) as f:
            per_class_conf = json.load(f)
    else:
        # 默认阈值（来自 baseline.yaml）
        per_class_conf = {
            "jieba": 0.25, "zonglie": 0.15, "qilie": 0.05,
            "jiaza": 0.20, "yiwuyaru": 0.25, "huashang": 0.10,
            "mamianmakeng": 0.25, "yanghuatiepi": 0.25, "gunyin": 0.25,
        }

    predictor = RTDETRv2Predictor(
        weights_path=args.weights,
        device=args.device,
        conf_threshold=args.conf,
        max_det=300,
        batch_size=args.batch,
    )

    subs = generate_submission(predictor, test_dir, output_path, per_class_conf)
    ok, errs = validate_submission(subs)

    print(f"\n[RT-DETRv2-X Inference Complete]")
    print(f"  Images:      {len(list(test_dir.glob('*.jpg')))}")
    print(f"  Predictions: {len(subs)}")
    print(f"  Output:      {output_path}")
    print(f"  Validation:  {'PASS' if ok else 'FAIL'}")
    if errs:
        for e in errs[:10]:
            print(f"    - {e}")


if __name__ == "__main__":
    main()
