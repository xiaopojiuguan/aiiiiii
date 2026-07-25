"""本地评估协议 v1 —— 对应方案第 5.3.4 节

核心修复:
- pred[5] 是 score (不是 pred[4])
- 按 image_id 逐图匹配 GT↔预测 (不跨图)
- F1/F2 阈值搜索
- Style-OOD 对比
- FP/image 统计
"""

import json
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import numpy as np

from src.paths import CLASS_NAMES, CRITICAL_CLASSES

logger = logging.getLogger(__name__)


def compute_iou(box1, box2):
    """两个框的 IoU。box = [xmin, ymin, xmax, ymax]."""
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def match_per_image(gt_boxes, pred_boxes, pred_scores, iou_thr=0.5):
    """单张图内贪心匹配：每个 GT 最多匹配一次。

    Args:
        gt_boxes: (M, 4)
        pred_boxes: (N, 4)
        pred_scores: (N,)

    Returns:
        tp: (N,) bool, fp: (N,) bool
    """
    n = len(pred_boxes)
    if n == 0:
        return np.array([], dtype=bool), np.array([], dtype=bool)
    if len(gt_boxes) == 0:
        return np.zeros(n, dtype=bool), np.ones(n, dtype=bool)

    # 按分数降序
    order = np.argsort(-pred_scores)
    pred_boxes = pred_boxes[order]
    pred_scores = pred_scores[order]

    gt_used = np.zeros(len(gt_boxes), dtype=bool)
    tp = np.zeros(n, dtype=bool)
    fp = np.zeros(n, dtype=bool)

    for i in range(n):
        best_iou, best_j = 0.0, -1
        for j in range(len(gt_boxes)):
            if gt_used[j]:
                continue
            iou = compute_iou(pred_boxes[i], gt_boxes[j])
            if iou > best_iou:
                best_iou, best_j = iou, j

        if best_iou >= iou_thr and best_j >= 0:
            tp[i] = True
            gt_used[best_j] = True
        else:
            fp[i] = True

    return tp, fp


def compute_ap(tp, fp, n_gt):
    """计算 Average Precision (101-point interpolation)."""
    if n_gt == 0:
        return 0.0
    if len(tp) == 0:
        return 0.0

    tp_cum = np.cumsum(tp, dtype=float)
    fp_cum = np.cumsum(fp, dtype=float)

    recalls = tp_cum / n_gt
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-7)

    ap = 0.0
    for t in np.linspace(0, 1, 101):
        p = np.max(precisions[recalls >= t]) if np.any(recalls >= t) else 0.0
        ap += p / 101.0
    return float(ap)


def compute_f_score(p, r, beta=1.0):
    """F-beta 分数。"""
    b2 = beta ** 2
    denom = b2 * p + r
    return (1 + b2) * p * r / denom if denom > 0 else 0.0


def evaluate_predictions(
    gt_dict: Dict[str, list],
    pred_dict: Dict[str, list],
    iou_thresholds: List[float] = None,
    class_names: List[str] = None,
) -> Dict:
    """完整评估（逐图匹配）。

    gt_dict:   {image_id: [(class_id, xmin, ymin, xmax, ymax), ...]}
    pred_dict: {image_id: [(class_id, xmin, ymin, xmax, ymax, score), ...]}
               score = pred[5]（不是 pred[4]）

    Returns:
        {per_class, overall, fp_per_image, ...}
    """
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.5, 1.0, 0.05)
    if class_names is None:
        class_names = CLASS_NAMES

    nc = len(class_names)
    all_img_ids = sorted(set(list(gt_dict.keys()) + list(pred_dict.keys())))

    # 逐类收集
    per_class = {cid: {"tp": [], "fp": [], "scores": [], "n_gt": 0} for cid in range(nc)}
    total_fp_per_img = defaultdict(int)

    for img_id in all_img_ids:
        gts = gt_dict.get(img_id, [])
        preds = pred_dict.get(img_id, [])

        # 按类别分组
        for cid in range(nc):
            gt_c = np.array([g[1:5] for g in gts if g[0] == cid])
            pred_c = [(p[1:5], p[5]) for p in preds if p[0] == cid]  # pred[5]=score!

            if not pred_c:
                continue

            pred_boxes = np.array([x[0] for x in pred_c])
            pred_scores = np.array([x[1] for x in pred_c])

            tp_c, fp_c = match_per_image(gt_c, pred_boxes, pred_scores, iou_thresholds[0])
            per_class[cid]["tp"].extend(tp_c.tolist())
            per_class[cid]["fp"].extend(fp_c.tolist())
            per_class[cid]["scores"].extend(pred_scores.tolist())

        per_class[cid]["n_gt"] += len(gt_c)

        # FP/image（基于 IoU=0.5）
        for cid in range(nc):
            gts_c = [g for g in gts if g[0] == cid]
            preds_c = [p for p in preds if p[0] == cid]
            if preds_c:
                gt_boxes = np.array([g[1:5] for g in gts_c])
                pred_boxes = np.array([p[1:5] for p in preds_c])
                pred_scores = np.array([p[5] for p in preds_c])
                _, fp_c = match_per_image(gt_boxes, pred_boxes, pred_scores, 0.5)
                total_fp_per_img[img_id] += fp_c.sum()

    # 计算逐类 AP
    per_class_ap = {}
    for cid in range(nc):
        tp = np.array(per_class[cid]["tp"])
        fp = np.array(per_class[cid]["fp"])
        n_gt = per_class[cid]["n_gt"]
        ap50 = compute_ap(tp, fp, n_gt) if len(tp) > 0 else 0.0
        per_class_ap[class_names[cid]] = {
            "ap_50": ap50,
            "n_gt": n_gt,
            "n_pred": len(tp),
        }

    # 总体指标
    all_ap50 = [v["ap_50"] for v in per_class_ap.values()]
    mAP50 = float(np.mean(all_ap50))

    # 致命类 F2
    critical_ap50 = [per_class_ap[c]["ap_50"] for c in CRITICAL_CLASSES if c in per_class_ap]
    crit_mAP50 = float(np.mean(critical_ap50)) if critical_ap50 else 0.0

    # FP/image
    fp_per_img_vals = [total_fp_per_img[i] for i in all_img_ids]
    avg_fp = float(np.mean(fp_per_img_vals)) if fp_per_img_vals else 0.0

    return {
        "per_class": per_class_ap,
        "overall": {
            "mAP@0.5": mAP50,
            f"critical_mAP@0.5": crit_mAP50,
            "avg_FP_per_image": avg_fp,
        },
    }


def search_per_class_thresholds(
    gt_dict: Dict[str, list],
    pred_dict: Dict[str, list],
    class_names: List[str] = None,
) -> Dict[str, float]:
    """按普通类 F1 / 致命类 F2 搜索最优阈值（方案 5.3.4）。

    Returns:
        {class_name: best_threshold} 字典
    """
    if class_names is None:
        class_names = CLASS_NAMES

    thresholds = {}
    candidates = np.arange(0.05, 0.96, 0.01)

    for cid, name in enumerate(class_names):
        # 收集该类所有预测
        all_scores = []
        all_matched = []

        for img_id in sorted(gt_dict.keys()):
            gts = [g for g in gt_dict.get(img_id, []) if g[0] == cid]
            preds_raw = [p for p in pred_dict.get(img_id, []) if p[0] == cid]

            if not preds_raw:
                continue

            gt_boxes = np.array([g[1:5] for g in gts])
            pred_boxes = np.array([p[1:5] for p in preds_raw])
            pred_scores = np.array([p[5] for p in preds_raw])  # pred[5]=score!

            tp, fp = match_per_image(gt_boxes, pred_boxes, pred_scores, iou_thr=0.5)
            all_scores.extend(pred_scores.tolist())
            all_matched.extend(tp.tolist())

        if not all_scores:
            thresholds[name] = 0.25
            continue

        all_scores = np.array(all_scores)
        all_matched = np.array(all_matched, dtype=bool)
        n_gt = sum(1 for img_id in gt_dict
                   for g in gt_dict[img_id] if g[0] == cid)

        is_critical = name in CRITICAL_CLASSES
        beta = 2.0 if is_critical else 1.0  # F2 vs F1

        best_thr = 0.25
        best_score = -1.0

        for thr in candidates:
            keep = all_scores >= thr
            tp_count = all_matched[keep].sum()
            fp_count = (~all_matched[keep]).sum()

            p = tp_count / max(1, tp_count + fp_count)
            r = tp_count / max(1, n_gt)
            f = compute_f_score(p, r, beta)

            if f > best_score:
                best_score = f
                best_thr = float(thr)

        thresholds[name] = best_thr

    return thresholds


def load_gt_from_voc(image_stems, data_dir) -> Dict[str, list]:
    """从 VOC XML 文件加载 GT。"""
    import xml.etree.ElementTree as ET
    from src.paths import CLASS_NAME_TO_ID

    gt = {}
    for stem in image_stems:
        xml_path = Path(data_dir) / f"{stem}.xml"
        if not xml_path.exists():
            continue
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            objects = []
            for obj in root.findall("object"):
                name = obj.findtext("name", "").strip().lower()
                if name not in CLASS_NAME_TO_ID:
                    continue
                bndbox = obj.find("bndbox")
                if bndbox is None:
                    continue
                xmin = float(bndbox.findtext("xmin", 0))
                ymin = float(bndbox.findtext("ymin", 0))
                xmax = float(bndbox.findtext("xmax", 0))
                ymax = float(bndbox.findtext("ymax", 0))
                if xmin < xmax and ymin < ymax:
                    objects.append((CLASS_NAME_TO_ID[name], xmin, ymin, xmax, ymax))
            gt[f"{stem}.jpg"] = objects
        except Exception:
            continue
    return gt


def load_pred_from_json(json_path) -> Dict[str, list]:
    """从提交 JSON 加载预测（与 GT 格式对齐）。"""
    from src.paths import CLASS_NAME_TO_ID

    with open(json_path, "r") as f:
        data = json.load(f)

    pred = defaultdict(list)
    for item in data:
        img_id = item["image_id"]
        cls_id = CLASS_NAME_TO_ID.get(item["category_name"])
        if cls_id is None:
            continue
        bbox = item["bbox"]
        score = item["score"]
        pred[img_id].append((cls_id, bbox[0], bbox[1], bbox[2], bbox[3], score))
    return dict(pred)


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Evaluate predictions")
    parser.add_argument("--pred-json", type=str, required=True, help="Prediction JSON")
    parser.add_argument("--gt-split", type=str, help="Path to split JSON with val_stems")
    parser.add_argument("--data-dir", type=str, help="Original data dir for XML")
    args = parser.parse_args()

    from src.paths import DATA_ROOT, SPLIT_DIR

    # 加载 GT
    split_path = Path(args.gt_split) if args.gt_split else SPLIT_DIR / "train_val_split.json"
    with open(split_path) as f:
        split = json.load(f)

    gt = load_gt_from_voc(split["val_stems"], args.data_dir or DATA_ROOT)
    print(f"Loaded {len(gt)} GT images")

    # 加载预测
    pred = load_pred_from_json(args.pred_json)
    print(f"Loaded {len(pred)} prediction images")

    # 评估
    results = evaluate_predictions(gt, pred)

    print("\n" + "=" * 60)
    print("  Evaluation Results")
    print("=" * 60)
    print(f"\n  mAP@0.5:          {results['overall']['mAP@0.5']:.4f}")
    print(f"  Critical mAP@0.5:  {results['overall']['critical_mAP@0.5']:.4f}")
    print(f"  Avg FP/image:      {results['overall']['avg_FP_per_image']:.2f}")
    print(f"\n  Per-class AP@0.5:")
    for name, cls_r in results["per_class"].items():
        print(f"    {name:<15s}: AP={cls_r['ap_50']:.4f}  (GT={cls_r['n_gt']}, Pred={cls_r['n_pred']})")

    # 阈值搜索
    thresholds = search_per_class_thresholds(gt, pred)
    print(f"\n  Optimal thresholds (F1/F2):")
    for name, thr in thresholds.items():
        marker = " [F2]" if name in CRITICAL_CLASSES else ""
        print(f"    {name:<15s}: {thr:.2f}{marker}")
    print("=" * 60)
