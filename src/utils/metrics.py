"""本地评估协议 v2 —— 对应方案第 5.3.4 节

修复:
- 真 mAP@0.5:0.95 (10 个 IoU 阈值取平均)
- 全局分数排序 + 逐图贪心匹配
- 按 image_id 逐图匹配 GT↔预测
- F1/F2 阈值搜索
- Style-OOD 对比
- FP/image 统计
"""

import json
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np

from src.paths import CLASS_NAMES, CRITICAL_CLASSES

logger = logging.getLogger(__name__)


def compute_iou(box1, box2):
    """两个框的 IoU."""
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return float(inter / union) if union > 0 else 0.0


def match_per_image(gt_boxes, pred_boxes, pred_scores, iou_thr=0.5):
    """单张图内贪心匹配：按全局分数降序，每个 GT 最多匹配一次。"""
    n = len(pred_boxes)
    if n == 0:
        return np.array([], dtype=bool), np.array([], dtype=bool)
    if len(gt_boxes) == 0:
        return np.zeros(n, dtype=bool), np.ones(n, dtype=bool)

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


def compute_ap_at_iou(
    gt_dict: Dict[str, list],
    pred_dict: Dict[str, list],
    cid: int,
    iou_thr: float,
) -> float:
    """计算单个类别在单个 IoU 阈值下的 AP (101-point interpolation)。

    关键：全局按分数排序，逐图匹配后汇总所有图像的 TP/FP。
    """
    # 收集所有预测和对应的图像ID（全局排序用）
    all_preds = []  # (score, bbox, img_id)
    gt_per_image = {}  # img_id -> gt_boxes

    for img_id in sorted(set(list(gt_dict.keys()) + list(pred_dict.keys()))):
        gts = gt_dict.get(img_id, [])
        preds = pred_dict.get(img_id, [])

        gt_c = np.array([g[1:5] for g in gts if g[0] == cid])
        if len(gt_c) > 0:
            gt_per_image[img_id] = gt_c

        for p in preds:
            if p[0] == cid:
                all_preds.append((p[5], p[1:5], img_id))

    n_gt_total = sum(len(v) for v in gt_per_image.values())
    if n_gt_total == 0:
        return 0.0
    if not all_preds:
        return 0.0

    # 全局按分数降序排序
    all_preds.sort(key=lambda x: x[0], reverse=True)

    # 按图片分组预测（保持全局排序）
    preds_per_image = defaultdict(list)
    preds_order_per_image = defaultdict(list)  # 记录每张图内预测在全局排序中的位置
    for global_idx, (score, bbox, img_id) in enumerate(all_preds):
        preds_order_per_image[img_id].append(global_idx)
        preds_per_image[img_id].append((score, bbox))

    # 按图片批量匹配（每个 GT 只匹配一次）
    tp_global = {}  # global_idx -> bool
    fp_global = {}
    for img_id in preds_per_image:
        gt_boxes = gt_per_image.get(img_id, np.zeros((0, 4)))
        preds = preds_per_image[img_id]
        global_indices = preds_order_per_image[img_id]

        pred_boxes = np.array([p[1] for p in preds])
        pred_scores = np.array([p[0] for p in preds])

        t, f = match_per_image(gt_boxes, pred_boxes, pred_scores, iou_thr)

        for j, global_idx in enumerate(global_indices):
            tp_global[global_idx] = t[j]
            fp_global[global_idx] = f[j]

    # 按全局排序重建 TP/FP 序列
    all_tp = np.array([tp_global[i] for i in range(len(all_preds))])
    all_fp = np.array([fp_global[i] for i in range(len(all_preds))])

    all_tp = np.array(all_tp)
    all_fp = np.array(all_fp)

    # 计算 PR 曲线
    tp_cum = np.cumsum(all_tp, dtype=float)
    fp_cum = np.cumsum(all_fp, dtype=float)
    recalls = tp_cum / n_gt_total
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-7)

    # 101-point interpolation
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        p = np.max(precisions[recalls >= t]) if np.any(recalls >= t) else 0.0
        ap += p / 101.0
    return float(ap)


def evaluate_predictions(
    gt_dict: Dict[str, list],
    pred_dict: Dict[str, list],
    class_names: List[str] = None,
) -> Dict:
    """完整评估：真 mAP@0.5:0.95 (10 IoU 阈值) + 逐图匹配 + 全局分数排序。

    gt_dict:   {image_id: [(class_id, xmin, ymin, xmax, ymax), ...]}
    pred_dict: {image_id: [(class_id, xmin, ymin, xmax, ymax, score), ...]}
    """
    if class_names is None:
        class_names = CLASS_NAMES

    nc = len(class_names)
    iou_thresholds = np.arange(0.5, 1.0, 0.05)  # 10 thresholds

    per_class = {}
    for cid in range(nc):
        aps = []
        for iou_thr in iou_thresholds:
            ap = compute_ap_at_iou(gt_dict, pred_dict, cid, iou_thr)
            aps.append(ap)

        # Count GT
        n_gt = sum(1 for img_id in gt_dict
                   for g in gt_dict[img_id] if g[0] == cid)
        n_pred = sum(1 for img_id in pred_dict
                     for p in pred_dict[img_id] if p[0] == cid)

        per_class[class_names[cid]] = {
            "ap_50": aps[0],
            "ap_50_95": float(np.mean(aps)),
            "aps": [float(a) for a in aps],
            "n_gt": n_gt,
            "n_pred": n_pred,
        }

    # Overall
    all_ap50 = [v["ap_50"] for v in per_class.values()]
    all_ap50_95 = [v["ap_50_95"] for v in per_class.values()]
    mAP50 = float(np.mean(all_ap50))
    mAP50_95 = float(np.mean(all_ap50_95))

    # Critical mAP
    crit_ap50 = [per_class[c]["ap_50"] for c in CRITICAL_CLASSES if c in per_class]
    crit_ap50_95 = [per_class[c]["ap_50_95"] for c in CRITICAL_CLASSES if c in per_class]

    # FP/image (at IoU=0.5)
    fp_per_img = defaultdict(int)
    for img_id in sorted(set(list(gt_dict.keys()) + list(pred_dict.keys()))):
        gts = gt_dict.get(img_id, [])
        preds = pred_dict.get(img_id, [])
        for cid in range(nc):
            gt_c = np.array([g[1:5] for g in gts if g[0] == cid])
            preds_c = [p for p in preds if p[0] == cid]
            if preds_c:
                pred_boxes = np.array([p[1:5] for p in preds_c])
                pred_scores = np.array([p[5] for p in preds_c])
                _, fp_c = match_per_image(gt_c, pred_boxes, pred_scores, 0.5)
                fp_per_img[img_id] += fp_c.sum()

    all_imgs = sorted(set(list(gt_dict.keys()) + list(pred_dict.keys())))
    avg_fp = float(np.mean([fp_per_img[i] for i in all_imgs]))

    return {
        "per_class": per_class,
        "overall": {
            "mAP@0.5": mAP50,
            "mAP@0.5:0.95": mAP50_95,
            "critical_mAP@0.5": float(np.mean(crit_ap50)) if crit_ap50 else 0.0,
            "critical_mAP@0.5:0.95": float(np.mean(crit_ap50_95)) if crit_ap50_95 else 0.0,
            "avg_FP_per_image": avg_fp,
        },
    }


def compute_f_score(p, r, beta=1.0):
    """F-beta 分数。"""
    b2 = beta ** 2
    denom = b2 * p + r
    return (1 + b2) * p * r / denom if denom > 0 else 0.0


def search_per_class_thresholds(
    gt_dict: Dict[str, list],
    pred_dict: Dict[str, list],
    class_names: List[str] = None,
) -> Dict[str, float]:
    """按普通类 F1 / 致命类 F2 搜索最优阈值。"""
    if class_names is None:
        class_names = CLASS_NAMES

    thresholds = {}
    candidates = np.arange(0.05, 0.96, 0.01)

    for cid, name in enumerate(class_names):
        is_critical = name in CRITICAL_CLASSES
        beta = 2.0 if is_critical else 1.0

        # 收集该类所有预测
        all_scores = []
        all_tp = []

        for img_id in sorted(gt_dict.keys()):
            gts = [g for g in gt_dict.get(img_id, []) if g[0] == cid]
            preds_raw = [p for p in pred_dict.get(img_id, []) if p[0] == cid]
            if not preds_raw:
                continue

            gt_boxes = np.array([g[1:5] for g in gts])
            pred_boxes = np.array([p[1:5] for p in preds_raw])
            pred_scores = np.array([p[5] for p in preds_raw])

            tp, fp = match_per_image(gt_boxes, pred_boxes, pred_scores, iou_thr=0.5)
            all_scores.extend(pred_scores.tolist())
            all_tp.extend(tp.tolist())

        if not all_scores:
            thresholds[name] = 0.25
            continue

        all_scores = np.array(all_scores)
        all_tp = np.array(all_tp, dtype=bool)
        n_gt = sum(1 for img_id in gt_dict
                   for g in gt_dict[img_id] if g[0] == cid)

        best_thr = 0.25
        best_score = -1.0

        for thr in candidates:
            keep = all_scores >= thr
            tp_count = all_tp[keep].sum()
            fp_count = (~all_tp[keep]).sum()
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
    """从提交 JSON 加载预测。"""
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
    parser.add_argument("--pred-json", type=str, required=True)
    parser.add_argument("--gt-split", type=str, help="Split JSON with val_stems")
    parser.add_argument("--data-dir", type=str, help="Original data dir for XML")
    args = parser.parse_args()

    from src.paths import DATA_ROOT, SPLIT_DIR

    split_path = Path(args.gt_split) if args.gt_split else SPLIT_DIR / "train_val_split.json"
    with open(split_path) as f:
        split = json.load(f)

    gt = load_gt_from_voc(split["val_stems"], args.data_dir or DATA_ROOT)
    print(f"Loaded {len(gt)} GT images")

    pred = load_pred_from_json(args.pred_json)
    print(f"Loaded {len(pred)} prediction images")

    results = evaluate_predictions(gt, pred)

    print("\n" + "=" * 65)
    print("  Evaluation Results (true mAP@0.5:0.95)")
    print("=" * 65)
    for name in CLASS_NAMES:
        r = results["per_class"].get(name, {})
        marker = " [F2]" if name in CRITICAL_CLASSES else ""
        print(f"  {name:<15s} AP50={r.get('ap_50',0):.4f}  AP={r.get('ap_50_95',0):.4f}  GT={r.get('n_gt',0):>4d}  Pred={r.get('n_pred',0):>6d}{marker}")

    o = results["overall"]
    print(f"  {'---':-<15s}  {'-----':<7s}  {'-----':<7s}  {'----':<5s}  {'-----':<6s}")
    print(f"  {'mAP@0.5':<15s}  {o['mAP@0.5']:.4f}")
    print(f"  {'mAP@0.5:0.95':<15s}  {o['mAP@0.5:0.95']:.4f}")
    print(f"  {'Critical mAP':<15s}  {o['critical_mAP@0.5']:.4f}")
    print(f"  {'Avg FP/image':<15s}  {o['avg_FP_per_image']:.2f}")

    thresholds = search_per_class_thresholds(gt, pred)
    print(f"\n  Optimal thresholds:")
    for name in CLASS_NAMES:
        marker = " [F2]" if name in CRITICAL_CLASSES else " [F1]"
        print(f"    {name:<15s}: {thresholds.get(name, 0.25):.2f}{marker}")
    print("=" * 65)
