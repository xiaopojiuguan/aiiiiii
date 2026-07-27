#!/usr/bin/env python3
"""
模型对比评估脚本
================
对比 YOLO 基线 / RT-DETRv2-X / DEIM-D-FINE-X 的 submission.json，
输出类别级和全局指标对比表。

用法:
  python plans/compare_results.py \
      --baseline outputs/submission.json \
      --rtdetrv2 outputs/submission_rtdetrv2_XXX.json \
      --dfine outputs/submission_dfine_XXX.json
"""

import json, argparse, sys
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import CLASS_NAMES


def load_submission(path: str) -> dict:
    """加载 submission.json，按 image_id 组织。"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    by_image = defaultdict(list)
    by_class = defaultdict(list)
    for entry in data:
        by_image[entry["image_id"]].append(entry)
        by_class[entry["category_name"]].append(entry)

    return {
        "raw": data,
        "by_image": dict(by_image),
        "by_class": dict(by_class),
    }


def compute_stats(submission: dict) -> dict:
    """计算统计信息。"""
    by_class = submission["by_class"]
    total = len(submission["raw"])
    images = len(submission["by_image"])

    class_stats = {}
    for cls_name in CLASS_NAMES:
        preds = by_class.get(cls_name, [])
        class_stats[cls_name] = {
            "count": len(preds),
            "per_image": len(preds) / images if images > 0 else 0,
        }

    return {
        "total_preds": total,
        "total_images": images,
        "per_image": total / images if images > 0 else 0,
        "by_class": class_stats,
    }


def print_comparison(results: dict):
    """打印对比表。"""
    models = list(results.keys())
    print("\n" + "=" * 80)
    print("Model Comparison")
    print("=" * 80)

    # 全局统计
    print(f"\n{'Metric':<30}", end="")
    for m in models:
        print(f"{m:>18}", end="")
    print()
    print("-" * (30 + 18 * len(models)))

    metrics = [
        ("Total Predictions", "total_preds"),
        ("Total Images", "total_images"),
        ("Avg preds/image", "per_image"),
    ]
    for label, key in metrics:
        print(f"{label:<30}", end="")
        for m in models:
            val = results[m].get(key, "-")
            if isinstance(val, float):
                print(f"{val:>18.2f}", end="")
            else:
                print(f"{val:>18}", end="")
        print()

    # 类别级统计
    print(f"\n--- Per-Class Predictions ---")
    print(f"{'Class':<20}", end="")
    for m in models:
        print(f"{m:>18}", end="")
    print()
    print("-" * (20 + 18 * len(models)))

    for cls_name in CLASS_NAMES:
        print(f"{cls_name:<20}", end="")
        for m in models:
            cls_stats = results[m].get("by_class", {}).get(cls_name, {})
            count = cls_stats.get("count", 0)
            per_img = cls_stats.get("per_image", 0)
            print(f"{count:>8} ({per_img:>5.1f}/img)", end="")
        print()

    # 稀有类汇总
    rare_classes = ["qilie", "zonglie", "huashang", "yiwuyaru"]
    print(f"\n--- Rare Class Summary ---")
    print(f"{'Class':<20}", end="")
    for m in models:
        print(f"{'Preds':>10} {'PerImg':>7}", end="")
    print()
    for cls_name in rare_classes:
        print(f"{cls_name:<20}", end="")
        for m in models:
            cls_stats = results[m].get("by_class", {}).get(cls_name, {})
            count = cls_stats.get("count", 0)
            per_img = cls_stats.get("per_image", 0)
            print(f"{count:>10} {per_img:>7.2f}", end="")
        print()

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Compare model submissions")
    parser.add_argument("--baseline", type=str, help="YOLO baseline submission.json")
    parser.add_argument("--rtdetrv2", type=str, help="RT-DETRv2-X submission.json")
    parser.add_argument("--dfine", type=str, help="DEIM-D-FINE-X submission.json")
    parser.add_argument("--files", type=str, nargs="+", help="Custom: NAME=PATH pairs")
    args = parser.parse_args()

    results = {}

    # 命名模型
    named_files = {}
    if args.baseline:
        named_files["YOLO11m-P2"] = args.baseline
    if args.rtdetrv2:
        named_files["RT-DETRv2-X"] = args.rtdetrv2
    if args.dfine:
        named_files["DEIM-D-FINE-X"] = args.dfine

    if args.files:
        for item in args.files:
            name, path = item.split("=", 1)
            named_files[name] = path

    if not named_files:
        print("Usage: compare_results.py --baseline <json> --rtdetrv2 <json> --dfine <json>")
        print("   or: compare_results.py --files YOLO=sub1.json DETR=sub2.json")
        return

    for name, path in named_files.items():
        if not Path(path).exists():
            print(f"WARNING: {name} not found: {path}")
            continue
        sub = load_submission(path)
        results[name] = compute_stats(sub)
        print(f"Loaded {name}: {results[name]['total_preds']} preds, "
              f"{results[name]['total_images']} images")

    if len(results) < 2:
        print("\nNeed at least 2 valid submission files to compare.")
        return

    print_comparison(results)


if __name__ == "__main__":
    main()
