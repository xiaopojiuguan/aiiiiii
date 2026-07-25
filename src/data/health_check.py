"""VOC 数据健康检查模块 —— 对应方案第 6.1 节

只解析 XML，不读取图像像素：
- JPG/XML 按 stem 一一对应检查
- XML 可解析性检查
- 类别严格属于 9 类
- 坐标合法性 (xmin < xmax, ymin < ymax)
- 坐标越界检查
- 完全重复框检查
- 空标注图统计（保留为负样本）
- 从 XML 的 size 字段统计尺寸和尺度
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional
import logging

from src.paths import DATA_ROOT, CLASS_NAMES, ORIGINAL_WIDTH, ORIGINAL_HEIGHT

logger = logging.getLogger(__name__)


def parse_voc_xml(xml_path: Path) -> Optional[dict]:
    """解析单个 VOC XML 文件，返回结构化标注信息。

    Args:
        xml_path: XML 文件路径

    Returns:
        dict 或 None（解析失败）
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        logger.warning(f"XML parse error: {xml_path.name} - {e}")
        return None

    # 图像尺寸
    size = root.find("size")
    if size is None:
        logger.warning(f"No <size> in {xml_path.name}")
        return None

    width = int(size.findtext("width", 0))
    height = int(size.findtext("height", 0))
    depth = int(size.findtext("depth", 0))

    # 标注对象
    objects = []
    for obj in root.findall("object"):
        name = obj.findtext("name", "").strip().lower()
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue

        try:
            xmin = int(bndbox.findtext("xmin", -1))
            ymin = int(bndbox.findtext("ymin", -1))
            xmax = int(bndbox.findtext("xmax", -1))
            ymax = int(bndbox.findtext("ymax", -1))
        except (ValueError, TypeError):
            continue

        objects.append({
            "name": name,
            "bbox": [xmin, ymin, xmax, ymax],
        })

    return {
        "filename": root.findtext("filename", xml_path.stem + ".jpg"),
        "width": width,
        "height": height,
        "depth": depth,
        "objects": objects,
    }


def check_jpg_xml_pairing(data_dir: Path) -> Tuple[List[str], List[str], List[str]]:
    """检查 JPG 和 XML 按 stem 一一对应。

    Returns:
        (paired_stems, orphan_jpg, orphan_xml)
    """
    jpg_stems = {p.stem for p in data_dir.glob("*.jpg")}
    xml_stems = {p.stem for p in data_dir.glob("*.xml")}

    paired = sorted(jpg_stems & xml_stems)
    orphan_jpg = sorted(jpg_stems - xml_stems)
    orphan_xml = sorted(xml_stems - jpg_stems)

    return paired, orphan_jpg, orphan_xml


def check_bbox_validity(objects: list, img_width: int, img_height: int) -> List[str]:
    """检查边界框合法性，返回问题列表。"""
    issues = []
    seen_boxes = set()

    for i, obj in enumerate(objects):
        xmin, ymin, xmax, ymax = obj["bbox"]

        # 坐标顺序
        if xmin >= xmax or ymin >= ymax:
            issues.append(f"Invalid box order: {obj['bbox']}")
            continue

        # 越界
        if xmin < 0 or ymin < 0 or xmax > img_width or ymax > img_height:
            issues.append(f"Box out of bounds: {obj['bbox']} vs ({img_width}x{img_height})")

        # 重复框
        box_key = (xmin, ymin, xmax, ymax, obj["name"])
        if box_key in seen_boxes:
            issues.append(f"Duplicate box: {obj}")
        seen_boxes.add(box_key)

    return issues


def run_health_check(data_dir: Path = None) -> dict:
    """执行完整数据健检，返回统计报告。"""
    if data_dir is None:
        data_dir = DATA_ROOT

    logger.info(f"Running health check on {data_dir}")

    report = {
        "data_dir": str(data_dir),
        "total": {},
        "pairing": {},
        "categories": {},
        "bbox_issues": [],
        "empty_annotations": [],
        "size_stats": defaultdict(list),
        "warnings": [],
    }

    # 1. 文件配对
    paired, orphan_jpg, orphan_xml = check_jpg_xml_pairing(data_dir)
    report["pairing"] = {
        "paired_count": len(paired),
        "orphan_jpg": orphan_jpg,
        "orphan_xml": orphan_xml,
    }
    logger.info(f"Paired: {len(paired)}, Orphan JPG: {len(orphan_jpg)}, Orphan XML: {len(orphan_xml)}")

    # 2. 类别统计
    class_counter = Counter()
    empty_count = 0
    issue_count = 0
    parse_errors = 0

    for stem in paired:
        xml_path = data_dir / f"{stem}.xml"
        parsed = parse_voc_xml(xml_path)

        if parsed is None:
            parse_errors += 1
            continue

        report["size_stats"]["width"].append(parsed["width"])
        report["size_stats"]["height"].append(parsed["height"])
        report["size_stats"]["depth"].append(parsed["depth"])

        objects = parsed["objects"]

        if not objects:
            empty_count += 1
            report["empty_annotations"].append(stem)
            continue

        # 逐框检查
        box_issues = check_bbox_validity(objects, parsed["width"], parsed["height"])
        if box_issues:
            issue_count += len(box_issues)
            report["bbox_issues"].extend([f"{stem}: {i}" for i in box_issues])

        for obj in objects:
            name = obj["name"]
            class_counter[name] += 1

            # 检查类别合法性
            if name not in CLASS_NAMES:
                report["warnings"].append(f"Unknown class '{name}' in {stem}")

    report["categories"] = dict(class_counter)
    report["total"] = {
        "paired_images": len(paired),
        "empty_annotations": empty_count,
        "parse_errors": parse_errors,
        "bbox_issue_count": issue_count,
    }

    # 3. 尺寸统计
    for k, v in report["size_stats"].items():
        if v:
            report["size_stats"][k] = {
                "min": min(v), "max": max(v),
                "mean": sum(v) / len(v),
                "unique": len(set(v)),
            }

    logger.info(f"Categories: {dict(class_counter)}")
    logger.info(f"Empty images: {empty_count}")
    logger.info(f"Parse errors: {parse_errors}")
    logger.info(f"BBox issues: {issue_count}")

    return report


def print_report(report: dict) -> None:
    """Print health check report."""
    print("\n" + "=" * 70)
    print("  SteelGuard-YOLO Data Health Check Report")
    print("=" * 70)

    print(f"\n[Data Dir] {report['data_dir']}")
    t = report["total"]
    p = report["pairing"]
    print(f"\n[Summary] {t['paired_images']} paired images")
    print(f"   Orphan JPG: {len(p['orphan_jpg'])} | Orphan XML: {len(p['orphan_xml'])}")
    print(f"   Empty annotations (negatives): {t['empty_annotations']}")
    print(f"   XML parse errors: {t['parse_errors']}")
    print(f"   BBox issues: {t['bbox_issue_count']}")

    print(f"\n[Class Distribution]:")
    for name in CLASS_NAMES:
        count = report["categories"].get(name, 0)
        bar = "#" * min(count // 20, 30)
        print(f"   {name:<15s}: {count:>5d}  {bar}")

    unknown = {k: v for k, v in report["categories"].items() if k not in CLASS_NAMES}
    if unknown:
        print(f"\n[WARNING] Unknown classes: {unknown}")

    if report["size_stats"]:
        print(f"\n[Image Size Stats]:")
        for k, v in report["size_stats"].items():
            if isinstance(v, dict):
                print(f"   {k}: min={v['min']}, max={v['max']}, "
                      f"unique_values={v['unique']}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    report = run_health_check()
    print_report(report)
