"""Group-Stratified 数据划分模块 —— 对应方案第 7.1 节

按板材/批次编号分组划分 train/val：
- 分组键 = 文件名开头的板材编号（兼容 -/_ 两种命名）
- 保证同板材/批次相邻帧不跨训练/验证
- 确保稀有类（气裂、划伤）在验证集中有代表
- 构建 Style-OOD 验证子集
- 划分泄漏检测
"""

import re
import xml.etree.ElementTree as ET
import json
import logging
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Set
import numpy as np

from src.paths import DATA_ROOT, CLASS_NAMES, SPLIT_DIR

logger = logging.getLogger(__name__)


def extract_batch_key(filename: str) -> str:
    """从文件名提取板材/批次分组键。

    文件名有两种格式:
      0001388270-Raw01-f_00001.jpg  (连字符命名)
      0001469108_Raw02_f_00001.jpg  (下划线命名)
    分组键 = 开头连续数字 (板材编号)，如 "0001388270"

    Args:
        filename: 图像文件名（不含路径）

    Returns:
        板材编号字符串
    """
    stem = Path(filename).stem
    # 匹配文件名开头的连续数字（板材编号）
    match = re.match(r"(\d+)", stem)
    if match:
        return match.group(1)
    # fallback
    return stem


def extract_raw_camera(filename: str) -> str:
    """提取相机位 RawXX。"""
    stem = Path(filename).stem
    match = re.search(r"(Raw\d+)", stem)
    return match.group(1) if match else "unknown"


def get_image_labels(xml_path: Path) -> Set[int]:
    """获取图像的类别标签集合（用于分层抽样）。"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        labels = set()
        for obj in root.findall("object"):
            name = obj.findtext("name", "").strip().lower()
            if name in CLASS_NAMES:
                labels.add(CLASS_NAMES.index(name))
        return labels
    except Exception:
        return set()


def compute_style_features(xml_path: Path) -> dict:
    """计算图像的风格特征，用于构建 Style-OOD 验证子集。"""
    stem = xml_path.stem
    return {
        "batch_group": extract_batch_key(stem),
        "raw_camera": extract_raw_camera(stem),
    }


def create_group_stratified_split(
    data_dir: Path = None,
    val_ratio: float = 0.2,
    random_state: int = 42,
) -> Dict:
    """创建 Group-Stratified 数据划分。

    核心约束：同一个板材编号的所有图像必须在同一集合中（全 train 或全 val）。

    Args:
        data_dir: 数据目录
        val_ratio: 验证集比例
        random_state: 随机种子

    Returns:
        划分结果字典
    """
    if data_dir is None:
        data_dir = DATA_ROOT

    logger.info(f"Creating group-stratified split on {data_dir}")
    rng = np.random.RandomState(random_state)

    # 1. 收集所有样本，按板材编号分组
    all_stems = sorted([p.stem for p in data_dir.glob("*.jpg")])
    plate_to_stems = defaultdict(list)
    for stem in all_stems:
        plate = extract_batch_key(stem)
        plate_to_stems[plate].append(stem)

    plates = sorted(plate_to_stems.keys())
    n_plates = len(plates)
    logger.info(f"Total images: {len(all_stems)}, Unique plates: {n_plates}")
    logger.info(f"Plates with >=10 images: {sum(1 for v in plate_to_stems.values() if len(v) >= 10)}")

    # 2. 收集每个板材的类别标签（用于分层）
    plate_labels = {}
    for plate, stems in plate_to_stems.items():
        all_labels = set()
        for stem in stems:
            xml_path = data_dir / f"{stem}.xml"
            all_labels |= get_image_labels(xml_path)
        plate_labels[plate] = all_labels

    # 3. 确保稀有类（气裂=2, 划伤=5, 夹杂=3）在 val 中有代表
    #    策略：对于包含稀有类的板材，优先分配一部分到 val
    rare_classes = [2, 5, 3]  # qilie, huashang, jiaza
    val_plates = set()
    remaining_plates = set(plates)

    for cid in rare_classes:
        plates_with_class = [p for p in plates if cid in plate_labels[p]]
        if plates_with_class:
            n_val = max(1, int(len(plates_with_class) * val_ratio))
            selected = rng.choice(plates_with_class, size=min(n_val, len(plates_with_class)), replace=False)
            for p in selected:
                if p in remaining_plates:
                    val_plates.add(p)
                    remaining_plates.discard(p)
            logger.info(f"  Class {CLASS_NAMES[cid]}: {len(plates_with_class)} plates, {n_val} selected for val")

    # 4. 补充验证集到目标比例
    target_val_plates = max(1, int(n_plates * val_ratio))
    remaining_list = sorted(remaining_plates)
    rng.shuffle(remaining_list)
    while len(val_plates) < target_val_plates and remaining_list:
        val_plates.add(remaining_list.pop())

    logger.info(f"Val plates: {len(val_plates)}/{n_plates} (target: {target_val_plates})")

    # 5. 泄漏检测：确认没有板材同时出现在训练和验证
    train_plates = set(plates) - val_plates
    overlap = train_plates & val_plates
    if overlap:
        logger.error(f"LEAK DETECTED: {len(overlap)} plates in both train and val!")
        logger.error(f"  Overlapping plates: {sorted(overlap)[:10]}...")
    else:
        logger.info("Leak check passed: 0 plates shared between train and val")

    # 6. 映射回 stem 列表
    train_stems = []
    val_stems = []
    for plate in train_plates:
        train_stems.extend(plate_to_stems[plate])
    for plate in val_plates:
        val_stems.extend(plate_to_stems[plate])

    # 排序保证可复现
    train_stems = sorted(train_stems)
    val_stems = sorted(val_stems)

    logger.info(f"Split result: train={len(train_stems)}, val={len(val_stems)}")

    # 7. 统计验证集类别
    val_labels = Counter()
    train_labels = Counter()
    for stem in val_stems:
        xml_path = data_dir / f"{stem}.xml"
        for lid in get_image_labels(xml_path):
            val_labels[CLASS_NAMES[lid]] += 1
    for stem in train_stems:
        xml_path = data_dir / f"{stem}.xml"
        for lid in get_image_labels(xml_path):
            train_labels[CLASS_NAMES[lid]] += 1

    logger.info(f"Val class distribution: {dict(val_labels)}")

    # 检查每个类是否都在 val 中有代表
    for cid, name in enumerate(CLASS_NAMES):
        if val_labels.get(name, 0) == 0:
            logger.warning(f"Class '{name}' has 0 instances in val set!")

    # 8. 构建 Style-OOD 子集（基于相机位划分）
    val_camera_groups = defaultdict(list)
    for stem in val_stems:
        cam = extract_raw_camera(stem)
        val_camera_groups[cam].append(stem)

    # Style-OOD: 取样本数最少的 1/3 相机位
    camera_sizes = {k: len(v) for k, v in val_camera_groups.items()}
    sorted_cams = sorted(camera_sizes, key=camera_sizes.get)
    n_ood_cams = max(1, len(sorted_cams) // 3)
    ood_cameras = sorted_cams[:n_ood_cams]

    style_ood_stems = []
    for cam in ood_cameras:
        style_ood_stems.extend(val_camera_groups[cam])

    logger.info(f"Style-OOD: {len(style_ood_stems)} stems from cameras {ood_cameras}")

    # 9. 保存划分
    split_data = {
        "config": {
            "val_ratio": val_ratio,
            "random_state": random_state,
            "total_plates": n_plates,
            "train_plates": len(train_plates),
            "val_plates": len(val_plates),
        },
        "train_stems": train_stems,
        "val_stems": val_stems,
        "style_ood_stems": style_ood_stems,
        "train_class_distribution": dict(train_labels),
        "val_class_distribution": dict(val_labels),
        "plate_groups": {
            "train_plates": sorted(train_plates),
            "val_plates": sorted(val_plates),
        },
    }

    # 保存到文件
    split_path = SPLIT_DIR / "train_val_split.json"
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split_data, f, ensure_ascii=False, indent=2)

    logger.info(f"Split saved to {split_path}")

    return split_data


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    split = create_group_stratified_split()

    # 摘要
    print(f"\n[Split Complete]:")
    print(f"   Train: {len(split['train_stems'])} images ({split['config']['train_plates']} plates)")
    print(f"   Val:   {len(split['val_stems'])} images ({split['config']['val_plates']} plates)")
    print(f"   Style-OOD: {len(split['style_ood_stems'])} images")
    print(f"   Val class dist: {split['val_class_distribution']}")
