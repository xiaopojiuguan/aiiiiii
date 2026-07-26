"""稀有缺陷 Copy-Paste 增强 —— 对应方案第 5.3.2 节

递进实验：
- CP-0: 矩形基线（复制目标框，粘贴到亮度/纹理相近的 ROI）
- CP-1: 边缘羽化（软边消除硬边）
"""

import logging
from pathlib import Path
from typing import List, Dict, Tuple
import json

import cv2
import numpy as np

from src.paths import CLASS_NAMES, DATA_ROOT, SPLIT_DIR
from src.data.health_check import parse_voc_xml

logger = logging.getLogger(__name__)

# 稀有类 ID（需要 Copy-Paste 的类）
RARE_CLASS_IDS = {2, 5, 3, 1}  # qilie=2, huashang=5, jiaza=3, zonglie=1


def extract_defect_patches(
    data_dir: Path,
    stem_list: List[str],
    target_class_ids: set = None,
) -> Dict[int, list]:
    """从训练集中提取稀有缺陷的局部图像块。

    Returns:
        {class_id: [(image_patch, width, height), ...]}
    """
    if target_class_ids is None:
        target_class_ids = RARE_CLASS_IDS

    patches = {cid: [] for cid in target_class_ids}

    for stem in stem_list:
        xml_path = data_dir / f"{stem}.xml"
        jpg_path = data_dir / f"{stem}.jpg"

        parsed = parse_voc_xml(xml_path)
        if not parsed or not parsed["objects"]:
            continue

        image = cv2.imread(str(jpg_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue

        for obj in parsed["objects"]:
            from src.paths import CLASS_NAME_TO_ID
            name = obj["name"]
            if name not in CLASS_NAME_TO_ID:
                continue
            cid = CLASS_NAME_TO_ID[name]
            if cid not in target_class_ids:
                continue

            xmin, ymin, xmax, ymax = [int(v) for v in obj["bbox"]]
            xmin = max(0, xmin); ymin = max(0, ymin)
            xmax = min(image.shape[1], xmax); ymax = min(image.shape[0], ymax)

            if xmax <= xmin + 2 or ymax <= ymin + 2:
                continue

            patch = image[ymin:ymax, xmin:xmax].copy()
            patches[cid].append({
                "patch": patch,
                "width": xmax - xmin,
                "height": ymax - ymin,
                "source_stem": stem,
                "bbox": (xmin, ymin, xmax, ymax),
            })

    total = sum(len(v) for v in patches.values())
    logger.info(f"Extracted {total} defect patches from rare classes")
    for cid in target_class_ids:
        name = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else f"cls_{cid}"
        logger.info(f"  {name}: {len(patches[cid])} patches")

    return patches


def cp0_rectangular_paste(
    tile_image: np.ndarray,
    patch_info: dict,
    paste_x: int,
    paste_y: int,
    tile_size: int = 1280,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """CP-0: 矩形基线粘贴。

    直接将缺陷块复制到目标位置（硬边）。

    Returns:
        (augmented_tile, (xmin, ymin, xmax, ymax))
    """
    patch = patch_info["patch"]
    pw, ph = patch.shape[1], patch.shape[0]

    # 确保不越界
    pw = min(pw, tile_size - paste_x)
    ph = min(ph, tile_size - paste_y)

    if pw <= 2 or ph <= 2:
        return tile_image, (0, 0, 0, 0)

    resized_patch = cv2.resize(patch, (pw, ph))

    result = tile_image.copy()
    result[paste_y:paste_y + ph, paste_x:paste_x + pw] = resized_patch

    bbox = (paste_x, paste_y, paste_x + pw, paste_y + ph)
    return result, bbox


def cp1_feathered_paste(
    tile_image: np.ndarray,
    patch_info: dict,
    paste_x: int,
    paste_y: int,
    tile_size: int = 1280,
    feather_width: int = 8,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """CP-1: 边缘羽化粘贴。

    使用距离变换创建软边掩膜，消除矩形硬边。
    """
    patch = patch_info["patch"]
    pw, ph = patch.shape[1], patch.shape[0]

    pw = min(pw, tile_size - paste_x)
    ph = min(ph, tile_size - paste_y)

    if pw <= 2 or ph <= 2:
        return tile_image, (0, 0, 0, 0)

    resized_patch = cv2.resize(patch, (pw, ph))

    # 创建羽化掩膜（边界处透明度渐变）
    mask = np.ones((ph, pw), dtype=np.float32)
    fw = min(feather_width, pw // 4, ph // 4)
    if fw > 0:
        mask[:fw, :] = np.linspace(0, 1, fw)[:, np.newaxis]
        mask[-fw:, :] = np.linspace(1, 0, fw)[:, np.newaxis]
        mask[:, :fw] = np.minimum(mask[:, :fw], np.linspace(0, 1, fw)[np.newaxis, :])
        mask[:, -fw:] = np.minimum(mask[:, -fw:], np.linspace(1, 0, fw)[np.newaxis, :])

    # Alpha 混合
    result = tile_image.copy()
    bg = result[paste_y:paste_y + ph, paste_x:paste_x + pw].astype(np.float32)
    fg = resized_patch.astype(np.float32)
    blended = (fg * mask + bg * (1 - mask)).astype(np.uint8)
    result[paste_y:paste_y + ph, paste_x:paste_x + pw] = blended

    bbox = (paste_x, paste_y, paste_x + pw, paste_y + ph)
    return result, bbox


def apply_copy_paste_to_tiles(
    tile_src_dir: Path,
    patches: Dict[int, list],
    output_img_dir: Path,
    output_lbl_dir: Path,
    max_per_tile: int = 3,
    max_new_tiles: int = 3000,
    cp_version: str = "cp1",
) -> int:
    """将 Copy-Paste 应用到现有的 tile 训练集。

    选择部分背景 tile，粘贴稀有缺陷，生成新的合成训练样本。

    Args:
        tile_src_dir: 源 tile 目录（含 images/ 和 labels/）
        patches: extract_defect_patches() 的输出
        output_img_dir, output_lbl_dir: 输出路径
        max_per_tile: 每张 tile 最多粘贴数
        max_new_tiles: 最多生成新 tile 数
        cp_version: "cp0" 或 "cp1"

    Returns:
        生成的新 tile 数
    """
    import random
    random.seed(42)

    paste_fn = cp1_feathered_paste if cp_version == "cp1" else cp0_rectangular_paste

    img_dir = tile_src_dir / "images"
    lbl_dir = tile_src_dir / "labels"
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_lbl_dir.mkdir(parents=True, exist_ok=True)

    # 找背景 tile（标签为空的 tile）
    bg_tiles = []
    for lbl_path in sorted(lbl_dir.glob("*.txt")):
        if lbl_path.stat().st_size == 0:  # 空文件 = 背景
            stem = lbl_path.stem
            img_path = img_dir / f"{stem}.jpg"
            if img_path.exists():
                bg_tiles.append(stem)

    logger.info(f"Found {len(bg_tiles)} background tiles for Copy-Paste")
    random.shuffle(bg_tiles)

    # 构建按类别索引的 patch 列表
    class_patches = {cid: plist for cid, plist in patches.items() if plist}

    # 计算每类目标生成数（稀有类更多）
    total_patches = sum(len(v) for v in class_patches.values())
    class_weights_inv = {cid: 1.0 / max(len(class_patches[cid]), 1) for cid in class_patches}
    total_w = sum(class_weights_inv.values())
    class_probs = {cid: w / total_w for cid, w in class_weights_inv.items()}

    count = 0
    class_counts = defaultdict(int)
    for stem in bg_tiles:
        if count >= max_new_tiles:
            break

        img_path = img_dir / f"{stem}.jpg"
        tile = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if tile is None:
            continue

        new_labels = []
        augmented = tile.copy()
        n_pasted = 0

        for _ in range(max_per_tile):
            # ---- 修复：先选类别（稀有类优先），再选该类的 patch ----
            cids = list(class_probs.keys())
            probs = [class_probs[c] for c in cids]
            cid = random.choices(cids, weights=probs, k=1)[0]
            patch_info = random.choice(class_patches[cid])

            # 随机粘贴位置（ROI 内）
            paste_x = random.randint(0, 1280 - min(patch_info["width"], 200) - 50)
            paste_y = random.randint(0, 1280 - min(patch_info["height"], 200) - 50)

            result, bbox = paste_fn(augmented, patch_info, paste_x, paste_y)
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue

            augmented = result
            # YOLO 归一化标签
            xc = (bbox[0] + bbox[2]) / 2.0 / 1280
            yc = (bbox[1] + bbox[3]) / 2.0 / 1280
            nw = (bbox[2] - bbox[0]) / 1280
            nh = (bbox[3] - bbox[1]) / 1280
            new_labels.append((cid, xc, yc, nw, nh))
            n_pasted += 1
            class_counts[cid] += 1

        if n_pasted == 0:
            continue

        # 保存
        new_name = f"cp_{stem}"
        tile_rgb = cv2.cvtColor(augmented, cv2.COLOR_GRAY2RGB)
        cv2.imwrite(str(output_img_dir / f"{new_name}.jpg"), tile_rgb)

        with open(output_lbl_dir / f"{new_name}.txt", "w") as f:
            for cid, xc, yc, nw, nh in new_labels:
                f.write(f"{cid} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n")

        count += 1

    logger.info(f"Generated {count} Copy-Paste tiles ({cp_version})")
    for cid, cnt in sorted(class_counts.items()):
        name = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else f"cls_{cid}"
        logger.info(f"  {name}: {cnt} pasted")
    return count


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)

    # 加载训练 stem
    with open(SPLIT_DIR / "train_val_split.json", "r") as f:
        split = json.load(f)

    # 提取稀有缺陷块
    patches = extract_defect_patches(DATA_ROOT, split["train_stems"])

    # 应用到 tile 训练集
    tile_src = DATA_ROOT.parent / "tiles_train"
    cp_output = DATA_ROOT.parent / "tiles_train_cp"
    n = apply_copy_paste_to_tiles(
        tile_src, patches,
        output_img_dir=cp_output / "images",
        output_lbl_dir=cp_output / "labels",
        cp_version="cp1",
    )

    # 合并到训练集
    import shutil
    for f in cp_output.glob("images/*.jpg"):
        shutil.copy2(f, tile_src / "images" / f.name)
    for f in cp_output.glob("labels/*.txt"):
        shutil.copy2(f, tile_src / "labels" / f.name)
    print(f"Merged {n} Copy-Paste tiles into training set")
