#!/usr/bin/env python3
"""
方案1: RT-DETRv2-X (ResNet101) 全量训练
========================================
- 仓库: lyuwenyu/RT-DETR (PyTorch)
- 数据: COCO JSON (自动调用 convert_to_coco.py)
- 配置: configs/{dataset,model,runtime}.yml 三个文件分拆
- 断点续训: --resume 自动找最近 checkpoint

用法:
  python plans/rtdetrv2_x/train.py              # 首次训练
  python plans/rtdetrv2_x/train.py --resume     # 打断后继续
  python plans/rtdetrv2_x/train.py --epochs 100 # 自定义轮数
"""

import sys, os, argparse, logging, subprocess, json, shutil
from pathlib import Path
from datetime import datetime

# ==== 路径配置 ====
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLAN_DIR = Path(__file__).resolve().parent
CONFIG_DIR = PLAN_DIR / "configs"
CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"
COCO_DIR = PROJECT_ROOT / "outputs" / "coco_annotations"
RTDETR_DIR = PROJECT_ROOT / "RT-DETR"          # lyuwenyu/RT-DETR 仓库位置

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rtdetrv2_x")


def print_env():
    import torch
    logger.info("=" * 60)
    logger.info("RT-DETRv2-X (ResNet101) Training")
    logger.info(f"  Python:   {sys.version.split()[0]}")
    logger.info(f"  PyTorch:  {torch.__version__}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            logger.info(f"  GPU[{i}]:   {p.name} ({p.total_memory/1024**3:.1f} GB)")
    logger.info(f"  Configs:  {CONFIG_DIR}")
    logger.info(f"  Output:   {CHECKPOINT_DIR}")
    logger.info("=" * 60)


def setup_rtdetr() -> bool:
    """确保 lyuwenyu/RT-DETR 仓库可用 (PyTorch 分支)。"""
    if RTDETR_DIR.exists() and (RTDETR_DIR / "src").exists():
        logger.info("RT-DETR repo found.")
        return True

    logger.info("Cloning RT-DETR (PyTorch) from GitHub...")
    try:
        subprocess.run(
            ["git", "clone", "https://github.com/lyuwenyu/RT-DETR.git",
             str(RTDETR_DIR)],
            check=True, cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        # 安装依赖
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", "."],
            check=False, cwd=str(RTDETR_DIR),
        )
        logger.info("RT-DETR setup complete.")
        return True
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        return False


def convert_data() -> bool:
    """调用 dfine 的转换脚本生成 COCO JSON。"""
    convert_script = PROJECT_ROOT / "plans" / "dfine" / "convert_to_coco.py"

    if COCO_DIR.exists() and (COCO_DIR / "train.json").exists():
        train_size = (COCO_DIR / "train.json").stat().st_size
        logger.info(f"COCO annotations exist ({train_size/1024/1024:.1f} MB)")
        return True

    logger.info("Converting YOLO → COCO...")
    result = subprocess.run(
        [sys.executable, str(convert_script), "--split", "all"],
        cwd=str(PROJECT_ROOT),
    )
    return result.returncode == 0


def build_final_config(args, exp_name: str, output_dir: Path) -> Path:
    """把三个分拆 yml 拼成 RT-DETR 可读的单一配置文件。"""
    import yaml

    with open(PROJECT_ROOT / "tile_dataset.yaml") as f:
        tile_cfg = yaml.safe_load(f)

    data_root = PROJECT_ROOT / tile_cfg["path"]
    train_img = str((data_root / tile_cfg["train"]).resolve())
    val_img = str((data_root / tile_cfg["val"]).resolve())
    train_json = str((COCO_DIR / "train.json").resolve())
    val_json = str((COCO_DIR / "val.json").resolve())

    epochs = args.epochs or 72
    batch = args.batch or 4
    val_batch = batch * 2

    dynamic_aug_end = max(1, epochs - 2)  # 最后 2 epoch 关强增强

    # 读取三个配置模板, 做变量替换
    configs = {}
    for name in ["dataset", "model", "runtime"]:
        with open(CONFIG_DIR / f"{name}.yml", encoding="utf-8") as f:
            raw = f.read()
        # 替换占位符
        raw = raw.replace("{TRAIN_IMG_DIR}", train_img)
        raw = raw.replace("{VAL_IMG_DIR}", val_img)
        raw = raw.replace("{TRAIN_JSON}", train_json)
        raw = raw.replace("{VAL_JSON}", val_json)
        raw = raw.replace("{BATCH_SIZE}", str(batch))
        raw = raw.replace("{VAL_BATCH_SIZE}", str(val_batch))
        raw = raw.replace("{EPOCHS}", str(epochs))
        raw = raw.replace("{DYNAMIC_AUG_END}", str(dynamic_aug_end))
        raw = raw.replace("{OUTPUT_DIR}", str(output_dir.resolve()))
        configs[name] = yaml.safe_load(raw)

    # 合并
    final = {}
    for name in ["model", "runtime", "dataset"]:
        final.update(configs[name])

    # 存到 output 目录下
    config_path = output_dir / "config.yml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(final, f, default_flow_style=False, allow_unicode=True)

    logger.info(f"Config merged → {config_path}")
    logger.info(f"  epochs={epochs}, batch={batch}, imgsz=1280")
    logger.info(f"  backbone lr: 1e-6, decoder lr: 1e-4")
    logger.info(f"  dynamic aug: off at epoch {dynamic_aug_end}")
    logger.info(f"  Train: {train_img}")
    logger.info(f"  Val:   {val_img}")

    return config_path


def train(args, config_path: Path):
    """启动 RT-DETR PyTorch 训练, 实时输出, 支持断点续训。"""
    train_script = RTDETR_DIR / "tools" / "train.py"

    if not train_script.exists():
        logger.error(f"train.py not found: {train_script}")
        sys.exit(1)

    # 查找 resume checkpoint
    resume_flag = []
    if args.resume:
        candidates = sorted(
            (config_path.parent).glob("checkpoint*.pth"),
            key=lambda p: p.stat().st_mtime, reverse=True,
        )
        if candidates:
            resume_flag = ["--resume", str(candidates[0])]
            logger.info(f"Resuming from: {candidates[0]}")
        else:
            logger.warning("No checkpoint found, starting fresh")

    logger.info("-" * 60)
    logger.info("Launching RT-DETRv2-X training...")
    logger.info(f"  Ctrl+C to interrupt → checkpoint auto-saved")
    logger.info("-" * 60)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["CUDA_LAUNCH_BLOCKING"] = "1"   # 方便定位 CUDA 错误

    cmd = [
        sys.executable, "-u", str(train_script),
        "--config", str(config_path),
        "--use-amp",
    ] + resume_flag

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(RTDETR_DIR),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                print(line, flush=True)
        proc.wait()

        if proc.returncode == 0:
            logger.info("\n" + "=" * 60)
            logger.info("Training COMPLETE!")
            logger.info(f"  Best: {config_path.parent}/best.pth")
            logger.info("=" * 60)
        else:
            logger.error(f"\nExited with code {proc.returncode}")

    except KeyboardInterrupt:
        proc.terminate()
        logger.info("\n" + "=" * 60)
        logger.info("INTERRUPTED by user.")
        logger.info(f"Checkpoint: {config_path.parent}/")
        logger.info(f"Resume: python plans/rtdetrv2_x/train.py --resume")
        logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="RT-DETRv2-X Training")
    parser.add_argument("--resume", action="store_true", help="断点续训")
    parser.add_argument("--epochs", type=int, default=None, help="训练轮数 (默认72)")
    parser.add_argument("--batch", type=int, default=None, help="Batch size (默认4)")
    parser.add_argument("--name", type=str, default=None, help="实验名")
    args = parser.parse_args()

    print_env()

    # 1. 环境
    if not setup_rtdetr():
        sys.exit(1)

    # 2. 数据
    if not convert_data():
        sys.exit(1)

    # 3. 配置
    exp_name = args.name or f"rtdetrv2_x_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = CHECKPOINT_DIR / exp_name
    config_path = build_final_config(args, exp_name, output_dir)

    # 4. 训练
    train(args, config_path)


if __name__ == "__main__":
    main()
