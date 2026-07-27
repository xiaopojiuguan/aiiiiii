#!/usr/bin/env python3
"""
方案2: DEIM-D-FINE-X 全量训练
==============================
- 模型: DEIM + D-FINE (密集匹配 + 细粒度分布回归)
- 骨干: HGNetv2-X (62M params)
- 换掉了什么:
    1. DEIM 密集 O2O 匹配 → 前半段训练多对一正样本, 解决小数据集稀疏监督
    2. D-FINE FDR → 框坐标从4个标量变成概率分布精修, 提升 mAP50-95
    3. 两者叠加: 36 epoch = 原生 D-FINE 72 epoch 的效果
- 数据: COCO JSON (自动转换)
- 显存: 48GB 4090, batch=2, imgsz=1280 (HGNetv2-X 较大)

用法:
  python plans/dfine/train.py                  # 首次训练
  python plans/dfine/train.py --resume         # 打断后继续
  python plans/dfine/train.py --epochs 50      # 自定义轮数
  python plans/dfine/train.py --convert-only   # 只转换数据
"""

import sys, os, argparse, logging, subprocess, json, yaml
from pathlib import Path
from datetime import datetime

# ==== 路径配置 ====
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLAN_DIR = Path(__file__).resolve().parent
CONFIG_DIR = PLAN_DIR / "configs"
CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"
COCO_DIR = PROJECT_ROOT / "outputs" / "coco_annotations"
DEIM_DIR = PROJECT_ROOT / "DEIM"                 # Intellindust-AI-Lab/DEIM 仓库

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("deim-dfine")


def print_env():
    import torch
    logger.info("=" * 60)
    logger.info("DEIM-D-FINE-X (HGNetv2-X) Training")
    logger.info(f"  Python:   {sys.version.split()[0]}")
    logger.info(f"  PyTorch:  {torch.__version__}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            logger.info(f"  GPU[{i}]:   {p.name} ({p.total_memory/1024**3:.1f} GB)")
    logger.info(f"  Configs:  {CONFIG_DIR}")
    logger.info(f"  Output:   {CHECKPOINT_DIR}")
    logger.info("=" * 60)


def setup_deim() -> bool:
    """克隆/检查 DEIM 仓库 (内含 D-FINE + DEIM 代码)。"""
    if DEIM_DIR.exists() and (DEIM_DIR / "tools" / "train.py").exists():
        logger.info("DEIM repo found.")
        return True

    logger.info("Cloning DEIM from GitHub...")
    logger.info("  git clone https://github.com/Intellindust-AI-Lab/DEIM")

    try:
        subprocess.run(
            ["git", "clone", "https://github.com/Intellindust-AI-Lab/DEIM",
             str(DEIM_DIR)],
            check=True, cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        logger.info("Clone successful, installing dependencies...")
        req_file = DEIM_DIR / "requirements.txt"
        if req_file.exists():
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", str(req_file)],
                check=False, cwd=str(DEIM_DIR),
            )
        logger.info("DEIM setup complete.")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Clone failed: {e}")
        logger.error("Manual: git clone https://github.com/Intellindust-AI-Lab/DEIM")
        return False
    except FileNotFoundError:
        logger.error("git not found. Manual clone required.")
        return False


def convert_data() -> bool:
    """YOLO txt → COCO JSON。"""
    convert_script = PLAN_DIR / "convert_to_coco.py"

    if COCO_DIR.exists() and (COCO_DIR / "train.json").exists():
        logger.info(f"COCO annotations exist ({(COCO_DIR/'train.json').stat().st_size/1024/1024:.1f} MB)")
        return True

    logger.info("Converting YOLO → COCO...")
    result = subprocess.run(
        [sys.executable, str(convert_script), "--split", "all"],
        cwd=str(PROJECT_ROOT),
    )
    return result.returncode == 0


def build_final_config(args, exp_name: str, output_dir: Path) -> Path:
    """合并 configs/{dataset,model,runtime}.yml → 最终配置文件。"""
    with open(PROJECT_ROOT / "tile_dataset.yaml") as f:
        tile_cfg = yaml.safe_load(f)

    data_root = PROJECT_ROOT / tile_cfg["path"]
    train_img = str((data_root / tile_cfg["train"]).resolve())
    val_img = str((data_root / tile_cfg["val"]).resolve())
    train_json = str((COCO_DIR / "train.json").resolve())
    val_json = str((COCO_DIR / "val.json").resolve())

    epochs = args.epochs or 36      # DEIM 减半训练
    batch = args.batch or 2         # HGNetv2-X 62M, 1280 分辨率保守 batch
    val_batch = batch * 2
    deim_stop = epochs // 2         # 前半段密集匹配
    flat_epoch = int(epochs * 0.6)  # FlatCosine 平顶区

    # 读取三个分拆配置, 替换占位符, 合并
    merged = {}
    for name in ["dataset", "model", "runtime"]:
        cfg_path = CONFIG_DIR / f"{name}.yml"
        if not cfg_path.exists():
            logger.error(f"Config missing: {cfg_path}")
            sys.exit(1)
        with open(cfg_path, encoding="utf-8") as f:
            raw = f.read()

        replacements = {
            "{TRAIN_IMG_DIR}": train_img,
            "{VAL_IMG_DIR}": val_img,
            "{TRAIN_JSON}": train_json,
            "{VAL_JSON}": val_json,
            "{BATCH_SIZE}": str(batch),
            "{VAL_BATCH_SIZE}": str(val_batch),
            "{EPOCHS}": str(epochs),
            "{DEIM_STOP_EPOCH}": str(deim_stop),
            "{FLAT_EPOCH}": str(flat_epoch),
            "{OUTPUT_DIR}": str(output_dir.resolve()),
        }
        for k, v in replacements.items():
            raw = raw.replace(k, v)

        merged.update(yaml.safe_load(raw))

    config_path = output_dir / "config.yml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(merged, f, default_flow_style=False, allow_unicode=True)

    logger.info(f"Config merged → {config_path}")
    logger.info(f"  epochs={epochs}, batch={batch}, imgsz=1280")
    logger.info(f"  DEIM dense match: epoch 0-{deim_stop}, then off")
    logger.info(f"  FlatCosine: flat={flat_epoch}ep, warmup=2000iter")
    logger.info(f"  backbone lr=1e-6, encoder/decoder lr=1e-4")
    return config_path


def train(args, config_path: Path):
    """启动 DEIM-D-FINE 训练, 实时输出。"""
    train_script = DEIM_DIR / "tools" / "train.py"

    if not train_script.exists():
        logger.error(f"train.py not found: {train_script}")
        logger.error("Run without --skip-setup to auto-clone DEIM repo.")
        sys.exit(1)

    # resume 处理
    extra_args = []
    if args.resume:
        candidates = sorted(
            (config_path.parent).glob("checkpoint*.pth"),
            key=lambda p: p.stat().st_mtime, reverse=True,
        )
        if candidates:
            extra_args = ["--resume", str(candidates[0])]
            logger.info(f"Resuming from: {candidates[0]}")
        else:
            logger.warning("No checkpoint found, starting fresh")

    logger.info("-" * 60)
    logger.info("Launching DEIM-D-FINE-X training...")
    logger.info(f"  Ctrl+C to interrupt → checkpoint auto-saved")
    logger.info("-" * 60)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["CUDA_LAUNCH_BLOCKING"] = "1"

    cmd = [
        sys.executable, "-u", str(train_script),
        "--config", str(config_path),
    ] + extra_args

    try:
        proc = subprocess.Popen(
            cmd, cwd=str(DEIM_DIR), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                print(line, flush=True)
        proc.wait()

        if proc.returncode == 0:
            logger.info("\n" + "=" * 60)
            logger.info("DEIM-D-FINE-X Training COMPLETE!")
            logger.info(f"  Best: {config_path.parent}/best.pth")
            logger.info("=" * 60)
        else:
            logger.error(f"\nExited with code {proc.returncode}")

    except KeyboardInterrupt:
        proc.terminate()
        logger.info("\n" + "=" * 60)
        logger.info("INTERRUPTED by user.")
        logger.info(f"Checkpoint: {config_path.parent}/")
        logger.info(f"Resume: python plans/dfine/train.py --resume")
        logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="DEIM-D-FINE-X Full Training")
    parser.add_argument("--resume", action="store_true", help="断点续训")
    parser.add_argument("--convert-only", action="store_true", help="只转换数据")
    parser.add_argument("--skip-setup", action="store_true", help="跳过 DEIM 安装")
    parser.add_argument("--epochs", type=int, default=None, help="训练轮数 (默认36)")
    parser.add_argument("--batch", type=int, default=None, help="Batch size (默认2)")
    parser.add_argument("--name", type=str, default=None, help="实验名")
    args = parser.parse_args()

    print_env()

    # 1. 环境
    if not args.skip_setup:
        if not setup_deim():
            logger.error("DEIM setup failed.")
            sys.exit(1)

    # 2. 数据
    if not convert_data():
        logger.error("Data conversion failed.")
        sys.exit(1)

    if args.convert_only:
        logger.info("--convert-only: Done.")
        return

    # 3. 配置
    exp_name = args.name or f"deim_dfine_x_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = CHECKPOINT_DIR / exp_name
    config_path = build_final_config(args, exp_name, output_dir)

    # 4. 训练
    train(args, config_path)


if __name__ == "__main__":
    main()
