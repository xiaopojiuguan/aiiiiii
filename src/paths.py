"""项目路径管理模块 —— 对应方案第 16.1 节

所有路径相对于项目根目录，禁止硬编码绝对路径。
"""

from pathlib import Path

# 项目根目录：从本文件向上两级（src/paths.py → 项目根目录）
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 数据目录
DATA_ROOT = PROJECT_ROOT / "train" / "train"  # 训练集（JPG+XML）
TEST_ROOT = PROJECT_ROOT / "test" / "初赛"    # 初赛测试集（JPG）

# 输出目录
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = OUTPUT_ROOT / "checkpoints"
LOG_DIR = OUTPUT_ROOT / "logs"
EVAL_DIR = OUTPUT_ROOT / "eval"

# 配置目录
CONFIG_DIR = PROJECT_ROOT / "configs"

# 划分文件
SPLIT_DIR = PROJECT_ROOT / "data_splits"

# 确保必要目录存在
for d in [OUTPUT_ROOT, CHECKPOINT_DIR, LOG_DIR, EVAL_DIR, SPLIT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# 类别映射：中文 → 拼音标签
CLASS_NAMES = [
    "jieba",          # 结疤
    "zonglie",        # 纵裂
    "qilie",          # 气裂
    "jiaza",          # 夹杂
    "yiwuyaru",       # 异物压入
    "huashang",       # 划伤
    "mamianmakeng",   # 麻面麻坑
    "yanghuatiepi",   # 氧化铁皮
    "gunyin",         # 辊印
]

CLASS_NAME_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}
CLASS_ID_TO_NAME = {i: name for i, name in enumerate(CLASS_NAMES)}

# 致命缺陷类别（F2 评分）
CRITICAL_CLASSES = ["qilie", "zonglie"]

# 图像尺寸
ORIGINAL_WIDTH = 4096
ORIGINAL_HEIGHT = 3000

# 切片参数
TILE_SIZE = 1280
TILE_OVERLAP = 256
TILE_STRIDE = TILE_SIZE - TILE_OVERLAP  # 1024

# 全局视图
GLOBAL_LONG_SIDE = 1536
