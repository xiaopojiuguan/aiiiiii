# SteelGuard-YOLO 项目状态文档

> 更新时间：2026-07-26  
> 赛题：2026 AIC·"AI＋钢铁"板材表面缺陷检测  
> 方案文档：`板材表面缺陷检测方案.md`

---

## 一、环境

| 项目 | 值 |
|---|---|
| Python | 3.14.6 |
| PyTorch | 2.13.0+cu132 |
| CUDA | 13.2 |
| GPU | NVIDIA GeForce RTX 5060 Laptop（8GB） |
| ultralytics | 8.4.105 |

---

## 二、项目文件结构

```
aiiiiii_clone/
├── 板材表面缺陷检测方案.md          # 完整设计方案
├── PROJECT_STATUS.md               # 本文件
├── requirements.txt                # 依赖清单
├── .gitignore
├── yolo11m.pt                      # COCO 预训练权重 (40.7 MB)
│
├── configs/
│   ├── baseline.yaml               # 强基线训练/推理/评估配置
│   └── yolo11m-p2.yaml             # YOLO11-P2 模型架构 (depth=0.67, width=0.50)
│
├── tile_dataset.yaml               # 训练数据指向 tiles_train / tiles_val
├── train/dataset.yaml              # (旧) 全图训练配置，含绝对路径
│
├── data_splits/
│   └── train_val_split.json        # Group Split: 2529 train / 671 val / 51 Style-OOD
│
├── scripts/
│   ├── train_baseline.py           # 训练入口 (158行)
│   └── predict.py                  # 推理入口 (336行) — 双尺度 + Soft-NMS + JSON
│
├── src/
│   ├── paths.py                    # 路径管理、常量、类别映射 (60行)
│   ├── data/
│   │   ├── health_check.py         # VOC XML 健检 (252行)
│   │   ├── split.py                # Group-Stratified 划分 (248行)
│   │   ├── convert.py              # VOC → YOLO 格式转换 (187行)
│   │   ├── tiling.py               # 1280×1280 切片 + 标签映射 (447行)
│   │   ├── hard_negatives.py       # 误检收集 → 硬负样本 (227行)
│   │   └── copypaste.py            # CP-0/CP-1 Copy-Paste 增强 (296行)
│   ├── utils/
│   │   ├── roi.py                  # ROI 掩膜、切片生成、全局视图 (251行)
│   │   └── metrics.py              # 评估协议 v1: AP/F1/F2/阈值搜索 (364行)
│   ├── train/
│   │   └── losses.py               # QFL + NWD 损失函数 (180行)
│   ├── models/                     # (空 — AAFM/MFFM 未实现)
│   └── inference/                  # (空)
│
├── train/
│   ├── train/                      # 原始 3200 JPG + 3200 XML
│   ├── train_yolo/                 # VOC→YOLO 全图转换: 2554 images + labels
│   ├── val_yolo/                   # 验证集全图: 646 images + labels
│   ├── tiles_train/                # 训练切片: 18,964 images + labels (~13 GB)
│   └── tiles_val/                  # 验证切片: 6,627 images + labels (~4.4 GB)
│
├── test/初赛/                      # 669 张测试图
│
└── outputs/
    ├── checkpoints/
    │   └── baseline_p2_20260725_153654/   # 当前训练结果
    │       ├── weights/best.pt            # 最优权重 E31 (76 MB)
    │       ├── weights/last.pt            # 最新权重 E45 (76 MB)
    │       ├── results.csv                # 逐 epoch 指标
    │       ├── args.yaml                  # 训练参数快照
    │       └── config.yaml               # 配置快照
    ├── logs/
    │   └── train_output.log               # 完整训练日志 (75 MB)
    ├── submission.json                    # 测试集提交文件 (23,614 predictions)
    └── val_predictions.json               # 验证集预测 + GT
```

---

## 三、模型配置

| 参数 | 值 |
|---|---|
| 架构 | **YOLO11-P2** (4 检测头: stride=4,8,16,32) |
| depth_multiple | 0.67 |
| width_multiple | 0.50 |
| 设计理念 | 空间优先：1280 分辨率 + P2 头 > 通道宽度 |
| 输入尺寸 | **1280×1280 切片**（非整图） |
| 预训练权重 | yolo11m.pt (COCO)，P2 头随机初始化 |
| 迁移情况 | 182/803 层从 COCO 迁移 |

---

## 四、数据管线

```
3200 原图 (4096×3000, 灰度, VOC XML)
    │
    ├── health_check.py → 健检通过
    ├── split.py → 按板材编号 Group Split
    │   ├── train: 2529 原图 (1720 块板材)
    │   ├── val:   671 原图 (429 块板材)
    │   └── Style-OOD: 51 原图
    │
    └── tiling.py → ROI + 1280×1280 切片 (stride=1024, overlap=256)
        ├── tiles_train: 18,964 切片
        └── tiles_val:   6,627 切片
```

### 验证集类别分布

| 类别 | GT 数量 |
|---|---|
| jieba 结疤 | 130 |
| yiwuyaru 异物压入 | 85 |
| yanghuatiepi 氧化铁皮 | 68 |
| mamianmakeng 麻面麻坑 | 68 |
| gunyin 辊印 | 61 |
| zonglie 纵裂 | 53 |
| jiaza 夹杂 | 35 |
| huashang 划伤 | 18 |
| qilie 气裂 | 10 |

---

## 五、强基线训练结果

### 训练参数

| 参数 | 值 |
|---|---|
| epochs | 设定 100，E31 最优，E45 早停 |
| batch | 2 |
| 梯度累积 | 自动 (nbs=64 → ~32 step/update) |
| optimizer | AdamW, lr=2e-4, wd=5e-4 |
| scheduler | Cosine + 3 epoch warmup |
| amp | FP16 |
| mosaic | 0.5，前 30 epoch 启用 |
| 增强 | hsv_v=0.3, translate=0.1, scale=0.2, fliplr=0.5 |

### 关键指标

| 指标 | 最优值 | Epoch |
|---|---|---|
| **mAP50 (切片级)** | **0.208** | E31 |
| mAP50-95 (切片级) | 0.121 | E31 |
| Best Precision | 0.324 | E29 |
| Best Recall | 0.258 | E30 |

### 训练曲线特征

```
E1-E5:  快速上升  (mAP50 0.002→0.098)
E6-E10: 波动上升  (mAP50 0.074-0.112)
E11-E20: 稳步爬升 (mAP50 0.073→0.140)
E21-E31: 继续上升  (mAP50 0.164→0.208  ← 峰值)
E32-E45: 平台+过拟合 (v_cls 从 1.5 恶化到 3.2，mAP 停滞)
```

- **总耗时**: ~30 小时
- **单 epoch**: ~40 分钟

---

## 六、推理结果

| 项目 | 值 |
|---|---|
| 测试集 | 669 张 |
| 总预测框 | 23,614 |
| 推理方式 | ROI + 切片 batch 推理 + 全局视图 + Soft-NMS |
| 提交文件 | `outputs/submission.json` |
| 格式校验 | ✅ 通过 |

---

## 七、实现情况对照

### ✅ 已完成

| 模块 | 文件 | 状态 |
|---|---|---|
| 数据健检 | health_check.py | 跑通 |
| Group Split | split.py | 跑通，无泄漏 |
| VOC→YOLO | convert.py | 跑通 |
| ROI + 1280 切片 | roi.py + tiling.py | 跑通，25,591 张 |
| P2 检测头 | yolo11m-p2.yaml | 已构建，训练正常 |
| 强基线训练 | train_baseline.py | E31 mAP50=0.208 |
| 双尺度推理 | predict.py | 测试集推理完成 |
| Soft-NMS | predict.py | 已实现 |
| 提交格式校验 | predict.py | 通过 |
| QFL 损失 | losses.py | 代码已有 |
| NWD 损失 | losses.py | 代码已有（未接入训练） |
| Copy-Paste CP-0/1 | copypaste.py | 代码已有（未接入训练） |
| Hard Negative Mining | hard_negatives.py | 代码已有 |
| 本地评估协议 | metrics.py | F1/F2 阈值搜索已实现 |

### ❌ 未实现

| 模块 | 说明 |
|---|---|
| AAFM 抗伪影门控 | 方案 5.2.1，无代码 |
| MFFM 形态融合 | 方案 5.2.2，无代码 |
| MixStyle 域泛化 | 方案 5.4.2，无代码 |
| RT-DETR 对照 | 方案 3.2，无代码 |
| 多 seed 复验 | 方案要求 2-3 seed，当前仅 1 |
| 整图级验证 | 仍为切片级 mAP |
| Style-OOD 评估 | metrics.py 支持但未实际运行 |
| TensorBoard | 未配置 |

---

## 八、当前已知问题

1. **v_cls 过拟合**: E32 后验证分类损失持续恶化，模型在记忆训练集
2. **切片级验证 ≠ 整图级**: 当前 mAP 基于切片，需补齐整图验证
3. **训练和推理链路不一致**: 训练用切片标签，推理用坐标还原 — 中间没有对齐验证
4. **NWD/QFL/Copy-Paste 代码已有但未接入训练管线**
5. **AAFM/MFFM 核心创新模块完全未动工**
6. **单 seed 结论不可靠** — 0.208 可能是运气好/坏

---

## 九、下一步行动

1. **整图级验证**: 用 `predict.py` 对 val 原图推理 → `metrics.py` 出逐类诊断
2. **补 Style-OOD 评估**: 看域泛化缺口多大
3. **接入已有模块**: NWD、QFL、Copy-Paste 代码已有，接入训练验证收益
4. **按诊断结果选消融方向**: 长尾严重 → Copy-Paste；伪影多 → AAFM；小框不准 → NWD
