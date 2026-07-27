# SteelGuard-YOLO 项目状态文档

> 更新时间：2026-07-27
> 赛题：2026 AIC·"AI＋钢铁"板材表面缺陷检测
> 方案文档：`板材表面缺陷检测方案.md`

---

## 一、环境

| 项目 | 值 |
|---|---|
| Python | 3.14.6 |
| PyTorch | 2.13.0+cu132 |
| CUDA | 13.2 |
| GPU | NVIDIA GeForce RTX 5060 Laptop（8GB VRAM） |
| ultralytics | 8.4.105 |

---

## 二、项目文件结构

```
aiiiiii_clone/
├── 板材表面缺陷检测方案.md          # 完整设计方案（48KB）
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
│
├── data_splits/
│   └── train_val_split.json        # Group Split: 2529 train / 671 val / 51 Style-OOD
│
├── scripts/
│   ├── train_baseline.py           # Phase 1 训练入口 (159行)
│   ├── train_phase3.py             # Phase 2-3 消融训练入口 (240行)
│   └── predict.py                  # 推理入口 (337行) — 双尺度 + Soft-NMS + JSON
│
├── src/
│   ├── paths.py                    # 路径管理、常量、类别映射 (60行)
│   ├── data/
│   │   ├── health_check.py         # VOC XML 健检 (252行)
│   │   ├── split.py                # Group-Stratified 划分 (248行)
│   │   ├── convert.py              # VOC → YOLO 格式转换 (187行)
│   │   ├── tiling.py               # 1280×1280 切片 + 标签映射 (447行)
│   │   ├── hard_negatives.py       # HN v2: 训练集 FP + GT 保留 + val 过滤 (267行)
│   │   └── copypaste.py            # CP-0/CP-1: 先选类别再选 patch (306行)
│   ├── utils/
│   │   ├── roi.py                  # ROI 掩膜、切片生成、全局视图 (251行)
│   │   └── metrics.py              # 评估协议 v2: 真 mAP@0.5:0.95 + 全局排序 + GT 匹配修复 (298行)
│   ├── train/
│   │   └── losses.py               # QFL + NWD 损失函数 (181行)
│   ├── models/                     # (空 — AAFM/MFFM 未实现)
│   └── inference/                  # (空)
│
├── train/
│   ├── train/                      # 原始 3200 JPG + 3200 XML
│   ├── tiles_train/                # 训练切片: 18,963 images + labels
│   └── tiles_val/                  # 验证切片: 6,627 images + labels
│
├── test/初赛/                      # 669 张测试图
│
└── outputs/
    ├── checkpoints/
    │   ├── baseline_p2_20260725_153654/   # Phase 1 强基线
    │   │   ├── weights/best.pt            # E31 mAP50=0.208
    │   │   └── weights/last.pt            # E45
    │   └── phase3_qfl_20260726_234655/    # Phase 2 QFL 消融（训练中）
    ├── logs/
    └── submission.json                    # 测试集提交文件
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
        ├── tiles_train: 18,963 切片
        └── tiles_val:   6,627 切片
```

---

## 五、Phase 1 强基线训练结果

### 训练参数

| 参数 | 值 |
|---|---|
| epochs | 设定 100，E31 最优，E45 早停 |
| batch | 2 × 梯度累积 8 = 等效 16 |
| optimizer | AdamW, lr=2e-4, wd=5e-4 |
| scheduler | Cosine + 3 epoch warmup |
| amp | FP16 |
| mosaic | 0.5，前 30 epoch |

### 关键指标

| 指标 | 最优值 | Epoch |
|---|---|---|
| **mAP50 (切片级)** | **0.208** | E31 |
| mAP50-95 (切片级) | 0.121 | E31 |
| Best Precision | 0.324 | E29 |
| Best Recall | 0.258 | E30 |

- **总耗时**: ~30 小时
- **单 epoch**: ~40 分钟

> ⚠️ 注意：以上 mAP 由旧版 metrics.py 计算（存在 GT 匹配状态重置 bug），实际值可能略低于报告值。后续所有实验将使用修复后的 metrics.py 重新评估。

---

## 六、Bug 修复记录（2026-07-27）

本轮对话中修复的关键问题：

| # | 问题 | 严重度 | 修复位置 | 状态 |
|---|---|---|---|---|
| 1 | 训练集/验证集泄漏：HN 生成可用验证集图片 | 🔴 | `hard_negatives.py:243-255` — 添加 train_stems 硬过滤 | ✅ |
| 2 | HN 切片丢弃真实 GT 标签 | 🔴 | `hard_negatives.py:166-194` — 保留 IoU≥0.4 的 GT | ✅ |
| 3 | metrics.py GT 匹配状态重置 → AP 虚高 | 🔴 Critical | `metrics.py:103-136` — 改为按图片分组批量匹配 | ✅ |
| 4 | 未实现真 mAP@0.5:0.95（10 IoU 阈值） | 🟡 | `metrics.py:170` — 10 个 IoU 阈值取平均 | ✅ |
| 5 | Phase 3 应从 baseline 微调，非 COCO 重训 | 🟡 | `train_phase3.py:146-148` — 加载 best.pt | ✅ |
| 6 | QFL 只写了 cls_pw，实际未接入 | 🟡 | `train_phase3.py:155-168` — cls_pw=0 + class_weights 手动设 | ✅ |
| 7 | CP 应该先选类别再选 patch | 🟢 | `copypaste.py:232-237` — 逆频率概率选类 | ✅ |
| 8 | 训练前数据一致性检查缺失 | 🟡 | `train_phase3.py:51-58` — check_data_consistency + 自动清 cache | ✅ |
| 9 | 消融实验 flags 未拆分 | 🟢 | `train_phase3.py:220-222` — --qfl/--hn/--cp 独立 flags | ✅ |

### 额外修复
| 问题 | 位置 | 状态 |
|---|---|---|
| metrics.py 死代码残留（lines 116-127） | `metrics.py` — 已删除 | ✅ |
| copypaste.py class_counts 从未递增 | `copypaste.py:264` — 添加 class_counts[cid] += 1 | ✅ |
| labels.cache 过期导致 FileNotFoundError | `train_phase3.py:54-57` — 训练前自动删除 stale cache | ✅ |
| 管道缓冲导致训练输出卡死 | 改用 `PYTHONUNBUFFERED=1 python -u` 启动 | ✅ |

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
| **Phase 1 强基线** | train_baseline.py | **E31 mAP50=0.208** |
| 双尺度推理 | predict.py | 测试集推理完成 |
| Soft-NMS | predict.py | 已实现 |
| 提交格式校验 | predict.py | 通过 |
| QFL / NWD 损失函数 | losses.py | 代码已有 |
| Copy-Paste CP-0/1 | copypaste.py | 代码已有（类别优先 + class_counts 修复） |
| Hard Negative Mining v2 | hard_negatives.py | 代码已有（train_stems 过滤 + GT 保留） |
| 本地评估协议 v2 | metrics.py | **GT 匹配 bug 已修复** + 真 mAP@0.5:0.95 |

### 🔄 进行中

| 模块 | 说明 |
|---|---|
| **Phase 2: QFL 消融实验** | 从 baseline 微调 40 epoch，验证类别加权独立收益 |
| 整图级验证评估 | 待 Phase 2 完成后用修复版 metrics.py 重评所有实验 |

### ❌ 未实现

| 模块 | 说明 | 优先级 |
|---|---|---|
| **AAFM** 抗伪影门控 | 方案 5.2.1，核心创新 | P1 |
| **MFFM** 形态融合 | 方案 5.2.2，核心创新 | P1 |
| NWD 损失接入训练 | losses.py 有代码但未接入 train_phase3.py | P2 |
| Copy-Paste CP-2/3/4 | 灰度匹配/前景掩膜/残差版本 | P2 |
| MixStyle 域泛化 | 方案 5.4.2 | P2 |
| 灰度域增强对照 | vs MixStyle | P2 |
| RT-DETR 独立对照 | 方案 3.2 | P3 |
| 多 seed 复验 | 关键配置 2-3 seed | P3 |
| Style-OOD 评估 | metrics.py 支持但未实际运行 | P2 |
| TensorBoard | 未配置 | P3 |

---

## 八、完整实验流程与路线图

```
Phase 0: 数据 + 闭环 ✅
├── XML 健检、Group Split、VOC→YOLO
├── ROI + 1280 切片管线
├── 冻结评估协议 v2 (mAP@0.5:0.95, F1/F2, Style-OOD)
└── 推理 + JSON 校验闭环

Phase 1: 强基线 ✅
├── YOLO11m-P2 + 类别感知采样 + 灰度增强
├── E31 mAP50=0.208 (切片级，旧 metrics)
└── 输出 baseline best.pt + 推理 submission.json

Phase 2: 消融实验（长尾 + 难例）🔄
├── 🔄 T1: QFL 类别重加权            ← 训练中 (~27h)
├── ⏳ T2: NWD 小框回归
├── ⏳ T3: Copy-Paste CP-0 → CP-1
├── ⏳ T8: Hard Negative Mining
└── ⏳ 整图级 mAP@0.5:0.95 重评所有实验

Phase 3: 结构模块（AAFM/MFFM）⏳
├── ⏳ S1: AAFM-only
├── ⏳ S2: MFFM-only
└── ⏳ S3: AAFM + MFFM

Phase 4: 域泛化 ⏳
├── ⏳ T5: 直接灰度增强
├── ⏳ T6: MixStyle
└── ⏳ T7: 灰度增强 + MixStyle

Phase 5: 组合 + 对照 ⏳
├── ⏳ 最优模块组合（仅保留 Phase 2-4 中独立有效的）
├── ⏳ RT-DETR-R50 独立对照
└── ⏳ 多 seed 复验（2-3 seed）

Phase 6: 后处理 + 加速 ⏳
├── ⏳ Soft-NMS / WBF 优化
├── ⏳ F1/F2 类别阈值搜索
└── ⏳ batch inference / FP16 / TensorRT

Phase 7: 最终固化与提交 ⏳
├── ⏳ 全数据重训（一份权重）
├── ⏳ 盲测 / 最坏时延检查
└── ⏳ 技术报告 + 提交
```

### 选模原则

最终模型 = **强基线 + 被 ID/Style-OOD/稳定性/复杂度共同证明有效的模块**，不是所有模块的堆叠。

候选模块晋级条件：
1. ID 验证集和 Style-OOD 验证集上方向一致
2. 至少 2-3 seed 复验，增益超过标准差
3. 对目标错误类型有解释力
4. 额外参数量/时延在预算内
5. 无合成伪影、数据泄漏或规则合规问题

内部选模分数：
```
Score = 0.35 × mAP@0.5:0.95
      + 0.20 × mAP@0.5
      + 0.15 × MacroF1@IoU0.5
      + 0.15 × CriticalMacroF2@IoU0.5
      + 0.15 × StyleOOD_mAP@0.5:0.95
```

---

## 九、时间估算

| 阶段 | 内容 | 预估 |
|---|---|---|
| Phase 2 消融实验 | QFL + NWD + CP + HN（~4 实验 × 28h） | ~5 天 |
| Phase 3 AAFM/MFFM | 实现（~2 天）+ 训练（~3 实验 × 30h） | ~5 天 |
| Phase 4 域泛化 | MixStyle + 灰度增强对照（~3 实验） | ~3 天 |
| Phase 5 组合 | 最优模块组合 + RT-DETR 对照 | ~3 天 |
| Phase 6 后处理 | NMS/阈值搜索 + 多 seed | ~2 天 |
| Phase 7 最终提交 | 全数据重训 + 报告 | ~2 天 |
| **总计** | | **~2.5-3 周** |

> 瓶颈：单 GPU (RTX 5060 8GB)，每实验 ~28-30 小时，无法并行训练。

---

## 十、当前已知风险

| 风险 | 控制措施 |
|---|---|
| 旧 baseline mAP 由有 bug 的 metrics 计算，真实值可能更低 | Phase 2 完成后用修复版 metrics 统一重评 |
| QFL 仅通过 class_weights 生效，非完整 QFL | 如果收益不显著，考虑接入 losses.py 的 quality_focal_loss |
| CP/HN 需先推理生成再训练，增加实验周期 | 提前准备推理脚本和预测文件 |
| AAFM/MFFM 需修改 ultralytics 模型架构，工程量大 | 预留充足实现时间 |
| 单 seed 结论不可靠 | 关键配置至少 2 seed 复验 |
| Style-OOD 未实际评估 | Phase 2 完成后立即补齐 |
