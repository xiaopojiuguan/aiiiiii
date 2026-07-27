# 换模型方案详解：RT-DETRv2-X vs DEIM-D-FINE-X

> 背景：YOLO11m-P2 mAP50=0.208，YOLO11x-P2 mAP50=0.206。参数量 ×2.8，指标降 0.002。容量不是瓶颈，范式才是。

---

## 一、为什么是这两个方案

YOLO 系列的核心问题在四个方面，这两个方案各击中不同的几项：

```
                       RT-DETRv2-X    DEIM-D-FINE-X
TAL 一对多分配          ✅ 换掉          ✅ 换掉 (更彻底)
NMS 去重                ✅ 去掉          ✅ 去掉
框回归直接坐标          ❌ 保留          ✅ 换成分布精修
监督信号稀疏            ❌              ✅ DEIM 密集匹配
```

所有实验用全量数据（2529 原图 → ~22,000 张 1280×1280 切片），不搞速筛子集。

---

## 二、方案1：RT-DETRv2-X

### 2.1 代码结构

```
plans/rtdetrv2_x/
├── configs/
│   ├── dataset.yml    # 数据路径 + 增强 + dataloader
│   ├── model.yml      # PResNet101-vd + HybridEncoder + v2 Decoder + 损失
│   └── runtime.yml    # AdamW 分层 lr + CosineDecay + EMA + 动态增强
└── train.py           # 训练启动（clone 仓库 → 转 COCO → 合并 yml → 训练）
```

训练入口 `plans/rtdetrv2_x/train.py` 核心流程：

```python
# 1. 自动 clone lyuwenyu/RT-DETR (PyTorch)
setup_rtdetr()

# 2. YOLO txt → COCO JSON (复用 plans/dfine/convert_to_coco.py)
convert_data()

# 3. 读取三个分拆 yml → 替换占位符 → 合并为单一配置
build_final_config(args, exp_name, output_dir)

# 4. 启动训练，实时逐行输出
subprocess.Popen([sys.executable, "tools/train.py", "--config", config_path])
```

### 2.2 换掉了什么

#### 标签分配：TAL 一对多 → 匈牙利一对一

YOLO 的 TAL（Task-Aligned Assigner）按 anchor 中心和 IoU 给每个 GT 分配多个正样本：

```
真实框 ──→ 预测框 A (正样本, FPN P3)
       └→ 预测框 B (正样本, FPN P4)  
       └→ 预测框 C (正样本, FPN P5)
```

RT-DETR 的匈牙利匹配保证每个 GT 只有 1 个预测框 + 每个预测框只负责 1 个 GT：

```yaml
# plans/rtdetrv2_x/configs/model.yml:94-102
RTDETRCriterionv2:
  matcher:
    type: HungarianMatcher        # 匈牙利算法做全局最优匹配
    cost_class: 2                 # 分类代价权重
    cost_bbox: 5                  # 框位置代价权重
    cost_giou: 2                  # GIoU 代价权重
```

匈牙利算法的输入是一个 cost matrix（300 query × 所有 GT），输出是全局代价最小的匹配方案。除了匹配 GT 的 query，其余 query 匹配到 `∅`（背景类）。

#### NMS：有 → 无

因为每个 query 天然只输出一个目标，推理时直接取 conf top-300，不存在重复框，NMS 被彻底消除。这在密集缺陷场景（麻面麻坑一片一片）是核心优势。

#### Decoder 升级：RTDETRTransformerv2

```yaml
# plans/rtdetrv2_x/configs/model.yml:65-82
RTDETRTransformerv2:
  num_queries: 300               # 每张图 300 个 query，逐个认领目标
  num_layers: 6                  # 6 层解码器
  num_points: [4, 8, 16]        # C3=4点 C4=8点 C5=16点 多尺度采样
  sample_type: discrete_sample   # v2 特色：离散采样，省插值
  num_denoising: 100             # CDN 去噪训练，加速收敛
  aux_loss: True                 # 中间层加辅助损失
```

v2 相比 v1 的改进：
- **多尺度独立采样点数**：C3=4, C4=8, C5=16，不再所有尺度统一，对小目标更友好
- **discrete_sample**：舍入采样偏移量到整数，省掉 grid_sample 的插值，部署更友好
- **动态数据增强**：前 N-2 epoch 开启强增强，最后 2 epoch 关闭让模型适配目标域

#### Backbone：ResNet101-vd

```yaml
# plans/rtdetrv2_x/configs/model.yml:27-35
PResNet:
  name: ResNet
  depth: 101           # 101 层
  variant: d           # vd 变体：改进下采样，减少信息丢失
  freeze_at: 0         # 不冻结任何层（全量微调）
  return_idx: [1, 2, 3]  # 输出 C3 C4 C5 三个尺度
  pretrained: True     # ImageNet 预训练
```

### 2.3 学习率策略

```yaml
# plans/rtdetrv2_x/configs/runtime.yml:21-28
optimizer:
  type: AdamW
  lr: 0.0001                            # decoder 基础 lr
  param_groups:
    - params: backbone
      lr_mult: 0.01                     # 0.0001 × 0.01 = 1e-6
    - params: encoder
      lr_mult: 1.0                      # = 1e-4
    - params: decoder
      lr_mult: 1.0                      # = 1e-4
```

**为什么 backbone lr 只有 1e-6？** ResNet101 的预训练特征已经很好，太大的 lr 会破坏这些特征。这个策略来自 RT-DETRv2 论文的 scale-adaptive hyperparameters：越大的骨干，backbone lr 越低。

### 2.4 动态数据增强

```yaml
# plans/rtdetrv2_x/configs/runtime.yml:54-59
use_dynamic_aug: True
dynamic_aug_policy:
  start_epoch: 0
  end_epoch: 70           # 第 70 epoch 关闭（72epoch 最后 2 轮）
```

训练前 70 epoch 用完整增强 pipeline：
```yaml
# plans/rtdetrv2_x/configs/dataset.yml:25-46
transforms:
  ops:
    - RandomPhotometricDistort    # 亮度/对比度/饱和度扰动
    - RandomZoomOut               # 随机缩小+padding（模拟全局上下文）
    - RandomIoUCrop               # 随机 IoU 裁剪
    - RandomHorizontalFlip        # 水平翻转
    - Resize: [1280, 1280]        # 统一到 1280
```

最后 2 epoch 关闭 RandomPhotometricDistort、RandomZoomOut、RandomIoUCrop，只保留 Resize + Flip，让模型干净地适配目标域分布。

### 2.5 预估

| 项 | 值 |
|---|---|
| 骨干 | ResNet101-vd (~67M 总参数) |
| 输入 | 1280×1280 |
| Batch | 4 |
| Epoch | 72 |
| 总迭代 | ~36-45 万 step |
| 预估时间 | **35-50 小时**（48GB 4090） |

---

## 三、方案2：DEIM-D-FINE-X

### 3.1 代码结构

```
plans/dfine/
├── configs/
│   ├── dataset.yml              # 数据（同上结构）
│   ├── model.yml                # HGNetv2-X + DEIM decoder + FDR + MAL loss
│   └── runtime.yml              # AdamW 分层 lr + FlatCosine 调度 + EMA
├── convert_to_coco.py           # YOLO txt → COCO JSON（两个方案共用）
└── train.py                     # 训练启动
```

训练入口 `plans/dfine/train.py` 核心流程：

```python
# 1. 自动 clone Intellindust-AI-Lab/DEIM
setup_deim()

# 2. YOLO txt → COCO JSON
convert_data()

# 3. 合并 configs/{dataset,model,runtime}.yml
build_final_config(args, exp_name, output_dir)
#    关键参数: epochs=36, batch=2, deim_stop=18, flat_epoch=22

# 4. 训练
subprocess.Popen([sys.executable, "tools/train.py", "--config", config_path])
```

### 3.2 换掉了什么（比 RT-DETRv2 多换了两样）

#### 框回归：直接坐标 → FDR 分布精修

RT-DETR 直接输出 4 个数 `(x, y, w, h)`：

```
decoder → MLP → [x, y, w, h]  4 个标量
```

D-FINE 的 FDR（Fine-grained Distribution Refinement）把每个坐标变成**概率分布**：

```
decoder → 分布头 → [P(x₁), P(x₂), ..., P(x₁₆)]  16 个 bin
                  → 加权求和得亚像素坐标
```

```yaml
# plans/dfine/configs/model.yml:53-55
  reg_max: 16           # 每个坐标离散为 16 个 bin
  fdr_alpha: 1.0        # FDR 强度系数
```

**为什么你的场景需要它？** baseline mAP50-95 / mAP50 = 58%，正常比例应该是 65-70%。这个数字的含义是：框能找得到，但框的 IoU 不够高——贴得不准。FDR 正是通过把离散坐标变成连续分布来提升定位精度的。

对应的损失函数多了 `loss_ddf`：

```yaml
# plans/dfine/configs/model.yml:63-73
loss:
  weight_dict:
    loss_mal: 1.0        # DEIM 感知匹配分类
    loss_bbox: 5.0       # L1 框回归
    loss_giou: 2.0       # GIoU
    loss_ddf: 1.5        # ← D-FINE 核心：分布蒸馏损失
    loss_fgl: 0.15       # Focal 辅助
```

#### 监督信号：一对一匹配 → DEIM 密集 O2O

RT-DETR 的一对一匹配虽然干净，但在**小数据集上正样本太少**。你有 2529 张原图 + 极端长尾（qilie 31 例、huashang 35 例），一对一匹配给稀有类的训练信号极其稀疏。

DEIM 的解决方案：**前半段训练用密集匹配，后半段关掉去噪**。

```yaml
# plans/dfine/configs/model.yml:56-58
  use_deim: True
  dense_match_stop_epoch: 18    # epoch 0-17 密集匹配
                                 # epoch 18-35 关掉，干净收敛
```

前 18 epoch：
```
GT → 多个 query（密集多对一匹配）
     每个稀有类获得几倍的训练信号
```

后 18 epoch：
```
GT → 1 个 query（退化为标准一对一）
     去掉冗余信号，精修定位
```

训练时 `build_final_config` 中自动计算：

```python
# plans/dfine/train.py:120
deim_stop = epochs // 2       # 36/2 = 18，前半段密集
flat_epoch = int(epochs * 0.6)  # 36×0.6 = 22，FlatCosine 平顶区
```

#### 损失：VFL → MAL（Matching-Aware Loss）

标准 VFL（Varifocal Loss）对所有预测一视同仁。DEIM 的 MAL 根据匹配质量给不同预测不同的损失权重——匹配得好的预测承担更多 loss，匹配不好的少承担。

```yaml
# plans/dfine/configs/model.yml:65-66
    loss_mal: 1.0          # DEIM Matching-Aware Loss
```

#### Backbone：ResNet101 → HGNetv2-X

```yaml
# plans/dfine/configs/model.yml:22-25
backbone:
  name: HGNetv2
  variant: X                  # 62M params，对标 ResNet101
  freeze_at: [1, 2, 3]       # 冻结 stem+stage1-3（更多层冻结）
```

HGNetv2 是百度自研的高效骨干，推理速度比 ResNet101 快但准确率相当。`freeze_at: [1,2,3]` 冻结了浅层和前三个 stage，只微调深层，进一步降低小数据集上的过拟合风险。

### 3.3 学习率策略（FlatCosine）

```yaml
# plans/dfine/configs/runtime.yml:42-48
lr_scheduler:
  type: FlatCosine
  warmup_iter: 2000         # 前 2000 iter 线性爬升
  flat_epoch: 22            # 第 2-22 epoch 保持高 lr
  lr_gamma: 0.5             # 第 23 epoch 开始余弦衰减
  min_lr_ratio: 0.01
```

与 RT-DETR 的 CosineDecay 对比：

```
RT-DETRv2-X (CosineDecay):   ▁▃▅███████████▇▅▃▁▁
DEIM-D-FINE-X (FlatCosine):  ▁▃██████████████████▇▅▃▁▁
                              ↑  warmup    ↑  22ep 平顶  ↑ 衰减
```

FlatCosine 在高 lr 区停留更久，给 DEIM 密集匹配阶段充足的探索空间，然后快速衰减收敛。

### 3.4 数据增强

与 RT-DETRv2-X 一致但调度不同：

```yaml
# plans/dfine/configs/runtime.yml:18-20
aug_schedule:
  warm_epoch: 4             # 前 4 epoch 逐步开启增强
  no_aug_last_epoch: 2      # 最后 2 epoch 关闭
```

```yaml
# plans/dfine/configs/dataset.yml
MultiScaleResize:
  sizes: [960, 1088, 1216, 1280]   # 多尺度训练，让模型适应不同分辨率
```

### 3.5 预估

| 项 | 值 |
|---|---|
| 骨干 | HGNetv2-X (62M) |
| 输入 | 1280×1280 |
| Batch | 2（HGNetv2-X 更重） |
| Epoch | 36（DEIM 减半，效果≈D-FINE 72 epoch） |
| 总迭代 | ~36-45 万 step（与方案1接近） |
| 预估时间 | **40-60 小时**（48GB 4090） |

---

## 四、对比总结

| 维度 | RT-DETRv2-X | DEIM-D-FINE-X |
|---|---|---|
| **骨干** | ResNet101-vd (67M) | HGNetv2-X (62M) |
| **标签分配** | 匈牙利一对一 | DEIM 密集→一对一（两阶段） |
| **框回归** | 直接 xywh 标量 | FDR 16-bin 概率分布精修 |
| **分类损失** | VFL | MAL（匹配感知） |
| **NMS** | 无 | 无 |
| **LR 调度** | CosineDecay | FlatCosine（高 lr 停留更久） |
| **训练 epoch** | 72 | 36 |
| **batch (1280)** | 4 | 2 |
| **每 epoch 时长** | ~35 分钟 | ~70-100 分钟 |
| **预估总时长** | 35-50 小时 | 40-60 小时 |
| **代码仓库** | lyuwenyu/RT-DETR | Intellindust-AI-Lab/DEIM |
| **COCO 参考 AP** | 54.3 (v2-X @640) | 56.5 (DEIM-D-FINE-X @640) |
| **主要回答什么** | 标签分配+NMS 是不是瓶颈 | 框贴不准+小样本监督 是不是瓶颈 |
| **稀有类友好度** | 中（一对一匹配样本少） | 高（前半段密集匹配给多倍信号） |

### 判读预期

| 场景 | 结论 |
|---|---|
| 两个都 >0.25 | 范式是瓶颈，DETR 架构确认有效 |
| DEIM 明显 > RT-DETR | 框贴不准+监督稀疏是主因，后续重点做定位精修 |
| RT-DETR ≥ DEIM | 密集匹配未带来增益，一对一已足够 |
| 两个都在 0.20 附近 | 数据是天花板，停止折腾架构 |

---

## 五、运行方式

```bash
# 方案1：RT-DETRv2-X（~2天）
python plans/rtdetrv2_x/train.py

# 方案2：DEIM-D-FINE-X（~2天）  
python plans/dfine/train.py

# 打断后恢复（两个脚本都支持）
python plans/rtdetrv2_x/train.py --resume
python plans/dfine/train.py --resume

# 自定义 epoch（比如想少跑几轮快速看趋势）
python plans/rtdetrv2_x/train.py --epochs 36
python plans/dfine/train.py --epochs 18
```

两个脚本的行为：
- 首次运行自动 git clone 对应仓库 + 装依赖 + YOLO→COCO 转换
- `Ctrl+C` 中断后打印 checkpoint 位置和恢复命令
- 训练过程中每 10 batch 打印 loss，每 epoch 跑验证并打印 mAP
- 每 5 epoch 存一次 checkpoint
- mAP50 15 epoch 不涨自动早停
