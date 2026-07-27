# 换模型方案目录

> 用途:YOLO11m-P2 → 0.208、YOLO11x-P2 → 0.206 之后,换哪些模型、每个怎么跑
> 更新:2026-07-27
> 说明:安装命令与超参按通用经验写出,未在本机验证。环境风险等级见每节。

---

## 一、为什么换模型大小没用,换范式才可能有用

| 模型 | 参数量 | mAP50 |
|---|---|---|
| YOLO11m-P2 | ~20M | 0.208 |
| YOLO11x-P2 | ~57M | 0.206 |

参数量 ×2.8,指标 −0.002。容量不是瓶颈。

11m 和 11x 是**同一套设计**,只改 depth/width 两个缩放系数,共用:

- **TAL 标签分配**(Task-Aligned Assigner,一对多)
- **NMS 后处理**
- **耦合的分类/回归头**(共享 neck 特征)
- **COCO 彩色自然图像预训练**

在 YOLO 系列里换型号,这四样一个都不变。所以下面的候选按「换掉哪一样」来组织,而不是按「哪个模型 SOTA」。

| 方案 | 换掉的东西 |
|---|---|
| 1 RT-DETRv2 | 标签分配(→一对一匈牙利)+ NMS(→无) |
| 2 D-FINE / DEIM | 同上,再换框回归方式(→分布精修) |
| 3 Co-DINO | 同上,再加一对多辅助头补监督信号 |
| 4 Cascade R-CNN | 换成两阶段 + RoIAlign + 递进 IoU 精修 |
| 5 两阶段解耦 | 把分类从检测里拆出来,独立训 |
| 6 DINOv2 冻结骨干 | 换预训练域(COCO → 自监督通用特征) |

---

## 二、统一对比协议(必须先定,否则数字不可比)

所有候选用同一套规则,否则跑出来的数没法横向比。

**数据**:复用 `data_splits/train_val_split.json`,不重新划分(避免同板材泄漏)。

**速筛配置**(用于架构排序,不是最终成绩):

| 项 | 值 |
|---|---|
| 训练集 | `tiles_train` 随机抽 6,000 张(固定 seed=42) |
| 验证集 | `tiles_val` 随机抽 2,000 张(固定 seed=42) |
| epoch | 15 |
| imgsz | 1280 |
| batch | 2(显存不够降 1 + 梯度累积) |
| AMP | 开 |

单个候选约 4–6 小时,而不是全量的 28 小时。一天能比 2–3 个架构。

**评估**:全部用修复版 `src/utils/metrics.py`,报告切片级 + 整图级两套数,conf=0.001、max_det=300 固定。

**注意**:速筛的绝对值会低于全量训练,只能用于**排序**,不能直接和 0.208 比大小。选出赢家后再上全量重训拿真实成绩。

---

## 方案 1:RT-DETRv2-R50

**环境风险:低**(ultralytics 自带 RTDETR 类,你已装 8.4.105,不需要新仓库)

**换掉了什么**:一对多 TAL → 一对一匈牙利匹配;有 NMS → 无 NMS。

**为什么可能有用**:钢板缺陷常成片密集出现(麻面麻坑尤其)。密集同类目标下,NMS 的 IoU 阈值是个两难——调高留重复框,调低压掉真目标。RT-DETR 的一对一匹配从设计上绕开这个问题。另外 TAL 对小目标的正样本分配一直偏少,匈牙利匹配不依赖 anchor/中心先验。

### 流程

**1. 环境**
```bash
# 无需额外安装,ultralytics 内置
python -c "from ultralytics import RTDETR; print('ok')"
```

**2. 数据**:零改动。RT-DETR 吃的就是标准 YOLO txt 格式,直接指向 `tile_dataset.yaml`。

**3. 训练**
```bash
PYTHONUNBUFFERED=1 python -u -c "
from ultralytics import RTDETR
m = RTDETR('rtdetr-l.pt')
m.train(
    data='tile_dataset.yaml', epochs=15, imgsz=1280,
    batch=2, optimizer='AdamW', lr0=1e-4, weight_decay=1e-4,
    warmup_epochs=2, amp=True, cache=False,
    hsv_h=0.0, hsv_s=0.0, hsv_v=0.3,
    degrees=0.0, fliplr=0.5, flipud=0.0, mosaic=0.0,
    project='outputs/checkpoints', name='rtdetr_screen'
)
"
```
注意 `mosaic=0.0` — DETR 系列对 mosaic 普遍不友好,会打乱匈牙利匹配的全局一致性。

**4. 推理**:不要走 `scripts/predict.py` 的 Soft-NMS 分支。RT-DETR 直接输出 300 个 query,按 conf 排序取即可。需要在 predict.py 里加一个 `--no-nms` 开关。

**5. 判读**

| 速筛 mAP50 | 结论 |
|---|---|
| > 0.25 | 范式是瓶颈 → 上全量 + RT-DETRv2-R101 |
| 0.19–0.23 | 两种完全不同的 assignment 同分 → 强烈指向数据是天花板 |
| < 0.15 | 可能只是没收敛,先看曲线还在不在涨 |

**成本**:约 5–6 小时。

**坑**:DETR 系列公认收敛慢(原始 DETR 需要 500 epoch,RT-DETR 改善到 ~72)。15 epoch 对它不公平。**必须看 loss/mAP 曲线是否还在单调上升**——如果还在涨,结论只能是「未收敛」,不能是「不行」。这时给它 30 epoch 再判。

---

## 方案 2:D-FINE(或 DEIM)

**环境风险:中**(需要 clone 官方仓库;也可试 HuggingFace transformers 的 `DFineForObjectDetection`,后者对新版 torch 兼容性更好)

**换掉了什么**:在 RT-DETR 基础上,再把框回归从「直接回归坐标」换成「细粒度分布精修」(Fine-grained Distribution Refinement)。

**为什么可能有用**:你的 mAP50-95 = 0.121 vs mAP50 = 0.208,只有 58%。这个比例偏低,说明框找得到但**贴不准**。D-FINE 的核心改动正好是提升框的定位质量,理论上直接命中这个症状。

### 流程

**1. 环境**(优先试 HF 路线,避开老仓库的依赖地狱)
```bash
pip install -U transformers
python -c "from transformers import DFineForObjectDetection; print('ok')"
```
如果 HF 没有,退回官方仓库:
```bash
git clone https://github.com/Peterande/D-FINE
cd D-FINE && pip install -r requirements.txt
```

**2. 数据**:需要 **COCO JSON 格式**。写一个转换脚本:
```
tiles_train/labels/*.txt (YOLO 归一化 xywh)
    → annotations/train.json (COCO 绝对 xywh + images/categories 段)
```
约 60 行,注意 category_id 从 1 开始(COCO 惯例),而 YOLO 从 0 开始。

**3. 训练**:改官方 config 的 `num_classes: 9`、`eval_spatial_size: [1280,1280]`、`train_dataloader.batch_size: 2`,加载 `dfine_m_coco.pth` 预训练权重。

**4. 推理 + 评估**:输出转回你的 JSON 格式,用同一份 metrics.py。

**5. 判读**:重点不看 mAP50,**看 mAP50-95 的提升幅度**。如果 50-95 涨得比 50 多(比例从 58% 往 65%+ 走),说明「框不准」的诊断成立,后续所有方案都该加定位精修。

**成本**:半天转格式 + 调 config,约 6 小时训练。

---

## 方案 3:Co-DINO(高精度上限探测)

**环境风险:高**(依赖 mmdetection + mmcv,在 Python 3.14 上大概率装不上,mmcv 预编译轮子跟进慢。可能需要单独开一个 Python 3.11 的 conda 环境)

**换掉了什么**:一对一匹配 + **额外挂一对多辅助头**,训练时两种监督并存,推理时只留一对一分支。

**为什么可能有用**:一对一匹配的已知缺点是正样本太少、监督信号稀疏,这在小数据集上更致命(你只有 3200 原图)。Co-DETR 的设计就是补这个。它是 COCO 上长期的 SOTA 之一。

**这个方案的定位是「探上限」**,不是「求实用」。用它回答:堆最强的检测器,这份数据能到多少?如果 Co-DINO 也只有 0.22,那 0.2 附近就是数据决定的,可以停止折腾架构了。

### 流程

**1. 环境**
```bash
conda create -n mmdet python=3.11 -y && conda activate mmdet
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -U openmim && mim install mmengine "mmcv>=2.0.0" mmdet
```

**2. 数据**:COCO JSON,复用方案 2 的转换脚本。

**3. 训练**:用 `co_dino_5scale_swin_l_16xb1_16e_o365tococo.py` 作模板,改 `num_classes=9`、`batch_size=1`、开 `amp` 和 `checkpointing`。

**4. 显存现实**:Swin-L + 5scale 在 1280 分辨率上,8GB **几乎肯定 OOM**。两个选择:
- 降到 `R50` backbone + 800 分辨率(但分辨率下降本身会掉点,成了混淆变量,判读时要记住)
- 租一张 24GB 卡跑这一个实验(这是最干净的做法,几十块钱)

**5. 判读**:这是「上限值」。所有其他方案都应该低于它。如果它也在 0.22 附近 → 数据天花板确认。

**成本**:1 天环境 + 8–10 小时训练。环境装不上的风险实打实存在,时间紧就跳过。

---

## 方案 4:Cascade R-CNN-R50-FPN

**环境风险:高**(同方案 3,mmdetection)

**换掉了什么**:单阶段 anchor-free → **两阶段 + RoIAlign**,并且用三级递进 IoU 阈值(0.5→0.6→0.7)逐级精修框。

**为什么可能有用**:两个理由。一是 RoIAlign 对每个 proposal 单独抠特征做分类,比单阶段共享 neck 特征的耦合头更适合「框内是什么」这种判断——而你的 val/cls_loss = 3.2 vs train 0.44 说明分类是主要失效点。二是递进 IoU 精修同样针对 mAP50-95 偏低。

### 流程

**1. 环境**:同方案 3 的 conda 环境。

**2. 数据**:COCO JSON。

**3. 训练**
```bash
python tools/train.py configs/cascade_rcnn/cascade-rcnn_r50_fpn_1x_coco.py \
  --cfg-options model.roi_head.bbox_head.0.num_classes=9 \
                model.roi_head.bbox_head.1.num_classes=9 \
                model.roi_head.bbox_head.2.num_classes=9 \
                train_dataloader.batch_size=2 \
  --amp
```
三个 bbox_head 的 num_classes 都要改,漏一个就报错。

**4. 判读**:mAP50-95/mAP50 的比例是不是从 58% 往上走。

**成本**:半天到 1 天(含环境)+ 约 6 小时训练。

---

## 方案 5:两阶段解耦(class-agnostic 检测 + 独立分类)

**环境风险:极低**(纯用你现有的 ultralytics + timm)

**换掉了什么**:不换检测器,**把「定位」和「分类」拆成两个独立训练的模型**。

**为什么可能有用**:baseline 的 `val/cls_loss = 3.2` 对 `train/cls_loss = 0.44`,差 7 倍;而 box_loss 是 0.756 对 0.327,只差 2.3 倍。分类分支的过拟合程度远高于回归分支。单阶段检测器强迫两个任务共享特征,分类拖垮整体。

**这个方案最大的价值是诊断性**:它把一个含糊的 mAP=0.2 拆成两个能单独读的数——

- Stage1 的 class-agnostic mAP50 回答:**缺陷找得到吗?**
- Stage2 的分类准确率回答:**缺陷分得清吗?**

### 流程

**1. Stage1 数据准备**(约 10 行脚本)
```
把 tiles_train/labels/*.txt 每行的首个数字(类别 id)全部改成 0
→ tiles_train_agnostic/labels/
images 用软链接或直接复用,不必复制
新建 agnostic_dataset.yaml: nc=1, names={0: defect}
```

**2. Stage1 训练**:就用你现成的 YOLO11m-P2,只改 nc=1。
```bash
python scripts/train_baseline.py --data agnostic_dataset.yaml --nc 1 --epochs 15
```

**3. Stage2 数据准备**:按 GT 框裁 patch。
```
对每个 GT 框,向外扩 1.3 倍取正方形裁剪 → resize 到 128×128
按类别存入 crops/{train,val}/{class_name}/
train/val 划分严格沿用同一份 split json,防泄漏
```

**4. Stage2 训练**:标准图像分类,不是检测。
```python
import timm
m = timm.create_model('convnext_tiny', pretrained=True, num_classes=9, in_chans=3)
# 关键:类别平衡采样 WeightedRandomSampler,或 loss 上加 class weight
# patch 只有 128×128,batch 可以开到 64,单 epoch 几分钟,能跑 50+ epoch
```

**5. 联合推理**:Stage1 出框 → 按框裁 patch → Stage2 定类别 → 合并成最终检测结果 → 同一份 metrics.py 评估。

**6. 判读表**

| Stage1 mAP50 | Stage2 acc | 结论 |
|---|---|---|
| > 0.45 | > 0.70 | 两个子任务都可解,端到端 0.2 是**耦合问题** → 两阶段直接当最终方案 |
| > 0.45 | < 0.50 | 找得到但分不清 → **类别定义重叠**,看混淆矩阵定位是哪几类互相吃 |
| ~0.25 | 任意 | 缺陷本身找不到 → 数据问题,换架构都无用 |

**成本**:半天写脚本 + Stage1 约 5 小时 + Stage2 约 1 小时。

**坑**:Stage2 在**GT 框**上训练,却在 Stage1 的**预测框**上推理,存在 train-test 失配。缓解办法:把 Stage1 在训练集上的假正例框作为第 10 类「背景」混进 Stage2 训练集。

---

## 方案 6:DINOv2/v3 冻结骨干 + 轻量检测头

**环境风险:低**(timm 或 HF transformers 都有,不需要老仓库)

**换掉了什么**:COCO 彩色自然图像**有监督**预训练 → 大规模**自监督**通用特征。

**为什么可能有用**:两个理由叠加。一是域差距——COCO 是彩色自然场景,你的数据是灰度工业成像,ImageNet/COCO 有监督特征在这上面迁移性差;自监督 ViT 特征公认更通用。二是长尾——qilie 31 例、huashang 35 例,冻结骨干后可训练参数量从 20M 掉到几 M,小样本过拟合风险大幅下降,这对你的极端长尾正对症。

### 流程

**1. 环境**
```bash
pip install timm
python -c "import timm; m=timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=True); print('ok')"
```

**2. 架构组装**(需要自己写,约 1 天)
```
DINOv2 ViT-B/14 (完全冻结,eval 模式,requires_grad=False)
    ↓ 取第 [3,6,9,12] 层的 patch token,reshape 成 2D 特征图
简单 FPN / 多尺度融合 (可训练)
    ↓
YOLO 检测头 或 anchor-free 头 (可训练)
```

**3. 显存与分辨率**:ViT-B 在 1280 上需要 batch=1 + gradient checkpointing。ViT-L 需降到 896。注意 DINOv2 是 patch14,输入边长最好是 14 的倍数(1288 而非 1280),否则要处理插值。

**4. 关键优化 — 特征缓存**:骨干冻结意味着**同一张图的特征永远不变**,可以预先全部算好存磁盘,训练时只跑检测头。单 epoch 时间能降一个数量级,轻松跑 30+ epoch。代价是磁盘占用(算一下:6000 张 × 4 层特征图,可能几十 GB,注意剩余空间)。

**5. 判读**

| 结果 | 结论 |
|---|---|
| 比 YOLO 高 > 0.05 | 预训练域是关键 → 换更大 DINOv3 + 解冻最后几层微调 |
| 稀有类 AP 明显改善 | 长尾假设成立 → 后续重点做重采样/CP/合成 |
| 仍在 0.20 附近 | 两套完全不同的特征提取器同分 → 接近确认数据天花板 |

**成本**:1 天实现 + 约 4 小时训练(有缓存的话更短)。

---

## 三、汇总对比

| # | 方案 | 环境风险 | 实现成本 | 训练耗时 | 主要回答什么 |
|---|---|---|---|---|---|
| 5 | 两阶段解耦 | 极低 | 半天 | ~6h | 找不到 vs 分不清,哪个是瓶颈 |
| 1 | RT-DETRv2 | 低 | 无 | ~6h | assignment + NMS 是不是瓶颈 |
| 6 | DINOv2 冻结 | 低 | 1天 | ~4h | 预训练域 + 长尾是不是瓶颈 |
| 2 | D-FINE | 中 | 半天 | ~6h | 框贴不准是不是瓶颈 |
| 4 | Cascade R-CNN | 高 | 半天–1天 | ~6h | 两阶段 RoIAlign 是否更适合 |
| 3 | Co-DINO | 高 | 1天 | ~10h | 这份数据的性能上限在哪 |

### 建议顺序

**第 1 天**:方案 5(两阶段解耦)。零环境风险,而且是唯一能把「找得到吗」和「分得清吗」拆开单独读的方案。跑完就知道后面该往哪个方向使劲。

**第 2 天**:方案 1(RT-DETRv2)。ultralytics 内置,数据零改动,基本是纯跑一条命令的成本。

**第 3–4 天**:按第 1 天的结果分叉——
- 若「分不清」是瓶颈 → 方案 6(换预训练域)
- 若「框不准」是瓶颈 → 方案 2(D-FINE)
- 若两者都还行但端到端差 → 方案 4(两阶段检测器)

**可选**:方案 3(Co-DINO)租卡跑一次,拿到上限值。如果最强检测器也只有 0.22,就不必再试第 7 个架构了。

赢家再上全量数据重训(约 28h)拿最终成绩。

---

## 四、几个必须统一的细节

**评估口径**:切片级和整图级是两个不同的数,0.208 / 0.206 都是切片级。赛题最终按整图级算分,新实验**两个都报**,别混用。

**metrics 版本**:0.208 是修 bug 前的 metrics.py 算的,可能虚高(GT 匹配状态重置会导致同一 GT 被多个预测重复匹配,TP 虚高)。需要确认 0.206 用的是哪一版——如果是修复后的,那 11m 和 11x「打平」这个前提本身就要重算。**新实验全部用修复版**。

**RT-DETR 的 NMS 差异**:方案 1/2/3 无 NMS,方案 4/5/6 有。这是范式固有差异,如实记录,不要为了「对齐」硬给 DETR 加 NMS。

**seed**:架构筛选阶段单 seed 可接受(只求排序)。最终选型至少 2 seed,增益要超过 seed 间标准差才算真的有效。

**速筛子集固定**:6000/2000 的抽样必须固定 seed 存成文件,所有候选用同一份,否则数字不可比。

