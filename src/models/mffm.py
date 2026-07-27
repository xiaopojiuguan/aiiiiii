"""MFFM: Multi-Morphology Feature Fusion —— 方案 5.2.2

替换 Neck P2/P3 融合节点的标准卷积为多形态并行分支:
  - 3×3: 块状纹理    - 1×7: 横向细长
  - 7×1: 纵向细长    - DCNv2: 不规则轮廓
  可学习 softmax 权重 → 1×1 融合

修复:
  - DCNv2 offset 由独立 conv 生成
  - 模块设置 i/f/type 属性兼容 ultralytics
  - 模块为标准 nn.Module，checkpoint 自动持久化
"""

import torch
import torch.nn as nn


class MorphConv(nn.Module):
    """单形态卷积分支。"""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: tuple):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size,
                      padding=(kernel_size[0] // 2, kernel_size[1] // 2), bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DeformConvBlock(nn.Module):
    """DCNv2 分支：offset conv + deformable conv。"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.offset_conv = nn.Conv2d(in_ch, 2 * 3 * 3, kernel_size=3, padding=1)
        self.dcn = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)  # fallback
        try:
            from torchvision.ops import DeformConv2d
            self.dcn = DeformConv2d(in_ch, out_ch, 3, padding=1)
            self._use_dcn = True
        except ImportError:
            self._use_dcn = False
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        offset = self.offset_conv(x)
        if self._use_dcn:
            x = self.dcn(x, offset)
        else:
            x = self.dcn(x)
        return self.act(self.bn(x))


class MFFMBlock(nn.Module):
    """多形态特征融合块，替换 Neck 中的 C3k2。

    Args:
        in_channels: 输入通道
        out_channels: 输出通道
        use_dcn: 是否启用 DCNv2 分支
    """

    def __init__(self, in_channels: int, out_channels: int, use_dcn: bool = False):
        super().__init__()
        n_branches = 4 if use_dcn else 3
        mid_ch = out_channels // n_branches

        # 确保 mid_ch > 0
        mid_ch = max(mid_ch, 8)

        self.branch_3x3 = MorphConv(in_channels, mid_ch, (3, 3))
        self.branch_1x7 = MorphConv(in_channels, mid_ch, (1, 7))
        self.branch_7x1 = MorphConv(in_channels, mid_ch, (7, 1))
        self.use_dcn = use_dcn

        if use_dcn:
            self.branch_dcn = DeformConvBlock(in_channels, mid_ch)
            total = mid_ch * 4
            self.branch_weights = nn.Parameter(torch.ones(4))
        else:
            total = mid_ch * 3
            self.branch_weights = nn.Parameter(torch.ones(3))

        self.fuse = nn.Conv2d(total, out_channels, 1, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

        # ultralytics 兼容属性（由 apply_mffm 设置）
        self.i = -1
        self.f = -1
        self.type = "MFFMBlock"

    def forward(self, x):
        branches = [
            self.branch_3x3(x),
            self.branch_1x7(x),
            self.branch_7x1(x),
        ]
        if self.use_dcn:
            branches.append(self.branch_dcn(x))

        w = torch.softmax(self.branch_weights, dim=0)
        weighted = [w[i] * b for i, b in enumerate(branches)]
        fused = torch.cat(weighted, dim=1)
        return self.act(self.norm(self.fuse(fused)))
