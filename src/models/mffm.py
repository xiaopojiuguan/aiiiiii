"""MFFM: Multi-Morphology Feature Fusion —— 方案 5.2.2

形态感知融合模块，替换 Neck 中 P2/P3 层的标准卷积：
  - 3×3 卷积分支：局部块状纹理（点状缺陷）
  - 1×7 卷积分支：横向细长结构（横向划伤）
  - 7×1 卷积分支：纵向细长结构（纵裂）
  - 可学习权重归一化融合

放在 Neck 高分辨率层（P2/P3），避免显著增加整网计算量。
"""

import torch
import torch.nn as nn


class MorphConv(nn.Module):
    """单形态卷积分支：不同 kernel size 捕获不同形态。"""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: tuple, groups: int = 1):
        super().__init__()
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class MFFMBlock(nn.Module):
    """多形态特征融合块。

    替代 Neck 中的标准 C3k2/Conv 块，在 P2/P3 融合节点使用。
    四路并行 → 拼接 → 1×1 压缩回原通道数。

    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数（通过 1×1 压缩后）
        use_dcn: 是否启用 DCNv2 分支（P2 可选，P3 建议开启）
    """

    def __init__(self, in_channels: int, out_channels: int, use_dcn: bool = False):
        super().__init__()
        mid_channels = out_channels // 4

        self.branch_3x3 = MorphConv(in_channels, mid_channels, (3, 3))
        self.branch_1x7 = MorphConv(in_channels, mid_channels, (1, 7))
        self.branch_7x1 = MorphConv(in_channels, mid_channels, (7, 1))

        dcn_ch = mid_channels
        if use_dcn:
            try:
                from torchvision.ops import DeformConv2d
                self.branch_dcn = nn.Sequential(
                    DeformConv2d(in_channels, dcn_ch, 3, padding=1, bias=False),
                    nn.BatchNorm2d(dcn_ch),
                    nn.SiLU(inplace=True),
                )
            except ImportError:
                # DCNv2 不可用时退化为普通 3×3
                self.branch_dcn = MorphConv(in_channels, dcn_ch, (3, 3))
        else:
            dcn_ch = 0

        total_mid = mid_channels * 3 + dcn_ch
        self.fuse = nn.Conv2d(total_mid, out_channels, 1, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

        # 可学习权重（对四个分支做加权）
        self.branch_weights = nn.Parameter(torch.ones(4))

    def forward(self, x):
        b3 = self.branch_3x3(x)
        b17 = self.branch_1x7(x)
        b71 = self.branch_7x1(x)

        branches = [b3, b17, b71]

        if hasattr(self, 'branch_dcn'):
            b_dcn = self.branch_dcn(x)
            branches.append(b_dcn)

        # 可学习权重 + softmax 归一化
        weights = torch.softmax(self.branch_weights[:len(branches)], dim=0)
        weighted = [w * b for w, b in zip(weights, branches)]

        fused = torch.cat(weighted, dim=1)
        out = self.fuse(fused)
        out = self.norm(out)
        out = self.act(out)
        return out
