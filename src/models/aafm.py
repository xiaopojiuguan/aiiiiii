"""AAFM: Artifact-Aware Feature Modulation —— 方案 5.2.1

抗伪影门控融合：
  F_res = Proj( Stem( I - GaussianBlur(I) ) )
  F_out = F_raw + sigmoid(Conv([F_raw, F_res])) × F_res

修复:
  - 残差分支通过投影层匹配 backbone P2/P3 通道
  - 模块为标准 nn.Module，checkpoint 自动持久化
"""

import torch
import torch.nn as nn


class AAFMGate(nn.Module):
    """门控融合单元。无 BN，bias=-6 → sigmoid≈0.0025 → 初始 F_out≈F_raw。"""

    def __init__(self, channels: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1, bias=True),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.gate[0].bias, -6.0)
        nn.init.kaiming_normal_(self.gate[0].weight, mode='fan_out', nonlinearity='linear')

    def forward(self, f_raw, f_res):
        if f_res.shape[2:] != f_raw.shape[2:]:
            f_res = nn.functional.interpolate(f_res, size=f_raw.shape[2:], mode="bilinear", align_corners=False)
        g = self.gate(torch.cat([f_raw, f_res], dim=1))
        return f_raw + g * f_res

    def forward(self, f_raw, f_res):
        if f_res.shape[2:] != f_raw.shape[2:]:
            f_res = nn.functional.interpolate(f_res, size=f_raw.shape[2:], mode="bilinear", align_corners=False)
        g = self.gate(torch.cat([f_raw, f_res], dim=1))
        return f_raw + g * f_res


class AAFMBlock(nn.Module):
    """AAFM 模块：残差 stem + 按层投影 + 门控融合。

    Args:
        in_channels: 输入图像通道 (3)
        stem_channels: stem 中间通道
        p2_channels: backbone P2 输出通道
        p3_channels: backbone P3 输出通道
    """

    def __init__(self, in_channels: int = 3, stem_channels: int = 32,
                 p2_channels: int = 64, p3_channels: int = 128):
        super().__init__()
        # 残差特征提取 (H → H/2 → H/4)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(stem_channels, stem_channels * 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels * 2),
            nn.SiLU(inplace=True),
        )
        stem_out = stem_channels * 2  # 64 (at H/4)

        # 投影到 backbone 各级通道
        self.proj_p2 = nn.Conv2d(stem_out, p2_channels, 1, bias=False) if p2_channels != stem_out else nn.Identity()
        self.proj_p3 = nn.Conv2d(stem_out, p3_channels, 1, bias=False) if p3_channels != stem_out else nn.Identity()

        # 门控
        self.gate_p2 = AAFMGate(p2_channels)
        self.gate_p3 = AAFMGate(p3_channels)

    def forward(self, residual, backbone_p2, backbone_p3):
        """Args:
            residual: (B,3,H,W) I - blur(I)
            backbone_p2: (B,C2,H/4,W/4)
            backbone_p3: (B,C3,H/8,W/8)
        Returns:
            p2_out, p3_out
        """
        f = self.stem(residual)               # (B, 64, H/4, W/4)

        p2_out = self.gate_p2(backbone_p2, self.proj_p2(f))
        p3_out = self.gate_p3(backbone_p3, self.proj_p3(f))
        return p2_out, p3_out
