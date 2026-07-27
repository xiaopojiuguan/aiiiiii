"""AAFM: Artifact-Aware Feature Modulation —— 方案 5.2.1

抗伪影门控融合模块：
  - 主视图：原始灰度 → 3通道 → YOLO backbone
  - 辅助视图：局部对比度残差 R = I - GaussianBlur(I)
  - P2/P3 层门控融合：F_out = F_raw + sigmoid(Conv([F_raw, F_res])) × F_res
"""

import torch
import torch.nn as nn


class AAFMGate(nn.Module):
    """门控融合单元：由 F_raw 和 F_res 共同决定增强程度。

    F_out = F_raw + gate × F_res
    gate = sigmoid(ConvBN([F_raw, F_res]))
    """

    def __init__(self, channels: int):
        super().__init__()
        self.gate_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )

    def forward(self, f_raw, f_res):
        # f_raw: backbone features at this level
        # f_res: residual branch features (upsampled/interpolated to match)
        if f_res.shape[2:] != f_raw.shape[2:]:
            f_res = nn.functional.interpolate(f_res, size=f_raw.shape[2:], mode="bilinear", align_corners=False)

        gate_input = torch.cat([f_raw, f_res], dim=1)
        gate = self.gate_conv(gate_input)
        return f_raw + gate * f_res


class AAFMBlock(nn.Module):
    """AAFM：浅层残差分支 + 多层门控融合。

    Args:
        in_channels: 输入通道数（残差分支 stem 输出通道）
        p2_channels: P2 层 backbone 通道数
        p3_channels: P3 层 backbone 通道数
    """

    def __init__(self, in_channels: int = 3, stem_channels: int = 32,
                 p2_channels: int = 128, p3_channels: int = 256):
        super().__init__()
        # 浅层残差特征提取：输入为 R = I - GaussianBlur(I)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.SiLU(inplace=True),
        )
        self.res_conv = nn.Sequential(
            nn.Conv2d(stem_channels, stem_channels * 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels * 2),
            nn.SiLU(inplace=True),
        )

        # 门控单元
        self.gate_p2 = AAFMGate(p2_channels)  # stride=4
        self.gate_p3 = AAFMGate(p3_channels)  # stride=8

    def forward(self, residual_img, backbone_p2, backbone_p3):
        """Args:
            residual_img: (B, 3, H, W) 局部对比度残差图
            backbone_p2: (B, C2, H/4, W/4) backbone P2 特征
            backbone_p3: (B, C3, H/8, W/8) backbone P3 特征
        Returns:
            p2_out, p3_out: 门控融合后的特征
        """
        f_res = self.stem(residual_img)     # H/2
        f_res = self.res_conv(f_res)         # H/4

        p2_out = self.gate_p2(backbone_p2, f_res) if backbone_p2 is not None else None
        p3_out = self.gate_p3(backbone_p3, f_res) if backbone_p3 is not None else None
        return p2_out, p3_out
