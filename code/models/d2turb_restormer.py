"""Restormer-based D2Turb inference network."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .restormer import Restormer


class BasicConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
            ),
            nn.GELU(),
        )

    def forward(self, feature: Tensor) -> Tensor:
        return self.body(feature)


class ASPI(nn.Module):
    """Adaptive Structural Prior Injection through channel attention."""

    def __init__(self, guide_channels: int = 96, rectifier_channels: int = 32) -> None:
        super().__init__()
        self.query = nn.Conv2d(rectifier_channels, rectifier_channels, kernel_size=1, bias=False)
        self.key = nn.Conv2d(guide_channels, rectifier_channels, kernel_size=1, bias=False)
        self.value = nn.Conv2d(guide_channels, rectifier_channels, kernel_size=1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, rectifier_feature: Tensor, guide_feature: Tensor) -> Tensor:
        batch, channels, height, width = rectifier_feature.shape
        query = self.query(rectifier_feature).reshape(batch, channels, -1)
        key = self.key(guide_feature).reshape(batch, channels, -1)
        value = self.value(guide_feature).reshape(batch, channels, -1)
        injected = F.scaled_dot_product_attention(query, key, value)
        injected = injected.reshape(batch, channels, height, width)
        return rectifier_feature + self.gamma * injected


class TiltRectifier(nn.Module):
    """Deep pixel-flow rectifier guided by Restormer features through ASPI."""

    def __init__(self, base_channels: int = 32, guide_channels: int = 96) -> None:
        super().__init__()
        self.input_projection = BasicConv(3, base_channels, 3, 1)
        self.aspi = ASPI(guide_channels=guide_channels, rectifier_channels=base_channels)
        self.down_1 = BasicConv(base_channels, base_channels * 2, 3, 2)
        self.down_2 = BasicConv(base_channels * 2, base_channels * 4, 3, 2)
        self.down_3 = BasicConv(base_channels * 4, base_channels * 8, 3, 2)
        self.down_4 = BasicConv(base_channels * 8, base_channels * 8, 3, 2)
        self.down_5 = BasicConv(base_channels * 8, base_channels * 8, 3, 2)
        self.bottleneck = nn.Sequential(
            BasicConv(base_channels * 8, base_channels * 8, 3, 1),
            BasicConv(base_channels * 8, base_channels * 8, 3, 1),
        )
        self.up_sample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.decode_5 = BasicConv(base_channels * 16, base_channels * 8, 3, 1)
        self.decode_4 = BasicConv(base_channels * 16, base_channels * 8, 3, 1)
        self.decode_3 = BasicConv(base_channels * 12, base_channels * 4, 3, 1)
        self.decode_2 = BasicConv(base_channels * 6, base_channels * 2, 3, 1)
        self.decode_1 = BasicConv(base_channels * 3, base_channels, 3, 1)
        self.flow_head = nn.Conv2d(base_channels, 2, kernel_size=3, padding=1)
        nn.init.constant_(self.flow_head.weight, 0)
        nn.init.constant_(self.flow_head.bias, 0)

    @staticmethod
    def _base_grid(batch: int, height: int, width: int, image: Tensor) -> Tensor:
        row, column = torch.meshgrid(
            torch.arange(height, device=image.device, dtype=image.dtype),
            torch.arange(width, device=image.device, dtype=image.dtype),
            indexing="ij",
        )
        x_coordinate = 2.0 * column / (width - 1.0) - 1.0
        y_coordinate = 2.0 * row / (height - 1.0) - 1.0
        grid = torch.stack((x_coordinate, y_coordinate), dim=-1)
        return grid.unsqueeze(0).repeat(batch, 1, 1, 1)

    def forward(self, deblurred: Tensor, guide_feature: Tensor) -> tuple[Tensor, Tensor]:
        encoder_1 = self.input_projection(deblurred)
        encoder_2 = self.down_1(self.aspi(encoder_1, guide_feature))
        encoder_3 = self.down_2(encoder_2)
        encoder_4 = self.down_3(encoder_3)
        encoder_5 = self.down_4(encoder_4)
        encoder_6 = self.down_5(encoder_5)

        bottleneck = self.bottleneck(encoder_6)
        decoder_5 = self.decode_5(torch.cat([self.up_sample(bottleneck), encoder_5], dim=1))
        decoder_4 = self.decode_4(torch.cat([self.up_sample(decoder_5), encoder_4], dim=1))
        decoder_3 = self.decode_3(torch.cat([self.up_sample(decoder_4), encoder_3], dim=1))
        decoder_2 = self.decode_2(torch.cat([self.up_sample(decoder_3), encoder_2], dim=1))
        decoder_1 = self.decode_1(torch.cat([self.up_sample(decoder_2), encoder_1], dim=1))
        pixel_flow = self.flow_head(decoder_1)

        batch, _, height, width = deblurred.shape
        normalized_flow = torch.stack(
            (
                pixel_flow[:, 0] / ((width - 1.0) / 2.0),
                pixel_flow[:, 1] / ((height - 1.0) / 2.0),
            ),
            dim=-1,
        )
        sampling_grid = self._base_grid(batch, height, width, deblurred) + normalized_flow
        rectified = F.grid_sample(
            deblurred,
            sampling_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return rectified, pixel_flow


class D2TurbRestormer(nn.Module):
    """D2Turb inference path with Restormer restoration and ASPI rectification."""

    def __init__(self) -> None:
        super().__init__()
        self.restormer = Restormer()
        self.rectifier = TiltRectifier(guide_channels=self.restormer.guide_channels)

    @staticmethod
    def _pad_input(image: Tensor) -> tuple[Tensor, int, int]:
        height, width = image.shape[-2:]
        pad_h = (-height) % 32
        pad_w = (-width) % 32
        if not pad_h and not pad_w:
            return image, height, width
        mode = "reflect" if pad_h < height and pad_w < width else "replicate"
        return F.pad(image, (0, pad_w, 0, pad_h), mode=mode), height, width

    def forward(self, image: Tensor) -> dict[str, Tensor]:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("Expected an RGB tensor with shape (batch, 3, height, width).")

        padded, original_h, original_w = self._pad_input(image)
        deblurred, guide_feature = self.restormer(padded, return_features=True)
        rectified, pixel_flow = self.rectifier(deblurred, guide_feature)
        return {
            "deblurred": deblurred[:, :, :original_h, :original_w],
            "restored": rectified[:, :, :original_h, :original_w],
            "pixel_flow": pixel_flow[:, :, :original_h, :original_w],
        }
