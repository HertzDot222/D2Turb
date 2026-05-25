"""Restormer texture-restoration backbone for D2Turb inference."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class MDTA(nn.Module):
    """Multi-DConv head transposed attention."""

    def __init__(self, channels: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.qkv_conv = nn.Conv2d(
            channels * 3,
            channels * 3,
            kernel_size=3,
            padding=1,
            groups=channels * 3,
            bias=False,
        )
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, feature: Tensor) -> Tensor:
        batch, _, height, width = feature.shape
        query, key, value = self.qkv_conv(self.qkv(feature)).chunk(3, dim=1)
        query = query.reshape(batch, self.num_heads, -1, height * width)
        key = key.reshape(batch, self.num_heads, -1, height * width)
        value = value.reshape(batch, self.num_heads, -1, height * width)
        query = F.normalize(query, dim=-1)
        key = F.normalize(key, dim=-1)

        attention = torch.softmax(
            torch.matmul(query, key.transpose(-2, -1).contiguous()) * self.temperature,
            dim=-1,
        )
        output = torch.matmul(attention, value).reshape(batch, -1, height, width)
        return self.project_out(output)


class GDFN(nn.Module):
    """Gated-DConv feed-forward network."""

    def __init__(self, channels: int, expansion_factor: float) -> None:
        super().__init__()
        hidden_channels = int(channels * expansion_factor)
        self.project_in = nn.Conv2d(channels, hidden_channels * 2, kernel_size=1, bias=False)
        self.conv = nn.Conv2d(
            hidden_channels * 2,
            hidden_channels * 2,
            kernel_size=3,
            padding=1,
            groups=hidden_channels * 2,
            bias=False,
        )
        self.project_out = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)

    def forward(self, feature: Tensor) -> Tensor:
        feature_a, feature_b = self.conv(self.project_in(feature)).chunk(2, dim=1)
        return self.project_out(F.gelu(feature_a) * feature_b)


class TransformerBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int, expansion_factor: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attention = MDTA(channels, num_heads)
        self.norm2 = nn.LayerNorm(channels)
        self.feed_forward = GDFN(channels, expansion_factor)

    @staticmethod
    def _normalize(feature: Tensor, norm: nn.LayerNorm) -> Tensor:
        batch, channels, height, width = feature.shape
        normalized = norm(feature.reshape(batch, channels, -1).transpose(-2, -1).contiguous())
        return normalized.transpose(-2, -1).contiguous().reshape(batch, channels, height, width)

    def forward(self, feature: Tensor) -> Tensor:
        feature = feature + self.attention(self._normalize(feature, self.norm1))
        return feature + self.feed_forward(self._normalize(feature, self.norm2))


class DownSample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, feature: Tensor) -> Tensor:
        return self.body(feature)


class UpSample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, feature: Tensor) -> Tensor:
        return self.body(feature)


class Restormer(nn.Module):
    """Restormer restoration stage returning the ASPI guide feature on request."""

    def __init__(
        self,
        num_blocks: Sequence[int] = (4, 6, 6, 8),
        num_heads: Sequence[int] = (1, 2, 4, 8),
        channels: Sequence[int] = (48, 96, 192, 384),
        num_refinement: int = 4,
        expansion_factor: float = 2.66,
    ) -> None:
        super().__init__()
        self.guide_channels = channels[1]
        self.embed_conv = nn.Conv2d(3, channels[0], kernel_size=3, padding=1, bias=False)
        self.encoders = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        TransformerBlock(channel, head_count, expansion_factor)
                        for _ in range(block_count)
                    ]
                )
                for block_count, head_count, channel in zip(num_blocks, num_heads, channels)
            ]
        )
        self.downs = nn.ModuleList([DownSample(channel) for channel in channels[:-1]])
        self.ups = nn.ModuleList([UpSample(channel) for channel in reversed(channels[1:])])
        self.reduces = nn.ModuleList(
            [
                nn.Conv2d(channels[index], channels[index - 1], kernel_size=1, bias=False)
                for index in reversed(range(2, len(channels)))
            ]
        )
        self.decoders = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        TransformerBlock(channels[2], num_heads[2], expansion_factor)
                        for _ in range(num_blocks[2])
                    ]
                ),
                nn.Sequential(
                    *[
                        TransformerBlock(channels[1], num_heads[1], expansion_factor)
                        for _ in range(num_blocks[1])
                    ]
                ),
                nn.Sequential(
                    *[
                        TransformerBlock(channels[1], num_heads[0], expansion_factor)
                        for _ in range(num_blocks[0])
                    ]
                ),
            ]
        )
        self.refinement = nn.Sequential(
            *[
                TransformerBlock(channels[1], num_heads[0], expansion_factor)
                for _ in range(num_refinement)
            ]
        )
        self.output = nn.Conv2d(channels[1], 3, kernel_size=3, padding=1, bias=False)

    @staticmethod
    def _pad_input(image: Tensor) -> tuple[Tensor, int, int]:
        height, width = image.shape[-2:]
        pad_h = (-height) % 8
        pad_w = (-width) % 8
        if not pad_h and not pad_w:
            return image, height, width
        mode = "reflect" if pad_h < height and pad_w < width else "replicate"
        return F.pad(image, (0, pad_w, 0, pad_h), mode=mode), height, width

    def forward(self, image: Tensor, return_features: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        image, original_h, original_w = self._pad_input(image)
        encoder_1 = self.encoders[0](self.embed_conv(image))
        encoder_2 = self.encoders[1](self.downs[0](encoder_1))
        encoder_3 = self.encoders[2](self.downs[1](encoder_2))
        encoder_4 = self.encoders[3](self.downs[2](encoder_3))

        decoder_3 = self.decoders[0](
            self.reduces[0](torch.cat([self.ups[0](encoder_4), encoder_3], dim=1))
        )
        decoder_2 = self.decoders[1](
            self.reduces[1](torch.cat([self.ups[1](decoder_3), encoder_2], dim=1))
        )
        decoder_1 = self.decoders[2](torch.cat([self.ups[2](decoder_2), encoder_1], dim=1))
        guide_feature = self.refinement(decoder_1)
        restored = self.output(guide_feature) + image

        restored = restored[:, :, :original_h, :original_w]
        guide_feature = guide_feature[:, :, :original_h, :original_w]
        if return_features:
            return restored, guide_feature
        return restored
