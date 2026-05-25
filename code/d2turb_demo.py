"""D2Turb simplified architecture demonstration.

This public example preserves the main two-stage data flow:
texture deblurring supplies structural guide features, and a rectifier uses
ASPI to fuse those features before predicting a restored image. It is a compact
architecture demonstration, not the full training or evaluation implementation.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Small residual block used by both demonstration stages."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, features: Tensor) -> Tensor:
        return features + self.body(features)


class TextureDeblur(nn.Module):
    """Estimate a deblurred image while exposing structural guide features."""

    def __init__(self, channels: int = 32, blocks: int = 3) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.encoder = nn.Sequential(*(ResidualBlock(channels) for _ in range(blocks)))
        self.image_head = nn.Conv2d(channels, 3, kernel_size=3, padding=1)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor]:
        guide_features = self.encoder(self.stem(image))
        deblurred = torch.clamp(image + self.image_head(guide_features), 0.0, 1.0)
        return deblurred, guide_features


class ASPI(nn.Module):
    """Adaptive Structural Prior Injection using channel-wise attention."""

    def __init__(self, channels: int = 32) -> None:
        super().__init__()
        self.query = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.key = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.value = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, rectifier_features: Tensor, guide_features: Tensor) -> Tensor:
        batch, channels, height, width = rectifier_features.shape
        query = self.query(rectifier_features).flatten(2)
        key = self.key(guide_features).flatten(2)
        value = self.value(guide_features).flatten(2)
        injected = F.scaled_dot_product_attention(query, key, value)
        injected = injected.view(batch, channels, height, width)
        return rectifier_features + self.scale * injected


class TiltRectifier(nn.Module):
    """Fuse structural guidance and predict a residual geometric correction."""

    def __init__(self, channels: int = 32, blocks: int = 3) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.encoder = nn.Sequential(*(ResidualBlock(channels) for _ in range(blocks)))
        self.aspi = ASPI(channels)
        self.decoder = nn.Sequential(
            ResidualBlock(channels),
            nn.Conv2d(channels, 3, kernel_size=3, padding=1),
        )

    def forward(self, deblurred: Tensor, guide_features: Tensor) -> Tensor:
        rectifier_features = self.encoder(self.stem(deblurred))
        fused_features = self.aspi(rectifier_features, guide_features)
        residual = self.decoder(fused_features)
        return torch.clamp(deblurred + residual, 0.0, 1.0)


class D2TurbDemo(nn.Module):
    """Simplified public interface for the D2Turb restoration pipeline."""

    def __init__(self, channels: int = 32) -> None:
        super().__init__()
        self.deblur = TextureDeblur(channels=channels)
        self.rectifier = TiltRectifier(channels=channels)

    def forward(self, image: Tensor) -> dict[str, Tensor]:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("Expected input shaped [batch, 3, height, width].")
        deblurred, guide_features = self.deblur(image)
        restored = self.rectifier(deblurred, guide_features)
        return {"deblurred": deblurred, "restored": restored}


def main() -> None:
    torch.manual_seed(0)
    model = D2TurbDemo().eval()
    turbulent_image = torch.rand(1, 3, 128, 128)
    with torch.no_grad():
        outputs = model(turbulent_image)

    print(f"input:     {tuple(turbulent_image.shape)}")
    print(f"deblurred: {tuple(outputs['deblurred'].shape)}")
    print(f"restored:  {tuple(outputs['restored'].shape)}")


if __name__ == "__main__":
    main()
