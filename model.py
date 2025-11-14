import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Простой residual-блок без BatchNorm:
    Conv -> ReLU -> Conv + skip
    """
    def __init__(self, n_feats: int = 32):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1),
        )

    def forward(self, x):
        res = self.body(x)
        return x + res


class BaselineEDSRSmall(nn.Module):
    """
    Лёгкая EDSR-подобная модель для baseline SR.
    scale=2 => апскейл в 2 раза по ширине/высоте.
    """
    def __init__(self, scale: int = 2, n_feats: int = 32, n_blocks: int = 4):
        super().__init__()

        self.scale = scale

        # 1) Shallow feature extraction
        self.head = nn.Sequential(
            nn.Conv2d(3, n_feats, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # 2) Residual body
        body_blocks = [ResidualBlock(n_feats) for _ in range(n_blocks)]
        self.body = nn.Sequential(*body_blocks)

        # 3) Upsampling (PixelShuffle)
        self.upsample = nn.Sequential(
            nn.Conv2d(
                n_feats,
                n_feats * (scale ** 2),  # расширяем каналы для PixelShuffle
                kernel_size=3,
                padding=1,
            ),
            nn.PixelShuffle(scale),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
        )

    def forward(self, x):
        # x: [B, 3, H, W]
        x = self.head(x)
        x = self.body(x)
        x = self.upsample(x)
        # output: [B, 3, H*scale, W*scale]
        return x


# Быстрый тест, если запустить model.py напрямую
if __name__ == "__main__":
    model = BaselineEDSRSmall(scale=2, n_feats=32, n_blocks=4)
    x = torch.randn(1, 3, 64, 64)   # 64x64 LR
    y = model(x)
    print("Input shape :", x.shape)
    print("Output shape:", y.shape)   # должно быть [1, 3, 128, 128]
