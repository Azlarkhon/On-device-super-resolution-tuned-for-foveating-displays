import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, n_feats=32):
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
    def __init__(self, scale=2, n_feats=32, n_blocks=4):
        super().__init__()
        # shallow feature
        self.head = nn.Sequential(
            nn.Conv2d(3, n_feats, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # residual body
        body = [ResidualBlock(n_feats) for _ in range(n_blocks)]
        self.body = nn.Sequential(*body)

        # upsampling
        self.upsample = nn.Sequential(
            nn.Conv2d(n_feats, n_feats * (scale ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(n_feats, 3, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.upsample(x)
        return x
