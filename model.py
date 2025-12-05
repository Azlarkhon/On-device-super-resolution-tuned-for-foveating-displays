import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Simple residual block without BatchNorm:
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
    Lightweight EDSR-like model for baseline SR.
    scale=2 => 2x upscaling.
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
                n_feats * (scale ** 2),  # expand channels for PixelShuffle
                kernel_size=3,
                padding=1,
            ),
            nn.PixelShuffle(scale),
            nn.Conv2d(n_feats, 3, kernel_size=3, padding=1),
        )

        # output: [B, 3, H*scale, W*scale]

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.upsample(x)
        return x


# ----------------- RCAN FULL ----------------- #

class CALayer(nn.Module):
    """
    Channel Attention (CA) Layer.
    Global average pooling -> 1x1 conv -> ReLU -> 1x1 conv -> sigmoid -> rescale.
    """
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class RCAB(nn.Module):
    """
    Residual Channel Attention Block (RCAB).
    Conv -> ReLU -> Conv -> CA -> + skip
    """
    def __init__(self, n_feats, kernel_size=3, reduction=16, bias=True, res_scale=1.0):
        super().__init__()
        layers = [
            nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size // 2, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size // 2, bias=bias),
        ]
        self.body = nn.Sequential(*layers)
        self.ca = CALayer(n_feats, reduction)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res = self.ca(res)
        res = res * self.res_scale
        return x + res


class ResidualGroup(nn.Module):
    """
    Residual Group for RCAN: a stack of RCABs + a conv, with group-level residual.
    """
    def __init__(self, n_feats, kernel_size, reduction, n_resblocks, bias=True, res_scale=1.0):
        super().__init__()
        modules_body = [
            RCAB(n_feats, kernel_size, reduction, bias=bias, res_scale=res_scale)
            for _ in range(n_resblocks)
        ]
        modules_body.append(
            nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size // 2, bias=bias)
        )
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        return x + res


class RCAN(nn.Module):
    """
    Full RCAN-style model for image SR.


        Default configuration close to original:
        - n_resgroups = 10
        - n_resblocks = 20
        - n_feats = 64
        - reduction = 16
        """

    def __init__(
            self,
            scale: int = 2,
            n_resgroups: int = 10,
            n_resblocks: int = 20,
            n_feats: int = 64,
            reduction: int = 16,
            res_scale: float = 1.0,
            rgb_range: int = 1,
    ):
        super().__init__()

        self.scale = scale
        self.rgb_range = rgb_range

        kernel_size = 3
        bias = True

        # Head
        self.head = nn.Conv2d(3, n_feats, kernel_size, padding=kernel_size // 2, bias=bias)

        # Body
        modules_body = [
            ResidualGroup(
                n_feats=n_feats,
                kernel_size=kernel_size,
                reduction=reduction,
                n_resblocks=n_resblocks,
                bias=bias,
                res_scale=res_scale,
            )
            for _ in range(n_resgroups)
        ]
        modules_body.append(
            nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size // 2, bias=bias)
        )
        self.body = nn.Sequential(*modules_body)

        # Upsampler
        up_modules = []
        if scale in [2, 3]:
            up_modules.extend([
                nn.Conv2d(n_feats, n_feats * (scale ** 2), kernel_size, padding=kernel_size // 2),
                nn.PixelShuffle(scale),
            ])
        elif scale == 4:
            for _ in range(2):
                up_modules.extend([
                    nn.Conv2d(n_feats, n_feats * (2 ** 2), kernel_size, padding=kernel_size // 2),
                    nn.PixelShuffle(2),
                ])
        else:
            raise ValueError(f"Unsupported scale: {scale}")

        up_modules.append(nn.Conv2d(n_feats, 3, kernel_size, padding=kernel_size // 2))
        self.tail = nn.Sequential(*up_modules)

    def forward(self, x):
        # x in [0, 1]
        x = x * self.rgb_range

        f_head = self.head(x)
        res = self.body(f_head)
        res = res + f_head  # long skip

        out = self.tail(res)
        out = out / self.rgb_range
        return out


if __name__ == "__main__":
    model_edsr = BaselineEDSRSmall(scale=2, n_feats=32, n_blocks=4)
    x = torch.randn(1, 3, 64, 64)
    y = model_edsr(x)
    print("Baseline EDSR:", x.shape, "->", y.shape)

    model_rcan = RCAN(scale=2)
    y2 = model_rcan(x)
    print("RCAN:", x.shape, "->", y2.shape)