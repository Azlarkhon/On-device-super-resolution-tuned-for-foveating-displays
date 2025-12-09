# fovea_inference_lr_only.py

import os
from typing import Tuple

from PIL import Image, ImageDraw, ImageFont
import torch
import torchvision.transforms.functional as TF
from torch.amp import autocast
import numpy as np

from model import RCAN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SCALE = 2          # RCAN upscale factor
PATCH_LR = 256     # LR patch size fed to RCAN
PATCH_HR = PATCH_LR * SCALE  # 256

# Fovea point in normalized coords [0, 1]
FOVEA_X = 0.4
FOVEA_Y = 0.56

# "circle" or "square"
PATCH_SHAPE = "square"


def load_model(weights_path: str = "rcan_full_3050ti_final.pth") -> RCAN:
    model = RCAN(scale=SCALE).to(DEVICE)
    state = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


def crop_lr_at_fovea(
    lr_img: Image.Image,
    fovea_x: float,
    fovea_y: float,
) -> Tuple[Image.Image, Tuple[int, int]]:
    w_lr, h_lr = lr_img.size
    if w_lr < PATCH_LR or h_lr < PATCH_LR:
        raise ValueError(f"LR image too small ({w_lr}x{h_lr}) for PATCH_LR={PATCH_LR}")

    cx_lr = int(round(fovea_x * (w_lr - 1)))
    cy_lr = int(round(fovea_y * (h_lr - 1)))

    half_lr = PATCH_LR // 2
    x_lr = cx_lr - half_lr
    y_lr = cy_lr - half_lr

    x_lr = max(0, min(x_lr, w_lr - PATCH_LR))
    y_lr = max(0, min(y_lr, h_lr - PATCH_LR))

    lr_crop = lr_img.crop((x_lr, y_lr, x_lr + PATCH_LR, y_lr + PATCH_LR))
    return lr_crop, (x_lr, y_lr)


def make_circular_mask(size: Tuple[int, int], feather: float = 0.3) -> Image.Image:
    w, h = size
    yy, xx = np.ogrid[:h, :w]
    cy, cx = h / 2.0, w / 2.0
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    max_r = min(w, h) / 2.0

    inner = max_r * (1.0 - feather)
    mask = np.zeros_like(r, dtype=np.float32)
    inside_inner = r <= inner
    between = (r > inner) & (r < max_r)

    mask[inside_inner] = 1.0
    mask[between] = (max_r - r[between]) / (max_r - inner)

    return Image.fromarray((mask * 255).astype("uint8"), mode="L")


def make_square_mask(size: Tuple[int, int]) -> Image.Image:
    return Image.new("L", size, 255)


def draw_border(
    img: Image.Image,
    x: int,
    y: int,
    w: int,
    h: int,
    shape: str = "square",
    color=(255, 0, 0),
    thickness: int = 4,
) -> None:
    draw = ImageDraw.Draw(img)
    x0, y0, x1, y1 = x, y, x + w, y + h
    if shape.lower() == "circle":
        for t in range(thickness):
            draw.ellipse((x0 + t, y0 + t, x1 - t, y1 - t), outline=color)
    else:
        for t in range(thickness):
            draw.rectangle((x0 + t, y0 + t, x1 - t, y1 - t), outline=color)


def label_tile(img: Image.Image, text: str) -> Image.Image:
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 28)
    except:  # fallback if arial is missing
        font = ImageFont.load_default()

    draw.rectangle((0, 0, 120, 40), fill=(0, 0, 0, 180))
    draw.text((5, 5), text, fill=(255, 255, 255), font=font)
    return img


def run_one_example_at_fovea(
    model: RCAN,
    lr_path: str,
    fovea_x: float,
    fovea_y: float,
    out_dir: str = "examples_rcan_3050ti_fovea",
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # Full LR image
    lr_img_full = Image.open(lr_path).convert("RGB")

    # 1) LR fovea patch
    lr_crop, (x_lr, y_lr) = crop_lr_at_fovea(lr_img_full, fovea_x, fovea_y)

    # 2) RCAN inference (LR -> SR)
    lr_tensor = TF.to_tensor(lr_crop).unsqueeze(0).to(DEVICE)
    use_cuda = DEVICE == "cuda"
    with torch.no_grad():
        with autocast(device_type="cuda", enabled=use_cuda):
            sr_tensor = model(lr_tensor)[0].cpu().clamp(0, 1)

    sr_img = TF.to_pil_image(sr_tensor)  # PATCH_HR x PATCH_HR

    # 3) Full LR-upscaled image (what you would show in VR)
    w_lr, h_lr = lr_img_full.size
    disp_w, disp_h = w_lr * SCALE, h_lr * SCALE
    lr_up_full = lr_img_full.resize((disp_w, disp_h), Image.BICUBIC)

    # 4) Composite SR patch into the upscaled LR image
    x_disp = x_lr * SCALE
    y_disp = y_lr * SCALE

    sr_foveated_full = lr_up_full.copy()

    if PATCH_SHAPE.lower() == "circle":
        mask = make_circular_mask((PATCH_HR, PATCH_HR), feather=0.3)
    else:
        mask = make_square_mask((PATCH_HR, PATCH_HR))

    sr_foveated_full.paste(sr_img, (x_disp, y_disp), mask)
    draw_border(sr_foveated_full, x_disp, y_disp, PATCH_HR, PATCH_HR, PATCH_SHAPE)

    # 5) Comparison strip: LR-upscaled patch vs SR patch (both 256x256), with labels
    lr_up_patch = lr_crop.resize((PATCH_HR, PATCH_HR), Image.BICUBIC)

    tiles = [
        label_tile(lr_up_patch.copy(), "LR"),
        label_tile(sr_img.copy(), "SR"),
    ]

    comp_w = PATCH_HR * len(tiles)
    comp_h = PATCH_HR
    comparison = Image.new("RGB", (comp_w, comp_h))
    for i, t in enumerate(tiles):
        comparison.paste(t, (i * PATCH_HR, 0))

    # 6) Save outputs
    name = os.path.basename(lr_path)
    base = os.path.splitext(name)[0]
    tag = f"fx{fovea_x:.2f}_fy{fovea_y:.2f}_{PATCH_SHAPE}"

    lr_patch_save = os.path.join(out_dir, f"LR_patch_{base}_{tag}.png")
    sr_patch_save = os.path.join(out_dir, f"SR_patch_{base}_{tag}.png")
    lr_full_save = os.path.join(out_dir, f"LR_full_{base}_{tag}.png")
    sr_full_save = os.path.join(out_dir, f"SR_foveated_full_{base}_{tag}.png")
    comp_save = os.path.join(out_dir, f"comparison_{base}_{tag}.png")

    lr_crop.save(lr_patch_save)
    sr_img.save(sr_patch_save)
    lr_up_full.save(lr_full_save)
    sr_foveated_full.save(sr_full_save)
    comparison.save(comp_save)

    print("Saved:")
    print("  LR patch:         ", lr_patch_save)
    print("  SR patch:         ", sr_patch_save)
    print("  LR full upscaled: ", lr_full_save)
    print("  SR foveated full: ", sr_full_save)
    print("  Comparison:       ", comp_save)


if __name__ == "__main__":
    lr_dir = "data/val/LR"

    names = sorted(
        [n for n in os.listdir(lr_dir) if n.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]
    )
    if not names:
        raise RuntimeError("No images in LR")

    index = 1  # choose your image index
    img_name = names[index]
    lr_path = os.path.join(lr_dir, img_name)

    print("Using LR image:", img_name)
    print(f"Fovea (normalized): x={FOVEA_X}, y={FOVEA_Y}")
    print(f"Patch shape: {PATCH_SHAPE}")

    model = load_model()
    run_one_example_at_fovea(
        model,
        lr_path,
        fovea_x=FOVEA_X,
        fovea_y=FOVEA_Y,
    )
