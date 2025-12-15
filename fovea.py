# fovea.py

import os
from typing import Tuple, Optional

from PIL import Image, ImageDraw, ImageFont, ImageFilter
import torch
import torchvision.transforms.functional as TF
from torch.amp import autocast
import numpy as np
from scipy.ndimage import gaussian_filter

from model import RCAN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SCALE = 2  # RCAN upscale factor
PATCH_LR = 128  # LR patch size fed to RCAN
PATCH_HR = PATCH_LR * SCALE  # 256

# Fovea point in normalized coords
FOVEA_X = 0.4
FOVEA_Y = 0.56

# Blur gradient parameters
MAX_BLUR_RADIUS = 15  # Reduced for smoother gradient
BLUR_GRADIENT_SMOOTHNESS = 2.0  # Higher = smoother transition
GRADIENT_EXTENSION = 2.0  # How far beyond patch the gradient extends (multiple of patch size)


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


def create_smooth_blur_gradient_mask(
        size: Tuple[int, int],
        patch_center: Tuple[float, float],
        patch_size: Tuple[int, int],
        gradient_smoothness: float = 2.0,
        gradient_extension: float = 3.0
) -> np.ndarray:
    """
    Create a smooth blur gradient mask using distance-based sigmoid function.
    Returns mask where 1.0 = no blur, 0.0 = max blur.
    """
    w, h = size
    y, x = np.ogrid[:h, :w]

    # Patch center coordinates
    cx, cy = patch_center

    # Calculate distances from patch center
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    # Patch radius (using half diagonal for circular gradient)
    patch_radius = np.sqrt((patch_size[0] / 2) ** 2 + (patch_size[1] / 2) ** 2)

    # Extended radius where full blur starts
    extended_radius = patch_radius * gradient_extension

    # Create continuous gradient based on distance
    # Inside patch: no blur (mask = 1.0)
    # Between patch_radius and extended_radius: smooth transition from 1.0 to 0.0
    # Beyond extended_radius: full blur (mask = 0.0)

    # Initialize mask with 1.0 (no blur) inside patch
    mask = np.ones((h, w), dtype=np.float32)

    # Find points outside the patch
    outside_patch = dist > patch_radius

    if outside_patch.any():
        # For points outside patch, calculate normalized distance [0, 1]
        # where 0 = at patch edge, 1 = at extended radius
        normalized_dist = np.zeros_like(dist)
        normalized_dist[outside_patch] = (dist[outside_patch] - patch_radius) / (extended_radius - patch_radius)

        # Clip to [0, 1] range
        normalized_dist = np.clip(normalized_dist, 0, 1)

        # Use sigmoid function for smooth transition
        # We adjust the sigmoid to give values from 1.0 to 0.0
        sigmoid = 1.0 / (1.0 + np.exp(gradient_smoothness * (normalized_dist - 0.5) * 10))

        # Apply the gradient to mask
        mask[outside_patch] = sigmoid[outside_patch]

    return mask


def apply_smooth_progressive_blur_optimized(
        image: Image.Image,
        patch_center: Tuple[float, float],
        patch_size: Tuple[int, int],
        sr_patch: Optional[Image.Image] = None,
        patch_coords: Optional[Tuple[int, int, int, int]] = None
) -> Image.Image:
    """
    Apply smooth progressive blur using continuous blur interpolation (optimized).
    """
    result = image.copy()
    w, h = image.size

    # First composite the sharp SR patch
    if sr_patch and patch_coords:
        x_disp, y_disp, pw, ph = patch_coords
        # Create soft-edged mask for patch blending
        yy, xx = np.ogrid[:ph, :pw]
        cy, cx = ph // 2, pw // 2
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        max_r = min(pw, ph) / 2.0
        feather_mask = np.clip(1.0 - r / max_r, 0, 1)
        feather_mask = (feather_mask * 255).astype(np.uint8)

        feather_mask_img = Image.fromarray(feather_mask, mode="L")
        result.paste(sr_patch, (x_disp, y_disp), feather_mask_img)

    # Create smooth blur gradient mask
    blur_mask = create_smooth_blur_gradient_mask(
        (w, h),
        patch_center,
        patch_size,
        gradient_smoothness=BLUR_GRADIENT_SMOOTHNESS,
        gradient_extension=GRADIENT_EXTENSION
    )

    # Convert image to numpy array
    result_array = np.array(result, dtype=np.float32)

    # Create blurred versions at different radii
    blur_levels = 5  # Number of blur levels to precompute
    blurred_arrays = []

    for i in range(blur_levels):
        # Safer radius calculation to avoid division by zero
        if blur_levels > 1:
            radius = (i / (blur_levels - 1)) * MAX_BLUR_RADIUS
        else:
            radius = 0

        if radius == 0:
            blurred_arrays.append(result_array)
        else:
            blurred = np.zeros_like(result_array)
            for channel in range(3):
                blurred[:, :, channel] = gaussian_filter(
                    result_array[:, :, channel],
                    sigma=radius
                )
            blurred_arrays.append(blurred)

    # If only one blur level, just return the image
    if blur_levels == 1:
        return Image.fromarray(result_array.astype(np.uint8))

    # Interpolate between blur levels based on mask value
    final_array = np.zeros_like(result_array)

    # Map mask value (0 to 1) to blur index (0 to blur_levels-1)
    blur_indices = (1.0 - blur_mask) * (blur_levels - 1)

    # For each pixel, interpolate between the two nearest blur levels
    for i in range(blur_levels - 1):
        # Find pixels that should use blur level i and i+1
        mask_i = (blur_indices >= i) & (blur_indices <= i + 1)

        if np.any(mask_i):
            # Calculate interpolation weight
            weight = blur_indices[mask_i] - i
            # Handle 1D vs 2D weight arrays
            if weight.ndim == 1:
                weight_3d = weight[:, np.newaxis]
            else:
                weight_3d = weight[:, :, np.newaxis]

            # Linear interpolation
            interpolated = (1 - weight_3d) * blurred_arrays[i][mask_i] + weight_3d * blurred_arrays[i + 1][mask_i]
            final_array[mask_i] = interpolated

    return Image.fromarray(np.clip(final_array, 0, 255).astype(np.uint8))

# Add this function definition BEFORE create_foveated_composite
def apply_smooth_progressive_blur_protected(
        image: Image.Image,
        patch_center: Tuple[float, float],
        patch_size: Tuple[int, int],
        protected_mask: Optional[np.ndarray] = None
) -> Image.Image:
    """
    Apply smooth progressive blur while protecting certain areas.
    """
    result = image.copy()
    w, h = image.size

    # Create smooth blur gradient mask
    blur_mask = create_smooth_blur_gradient_mask(
        (w, h),
        patch_center,
        patch_size,
        gradient_smoothness=BLUR_GRADIENT_SMOOTHNESS,
        gradient_extension=GRADIENT_EXTENSION
    )

    # If we have a protected mask (e.g., SR patch), adjust the blur mask
    if protected_mask is not None:
        # Protected areas should have blur_mask = 1.0 (no blur)
        # Combine with existing blur_mask, taking maximum value
        combined_mask = np.maximum(blur_mask, protected_mask)
        blur_mask = combined_mask

    # Convert image to numpy array
    result_array = np.array(result, dtype=np.float32)

    # Create blurred versions at different radii
    blur_levels = 5
    blurred_arrays = []

    for i in range(blur_levels):
        if blur_levels > 1:
            radius = (i / (blur_levels - 1)) * MAX_BLUR_RADIUS
        else:
            radius = 0

        if radius == 0:
            blurred_arrays.append(result_array)
        else:
            blurred = np.zeros_like(result_array)
            for channel in range(3):
                blurred[:, :, channel] = gaussian_filter(
                    result_array[:, :, channel],
                    sigma=radius
                )
            blurred_arrays.append(blurred)

    if blur_levels == 1:
        return image

    # Interpolate between blur levels based on combined mask
    final_array = np.zeros_like(result_array)
    blur_indices = (1.0 - blur_mask) * (blur_levels - 1)

    for i in range(blur_levels - 1):
        mask_i = (blur_indices >= i) & (blur_indices <= i + 1)

        if np.any(mask_i):
            weight = blur_indices[mask_i] - i
            if weight.ndim == 1:
                weight_3d = weight[:, np.newaxis]
            else:
                weight_3d = weight[:, :, np.newaxis]

            interpolated = (1 - weight_3d) * blurred_arrays[i][mask_i] + weight_3d * blurred_arrays[i + 1][mask_i]
            final_array[mask_i] = interpolated

    return Image.fromarray(np.clip(final_array, 0, 255).astype(np.uint8))


def create_foveated_composite_simple(
        lr_up_full: Image.Image,
        sr_patch: Image.Image,
        x_disp: int,
        y_disp: int,
        fovea_x: float,
        fovea_y: float
) -> Image.Image:
    """
    Simple working solution:
    1. Blur the entire LR image
    2. Paste sharp SR patch on top
    """
    # Start with upscaled LR image
    composite = lr_up_full.copy()

    # Calculate patch center
    disp_w, disp_h = composite.size
    patch_cx = x_disp + PATCH_HR // 2
    patch_cy = y_disp + PATCH_HR // 2

    # 1. Apply blur to the ENTIRE LR image
    composite_blurred = apply_smooth_progressive_blur_optimized(
        composite,
        (patch_cx, patch_cy),
        (PATCH_HR, PATCH_HR),
        sr_patch=None,  # Don't paste SR patch during blurring
        patch_coords=None
    )

    # 2. Create feather mask for smooth blending
    yy, xx = np.ogrid[:PATCH_HR, :PATCH_HR]
    center_y, center_x = PATCH_HR // 2, PATCH_HR // 2
    dist = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
    max_dist = PATCH_HR / 2.0

    # Feather mask: 1.0 at center, 0.0 at edges
    feather_mask = np.clip(1.0 - dist / max_dist, 0, 1)
    feather_mask = (feather_mask * 255).astype(np.uint8)

    # 3. Convert to PIL mask
    feather_mask_img = Image.fromarray(feather_mask, mode="L")

    # 4. Paste SR patch on top of blurred image
    composite_blurred.paste(sr_patch, (x_disp, y_disp), feather_mask_img)

    return composite_blurred



def label_tile(img: Image.Image, text: str) -> Image.Image:
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 28)
    except:
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

    # 3) Full LR-upscaled image
    w_lr, h_lr = lr_img_full.size
    disp_w, disp_h = w_lr * SCALE, h_lr * SCALE
    lr_up_full = lr_img_full.resize((disp_w, disp_h), Image.BICUBIC)

    # 4) Create foveated composite with smooth progressive blur
    x_disp = x_lr * SCALE
    y_disp = y_lr * SCALE

    sr_foveated_full = create_foveated_composite_simple(  # Changed to simple version
        lr_up_full,
        sr_img,
        x_disp,
        y_disp,
        fovea_x,
        fovea_y
    )

    # 5) Create visualization of blur gradient
    patch_cx = x_disp + PATCH_HR // 2
    patch_cy = y_disp + PATCH_HR // 2

    blur_mask = create_smooth_blur_gradient_mask(
        (disp_w, disp_h),
        (patch_cx, patch_cy),
        (PATCH_HR, PATCH_HR),
        gradient_smoothness=BLUR_GRADIENT_SMOOTHNESS,
        gradient_extension=GRADIENT_EXTENSION
    )

    blur_mask_viz = (blur_mask * 255).astype(np.uint8)
    blur_mask_img = Image.fromarray(blur_mask_viz, mode="L")
    blur_mask_img = blur_mask_img.convert("RGB")

    # Mark patch location on visualization
    draw = ImageDraw.Draw(blur_mask_img)

    # Draw patch boundary
    draw.rectangle((x_disp, y_disp, x_disp + PATCH_HR, y_disp + PATCH_HR),
                   outline=(0, 255, 0), width=2)

    # Draw patch center
    draw.ellipse((patch_cx - 5, patch_cy - 5, patch_cx + 5, patch_cy + 5),
                 fill=(0, 255, 0), outline=(255, 255, 255))

    # Draw gradient extent
    patch_radius = np.sqrt((PATCH_HR / 2) ** 2 + (PATCH_HR / 2) ** 2)
    gradient_radius = patch_radius * GRADIENT_EXTENSION
    draw.ellipse((patch_cx - gradient_radius, patch_cy - gradient_radius,
                  patch_cx + gradient_radius, patch_cy + gradient_radius),
                 outline=(255, 0, 0), width=2)

    # 6) Comparison strip: LR-upscaled patch vs SR patch
    lr_up_patch = lr_crop.resize((PATCH_HR, PATCH_HR), Image.BICUBIC)

    tiles = [
        label_tile(lr_up_patch.copy(), "LR Patch"),
        label_tile(sr_img.copy(), "SR Patch"),
    ]

    comp_w = PATCH_HR * len(tiles)
    comp_h = PATCH_HR
    comparison = Image.new("RGB", (comp_w, comp_h))
    for i, t in enumerate(tiles):
        comparison.paste(t, (i * PATCH_HR, 0))

    # 7) Save outputs
    name = os.path.basename(lr_path)
    base = os.path.splitext(name)[0]
    tag = f"fx{fovea_x:.2f}_fy{fovea_y:.2f}"

    lr_patch_save = os.path.join(out_dir, f"LR_patch_{base}_{tag}.png")
    sr_patch_save = os.path.join(out_dir, f"SR_patch_{base}_{tag}.png")
    lr_full_save = os.path.join(out_dir, f"LR_full_{base}_{tag}.png")
    sr_full_save = os.path.join(out_dir, f"SR_foveated_full_{base}_{tag}.png")
    blur_viz_save = os.path.join(out_dir, f"blur_gradient_{base}_{tag}.png")
    comp_save = os.path.join(out_dir, f"comparison_{base}_{tag}.png")

    lr_crop.save(lr_patch_save)
    sr_img.save(sr_patch_save)
    lr_up_full.save(lr_full_save)
    sr_foveated_full.save(sr_full_save)
    blur_mask_img.save(blur_viz_save)
    comparison.save(comp_save)

    print("Saved:")
    print("  LR patch:         ", lr_patch_save)
    print("  SR patch:         ", sr_patch_save)
    print("  LR full upscaled: ", lr_full_save)
    print("  SR foveated full: ", sr_full_save)
    print("  Blur gradient viz:", blur_viz_save)
    print("  Comparison:       ", comp_save)

    # Show blur parameters
    print(f"\nBlur parameters:")
    print(f"  Max blur radius: {MAX_BLUR_RADIUS}px")
    print(f"  Gradient smoothness: {BLUR_GRADIENT_SMOOTHNESS}")
    print(f"  Gradient extension: {GRADIENT_EXTENSION}x patch size")
    print(f"  Blur levels: 5")  # Changed from "Blur samples: 20"


if __name__ == "__main__":
    lr_dir = "data/val/LR"

    names = sorted(
        [n for n in os.listdir(lr_dir) if n.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]
    )
    if not names:
        raise RuntimeError("No images in LR")

    index = 6  # choose your image index
    img_name = names[index]
    lr_path = os.path.join(lr_dir, img_name)

    print("Using LR image:", img_name)
    print(f"Fovea (normalized): x={FOVEA_X}, y={FOVEA_Y}")

    model = load_model()
    run_one_example_at_fovea(
        model,
        lr_path,
        fovea_x=FOVEA_X,
        fovea_y=FOVEA_Y,
    )