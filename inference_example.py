import os
from PIL import Image
import torch
import torchvision.transforms.functional as TF

from model import BaselineEDSRSmall

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SCALE = 2       # тот же scale, что в модели / train.py
PATCH_LR = 64   # размер патча LR (как в train.py)
PATCH_HR = PATCH_LR * SCALE


def load_model(weights_path="baseline_sr.pth"):
    model = BaselineEDSRSmall(scale=SCALE).to(DEVICE)
    state = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


def center_crop_pair(lr_img: Image.Image, hr_img: Image.Image):
    # центр-кроп LR 64x64
    w_lr, h_lr = lr_img.size
    x_lr = max(0, (w_lr - PATCH_LR) // 2)
    y_lr = max(0, (h_lr - PATCH_LR) // 2)
    lr_crop = lr_img.crop((x_lr, y_lr, x_lr + PATCH_LR, y_lr + PATCH_LR))

    # соответствующий кроп HR 128x128
    w_hr, h_hr = hr_img.size
    x_hr = x_lr * SCALE
    y_hr = y_lr * SCALE

    # защита от выхода за границы
    x_hr = min(max(0, x_hr), max(0, w_hr - PATCH_HR))
    y_hr = min(max(0, y_hr), max(0, h_hr - PATCH_HR))

    hr_crop = hr_img.crop((x_hr, y_hr, x_hr + PATCH_HR, y_hr + PATCH_HR))

    return lr_crop, hr_crop


def run_one_example(lr_path, hr_path, out_dir="examples"):
    os.makedirs(out_dir, exist_ok=True)

    # читаем исходные LR/HR
    lr_img_full = Image.open(lr_path).convert("RGB")
    hr_img_full = Image.open(hr_path).convert("RGB")

    # берём центр-патчи согласованных размеров
    lr_img, hr_img = center_crop_pair(lr_img_full, hr_img_full)

    # в тензор
    lr_tensor = TF.to_tensor(lr_img).unsqueeze(0).to(DEVICE)

    model = load_model()
    with torch.no_grad():
        sr_tensor = model(lr_tensor)[0].cpu().clamp(0, 1)

    sr_img = TF.to_pil_image(sr_tensor)

    name = os.path.basename(lr_path)
    base = os.path.splitext(name)[0]

    lr_save = os.path.join(out_dir, f"LR_patch_{base}.png")
    hr_save = os.path.join(out_dir, f"HR_patch_{base}.png")
    sr_save = os.path.join(out_dir, f"SR_patch_{base}.png")

    lr_img.save(lr_save)
    hr_img.save(hr_save)
    sr_img.save(sr_save)

    print("Saved:")
    print(lr_save)
    print(hr_save)
    print(sr_save)


if __name__ == "__main__":
    lr_dir = "data/val/LR"
    hr_dir = "data/val/HR"

    names = sorted(
        [n for n in os.listdir(lr_dir) if n.lower().endswith((".png", ".jpg", ".jpeg"))]
    )
    if not names:
        raise RuntimeError("No images in data/val/LR")

    name = names[0]
    lr_path = os.path.join(lr_dir, name)
    hr_path = os.path.join(hr_dir, name)

    run_one_example(lr_path, hr_path, out_dir="examples")
