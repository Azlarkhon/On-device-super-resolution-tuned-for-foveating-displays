import os
from PIL import Image

def make_lr(hr_dir, lr_dir, scale=2):
    os.makedirs(lr_dir, exist_ok=True)
    for name in os.listdir(hr_dir):
        if not name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        hr_path = os.path.join(hr_dir, name)
        img = Image.open(hr_path).convert("RGB")

        w, h = img.size
        img_lr = img.resize(
            (w // scale, h // scale),
            Image.Resampling.BICUBIC     # ← Правильный вариант
        )

        img_lr.save(os.path.join(lr_dir, name))

if __name__ == "__main__":
    make_lr("data/train/HR", "data/train/LR", scale=2)
    make_lr("data/val/HR",   "data/val/LR",   scale=2)

    print("LR images successfully created!")
