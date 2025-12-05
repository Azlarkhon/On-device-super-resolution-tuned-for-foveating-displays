import os
import random
from PIL import Image

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

# NEW CORRECT IMPORTS for PyTorch 2.4+
from torch.amp import autocast, GradScaler   # <-- this is the correct one now

from model import BaselineEDSRSmall


# ----------------- CONFIG (RTX 3050 Ti 4GB) ----------------- #
SCALE = 2
PATCH_LR = 64
PATCH_HR = PATCH_LR * SCALE
PATCHES_PER_IMAGE = 20

EPOCHS = 100
BATCH_SIZE = 48          # perfect for 4GB VRAM
LR_INIT = 2e-4


# ----------------- DATASET ----------------- #
class SRDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, scale=2, patch_lr=64, patches_per_image=20, is_train=True):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.scale = scale
        self.patch_lr = patch_lr
        self.patch_hr = patch_lr * scale
        self.patches_per_image = patches_per_image
        self.is_train = is_train

        self.names = sorted([n for n in os.listdir(hr_dir)
                            if n.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))])
        if not self.names:
            raise RuntimeError(f"No images in {hr_dir}")

    def __len__(self):
        return len(self.names) * self.patches_per_image if self.is_train else len(self.names)

    def _load_pair(self, idx):
        name = self.names[idx % len(self.names)]
        hr = Image.open(os.path.join(self.hr_dir, name)).convert("RGB")
        lr = Image.open(os.path.join(self.lr_dir, name)).convert("RGB")
        return hr, lr

    def __getitem__(self, idx):
        hr, lr = self._load_pair(idx)
        w, h = lr.size
        if w < self.patch_lr or h < self.patch_lr:
            lr = lr.resize((self.patch_lr*2, self.patch_lr*2), Image.Resampling.BICUBIC)
            hr = hr.resize((self.patch_hr*2, self.patch_hr*2), Image.Resampling.BICUBIC)
            w, h = lr.size

        i = random.randint(0, w - self.patch_lr) if self.is_train else (w - self.patch_lr)//2
        j = random.randint(0, h - self.patch_lr) if self.is_train else (h - self.patch_lr)//2

        lr_patch = lr.crop((i, j, i + self.patch_lr, j + self.patch_lr))
        hr_patch = hr.crop((i*self.scale, j*self.scale,
                            i*self.scale + self.patch_hr, j*self.scale + self.patch_hr))

        lr_t = TF.to_tensor(lr_patch)
        hr_t = TF.to_tensor(hr_patch)

        if self.is_train:
            if random.random() < 0.5: lr_t, hr_t = TF.hflip(lr_t), TF.hflip(hr_t)
            if random.random() < 0.5: lr_t, hr_t = TF.vflip(lr_t), TF.vflip(hr_t)
            if random.random() < 0.5:
                k = random.randint(0, 3)
                lr_t = torch.rot90(lr_t, k, [1, 2])
                hr_t = torch.rot90(hr_t, k, [1, 2])

        return lr_t, hr_t


# ----------------- TRAINING ----------------- #
def train_baseline(train_dataset, val_dataset, epochs=100, batch_size=48, lr_init=2e-4, device="cuda"):
    torch.backends.cudnn.benchmark = True

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=6, pin_memory=True, persistent_workers=True, drop_last=True)
    val_loader   = DataLoader(val_dataset, batch_size=1, shuffle=False,
                              num_workers=4, pin_memory=True)

    model = BaselineEDSRSmall(scale=SCALE).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=lr_init, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler()                     # <-- correct new syntax

    print(f"Training started → {device.upper()} | Batch: {batch_size} | 3050 Ti 4GB tuned")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for lr_img, hr_img in train_loader:
            lr_img = lr_img.to(device, non_blocking=True)
            hr_img = hr_img.to(device, non_blocking=True)

            optimizer.zero_grad()
            with autocast('cuda'):                         # <-- this is still correct
                sr = model(lr_img).clamp(0, 1)
                loss = criterion(sr, hr_img)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * lr_img.size(0)

        # Validation
        model.eval()
        val_psnr = 0.0
        with torch.no_grad():
            for lr_img, hr_img in val_loader:
                lr_img = lr_img.to(device, non_blocking=True)
                hr_img = hr_img.to(device, non_blocking=True)
                with autocast('cuda'):
                    sr = model(lr_img).clamp(0, 1)
                psnr = -10 * torch.log10(torch.mean((sr - hr_img) ** 2) + 1e-10)
                val_psnr += psnr.item()
        val_psnr /= len(val_loader)

        print(f"Epoch {epoch:03d} | Loss: {total_loss/len(train_loader.dataset):.5f} | "
              f"Val PSNR: {val_psnr:.2f} dB | LR: {optimizer.param_groups[0]['lr']:.1e}")
        scheduler.step()

    return model


# ----------------- MAIN ----------------- #
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)} | Total VRAM: {torch.cuda.get_device_properties(0).total_memory//1024**3} GB")

    train_dataset = SRDataset("data/train/HR", "data/train/LR", is_train=True)
    val_dataset   = SRDataset("data/val/HR",   "data/val/LR",   is_train=False)

    model = train_baseline(train_dataset, val_dataset,
                           epochs=EPOCHS, batch_size=BATCH_SIZE,
                           lr_init=LR_INIT, device=device)

    torch.save(model.state_dict(), "baseline_sr_3050ti_final.pth")
    print("Training finished — model saved as baseline_sr_3050ti_final.pth")