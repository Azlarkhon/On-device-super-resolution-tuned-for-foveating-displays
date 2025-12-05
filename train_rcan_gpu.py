import os
import random
from PIL import Image

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

from torch.amp import autocast, GradScaler
from model import RCAN

# ----------------- CONFIG (RCAN, RTX 3050 Ti) ----------------- #
SCALE = 2
PATCH_LR = 64
PATCH_HR = PATCH_LR * SCALE
PATCHES_PER_IMAGE = 20

EPOCHS = 25  # reduced from 200

# Effective batch kept at 48 (like baseline), but tuned for 4GB VRAM
BATCH_EFFECTIVE = 48
BATCH_MICRO = 8          # real batch per step on GPU
ACCUM_STEPS = BATCH_EFFECTIVE // BATCH_MICRO  # 6

LR_INIT = 1e-4           # slightly smaller for big RCAN

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

        self.names = sorted([
            n for n in os.listdir(hr_dir)
            if n.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
        ])
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
            lr = lr.resize((self.patch_lr * 2, self.patch_lr * 2), Image.Resampling.BICUBIC)
            hr = hr.resize((self.patch_hr * 2, self.patch_hr * 2), Image.Resampling.BICUBIC)
            w, h = lr.size

        i = random.randint(0, w - self.patch_lr) if self.is_train else (w - self.patch_lr) // 2
        j = random.randint(0, h - self.patch_lr) if self.is_train else (h - self.patch_lr) // 2

        lr_patch = lr.crop((i, j, i + self.patch_lr, j + self.patch_lr))
        hr_patch = hr.crop((
            i * self.scale,
            j * self.scale,
            i * self.scale + self.patch_hr,
            j * self.scale + self.patch_hr
        ))

        lr_t = TF.to_tensor(lr_patch)
        hr_t = TF.to_tensor(hr_patch)

        if self.is_train:
            if random.random() < 0.5:
                lr_t, hr_t = TF.hflip(lr_t), TF.hflip(hr_t)
            if random.random() < 0.5:
                lr_t, hr_t = TF.vflip(lr_t), TF.vflip(hr_t)
            if random.random() < 0.5:
                k = random.randint(0, 3)
                lr_t = torch.rot90(lr_t, k, [1, 2])
                hr_t = torch.rot90(hr_t, k, [1, 2])

        return lr_t, hr_t

# ----------------- TRAINING (RCAN + ACCUM + AMP) ----------------- #
def train_rcan(train_dataset, val_dataset, epochs=EPOCHS, batch_micro=BATCH_MICRO,
               accum_steps=ACCUM_STEPS, lr_init=LR_INIT, device="cuda"):

    use_cuda = (device.startswith("cuda") and torch.cuda.is_available())

    torch.backends.cudnn.benchmark = use_cuda

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_micro,
        shuffle=True,
        num_workers=6,
        pin_memory=use_cuda,
        persistent_workers=use_cuda,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=use_cuda,
    )

    model = RCAN(scale=SCALE).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=lr_init, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler(enabled=use_cuda)

    print(f"Training started → {device.upper()} | RCAN full | "
          f"micro-batch: {batch_micro}, accum: {accum_steps}, "
          f"effective: {batch_micro * accum_steps}")

    if use_cuda:
        print(f"GPU: {torch.cuda.get_device_name(0)} | "
              f"VRAM: {torch.cuda.get_device_properties(0).total_memory // (1024 ** 3)} GB")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()
        step_count = 0

        for i, (lr_img, hr_img) in enumerate(train_loader):
            lr_img = lr_img.to(device, non_blocking=use_cuda)
            hr_img = hr_img.to(device, non_blocking=use_cuda)

            with autocast(device_type="cuda", enabled=use_cuda):
                sr = model(lr_img).clamp(0, 1)
                raw_loss = criterion(sr, hr_img)
                loss = raw_loss / accum_steps

            if use_cuda:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            total_loss += raw_loss.item() * lr_img.size(0)
            step_count += 1

            if (i + 1) % accum_steps == 0:
                if use_cuda:
                    scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                if use_cuda:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad()

        if step_count % accum_steps != 0:
            if use_cuda:
                scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if use_cuda:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad()

        # ----------------- VALIDATION ----------------- #
        model.eval()
        val_psnr = 0.0
        with torch.no_grad():
            for lr_img, hr_img in val_loader:
                lr_img = lr_img.to(device, non_blocking=use_cuda)
                hr_img = hr_img.to(device, non_blocking=use_cuda)

                with autocast(device_type="cuda", enabled=use_cuda):
                    sr = model(lr_img).clamp(0, 1)
                    mse = torch.mean((sr - hr_img) ** 2)
                    psnr = -10 * torch.log10(mse + 1e-10)
                val_psnr += psnr.item()
        val_psnr /= len(val_loader)

        print(
            f"Epoch {epoch:03d} | "
            f"Loss: {total_loss / len(train_loader.dataset):.5f} | "
            f"Val PSNR: {val_psnr:.2f} dB | "
            f"LR: {optimizer.param_groups[0]['lr']:.1e}"
        )

        # ----------------- SAVE CHECKPOINT ----------------- #
        if epoch % 5 == 0:
            torch.save(model.state_dict(), f"rcan_epoch_{epoch}.pth")
            print(f"Checkpoint saved: rcan_epoch_{epoch}.pth")

        scheduler.step()

    # ----------------- FINAL MODEL ----------------- #
    torch.save(model.state_dict(), "rcan_full_3050ti_final.pth")
    print("Training finished — model saved as rcan_full_3050ti_final.pth")
    return model

# ----------------- MAIN ----------------- #
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_dataset = SRDataset("data/train/HR", "data/train/LR", is_train=True)
    val_dataset = SRDataset("data/val/HR", "data/val/LR", is_train=False)

    model = train_rcan(
        train_dataset,
        val_dataset,
        epochs=EPOCHS,
        batch_micro=BATCH_MICRO,
        accum_steps=ACCUM_STEPS,
        lr_init=LR_INIT,
        device=device,
    )
