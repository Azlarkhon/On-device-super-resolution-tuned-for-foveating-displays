import os
from PIL import Image

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

from model import BaselineEDSRSmall


# ----------------- CONFIG ----------------- #
SCALE = 2               # upscale factor
PATCH_LR = 64           # LR patch size (64x64)
PATCH_HR = PATCH_LR * SCALE
PATCHES_PER_IMAGE = 10  # how many random patches per image per "epoch"

EPOCHS = 100
BATCH_SIZE = 4
LR_INIT = 1e-4
STEP_SIZE = 50          # step for StepLR
GAMMA = 0.5             # LR decay factor


# ----------------- DATASET ----------------- #
class SRDataset(Dataset):
    """
    Super-resolution dataset with:
    - aligned HR/LR pairs
    - random patch extraction
    - data augmentation
    - virtual length multiplier: multiple patches per image per epoch
    """
    def __init__(
        self,
        hr_dir: str,
        lr_dir: str,
        scale: int = 2,
        patch_lr: int = 64,
        patches_per_image: int = 10,
        is_train: bool = True,
    ):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.scale = scale
        self.patch_lr = patch_lr
        self.patch_hr = patch_lr * scale
        self.patches_per_image = patches_per_image
        self.is_train = is_train

        self.names = sorted(
            [
                n for n in os.listdir(hr_dir)
                if n.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
        )
        if len(self.names) == 0:
            raise RuntimeError(f"No images found in {hr_dir}")

    def __len__(self):
        # each image will be sampled multiple times per "epoch"
        if self.is_train:
            return len(self.names) * self.patches_per_image
        return len(self.names)

    def _load_pair(self, idx):
        name = self.names[idx]
        hr_path = os.path.join(self.hr_dir, name)
        lr_path = os.path.join(self.lr_dir, name)

        if not os.path.exists(lr_path):
            raise RuntimeError(f"LR file not found for {name}: {lr_path}")

        hr = Image.open(hr_path).convert("RGB")
        lr = Image.open(lr_path).convert("RGB")
        return hr, lr

    def __getitem__(self, idx):
        if self.is_train:
            real_idx = idx % len(self.names)
        else:
            real_idx = idx

        hr, lr = self._load_pair(real_idx)

        # guarantee LR can provide at least patch_lr
        w_lr, h_lr = lr.size
        if w_lr < self.patch_lr or h_lr < self.patch_lr:
            lr = lr.resize((self.patch_lr, self.patch_lr), Image.Resampling.BICUBIC)
            hr = hr.resize((self.patch_hr, self.patch_hr), Image.Resampling.BICUBIC)
            w_lr, h_lr = lr.size

        # crop coordinates in LR
        if self.is_train:
            x_lr = torch.randint(0, w_lr - self.patch_lr + 1, (1,)).item()
            y_lr = torch.randint(0, h_lr - self.patch_lr + 1, (1,)).item()
        else:
            x_lr = max(0, (w_lr - self.patch_lr) // 2)
            y_lr = max(0, (h_lr - self.patch_lr) // 2)

        lr_patch = lr.crop((x_lr, y_lr, x_lr + self.patch_lr, y_lr + self.patch_lr))

        # corresponding HR patch
        x_hr = x_lr * self.scale
        y_hr = y_lr * self.scale
        w_hr, h_hr = hr.size

        if x_hr + self.patch_hr > w_hr or y_hr + self.patch_hr > h_hr:
            # fallback: resize HR to exactly patch size if something is inconsistent
            hr = hr.resize((self.patch_hr, self.patch_hr), Image.Resampling.BICUBIC)
            x_hr, y_hr = 0, 0

        hr_patch = hr.crop((x_hr, y_hr, x_hr + self.patch_hr, y_hr + self.patch_hr))

        # to tensors [0, 1]
        lr_tensor = TF.to_tensor(lr_patch)
        hr_tensor = TF.to_tensor(hr_patch)

        # augmentations
        if self.is_train:
            if torch.rand(1) < 0.5:
                lr_tensor = TF.hflip(lr_tensor)
                hr_tensor = TF.hflip(hr_tensor)
            if torch.rand(1) < 0.5:
                lr_tensor = TF.vflip(lr_tensor)
                hr_tensor = TF.vflip(hr_tensor)
            if torch.rand(1) < 0.5:
                k = torch.randint(0, 4, (1,)).item()
                lr_tensor = torch.rot90(lr_tensor, k, [1, 2])
                hr_tensor = torch.rot90(hr_tensor, k, [1, 2])

        return lr_tensor, hr_tensor


# ----------------- TRAINING ----------------- #
def train_baseline(
    train_dataset: Dataset,
    val_dataset: Dataset,
    epochs: int = 50,
    batch_size: int = 4,
    lr_init: float = 1e-4,
    device: str = "cuda",
):
    # cuDNN autotuner
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device == "cuda"),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=(device == "cuda"),
        drop_last=False,
    )

    model = BaselineEDSRSmall(scale=SCALE).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr_init)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=STEP_SIZE, gamma=GAMMA
    )

    for epoch in range(1, epochs + 1):
        # ---------- TRAIN ---------- #
        model.train()
        total_loss = 0.0

        for lr_img, hr_img in train_loader:
            lr_img = lr_img.to(device, non_blocking=True)
            hr_img = hr_img.to(device, non_blocking=True)

            optimizer.zero_grad()
            sr_img = model(lr_img)

            # clamp to valid range during training to stabilize
            sr_img = torch.clamp(sr_img, 0.0, 1.0)

            loss = criterion(sr_img, hr_img)
            loss.backward()

            # optional gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            total_loss += loss.item() * lr_img.size(0)

        train_loss = total_loss / len(train_loader.dataset)

        # ---------- VAL ---------- #
        model.eval()
        val_psnr_sum = 0.0

        with torch.no_grad():
            for lr_img, hr_img in val_loader:
                lr_img = lr_img.to(device, non_blocking=True)
                hr_img = hr_img.to(device, non_blocking=True)

                sr_img = model(lr_img)
                sr_img = torch.clamp(sr_img, 0.0, 1.0)

                mse = nn.functional.mse_loss(sr_img, hr_img)
                mse = torch.clamp(mse, min=1e-10)
                psnr = -10.0 * torch.log10(mse)
                val_psnr_sum += psnr.item()

        val_psnr = val_psnr_sum / len(val_loader)

        print(
            f"Epoch {epoch:03d}: "
            f"Train L1 = {train_loss:.4f}, "
            f"Val PSNR = {val_psnr:.2f} dB, "
            f"LR = {scheduler.get_last_lr()[0]:.1e}"
        )

        scheduler.step()

    return model


# ----------------- MAIN ----------------- #
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_hr_dir = "data/train/HR"
    train_lr_dir = "data/train/LR"
    val_hr_dir = "data/val/HR"
    val_lr_dir = "data/val/LR"

    train_dataset = SRDataset(
        train_hr_dir,
        train_lr_dir,
        scale=SCALE,
        patch_lr=PATCH_LR,
        patches_per_image=PATCHES_PER_IMAGE,
        is_train=True,
    )

    val_dataset = SRDataset(
        val_hr_dir,
        val_lr_dir,
        scale=SCALE,
        patch_lr=PATCH_LR,
        patches_per_image=1,
        is_train=False,
    )

    model = train_baseline(
        train_dataset,
        val_dataset,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr_init=LR_INIT,
        device=device,
    )

    torch.save(model.state_dict(), "baseline_sr_optimized.pth")
    print("Saved baseline_sr_optimized.pth")
