import os
from PIL import Image

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

from model import BaselineEDSRSmall


SCALE = 2          # апскейл в модели
PATCH_LR = 64      # размер патча в LR (64x64)
PATCH_HR = PATCH_LR * SCALE  # 128x128 в HR


class SRDataset(Dataset):
    """
    Датасет: читает HR/LR по имени, режет случайный патч фиксированного размера.
    Все выходные тензоры одного размера => DataLoader работает без ошибок.
    """
    def __init__(self, hr_dir: str, lr_dir: str, scale: int = 2, patch_lr: int = 64, is_train: bool = True):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.scale = scale
        self.patch_lr = patch_lr
        self.patch_hr = patch_lr * scale
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
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]

        hr_path = os.path.join(self.hr_dir, name)
        lr_path = os.path.join(self.lr_dir, name)

        if not os.path.exists(lr_path):
            raise RuntimeError(f"LR file not found for {name}: {lr_path}")

        hr = Image.open(hr_path).convert("RGB")
        lr = Image.open(lr_path).convert("RGB")

        # гарантируем, что LR >= patch_lr
        w_lr, h_lr = lr.size
        if w_lr < self.patch_lr or h_lr < self.patch_lr:
            # если картинка слишком маленькая — ресайзим до нужного размера
            lr = lr.resize((self.patch_lr, self.patch_lr), Image.Resampling.BICUBIC)
            hr = hr.resize((self.patch_hr, self.patch_hr), Image.Resampling.BICUBIC)
            w_lr, h_lr = lr.size

        # train: случайный кроп, val: центрированный кроп
        if self.is_train:
            # случайная позиция в LR
            x_lr = torch.randint(0, w_lr - self.patch_lr + 1, (1,)).item()
            y_lr = torch.randint(0, h_lr - self.patch_lr + 1, (1,)).item()
        else:
            # центр
            x_lr = max(0, (w_lr - self.patch_lr) // 2)
            y_lr = max(0, (h_lr - self.patch_lr) // 2)

        # вырезаем LR-патч
        lr_patch = lr.crop((x_lr, y_lr, x_lr + self.patch_lr, y_lr + self.patch_lr))

        # соответствующий HR-патч (координаты масштабируем)
        x_hr = x_lr * self.scale
        y_hr = y_lr * self.scale
        w_hr, h_hr = hr.size

        # на всякий случай гарантируем границы
        if x_hr + self.patch_hr > w_hr or y_hr + self.patch_hr > h_hr:
            # если геометрия поехала — ресайзим HR в нужный размер
            hr = hr.resize((self.patch_hr + 2, self.patch_hr + 2), Image.Resampling.BICUBIC)
            w_hr, h_hr = hr.size
            x_hr = 0
            y_hr = 0

        hr_patch = hr.crop((x_hr, y_hr, x_hr + self.patch_hr, y_hr + self.patch_hr))

        # в тензоры [0,1]
        lr_tensor = TF.to_tensor(lr_patch)
        hr_tensor = TF.to_tensor(hr_patch)

        return lr_tensor, hr_tensor


def train_baseline(
    train_dataset: Dataset,
    val_dataset: Dataset,
    epochs: int = 50,
    batch_size: int = 4,
    lr: float = 1e-4,
    device: str = "cuda",
):
    # num_workers=0 — безопасно для Windows
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if device == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device == "cuda" else False,
    )

    model = BaselineEDSRSmall(scale=SCALE).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        # ---------- TRAIN ----------
        model.train()
        total_loss = 0.0

        for lr_img, hr_img in train_loader:
            lr_img = lr_img.to(device)
            hr_img = hr_img.to(device)

            optimizer.zero_grad()
            sr_img = model(lr_img)
            loss = criterion(sr_img, hr_img)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * lr_img.size(0)

        train_loss = total_loss / len(train_loader.dataset)

        # ---------- VAL ----------
        model.eval()
        val_psnr_sum = 0.0

        with torch.no_grad():
            for lr_img, hr_img in val_loader:
                lr_img = lr_img.to(device)
                hr_img = hr_img.to(device)

                sr_img = model(lr_img)

                mse = nn.functional.mse_loss(sr_img, hr_img)
                mse = torch.clamp(mse, min=1e-10)
                psnr = -10.0 * torch.log10(mse)
                val_psnr_sum += psnr.item()

        val_psnr = val_psnr_sum / len(val_loader)

        print(
            f"Epoch {epoch}: "
            f"Train L1 = {train_loss:.4f}, "
            f"Val PSNR = {val_psnr:.2f} dB"
        )

    return model


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_hr_dir = "data/train/HR"
    train_lr_dir = "data/train/LR"
    val_hr_dir = "data/val/HR"
    val_lr_dir = "data/val/LR"

    train_dataset = SRDataset(train_hr_dir, train_lr_dir, scale=SCALE, patch_lr=PATCH_LR, is_train=True)
    val_dataset   = SRDataset(val_hr_dir,   val_lr_dir,   scale=SCALE, patch_lr=PATCH_LR, is_train=False)

    model = train_baseline(
        train_dataset,
        val_dataset,
        epochs=5,
        batch_size=4,
        lr=1e-4,
        device=device,
    )

    torch.save(model.state_dict(), "baseline_sr.pth")
    print("Saved baseline_sr.pth")
