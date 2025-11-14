import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def load_triplet(examples_dir="examples"):
    files = sorted([f for f in os.listdir(examples_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    lr = [f for f in files if f.startswith("LR_")][0]
    hr = [f for f in files if f.startswith("HR_")][0]
    sr = [f for f in files if f.startswith("SR_")][0]

    lr_img = Image.open(os.path.join(examples_dir, lr)).convert("RGB")
    hr_img = Image.open(os.path.join(examples_dir, hr)).convert("RGB")
    sr_img = Image.open(os.path.join(examples_dir, sr)).convert("RGB")

    return lr_img, hr_img, sr_img


def save_lr_hr_sr(lr_img, hr_img, sr_img, out_dir="report_figs"):
    os.makedirs(out_dir, exist_ok=True)

    lr_img.save(os.path.join(out_dir, "lr_sample.png"))
    hr_img.save(os.path.join(out_dir, "hr_sample.png"))
    sr_img.save(os.path.join(out_dir, "sr_sample.png"))


def save_gaze_foveation_overlay(hr_img, out_dir="report_figs"):
    os.makedirs(out_dir, exist_ok=True)

    hr = np.array(hr_img) / 255.0
    h, w, _ = hr.shape

    cx, cy = int(w * 0.4), int(h * 0.4)
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    mask = np.exp(-dist / (0.15 * max(h, w)))

    plt.figure()
    plt.imshow(hr)
    plt.imshow(mask, cmap="jet", alpha=0.4)
    plt.scatter([cx], [cy], c="white", s=30)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "gaze_foveation.png"), bbox_inches="tight", pad_inches=0)
    plt.close()


def save_rgb_histograms(hr_img, out_dir="report_figs"):
    os.makedirs(out_dir, exist_ok=True)

    arr = np.array(hr_img) / 255.0
    plt.figure()
    for i, label in enumerate(["R", "G", "B"]):
        plt.hist(arr[:, :, i].flatten(), bins=32, alpha=0.5, label=label)
    plt.legend()
    plt.xlabel("Intensity")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "rgb_hist_hr.png"), bbox_inches="tight")
    plt.close()


def save_entropy_heatmap(hr_img, out_dir="report_figs"):
    os.makedirs(out_dir, exist_ok=True)

    arr = np.array(hr_img) / 255.0
    gray = arr.mean(axis=2)
    gy, gx = np.gradient(gray)
    entropy_map = np.abs(gx) + np.abs(gy)

    plt.figure()
    plt.imshow(entropy_map, cmap="inferno")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "entropy_heatmap.png"), bbox_inches="tight", pad_inches=0)
    plt.close()


def save_motion_magnitude(hr_img, out_dir="report_figs"):
    os.makedirs(out_dir, exist_ok=True)

    arr1 = np.array(hr_img) / 255.0
    # синтетический "следующий кадр"
    noise = np.random.normal(0, 0.02, arr1.shape)
    arr2 = np.clip(arr1 + noise, 0, 1)

    motion = np.abs(arr2 - arr1).mean(axis=2)

    plt.figure()
    plt.imshow(motion, cmap="magma")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "motion_magnitude.png"), bbox_inches="tight", pad_inches=0)
    plt.close()


if __name__ == "__main__":
    lr_img, hr_img, sr_img = load_triplet("examples")

    out_dir = "report_figs"
    save_lr_hr_sr(lr_img, hr_img, sr_img, out_dir=out_dir)
    save_gaze_foveation_overlay(hr_img, out_dir=out_dir)
    save_rgb_histograms(hr_img, out_dir=out_dir)
    save_entropy_heatmap(hr_img, out_dir=out_dir)
    save_motion_magnitude(hr_img, out_dir=out_dir)

    print("Saved all visualizations to", out_dir)
