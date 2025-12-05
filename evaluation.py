"""
evaluation_complete.py - Evaluates ALL validation images
"""

import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from model import RCAN

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCALE = 2
PATCH_LR = 128
PATCH_HR = PATCH_LR * SCALE

class CompleteSREvaluator:
    """Complete evaluator for RCAN model - evaluates ALL images."""

    def __init__(self, weights_path="rcan_full_3050ti_final.pth"):
        self.device = DEVICE
        self.model = self._load_model(weights_path)

    def _load_model(self, weights_path):
        """Load trained model."""
        model = RCAN(scale=SCALE).to(self.device)
        state = torch.load(weights_path, map_location=self.device)
        model.load_state_dict(state)
        model.eval()
        print(f"✓ Model loaded: {weights_path}")
        return model

    def calculate_psnr(self, img1, img2):
        """Calculate PSNR between two images."""
        if isinstance(img1, torch.Tensor):
            img1 = img1.cpu().numpy()
        if isinstance(img2, torch.Tensor):
            img2 = img2.cpu().numpy()

        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * np.log10(1.0 / np.sqrt(mse))

    def calculate_ssim_simple(self, img1, img2):
        """Calculate simplified SSIM."""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        if isinstance(img1, torch.Tensor):
            img1 = img1.cpu().numpy()
        if isinstance(img2, torch.Tensor):
            img2 = img2.cpu().numpy()

        mu1 = img1.mean()
        mu2 = img2.mean()
        sigma1 = img1.std()
        sigma2 = img2.std()
        sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()

        ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1**2 + mu2**2 + C1) * (sigma1**2 + sigma2**2 + C2))

        return ssim

    def center_crop(self, lr_img, hr_img):
        """Center crop images to standard size."""
        w_lr, h_lr = lr_img.size
        w_hr, h_hr = hr_img.size

        x_lr = max(0, (w_lr - PATCH_LR) // 2)
        y_lr = max(0, (h_lr - PATCH_LR) // 2)
        lr_crop = lr_img.crop((x_lr, y_lr, x_lr + PATCH_LR, y_lr + PATCH_LR))

        x_hr = x_lr * SCALE
        y_hr = y_lr * SCALE
        x_hr = min(max(0, x_hr), max(0, w_hr - PATCH_HR))
        y_hr = min(max(0, y_hr), max(0, h_hr - PATCH_HR))
        hr_crop = hr_img.crop((x_hr, y_hr, x_hr + PATCH_HR, y_hr + PATCH_HR))

        return lr_crop, hr_crop

    def evaluate_all_images(self, lr_dir, hr_dir):
        """Evaluate ALL images in the directory."""
        # Get ALL image names
        names = sorted([
            n for n in os.listdir(lr_dir)
            if n.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
        ])

        if not names:
            print("✗ No images found!")
            return []

        print(f"📊 Evaluating ALL {len(names)} images...")

        results = []
        for name in tqdm(names, desc="Processing"):
            lr_path = os.path.join(lr_dir, name)
            hr_path = os.path.join(hr_dir, name)

            if not os.path.exists(hr_path):
                print(f"⚠ Warning: HR not found for {name}")
                continue

            try:
                # Load images
                lr_img = Image.open(lr_path).convert("RGB")
                hr_img = Image.open(hr_path).convert("RGB")

                # Center crop
                lr_crop, hr_crop = self.center_crop(lr_img, hr_img)

                # Convert to tensors
                lr_tensor = TF.to_tensor(lr_crop).unsqueeze(0).to(self.device)
                hr_tensor = TF.to_tensor(hr_crop).unsqueeze(0).to(self.device)

                # Generate SR
                with torch.no_grad():
                    sr_tensor = self.model(lr_tensor)[0].cpu().clamp(0, 1)

                # Calculate metrics
                psnr = self.calculate_psnr(sr_tensor, hr_tensor[0].cpu())
                ssim = self.calculate_ssim_simple(sr_tensor, hr_tensor[0].cpu())
                abs_error = torch.abs(sr_tensor - hr_tensor[0].cpu()).mean().item()

                results.append({
                    'filename': name,
                    'psnr': float(psnr),
                    'ssim': float(ssim),
                    'abs_error': float(abs_error),
                    'sr_image': sr_tensor,
                    'hr_image': hr_tensor[0].cpu(),
                    'lr_image': TF.to_tensor(lr_crop),
                })

            except Exception as e:
                print(f"✗ Error processing {name}: {e}")

        return results

    def generate_statistics(self, results):
        """Generate comprehensive statistics."""
        if not results:
            return None

        psnr_values = [r['psnr'] for r in results]
        ssim_values = [r['ssim'] for r in results]

        stats = {
            'total_images': len(results),
            'psnr_mean': np.mean(psnr_values),
            'psnr_std': np.std(psnr_values),
            'psnr_min': np.min(psnr_values),
            'psnr_max': np.max(psnr_values),
            'ssim_mean': np.mean(ssim_values),
            'ssim_std': np.std(ssim_values),
            'ssim_min': np.min(ssim_values),
            'ssim_max': np.max(ssim_values),
        }

        # Categorize image quality
        excellent_psnr = sum(1 for p in psnr_values if p > 40)
        good_psnr = sum(1 for p in psnr_values if 30 <= p <= 40)
        acceptable_psnr = sum(1 for p in psnr_values if 20 <= p < 30)
        poor_psnr = sum(1 for p in psnr_values if p < 20)

        excellent_ssim = sum(1 for s in ssim_values if s > 0.95)
        good_ssim = sum(1 for s in ssim_values if 0.90 <= s <= 0.95)
        acceptable_ssim = sum(1 for s in ssim_values if 0.80 <= s < 0.90)
        poor_ssim = sum(1 for s in ssim_values if s < 0.80)

        stats['psnr_categories'] = {
            'excellent': excellent_psnr,
            'good': good_psnr,
            'acceptable': acceptable_psnr,
            'poor': poor_psnr
        }

        stats['ssim_categories'] = {
            'excellent': excellent_ssim,
            'good': good_ssim,
            'acceptable': acceptable_ssim,
            'poor': poor_ssim
        }

        return stats

    def save_results(self, results, stats, output_dir="full_evaluation"):
        """Save all results to files."""
        os.makedirs(output_dir, exist_ok=True)

        # 1. Save metrics CSV
        df = pd.DataFrame([{
            'filename': r['filename'],
            'psnr': r['psnr'],
            'ssim': r['ssim'],
            'abs_error': r['abs_error']
        } for r in results])

        csv_path = os.path.join(output_dir, 'all_metrics.csv')
        df.to_csv(csv_path, index=False)
        print(f"✓ Metrics saved: {csv_path}")

        # 2. Save statistics report
        report_path = os.path.join(output_dir, 'statistics_report.txt')
        with open(report_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("RCAN SUPER-RESOLUTION - COMPLETE EVALUATION REPORT\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Total Images Evaluated: {stats['total_images']}\n")
            f.write(f"Evaluation Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Device: {self.device}\n\n")

            f.write("-" * 70 + "\n")
            f.write("PSNR STATISTICS (dB)\n")
            f.write("-" * 70 + "\n")
            f.write(f"Mean ± Std: {stats['psnr_mean']:.2f} ± {stats['psnr_std']:.2f}\n")
            f.write(f"Range: [{stats['psnr_min']:.2f}, {stats['psnr_max']:.2f}]\n\n")

            f.write("PSNR Quality Categories:\n")
            f.write(f"  Excellent (>40 dB): {stats['psnr_categories']['excellent']} images\n")
            f.write(f"  Good (30-40 dB): {stats['psnr_categories']['good']} images\n")
            f.write(f"  Acceptable (20-30 dB): {stats['psnr_categories']['acceptable']} images\n")
            f.write(f"  Poor (<20 dB): {stats['psnr_categories']['poor']} images\n\n")

            f.write("-" * 70 + "\n")
            f.write("SSIM STATISTICS\n")
            f.write("-" * 70 + "\n")
            f.write(f"Mean ± Std: {stats['ssim_mean']:.4f} ± {stats['ssim_std']:.4f}\n")
            f.write(f"Range: [{stats['ssim_min']:.4f}, {stats['ssim_max']:.4f}]\n\n")

            f.write("SSIM Quality Categories:\n")
            f.write(f"  Excellent (>0.95): {stats['ssim_categories']['excellent']} images\n")
            f.write(f"  Good (0.90-0.95): {stats['ssim_categories']['good']} images\n")
            f.write(f"  Acceptable (0.80-0.90): {stats['ssim_categories']['acceptable']} images\n")
            f.write(f"  Poor (<0.80): {stats['ssim_categories']['poor']} images\n\n")

            f.write("-" * 70 + "\n")
            f.write("INTERPRETATION GUIDE\n")
            f.write("-" * 70 + "\n")
            f.write("PSNR (dB):\n")
            f.write("  >40: Near perfect (indistinguishable from original)\n")
            f.write("  30-40: Good quality (minor artifacts)\n")
            f.write("  20-30: Acceptable (visible but not distracting)\n")
            f.write("  <20: Poor (significant quality loss)\n\n")

            f.write("SSIM (0-1):\n")
            f.write("  0.95-1.00: Excellent structural similarity\n")
            f.write("  0.90-0.95: Good structural preservation\n")
            f.write("  0.80-0.90: Moderate structural similarity\n")
            f.write("  <0.80: Poor structural preservation\n")

        print(f"✓ Report saved: {report_path}")

        # 3. Save visualizations
        self._save_visualizations(results, stats, output_dir)

        return csv_path, report_path

    def _save_visualizations(self, results, stats, output_dir):
        """Save visualization plots."""
        psnr_values = [r['psnr'] for r in results]
        ssim_values = [r['ssim'] for r in results]

        # Create subdirectory for plots
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        # Plot 1: PSNR Distribution
        plt.figure(figsize=(10, 6))
        plt.hist(psnr_values, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(stats['psnr_mean'], color='red', linestyle='--',
                   label=f'Mean: {stats["psnr_mean"]:.2f} dB')
        plt.xlabel('PSNR (dB)')
        plt.ylabel('Number of Images')
        plt.title(f'PSNR Distribution ({stats["total_images"]} images)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'psnr_distribution.png'), dpi=150)
        plt.close()

        # Plot 2: SSIM Distribution
        plt.figure(figsize=(10, 6))
        plt.hist(ssim_values, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        plt.axvline(stats['ssim_mean'], color='red', linestyle='--',
                   label=f'Mean: {stats["ssim_mean"]:.4f}')
        plt.xlabel('SSIM')
        plt.ylabel('Number of Images')
        plt.title(f'SSIM Distribution ({stats["total_images"]} images)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'ssim_distribution.png'), dpi=150)
        plt.close()

        # Plot 3: PSNR vs SSIM scatter
        plt.figure(figsize=(10, 6))
        plt.scatter(psnr_values, ssim_values, alpha=0.6, c='purple', s=30)
        plt.xlabel('PSNR (dB)')
        plt.ylabel('SSIM')
        plt.title('PSNR vs SSIM Correlation')
        plt.grid(True, alpha=0.3)

        # Add correlation info
        corr = np.corrcoef(psnr_values, ssim_values)[0, 1]
        plt.text(0.05, 0.95, f'Correlation: {corr:.3f}',
                transform=plt.gca().transAxes, fontsize=12,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'psnr_vs_ssim.png'), dpi=150)
        plt.close()

        print(f"✓ Visualizations saved to: {plots_dir}")

        # Plot 4: Quality categories (bar chart)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # PSNR categories
        psnr_cats = ['Excellent', 'Good', 'Acceptable', 'Poor']
        psnr_counts = [stats['psnr_categories'][k] for k in ['excellent', 'good', 'acceptable', 'poor']]
        colors_psnr = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']

        axes[0].bar(psnr_cats, psnr_counts, color=colors_psnr, edgecolor='black')
        axes[0].set_xlabel('PSNR Quality Category')
        axes[0].set_ylabel('Number of Images')
        axes[0].set_title('PSNR Quality Distribution')
        axes[0].grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for i, v in enumerate(psnr_counts):
            axes[0].text(i, v + max(psnr_counts)*0.01, str(v),
                        ha='center', va='bottom', fontweight='bold')

        # SSIM categories
        ssim_cats = ['Excellent', 'Good', 'Acceptable', 'Poor']
        ssim_counts = [stats['ssim_categories'][k] for k in ['excellent', 'good', 'acceptable', 'poor']]
        colors_ssim = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']

        axes[1].bar(ssim_cats, ssim_counts, color=colors_ssim, edgecolor='black')
        axes[1].set_xlabel('SSIM Quality Category')
        axes[1].set_ylabel('Number of Images')
        axes[1].set_title('SSIM Quality Distribution')
        axes[1].grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for i, v in enumerate(ssim_counts):
            axes[1].text(i, v + max(ssim_counts)*0.01, str(v),
                        ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'quality_categories.png'), dpi=150)
        plt.close()

        # Save example images (best and worst)
        self._save_example_images(results, output_dir)

    def _save_example_images(self, results, output_dir):
        """Save best and worst examples."""
        examples_dir = os.path.join(output_dir, "examples")
        os.makedirs(examples_dir, exist_ok=True)

        # Sort by PSNR
        sorted_by_psnr = sorted(results, key=lambda x: x['psnr'])

        # Get 2 worst and 2 best
        examples = []
        if len(sorted_by_psnr) >= 4:
            examples.extend(sorted_by_psnr[:2])  # 2 worst
            examples.extend(sorted_by_psnr[-2:]) # 2 best

        # Save each example
        for i, result in enumerate(examples):
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))

            # LR
            lr_img = result['lr_image'].permute(1, 2, 0).numpy()
            axes[0].imshow(lr_img)
            axes[0].set_title(f"LR Input")
            axes[0].axis('off')

            # SR
            sr_img = result['sr_image'].permute(1, 2, 0).numpy()
            axes[1].imshow(sr_img)
            axes[1].set_title(f"SR Output\nPSNR: {result['psnr']:.2f} dB\nSSIM: {result['ssim']:.4f}")
            axes[1].axis('off')

            # HR
            hr_img = result['hr_image'].permute(1, 2, 0).numpy()
            axes[2].imshow(hr_img)
            axes[2].set_title("Ground Truth HR")
            axes[2].axis('off')

            plt.suptitle(f"Example {'Worst' if i < 2 else 'Best'}: {result['filename']}",
                        fontsize=14, y=1.05)
            plt.tight_layout()

            example_name = f"{'worst' if i < 2 else 'best'}_{i%2+1}_{result['filename'].split('.')[0]}.png"
            plt.savefig(os.path.join(examples_dir, example_name), dpi=150, bbox_inches='tight')
            plt.close()

        print(f"✓ Example images saved to: {examples_dir}")

    def run(self):
        """Main execution method."""
        print("=" * 70)
        print("RCAN SUPER-RESOLUTION - COMPLETE EVALUATION")
        print("=" * 70)

        # Configuration
        LR_DIR = "data/val/LR"
        HR_DIR = "data/val/HR"
        OUTPUT_DIR = "complete_evaluation"

        # Check directories
        if not os.path.exists(LR_DIR):
            print(f"✗ LR directory not found: {LR_DIR}")
            return
        if not os.path.exists(HR_DIR):
            print(f"✗ HR directory not found: {HR_DIR}")
            return

        # Count total images
        total_images = len([n for n in os.listdir(LR_DIR)
                           if n.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))])
        print(f"📁 Found {total_images} images in validation set")

        # Evaluate ALL images
        results = self.evaluate_all_images(LR_DIR, HR_DIR)

        if not results:
            print("✗ No results generated!")
            return

        # Generate statistics
        stats = self.generate_statistics(results)

        # Save everything
        self.save_results(results, stats, OUTPUT_DIR)

        # Print summary
        print("\n" + "=" * 70)
        print("✅ EVALUATION COMPLETE - SUMMARY")
        print("=" * 70)
        print(f"Total Images Processed: {stats['total_images']}")
        print(f"Average PSNR: {stats['psnr_mean']:.2f} ± {stats['psnr_std']:.2f} dB")
        print(f"Average SSIM: {stats['ssim_mean']:.4f} ± {stats['ssim_std']:.4f}")
        print(f"PSNR Range: {stats['psnr_min']:.2f} - {stats['psnr_max']:.2f} dB")
        print(f"SSIM Range: {stats['ssim_min']:.4f} - {stats['ssim_max']:.4f}")
        print(f"\n📊 Results saved to: {OUTPUT_DIR}/")

if __name__ == "__main__":
    evaluator = CompleteSREvaluator(weights_path="rcan_full_3050ti_final.pth")
    evaluator.run()