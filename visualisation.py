import numpy as np
import matplotlib.pyplot as plt

# ----------- HR / LR SAMPLE IMAGES -----------
hr = np.random.rand(256,256,3)
lr = hr[::2,::2]

plt.figure(figsize=(6,6))
plt.title("High-Resolution (HR)")
plt.imshow(hr)
plt.axis('off')
plt.show()

plt.figure(figsize=(6,6))
plt.title("Low-Resolution (LR)")
plt.imshow(lr)
plt.axis('off')
plt.show()

# ----------- FOVEATION MASK + GAZE -----------
h, w = hr.shape[:2]
cx, cy = int(w*0.4), int(h*0.4)

yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
dist = np.sqrt((xx-cx)**2 + (yy-cy)**2)
mask = np.exp(-dist/60)

plt.figure(figsize=(6,6))
plt.title("Gaze Point + Foveation Mask")
plt.imshow(hr)
plt.imshow(mask, alpha=0.4)
plt.scatter([cx], [cy], c='red', s=40)
plt.axis('off')
plt.show()

# ----------- RGB HISTOGRAMS -----------
plt.figure(figsize=(7,4))
plt.title("RGB Histograms")
plt.hist(hr[:,:,0].ravel(), bins=30, alpha=0.6, label='R')
plt.hist(hr[:,:,1].ravel(), bins=30, alpha=0.6, label='G')
plt.hist(hr[:,:,2].ravel(), bins=30, alpha=0.6, label='B')
plt.legend()
plt.show()

# ----------- ENTROPY / TEXTURE HEATMAP -----------
gray = hr.mean(axis=2)
gradx, grady = np.gradient(gray)
entropy = np.abs(gradx) + np.abs(grady)

plt.figure(figsize=(6,6))
plt.title("Entropy / Texture Heatmap")
plt.imshow(entropy, cmap='viridis')
plt.axis('off')
plt.show()

# ----------- TRUE CONSECUTIVE-FRAME MOTION -----------
# simulate temporal shift: frame2 = hr shifted by 1 pixel
frame2 = np.roll(hr, shift=1, axis=0)  # vertical motion example

motion = np.abs(frame2 - hr).mean(axis=2)

plt.figure(figsize=(6,6))
plt.title("Consecutive-Frame Motion Magnitude")
plt.imshow(motion, cmap='inferno')
plt.axis('off')
plt.show()
