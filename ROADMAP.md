# Roadmap — On-device Super-Resolution Tuned for Foveated Displays

Repo: https://github.com/Azlarkhon/On-device-super-resolution-tuned-for-foveating-displays

## Scope
Gaze-aware super-resolution (SR) system optimized for foveated displays. Prioritizes the foveal region while reducing computational load in the periphery. Aims for real-time performance on embedded GPUs.  
Key metrics: FPS, end-to-end latency, LPIPS/SSIM (global and foveal), MOS.

## Milestones (Tashkent time)

**W1 · Oct 27 – Nov 2**  
Repository setup, base structure, CI skeleton, `ROADMAP.md`.  
Owner: Azlarkhon

**W2 · Nov 3 – Nov 9**  
Baseline SR (EDSR/RCAN) implemented end-to-end on small dataset; compute LPIPS/SSIM baseline.  
Owner: Muhammadsodiq

**W3 · Nov 10 – Nov 16**  
Integrate eye-tracking pipeline; generate gaze-based foveation maps.  
Owner: Mirsolih

**W4 · Nov 17 – Nov 23**  
Design foveated SR: high-capacity for fovea, lightweight for periphery; region-weighted perceptual losses.  
Owner: Azlarkhon

**W5 · Nov 24 – Nov 30**  
Desktop prototype, latency profiling, and performance trade-off table.  
Owner: Muhammadsodiq

**W6 · Dec 1 – Dec 7**  
On-device deployment (Jetson Orin/TensorRT); quantization and tuning.  
Owner: Mirsolih

**W7 · Dec 8 – Dec 14**  
User study design and pilot run (MOS protocol and ethics checklist).  
Owner: Azlarkhon

**W8 · Dec 15 – Dec 21**  
Ablations: scheduler vs. no-scheduler, radius variation, region-weighting experiments.  
Owners: Muhammadsodiq & Mirsolih

**W9 · Dec 22 – Dec 27**  
Final evaluation, report, poster, and demo video.  
Owners: All

## RACI Matrix

| Task | Responsible | Accountable | Consulted | Informed |
|------|--------------|-------------|------------|-----------|
| Baseline SR code | Muhammadsodiq | Azlarkhon | Mirsolih | All |
| Eye-tracker integration | Mirsolih | Azlarkhon | Muhammadsodiq | All |
| On-device optimization | Mirsolih | Azlarkhon | Muhammadsodiq | All |
| User study design | Azlarkhon | Azlarkhon | Mirsolih, Muhammadsodiq | All |
| Final report & demo | All | Azlarkhon | — | All |

## Success Criteria
- Latency < 20 ms  
- FPS ≥ 60  
- Foveal LPIPS improvement ≥ 10% over baseline  
- SSIM > 0.90 in foveal region  
- MOS ≥ +0.5 improvement in perceived sharpness  

## Weekly Check-ins
Each team member updates 3–6 bullet points weekly inside this file.  
Link commits/issues to milestones to ensure transparency and tracking.

## Risks and Mitigations
| Risk | Mitigation |
|------|-------------|
| Compute bottleneck | Quantization, pruning, reduced periphery SR load |
| Gaze tracking errors | Predictive smoothing, tolerance margin |
| Dataset shift | Fine-tune on small on-device dataset |
| High MOS variance | Pilot with ≥ 20 participants, balanced sampling |

## Deliverables
- Full training and deployment code for gaze-aware SR  
- Comprehensive report with experiments and ablations  
- Poster and video demo  
- Stretch goals: 90 FPS capability, adaptive upscale factors, power efficiency analysis

## Links
- Proposal document (PDF)  
- Maintained `ROADMAP.md` in GitHub repo  
- Weekly progress issues and PR references
