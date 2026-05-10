# EEEM068 — Applied Machine Learning
## Action Recognition Using Vision Transformers (ViT)
**University of Surrey | 2025/6 | Group Project**

---

## Group Members
| Name | Student ID | Email |
|------|-----------|-------|
| Sreekanth Akula | — | s.akula@surrey.ac.uk |
| Vamsi Kuchipudi | — | v.kuchipudi@surrey.ac.uk |
| Ravitheja Jadi | — | r.jadi@surrey.ac.uk |
| Vasavi Atkuri | — | v.atkuri@surrey.ac.uk |
| Sathwik Pyneni | — | s.pyneni@surrey.ac.uk |

---

## Project Overview
This project implements and evaluates two Vision Transformer architectures for video action recognition on the HMDB_simp dataset (25 classes, 1,250 clips).

**Models Used:**
- **TimeSFormer** (Facebook AI) — Divided space-time attention, supervised Kinetics-400 pre-training
- **VideoMAE** (Microsoft Research) — Self-supervised masked autoencoder pre-training (90% masking)

---

## Results Summary

| Part | Task | Result |
|------|------|--------|
| A | TimeSFormer Classification | Top-1: **81.91%** Top-5: 95.21% |
| A | VideoMAE Classification | Top-1: **93.09%** Top-5: 98.94% |
| B | Best Ablation (Partial FT) | Val: **90.91%** |
| C | Robustness (3 seeds) | **95.92% ± 2.05%** |
| D | DETR Localisation (JHMDB) | Mean IoU: **0.9687** |
| E | Interpretability | Attention maps + t-SNE + Error analysis |

**Winner: VideoMAE — +11.18% Top-1 with 35M fewer parameters!**

---

## Repository Structure
```
EEEM068-Action-Recognition/
├── README.md
├── notebooks/
│   └── EEEM068_project.ipynb      # Main notebook — all 5 parts
├── report/
│   └── EEEM068_IEEE_Report.pdf    # 5-page IEEE format report
├── training_logs/
│   ├── training_log_sreekanth_week1.md
│   ├── training_log_sreekanth_week2.md
│   ├── ... (13 weeks × 5 members = 65 files)
├── results/
│   ├── confusion_matrix_timesformer.png
│   ├── confusion_matrix_videomae.png
│   ├── training_curves.png
│   ├── ablation_results.png
│   ├── robustness_results.png
│   ├── attention_maps.png
│   ├── tsne_features.png
│   └── error_analysis.png
└── contributions/
    └── group_contributions.md
```

---

## How to Run

### Requirements
```bash
pip install transformers==4.40.0 accelerate timm einops
pip install scikit-learn seaborn tensorboard av gradio
```

### Setup
1. Mount Google Drive in Colab
2. Set data paths to HMDB_simp and JHMDB folders
3. Run all cells top to bottom

### Data Paths
```python
HMDB_DIR  = '/content/drive/MyDrive/EEEM068/data/HMDB_simp/HMDB_simp'
JHMDB_DIR = '/content/drive/MyDrive/EEEM068/data/JHMDB_video/ReCompress_Videos'
CHECKPOINT_DIR = '/content/drive/MyDrive/EEEM068/checkpoints'
```

---

## Key Technical Details

| Setting | TimeSFormer | VideoMAE |
|---------|-------------|----------|
| Pre-training | Supervised (Kinetics-400) | Self-supervised MAE |
| Input frames | 8 | 16 |
| Batch size | 4 | 2 |
| Learning rate | 1e-4 | 5e-5 |
| Parameters | 121.3M | 86.2M |
| Token strategy | CLS token | Mean pool all |

---

## GPU Used
- NVIDIA A100-SXM4-40GB (Google Colab Pro)
- Mixed precision training (float16) via GradScaler

---

## Module
EEEM068 Applied Machine Learning | SEMR2 2025/6 | University of Surrey
