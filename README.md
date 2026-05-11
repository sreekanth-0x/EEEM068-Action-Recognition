# EEEM068 — Applied Machine Learning
## Action Recognition Using Vision Transformers (ViT)
**University of Surrey | 2025/6 | Group Project**

---

## Group Members
| Name            | Student ID| Email           |
|------           |-----------|-------          |
| Sreekanth Akula | 6963945  | sa05335@surrey.ac.uk |
| Vamsi Kuchipudi | 6959104   | vk00672@surrey.ac.uk |
| Ravitheja Jadi  | 6961879   | rj00844@surrey.ac.uk |
| Vasavi Atkuri   | 6954763   | va00462@surrey.ac.uk |
| Sathwik Pyneni  | 6963308   | sp02890@surrey.ac.uk |

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
``    `
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
EEEM068-Action-Recognition/
│
├── README.md ← project description
│
├── notebooks/
│ ├── EEEM068_project.ipynb ← YOUR MAIN NOTEBOOK
│
├── report/
│ └── EEEM068_IEEE_Report.pdf ← YOUR REPORT PDF
│
├── training_logs/ ← ALL WEEKLY LOGS
│ ├── training_log_sreekanth_week1.md
│ ├── training_log_sreekanth_week2.md
│ ├── ... week3 to week13.md
│ ├── training_log_vamsi_week1.md
│ ├── ... vamsi week2 to week13.md
│ ├── training_log_ravitheja_week1.md
│ ├── training_log_vasavi_week1.md
│ └── training_log_sathwik_week1.md
│
├── results/ ← SCREENSHOTS AND CHARTS
│ ├── confusion_matrix_timesformer.png
│ ├── confusion_matrix_videomae.png
│ ├── training_curves.png
│ ├── ablation_results.png
│ ├── robustness_results.png
│ ├── attention_maps.png
│ ├── tsne_features.png
│ └── error_analysis.png
│
└── contributions/
└── group_contributions.md

← WHO DID WHATnotebooks/EEEM068_project.ipynb
Your MAIN notebook with ALL 5 parts (A, B, C, D, E) in one file. This is the most important file. When the professor runs it — it must work from top to bottom without errors. Include all code: data loading, TimeSFormer training, VideoMAE training, ablations, robustness, DETR, attention maps, t-SNE, error analysis, Gradio demo.
REQUIRED Most important file
📄report/EEEM068_IEEE_Report.pdf
Your 5-page IEEE format technical report PDF. The one I just created for you. Upload this directly to GitHub. Also attach this when replying to your TA's email TODAY.
REQUIRED Send to TA today
📊training_logs/training_log_[name]_week[N].md
All 13 weekly logs for EACH of the 5 members. You already have these — I created them all. Each log covers: what lectures were attended, what was coded, what experiments were run, hyperparameters used, results obtained, and next week plan. Professor checks these to see individual contribution.
REQUIRED 5 members × 13 weeks = 65 files
🖼️results/*.png
All output images saved from your Colab notebook: confusion matrices, training curves, ablation bar charts, robustness chart, attention maps, t-SNE plot, error analysis images. Download these from your Google Drive checkpoints folder and upload to GitHub.
Important Download from Drive first
📋contributions/group_contributions.md
A file listing WHO did WHAT in the project. Required by the professor — "Group leaders must confirm agreed contributions via email before submission." Put this in writing on GitHub too.
URGENT — do today
📝README.md
Describes what the project is, what each file does, how to run the notebook, and what results to expect. GitHub shows this automatically on the main page when the TA visits your repo.
Required

