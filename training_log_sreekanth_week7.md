# Training Log — Sreekanth Akula — Week 7
**Module:** EEEM068 Applied Machine Learning  
**Date:** Week 7  
**Task:** Project Setup + Dataset Preparation + TimeSFormer Training Start

---

## Environment Setup
- Platform: Google Colaboratory
- GPU: NVIDIA A100-SXM4-40GB
- Framework: PyTorch + HuggingFace Transformers

## Libraries Installed
```
transformers==4.40.0
accelerate, timm, einops, scipy
scikit-learn, seaborn, tensorboard
av, gradio, opencv-python
```

## Data Paths Configured
- HMDB_simp: `/content/drive/MyDrive/EEEM068/data/HMDB_simp/HMDB_simp`
- JHMDB: `/content/drive/MyDrive/EEEM068/data/JHMDB_video/ReCompress_Videos`
- Checkpoints: `/content/drive/MyDrive/EEEM068/checkpoints`

## Dataset: HMDB_simp
| Split | Clips |
|-------|-------|
| Total | 1,250 |
| Train | 875 (70%) |
| Val   | 187 (15%) |
| Test  | 188 (15%) |

- Classes: 25 action categories
- Format: Pre-extracted JPEG frames per clip
- Stratified split used (same class proportions in all sets)

## Data Augmentation (Training Set)
- Resize to 224×224
- RandomHorizontalFlip
- ColorJitter (brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
- Normalize: mean=[0.45,0.45,0.45], std=[0.225,0.225,0.225]

## Model 1: TimeSFormer — Setup
- Source: `facebook/timesformer-base-finetuned-k400`
- Total parameters: 121,277,977
- Input: [Batch=4, Channels=3, Frames=8, 224, 224]
- Classification head: Linear(768 → 25)
- Optimiser: AdamW (lr=1e-4, weight_decay=1e-4)
- Scheduler: CosineAnnealingLR (T_max=20, eta_min=1e-7)
- Loss: CrossEntropyLoss
- Mixed precision: float16 via GradScaler

## Training Started — Epoch 1 Results
| Step | Loss |
|------|------|
| 0/219 | 3.0493 |
| 30/219 | 2.3325 |
| 60/219 | 1.4460 |
| 90/219 | 1.3762 |
| 120/219 | 0.9372 |
| 150/219 | 0.6602 |
| 210/219 | 0.8684 |

- Epoch 1 Train Acc: 58.40% | Val Acc: 78.07% ← Best saved!
- Epoch 1 time: 3573 seconds

## Observations
- Loss drops significantly from 3.05 → 0.87 in just epoch 1
- Val accuracy already at 78% after 1 epoch — good pretrained features
- Training time ~1 hour per epoch on A100

## Next Week Plan
- Complete TimeSFormer training (all 20 epochs)
- Start VideoMAE setup and training
