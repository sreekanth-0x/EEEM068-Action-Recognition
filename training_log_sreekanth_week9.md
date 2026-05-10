# Training Log — Sreekanth Akula — Week 9
**Module:** EEEM068 Applied Machine Learning  
**Date:** Week 9  
**Task:** VideoMAE Training + Part A Final Comparison

---

## Model 2: VideoMAE — Setup
- Source: `MCG-NJU/videomae-base-finetuned-kinetics`
- Pre-training: Self-supervised MAE (90% token masking)
- Total parameters: 86,244,889
- Input: [Batch=2, Channels=3, Frames=16, 224, 224]
- Token strategy: Mean pooling of ALL tokens (not CLS only)
- Optimiser: AdamW (lr=5e-5, weight_decay=1e-4)
- Scheduler: CosineAnnealingLR (T_max=20, eta_min=1e-7)

## Why Different Settings from TimeSFormer?
- Frames=16 (not 8): VideoMAE pretraining used 16 frames — must match
- Batch=2 (not 4): 16 frames × 224×224 = 4× more GPU memory needed
- LR=5e-5 (not 1e-4): MAE features are richer — update more carefully
- Train batches: 438 (vs 219 for TimeSFormer — more updates per epoch)

## VideoMAE Training — Epoch 1 Results
| Step | Loss |
|------|------|
| 0/438 | 6.4258 |
| 30/438 | 3.5103 |
| 60/438 | 4.5176 |
| 150/438 | 0.0146 |
| 300/438 | 0.0002 |
| 420/438 | 0.0013 |

- Epoch 1 Train Acc: 51.43% | Val Acc: 81.28% ← Best saved!
- Epoch 1 time: 7616 seconds (~2 hours)

## VideoMAE Training — All 20 Epochs
| Epoch | Val Acc |
|-------|---------|
| 1  | 81.28% |
| 2  | improving |
| 5  | ~88% |
| 10 | ~91% |
| 20 | 93.09% |

## VideoMAE Test Set Evaluation
| Metric | Result |
|--------|--------|
| Top-1 Accuracy | **93.09%** |
| Top-5 Accuracy | **98.94%** |
| Total Parameters | 86,244,889 |

## Part A Final Comparison
| Model | Top-1 | Top-5 | Params | Frames |
|-------|-------|-------|--------|--------|
| TimeSFormer | 81.91% | 95.21% | 121M | 8 |
| **VideoMAE** | **93.09%** | **98.94%** | **86M** | **16** |

**Winner: VideoMAE — +11.18% Top-1 with 35M fewer parameters!**

## Why VideoMAE Wins
1. Self-supervised MAE pre-training learns richer video features
2. 16 frames captures more temporal information than 8
3. Mean pooling uses ALL patch information vs CLS token only

## Files Saved
- `checkpoints/timesformer_best.pt`
- `checkpoints/videomae_best.pt`
- `checkpoints/all_models_saved.pt` (830.3 MB)
- `checkpoints/confusion_timesformer.png`
- `checkpoints/confusion_videomae.png`
- `checkpoints/curves_timesformer.png`
- `checkpoints/curves_videomae.png`

## Observations
- VideoMAE starts slower (51% train acc epoch 1) but reaches much higher final accuracy
- Both models: Top-5 > 95% — correct class always in top 5 predictions
- VideoMAE per-class: perfect F1=1.00 for chew, golf, kiss, pullup, pushup, situp, smoke

## Next Week Plan
- Part B: Ablation studies (frames, fine-tuning strategy, sampling)
