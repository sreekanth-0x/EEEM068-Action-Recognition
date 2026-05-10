# Training Log — Sreekanth Akula — Week 8
**Module:** EEEM068 Applied Machine Learning  
**Date:** Week 8  
**Task:** Complete TimeSFormer Training + Evaluate on Test Set

---

## TimeSFormer Training — All 20 Epochs Complete

| Epoch | Train Acc | Val Acc | Notes |
|-------|-----------|---------|-------|
| 1  | 58.40% | 78.07% | Best saved |
| 2  | 86.63% | 79.68% | Best saved |
| 3  | ~90%   | ~80%   | Improving |
| 5  | ~93%   | ~81%   | Stable |
| 10 | ~96%   | ~81%   | Plateau beginning |
| 15 | ~97%   | ~81%   | Very small gains |
| 20 | ~98%   | 81.91% | Final |

- Training time per epoch: ~293s (from epoch 2 onwards — backbone cached)
- Best model saved: `timesformer_best.pt`

## TimeSFormer Test Set Evaluation
| Metric | Result |
|--------|--------|
| Top-1 Accuracy | **81.91%** |
| Top-5 Accuracy | **95.21%** |
| Total Parameters | 121,277,977 |

## Per-Class Results (TimeSFormer)
**Best classes (F1 = 1.00):**
- climb, draw_sword, kiss, pour, pullup, ride_bike, situp, wave

**Hardest classes:**
- cartwheel: F1=0.57 (confused with flic_flac)
- eat: F1=0.57 (confused with chew)
- smoke: F1=0.62

## Confusion Matrix
- Saved to: `checkpoints/confusion_timesformer.png`
- Training curves saved to: `checkpoints/curves_timesformer.png`

## Observations
- TimeSFormer converges fast (epoch 1 already 78% val)
- Val accuracy plateaus around epoch 5 — shows good generalisation
- Most errors between visually similar classes (cartwheel vs flic_flac)
- 81.91% is strong for 875 training clips

## Next Week Plan
- Train VideoMAE model (16 frames, lr=5e-5)
- Compare both models on test set
- Save combined model checkpoint
