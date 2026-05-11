# Training Log — Ravitheja_jadi(6961879) — Week 8
**Module:** EEEM068 Applied Machine Learning
**Date:** Week 8 — April 2026
**Task:** TimeSFormer Complete + Evaluation

---

## TimeSFormer All 20 Epochs
| Epoch | Train | Val |
|-------|-------|-----|
| 1 | 58.40% | 78.07% |
| 2 | 86.63% | 79.68% |
| 5 | ~93% | ~81% |
| 20 | ~98% | 81.91% |

## Test Results
- Top-1 Accuracy: 81.91%
- Top-5 Accuracy: 95.21%
- Parameters: 121,277,977

## Best Classes (F1=1.00)
climb, draw_sword, kiss, pour, pullup, ride_bike, situp, wave

## Hardest Classes
- cartwheel: F1=0.57 (confused with flic_flac)
- eat: F1=0.57 (confused with chew)

## Files Saved
- checkpoints/timesformer_best.pt
- checkpoints/confusion_timesformer.png
- checkpoints/curves_timesformer.png

## Next Week Plan
- Train VideoMAE: 16 frames, lr=5e-5, batch=2
