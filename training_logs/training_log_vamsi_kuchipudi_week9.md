# Training Log —  — Week 9
**Module:** EEEM068 Applied Machine Learning
**Date:** Week 9 — April 2026
**Task:** VideoMAE Training + Part A Comparison

---

## VideoMAE Setup
- Source: MCG-NJU/videomae-base-finetuned-kinetics
- Parameters: 86,244,889 (35M fewer than TimeSFormer)
- 16 frames, batch=2, lr=5e-5
- Mean pooling of ALL tokens (not CLS only)

## VideoMAE Training
| Epoch | Val Acc |
|-------|---------|
| 1 | 81.28% |
| 5 | ~88% |
| 10 | ~91% |
| 20 | 93.09% |

## Test Results
- Top-1: 93.09%
- Top-5: 98.94%

## Part A Final Comparison
| Model | Top-1 | Top-5 | Params |
|-------|-------|-------|--------|
| TimeSFormer | 81.91% | 95.21% | 121M |
| VideoMAE | 93.09% | 98.94% | 86M |

Winner: VideoMAE +11.18% Top-1 with 35M fewer parameters!

## Files Saved
- checkpoints/videomae_best.pt
- checkpoints/all_models_saved.pt

## Next Week Plan
- Part B: 3 ablation studies
