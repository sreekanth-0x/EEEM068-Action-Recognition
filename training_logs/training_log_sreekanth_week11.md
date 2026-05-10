# Training Log — Sreekanth Akula — Week 11
**Module:** EEEM068 Applied Machine Learning  
**Date:** Week 11  
**Task:** Part C — Statistical Robustness Testing (3 Seeds)

---

## Why Test Robustness?
A single test result of 93.09% could be lucky depending on which
188 clips happened to end up in the test set. By running the SAME
saved VideoMAE model with 3 different random seeds, we get 3
different data splits — proving results are consistent and reliable.

## What Changes Per Seed?
- CHANGES: data split (different videos in train/val/test), batch order
- STAYS SAME: model weights (same best VideoMAE checkpoint loaded every time)

## Experiment Setup
- Model: Best saved VideoMAE from Part A (`all_models_saved.pt`)
- Dataset: HMDB_simp, 16 frames, same preprocessing
- Seeds tested: 42, 123, 2024

## Results — All 3 Seeds

### Seed 42
- Test samples: 188
- Top-1 Accuracy: **93.09%**
- Top-5 Accuracy: **98.94%**
- Note: Same as original Part A result

### Seed 123
- Test samples: 188
- Top-1 Accuracy: **96.81%**
- Top-5 Accuracy: **98.94%**
- Note: Slightly easier test split

### Seed 2024
- Test samples: 188
- Top-1 Accuracy: **97.87%**
- Top-5 Accuracy: **100.00%** ← Perfect Top-5!
- Note: Best split — all correct classes in top 5

## Statistical Summary
| Metric | Mean | Std |
|--------|------|-----|
| Top-1 | **95.92%** | **±2.05%** |
| Top-5 | **99.29%** | **±0.49%** |

## Why Is There 2.05% Variance?
The test set has only 188 samples. Each seed creates a different split —
some splits contain harder examples (more draw_sword, flic_flac) while
others have easier examples. With 188 samples, 4 misclassified clips = 2%.
This is expected for small datasets.

## Conclusion
The model is ROBUST — consistently high accuracy (93%+) across all seeds.
High mean (95.92%) with low variance (±2.05%) proves results are genuine,
not just lucky from one particular test split.

## Files Saved
- `checkpoints/robustness_results.png` — line plot of 3 seed results

## Observations
- All 3 seeds give >93% Top-1 — model is consistently strong
- Top-5 near perfect in all cases (98.94–100%)
- Seed 2024 perfect Top-5 = correct class ALWAYS in top 5 predictions

## Next Week Plan
- Part D: DETR-style localisation on JHMDB dataset
- Part E: Interpretability — attention maps, t-SNE, error analysis
