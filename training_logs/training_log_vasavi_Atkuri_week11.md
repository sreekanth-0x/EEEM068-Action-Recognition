# Training Log —Vasavi Atkuri  — Week 11
**Module:** EEEM068 Applied Machine Learning
**Date:** Week 11 — April 2026
**Task:** Part C — Robustness Testing (3 Seeds)

---

## Why Test Robustness?
Single result could be lucky. Running same VideoMAE with 3 seeds
gives 3 different data splits — proves consistent results.

## Results
| Seed | Top-1 | Top-5 |
|------|-------|-------|
| 42 | 93.09% | 98.94% |
| 123 | 96.81% | 98.94% |
| 2024 | 97.87% | 100.00% |

## Summary
- Mean Top-1: 95.92%
- Std: +/-2.05%
- Mean Top-5: 99.29%

Conclusion: Model is ROBUST — consistently 93%+ across all splits.

## File Saved
- checkpoints/robustness_results.png

## Next Week Plan
- Part D: DETR localisation on JHMDB
- Part E: Interpretability start
