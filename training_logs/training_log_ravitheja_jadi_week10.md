# Training Log — Ravitheja_jadi(6961879) — Week 10
**Module:** EEEM068 Applied Machine Learning
**Date:** Week 10 — April 2026
**Task:** Part B — Ablation Studies

---

## Ablation 1: Number of Frames
| Frames | Val Acc |
|--------|---------|
| 4 | 74.87% |
| 8 | 81.82% |
| 16 | 83.96% BEST |

Conclusion: More frames = better

## Ablation 2: Fine-tuning Strategy
| Strategy | Val Acc |
|----------|---------|
| head only | 80.75% |
| partial (last 4 layers) | 90.91% BEST |
| full | 81.82% |

Surprising: partial beats full at 5 epochs!
Reason: 121M params on 875 clips in 5 epochs = overfitting

## Ablation 3: Temporal Sampling
| Strategy | Val Acc |
|----------|---------|
| uniform | 81.28% |
| dense (middle) | 81.82% BEST |
| random | 77.54% WORST |

Conclusion: Avoid random sampling

## File Saved
- checkpoints/ablation_results.png

## Next Week Plan
- Part C: robustness testing 3 seeds
