# Training Log — Ravitheja_jadi(6961879) — Week 13
**Module:** EEEM068 Applied Machine Learning
**Date:** Week 13 — May 2026
**Task:** Part E Complete + Gradio Demo + Final Summary

---

## Part E: Error Analysis Complete

Correct predictions:
- golf: 98% confidence
- situp: 95% confidence
- ride_bike: 93% confidence

Incorrect predictions (intelligent confusions):
- pick confused with throw (similar arm motion)
- shoot_gun confused with shoot_bow (both aiming)
- flic_flac confused with cartwheel (both acrobatic)

All errors between SIMILAR classes — model learned meaningful features.

## Gradio Demo Built
- Upload video -> predict top-5 actions with confidence
- Model used: VideoMAE (best performer)

## Final Results Summary
| Part | Result |
|------|--------|
| A TimeSFormer | Top-1: 81.91% Top-5: 95.21% |
| A VideoMAE | Top-1: 93.09% Top-5: 98.94% |
| B Best ablation | 90.91% partial fine-tuning |
| C Robustness | 95.92% +/-2.05% |
| D DETR IoU | 0.9687 |
| E Interpretability | Attention + t-SNE + Error + Gradio |

## All Files in GitHub
- EEEM068_project.ipynb
- training_log_ravitheja_jadi_week1.md through week13.md

## Next Steps
- Write IEEE report (5 pages, double column)
- Prepare 5-minute oral presentation
- Practice Q&A for all parts
