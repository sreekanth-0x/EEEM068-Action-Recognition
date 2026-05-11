# Training Log —Vasavi Atkuri  — Week 12
**Module:** EEEM068 Applied Machine Learning
**Date:** Week 12 — April/May 2026
**Task:** Part D Localisation + Part E Start

---

## Part D: DETR on JHMDB
- Frozen VideoMAE backbone: 86M (NOT trained)
- Trainable decoder: 3,430,938 params only
- Loss: CrossEntropy(class) + 5.0 x L1(bbox)

## DETR Training (10 epochs)
| Epoch | Loss |
|-------|------|
| 1 | 1.4043 Best saved |
| 2 | 0.3405 New best |
| 10 | 0.0344 Final best |

Mean IoU on JHMDB test: 0.9687

## Part E: Interpretability Started
- Attention maps: model attends correctly to body/limbs
- t-SNE: features shape (188, 768), KL divergence converged 0.382
- Well-separated clusters match classification results

## Files Saved
- checkpoints/localiser_best.pt
- checkpoints/localisation_results.png
- checkpoints/attention_maps.png
- checkpoints/tsne_features.png

## Next Week Plan
- Complete error analysis
- Build Gradio demo
- Start writing report
