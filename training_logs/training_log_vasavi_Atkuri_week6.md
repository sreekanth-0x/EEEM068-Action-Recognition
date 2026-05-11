# Training Log —Vasavi Atkuri  — Week 6
**Module:** EEEM068 Applied Machine Learning
**Date:** Week 6 — March/April 2026
**Task:** Attention Visualisation + Full Code Implementation

---

## Lecture Topics
- Attention maps: reshape 196 patch tokens to 14x14 grid, resize to 224x224
- Overlay as heatmap: red = high attention, blue = low
- t-SNE: reduce 768-dim features to 2D for visualisation
- Well-separated clusters = model learned distinct features per class

## Code Implemented
- HMDBFrameDataset class: uniform sampling, augmentation pipeline
- Data verified: 1250 clips, 25 classes, 875/187/188 split
- JHMDBDatasetSimple: cv2 video decoding for .avi files
- JHMDB: Train=742, Test=186, Classes=21

## Formative Feedback Prepared
- 3-minute project plan ready
- Covers: models, dataset, 5 parts, timeline

## Next Week Plan
- Week 7 formative feedback presentation
- Create GitHub repo, add all teammates and TA
- Begin TimeSFormer training
