# Training Log —  — Week 5
**Module:** EEEM068 Applied Machine Learning
**Date:** Week 5 — March 2026
**Task:** Object Detection + DETR + Planning Part D

---

## Lecture Topics
- Detection: predict WHAT and WHERE (bounding box + class)
- Bounding box: [x_centre, y_centre, width, height] all 0-1 normalised
- DETR: N learnable object queries cross-attend to backbone features
- Each query outputs class label + bounding box
- IoU = Intersection over Union (1.0 = perfect, 0.0 = no overlap)

## Part D Architecture Planned
- Frozen VideoMAE backbone (86M params)
- Linear(768 to 256)
- 5 Learnable Object Queries
- TransformerDecoder: 3 layers, 8 heads
- Class head: Linear(256 to 22)
- Bbox head: Linear(256 to 4) + Sigmoid
- Trainable params: only 3.4M

## JHMDB Dataset
- 21 classes, 928 .avi video files
- Train: 742 / Test: 186

## Next Week Plan
- Complete dataset pipeline code
- Prepare formative feedback presentation
