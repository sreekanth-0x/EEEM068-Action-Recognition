# Training Log — Sreekanth Akula — Week 12
**Module:** EEEM068 Applied Machine Learning  
**Date:** Week 12  
**Task:** Part D — DETR-Style Spatio-Temporal Localisation on JHMDB

---

## What is Part D?
Part A answered: WHAT action is happening?
Part D answers: WHERE in the frame is the action happening?
We predict a bounding box [x_centre, y_centre, width, height] (0-1 normalised).

## Dataset: JHMDB
| Split | Videos |
|-------|--------|
| Total | 928 |
| Train | 742 (80%) |
| Test  | 186 (20%) |
- Classes: 21 action classes
- Format: Raw .avi video files (decoded with cv2)
- Ground truth: Simplified bounding boxes [0.5, 0.5, 0.8, 0.8]

## DETR Architecture — SimpleLocaliser
```
Frozen VideoMAE Backbone (86M params — NOT trained)
        ↓
Linear Projection: 768 → 256
        ↓
5 Learnable Object Queries [5, 256]
        ↓
Transformer Decoder: 3 layers, 8 heads, 1024 FFN dim
        ↓
Classification Head: Linear(256 → 22)   [21 classes + no-object]
Bounding Box Head:   Linear(256→256→4) + Sigmoid [x,y,w,h]
```

**Trainable parameters: 3,430,938 (only decoder — backbone frozen)**

## Training Configuration
- Optimiser: AdamW (lr=1e-4, weight_decay=1e-4)
- Epochs: 10
- Batch size: 2
- Loss: CrossEntropyLoss(class) + 5.0 × L1Loss(bbox)
- Grad clipping: 0.1

## Training Results — All 10 Epochs

| Epoch | Step | Loss | Notes |
|-------|------|------|-------|
| 1 | 0/371 | 3.3056 | High at start |
| 1 | 50/371 | 1.8195 | Drops fast |
| 1 | 350/371 | 0.2074 | Very low! |
| 1 | END | 1.4043 | Best saved |
| 2 | 0/371 | 0.4947 | Starts much lower |
| 2 | END | 0.3405 | New best saved |
| 3 | END | 0.1659 | New best |
| 5 | END | ~0.08 | Converging |
| 10 | END | **0.0344** | Final best |

**Best localisation loss: 0.0344**

## Evaluation — IoU Results

IoU (Intersection over Union) = Area of Overlap ÷ Area of Union
- IoU = 1.0 → Perfect predicted box matches ground truth
- IoU = 0.0 → No overlap at all

**Mean IoU on JHMDB test set: 0.9687**

## Why High IoU?
Ground truth boxes are simplified [0.5, 0.5, 0.8, 0.8] for all clips
(large centred boxes). The model quickly learns to predict similar
large centred boxes → high IoU. In production with precise per-frame
annotations, IoU would typically be 0.50–0.70.

## Files Saved
- `checkpoints/localiser_best.pt`
- `checkpoints/localisation_results.png` — visualisation with GT and predicted boxes

## Observations
- Frozen backbone converges faster — stable VideoMAE features from the start
- Only 3.4M params trained vs 86M backbone = very fast training
- Loss drops dramatically in epoch 1 then refines slowly
- Dual loss (classification + bbox) both contribute to learning

## Next Week Plan
- Part E: Interpretability — attention maps, t-SNE, error analysis, Gradio demo
- Start writing the technical report
