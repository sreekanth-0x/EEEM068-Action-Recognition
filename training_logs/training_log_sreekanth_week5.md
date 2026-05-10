# Training Log — Sreekanth Akula — Week 5
**Module:** EEEM068 Applied Machine Learning  
**Date:** Week 5 — March 2026  
**Task:** Object Detection + DETR + Planning Part D Architecture

---

## Lecture Topics Covered

### Object Detection Fundamentals
- Classification: WHAT is in the image?
- Detection: WHAT and WHERE (bounding box + class label)
- Bounding box format: [x_centre, y_centre, width, height] — all normalised 0-1
- Traditional approach: sliding window + CNN → very slow
- Region-based: RCNN → Fast RCNN → Faster RCNN → anchor boxes

### DETR — Detection Transformer (Carion et al., 2020)
- Reformulates detection as a SET PREDICTION problem
- No anchor boxes, no NMS (non-maximum suppression) needed
- Key innovation: N learnable OBJECT QUERIES
  - Each query learns to detect one object
  - Queries attend to image features via CROSS-ATTENTION
  - Each query outputs: class label + bounding box
- Encoder: processes image features (from CNN backbone)
- Decoder: N queries cross-attend to encoder output
- Set loss: Hungarian matching between predicted and ground truth sets

### Cross-Attention vs Self-Attention
- Self-attention: tokens attend to each other (same sequence)
- Cross-attention: one sequence attends to another sequence
  - In DETR: object queries (sequence 1) attend to backbone features (sequence 2)
  - Query asks: "is there an object here in the features?"

### Evaluation Metric: IoU (Intersection over Union)
- IoU = Area of Overlap ÷ Area of Union
- IoU = 1.0 → perfect match (predicted = ground truth)
- IoU = 0.5 → commonly used threshold for "correct detection"
- IoU = 0.0 → no overlap at all

## Lab Session — Transfer Learning
- Practised loading pretrained ResNet and freezing layers
- Understood requires_grad=False to freeze parameters
- Compared full fine-tuning vs head-only vs partial fine-tuning
- Measured trainable parameters with: sum(p.numel() for p in model.parameters() if p.requires_grad)

## Planning Part D Architecture
Designed DETR-style localisation model for JHMDB:

```
Frozen VideoMAE backbone (86M params, requires_grad=False)
    ↓ extract features
nn.Linear(768 → 256)  ← project to smaller dim
    ↓
nn.Embedding(5, 256)  ← 5 learnable object queries
    ↓
TransformerDecoder(3 layers, 8 heads, 1024 FFN)
    ↓
Classification head: Linear(256 → 22)  [21 + no-object]
Bounding box head:   Linear(256→256→4) + Sigmoid
```

**Trainable params: ~3.4M (decoder only)**
**Frozen params: 86M (backbone)**

## JHMDB Dataset Understanding
- 21 action classes, 928 videos total
- Format: raw .avi files (decoded with cv2 frame-by-frame)
- Ground truth: bounding box annotations per clip
- Split: 80% train (742) / 20% test (186)

## Observations
- Freezing backbone = stable features, fast training, avoids overfitting on 742 clips
- Only training 3.4M params on 742 clips is much safer than training 86M
- Loss = classification loss + 5.0 × bbox L1 loss (bbox weighted higher)
- Why weight=5.0 on bbox? Encourages model to localise precisely

## Next Week Plan
- Lab: implement full dataset pipeline
- Code HMDBFrameDataset class with augmentation
- Code JHMDBDatasetSimple for Part D
- Plan ablation study design
