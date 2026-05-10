# Training Log — Sreekanth Akula — Week 6
**Module:** EEEM068 Applied Machine Learning  
**Date:** Week 6 — March/April 2026  
**Task:** CNN Attention + Interpretability + Full Code Implementation Started

---

## Lecture Topics Covered — Week 6 (CNNs and Attention)

### Attention in CNNs
- CAM (Class Activation Maps): visualise which regions CNN uses for classification
- Grad-CAM: use gradients to generate class activation maps
- Shows WHERE the network focuses — validates model is not "cheating"

### Transformer Attention Visualisation
- Attention weights stored per layer per head
- Last layer attention most task-relevant
- Extract attention from [CLS] token → to all patch tokens
- Reshape 196 attention weights → 14×14 grid
- Resize to 224×224 for overlay on original image
- Jet colourmap: red = high attention, blue = low attention

### t-SNE (t-distributed Stochastic Neighbour Embedding)
- Dimensionality reduction: high-dim → 2D for visualisation
- Preserves LOCAL structure: similar points stay close in 2D
- Parameters: perplexity (local vs global balance), random_state
- Good t-SNE: well-separated clusters = distinct features per class
- Overlapping clusters = model finds these classes similar
- KL divergence measures how well 2D preserves high-dim structure

### Error Analysis
- Collect correct and incorrect predictions
- Show video frames + top-5 confidence scores
- Helps understand failure modes
- Pattern: errors should be between similar classes, not random

## Full Code Implementation Started
This week: wrote and tested the complete dataset loading pipeline.

### HMDBFrameDataset Class — Implemented
```python
class HMDBFrameDataset(Dataset):
    # Loads pre-extracted JPEG frames from folders
    # Uniform sampling: np.linspace(0, n-1, num_frames)
    # Returns: [C, T, H, W] tensor + label + path
```

### Data Pipeline Verified
```
HMDB_simp path: /content/drive/MyDrive/EEEM068/data/HMDB_simp/HMDB_simp
Total clips found: 1,250
Classes: 25
Train: 875   Val: 187   Test: 188
```

### Augmentation Pipeline Coded
Training:
- Resize(224,224) + RandomHorizontalFlip + ColorJitter(0.4,0.4,0.4,0.1)
- ToTensor + Normalize([0.45,0.45,0.45], [0.225,0.225,0.225])

Validation/Test:
- Resize(224,224) + ToTensor + Normalize (no augmentation)

### DataLoaders Created
- train_loader: batch=4, shuffle=True, num_workers=2, pin_memory=True
- val_loader: batch=4, shuffle=False
- test_loader: batch=4, shuffle=False

### JHMDBDatasetSimple — Implemented
```python
class JHMDBDatasetSimple(Dataset):
    # Loads .avi files with cv2.VideoCapture
    # Decodes frames: cap.read() loop
    # Returns: [C, T, H, W] + bbox [0.5,0.5,0.8,0.8] + label
```
JHMDB: Train=742, Test=186, Classes=21

## Pre-formative Feedback Preparation
- Week 7 formative feedback session coming up
- Prepared 3-minute project plan presentation outline:
  1. Project topic: Action Recognition using ViT
  2. Models: TimeSFormer + VideoMAE
  3. Dataset: HMDB_simp (25 classes, 1250 clips)
  4. Parts: A (classification) B (ablations) C (robustness) D (localisation) E (interpretability)
  5. Timeline: training week 7-8, experiments week 9-10, report week 11-12

## Observations
- pin_memory=True in DataLoader speeds up GPU transfer significantly
- np.linspace for frame sampling is better than random — systematic coverage
- cv2.VideoCapture works well for .avi decoding in JHMDB
- Stratified split ensures each class has same proportion in train/val/test

## Next Week Plan
- Formative feedback presentation (Week 7)
- Create GitHub repository and add TA
- Begin TimeSFormer training on full HMDB_simp dataset
- Upload first notebook to GitHub
