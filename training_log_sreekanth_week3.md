# Training Log — Sreekanth Akula — Week 3
**Module:** EEEM068 Applied Machine Learning  
**Date:** Week 3 — February/March 2026  
**Task:** Video Understanding + TimeSFormer Paper + Project Group Formed

---

## Lecture Topics Covered

### Video Understanding Challenges
- Video = Images + Time (3D data problem)
- Spatial dimension: what things look like in each frame
- Temporal dimension: how things move across frames
- Cannot just apply image model to single frames — loses motion information
- Challenge: same action looks different depending on viewpoint, speed, person

### Two-Stream Networks (Simonyan & Zisserman, 2014)
- Stream 1: spatial (RGB frames) — what things look like
- Stream 2: temporal (optical flow) — how things move
- Combine both streams for classification
- Limitation: optical flow is expensive to compute

### 3D Convolutions (I3D — Carreira & Zisserman, 2017)
- Inflate 2D ImageNet filters to 3D (adds temporal dimension)
- Processes spatiotemporal volumes together
- Limitation: limited receptive field, high memory cost

### TimeSFormer (Bertasius et al., 2021)
- Extends ViT to video with Divided Space-Time Attention
- SPATIAL attention: each patch attends to all patches in SAME frame (196×196)
- TEMPORAL attention: each patch attends to SAME position across all frames (8×8)
- Alternates between spatial and temporal attention in each layer
- Result: 10× cheaper than full space-time attention

**Full space-time attention cost:**
- 8 frames × 196 patches = 1,568 tokens
- 1,568² = 2,456,704 operations per layer — TOO EXPENSIVE

**Divided attention cost:**
- Spatial: 196² = 38,416 per frame
- Temporal: 8² = 64 per patch position
- Much more efficient!

## Project Group Formation
- Module project groups confirmed on SurreyLearn
- Topic selected: Action Recognition using Vision Transformers
- TA assigned to our group
- Initial communication with TA established

## Literature Progress
Papers read this week:
1. Simonyan & Zisserman (2014) — Two-stream networks ✓
2. Carreira & Zisserman (2017) — I3D ✓
3. Bertasius et al. (2021) — TimeSFormer ✓

## Lab Session
- PyTorch Dataset and DataLoader practice
- Implemented custom __getitem__ for image loading
- Practised transforms pipeline end to end

## Key Understanding Gained
- Divided space-time attention solves the computational cost problem
- TimeSFormer pretrained on Kinetics-400 (400 classes, 240K videos)
- Fine-tuning: load pretrained weights, replace classification head

## Observations
- TimeSFormer uses 8 frames as default — good balance of speed vs accuracy
- CLS token at position 0 aggregates video-level information
- Linear(768 → num_classes) is the only new layer we train from scratch

## Next Week Plan
- Read VideoMAE paper — understand masked autoencoding
- Lab: implement first video dataset loading
- Start planning project methodology
