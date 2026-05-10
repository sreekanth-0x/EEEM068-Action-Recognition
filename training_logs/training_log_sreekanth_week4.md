# Training Log — Sreekanth Akula — Week 4
**Module:** EEEM068 Applied Machine Learning  
**Date:** Week 4 — March 2026  
**Task:** VideoMAE + Self-Supervised Learning + Training Techniques

---

## Lecture Topics Covered

### Self-Supervised Learning
- No labels needed — the data itself provides supervision
- Pretext tasks: predict masked patches, predict rotation, predict colourisation
- Learns general representations that transfer to many downstream tasks
- Advantage over supervised: can use UNLIMITED unlabelled data

### Masked Autoencoder (MAE — He et al., 2022)
- For images: mask 75% of patches → reconstruct missing pixels
- Forces model to deeply understand image content
- Result: much better representations than supervised training on same data
- The "understanding" comes from having to fill in what is missing

### VideoMAE (Wang et al., 2022)
- Extends MAE concept to video
- Mask 90% of video tokens (much higher than image MAE's 75%)
- Why 90%? Video has temporal redundancy — adjacent frames are similar
  Masking 90% ensures model cannot just copy nearby frames
- Reconstruct pixel values of hidden patches from only 10% visible
- Result: extremely rich video representations without ANY labels

**VideoMAE vs TimeSFormer Pre-training:**
| Aspect | TimeSFormer | VideoMAE |
|--------|-------------|----------|
| Pre-training | Supervised (Kinetics-400 labels) | Self-supervised (MAE) |
| Labels needed | Yes — 240K labelled videos | No — any raw video |
| What it learns | Class-specific features | General video understanding |
| Transfer ability | Good | Excellent |

### Training Techniques Studied
- **AdamW**: Adam + weight decay (L2 regularisation)
  - Adam: adaptive learning rate per parameter + momentum
  - Weight decay: penalises large weights → prevents overfitting
- **Cosine Annealing**: smoothly reduces LR from initial to near-zero
  - Large updates early (fast convergence) → small updates later (precision)
- **Mixed Precision (float16)**: halves memory, doubles speed on modern GPUs
  - GradScaler handles numerical precision automatically
- **Gradient Clipping**: limits maximum gradient norm
  - Prevents exploding gradients in deep transformer layers

## Lab Session — HuggingFace Transformers
- Installed transformers library
- Loaded pretrained model: `from_pretrained('facebook/timesformer-base-finetuned-k400')`
- Understood model output structure: `out.last_hidden_state`
- Practised replacing classification head: `nn.Linear(768, num_classes)`

## Literature Completed
Papers read this week:
4. He et al. (2022) — MAE (Masked Autoencoders) ✓
5. Wang et al. (2022) — VideoMAE ✓

**5 papers total now read — satisfies minimum requirement!**

## Key Design Decisions Made for Project
- Use TimeSFormer AND VideoMAE (compare both)
- TimeSFormer: 8 frames, batch=4, lr=1e-4
- VideoMAE: 16 frames (matches pretraining), batch=2, lr=5e-5
- Dataset: HMDB_simp (25 classes, 1250 clips)
- Split: 70% train / 15% val / 15% test stratified

## Observations
- VideoMAE's 90% masking ratio seems extreme but works because
  video has temporal redundancy unlike images
- Lower LR for VideoMAE (5e-5 vs 1e-4) because features are already
  very good — careful adaptation preserves what was learned
- Mean pooling of all tokens (VideoMAE) vs CLS token only (TimeSFormer)

## Next Week Plan
- Lab: implement full HMDBFrameDataset class
- Code the data loading pipeline with augmentations
- Set up Google Drive folder structure
