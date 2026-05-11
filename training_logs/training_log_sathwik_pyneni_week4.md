# Training Log —  — Week 4
**Module:** EEEM068 Applied Machine Learning
**Date:** Week 4 — March 2026
**Task:** VideoMAE + Self-Supervised Learning + Training Techniques

---

## Lecture Topics
- Self-supervised learning: no labels needed
- VideoMAE: mask 90% of video tokens, reconstruct missing pixels
- Why 90%? Video has temporal redundancy unlike images
- AdamW: Adam + weight decay (L2 regularisation)
- Cosine Annealing: reduces LR from 1e-4 to near zero over 20 epochs
- Mixed Precision float16: halves memory, doubles speed on GPU
- Gradient Clipping: prevents exploding gradients

## Literature Completed
4. He et al. 2022 — MAE done
5. Wang et al. 2022 — VideoMAE done

Minimum 5 papers requirement met!

## Project Decisions Agreed With Group
- TimeSFormer: 8 frames, batch=4, lr=1e-4
- VideoMAE: 16 frames, batch=2, lr=5e-5
- Dataset: HMDB_simp 25 classes 1250 clips
- Split: 70 train / 15 val / 15 test

## Next Week Plan
- Implement HMDBFrameDataset class
- Set up Google Drive folder structure
