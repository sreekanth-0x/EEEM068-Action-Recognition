# Training Log — Sreekanth Akula — Week 2
**Module:** EEEM068 Applied Machine Learning  
**Date:** Week 2 — February 2026  
**Task:** Deep Learning Fundamentals + CNN Review + ViT Introduction

---

## Lecture Topics Covered This Week

### Deep Learning Basics Revised
- Neural network forward pass and backpropagation
- Gradient descent and weight updates
- Loss functions for classification (CrossEntropyLoss)
- Activation functions: ReLU, Softmax
- Overfitting vs underfitting — regularisation techniques

### CNN Architecture Review
- Convolutional layers: local receptive fields, weight sharing
- Pooling layers: MaxPool, AvgPool
- Classic CNNs: AlexNet, VGG, ResNet
- Why CNNs have inductive biases (translation invariance)
- Limitation: CNNs cannot model long-range spatial relationships

### Introduction to Transformers
- Attention mechanism: Q, K, V matrices explained
- Self-attention: every token attends to every other token
- Positional encoding: how the model knows WHERE each token is
- Multi-head attention: multiple attention perspectives in parallel
- Feed-forward network in transformer blocks

### Vision Transformer (ViT) — Key Concepts
- Image → patches (16×16 pixels each)
- 224×224 image → 14×14 = 196 patches
- Each patch projected to 768-dim embedding vector
- [CLS] token added — collects info from all patches
- All 197 tokens pass through transformer encoder
- [CLS] token output used for classification

## Lab Session
- Ran basic PyTorch classification example
- Practised DataLoader, Dataset class structure
- Understood train/val/test splitting with train_test_split
- Practised transforms.Compose pipeline

## Key Concepts Understood
| Concept | Understanding |
|---------|--------------|
| Patch embedding | Image cut into 16×16 patches, each → 768-dim vector |
| Self-attention | Every patch looks at every other patch simultaneously |
| CLS token | Special token that aggregates all patch information |
| Why ViT > CNN | Global receptive field from layer 1, no inductive bias |

## Literature Started
- Started reading: Dosovitskiy et al. "An Image is Worth 16×16 Words" (ViT paper)
- Key finding: ViT matches CNN performance when trained on enough data

## Observations
- Self-attention is computationally expensive: O(n²) for n tokens
- ViT patch size 16×16 → 196 patches → 196² = 38,416 attention operations
- This is why divided space-time attention is needed for video

## Next Week Plan
- Read TimeSFormer paper (Bertasius et al.)
- Understand how ViT is extended for video
- Lab: first PyTorch training loop practice
