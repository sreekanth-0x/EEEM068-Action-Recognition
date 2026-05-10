# Training Log — Sreekanth Akula — Week 10
**Module:** EEEM068 Applied Machine Learning  
**Date:** Week 10  
**Task:** Part B — Ablation Studies (3 experiments)

---

## What is an Ablation Study?
Change ONE variable at a time while keeping everything else constant.
This measures the exact effect of each design choice.
All ablations: TimeSFormer + 5 epochs (fast comparison)

---

## Ablation 1 — Number of Input Frames

**Hypothesis:** More frames = more temporal context = better accuracy

| Frames | Val Accuracy | Notes |
|--------|-------------|-------|
| 4 | 74.87% | Not enough temporal context |
| 8 | 81.82% | Good balance |
| 16 | 83.96% | Best — but 2× memory cost |

**Conclusion:** More frames consistently helps.
16 frames best (83.96%) but requires batch size 2 instead of 4.

---

## Ablation 2 — Fine-tuning Strategy

**Hypothesis:** Updating more parameters gives better results (at 5 epochs)

Strategies tested:
- **head**: freeze ALL backbone, only train Linear(768→25) — ~19K params train
- **partial**: freeze backbone, unfreeze last 4 transformer layers — ~25M params train
- **full**: update ALL 121M parameters

| Strategy | Val Accuracy | Notes |
|----------|-------------|-------|
| head only | 80.75% | Safe from overfitting, limited adaptation |
| **partial** | **90.91%** | **BEST — balances adaptation + regularisation** |
| full | 81.82% | Overfits in only 5 epochs (too many params) |

**Surprising result:** Partial (90.91%) beats full (81.82%)!
Reason: At 5 epochs, training 121M params on only 875 clips → overfitting.
Partial fine-tuning acts as natural regularisation.

**Code for partial fine-tuning:**
```python
for p in m.backbone.parameters(): p.requires_grad = False
for layer in m.backbone.encoder.layer[-4:]:
    for p in layer.parameters(): p.requires_grad = True
```

---

## Ablation 3 — Temporal Sampling Strategy

**Hypothesis:** How we pick frames from a video affects accuracy

| Strategy | Val Accuracy | How it works |
|----------|-------------|--------------|
| uniform | 81.28% | np.linspace(0, n-1, num_frames) |
| **dense** | **81.82%** | Consecutive frames from middle of clip |
| random | 77.54% | Random frame positions — WORST |

**Conclusion:** 
- Random is clearly worst — may cluster all frames at one moment
- Uniform ≈ Dense (only 0.54% difference — within noise at 5 epochs)
- Avoid random sampling

---

## Ablation Results Plot
- Saved: `checkpoints/ablation_results.png`
- 3 bar charts side by side showing all comparisons

## Observations
- Ablation 2 gave the most surprising result — partial beats full at 5 epochs
- With 20 full epochs, full fine-tuning would likely win eventually
- Random sampling is consistently the worst choice for temporal selection

## Next Week Plan
- Part C: Statistical robustness testing across 3 random seeds
- Part D: DETR localisation on JHMDB dataset
