# Training Log — Sreekanth Akula — Week 13
**Module:** EEEM068 Applied Machine Learning  
**Date:** Week 13  
**Task:** Part E — Interpretability + Gradio Demo + Final Summary

---

## Part E Overview
Three methods to understand WHY the model makes its predictions:
1. Attention Maps — WHERE does the model look?
2. t-SNE Visualisation — WHAT features did it learn?
3. Error Analysis — WHICH predictions fail and why?

---

## Method 1 — Attention Map Visualisation

**How it works:**
1. Load VideoMAEWithAttn (output_attentions=True)
2. Run inference: logits, attns = model(video, return_attn=True)
3. Take last transformer layer attention: attns[-1]
4. Average over all attention heads: .mean(0)
5. Extract CLS→patch attention: last_attn[0, 1:][:196]
6. Reshape 196 tokens → 14×14 grid
7. Resize to 224×224 with cv2
8. Normalise to 0–1
9. Overlay as jet colourmap on original frame (50% blend)

**Result:** Bright red = where model focused most

**Observation:**
- Model correctly attends to body/limbs for physical actions
- For "golf": focuses on arm and club region
- For "fencing": focuses on arm and weapon
- Model is NOT cheating by looking at backgrounds
- Confirms model learned semantically meaningful features

**File saved:** `checkpoints/attention_maps.png`

---

## Method 2 — t-SNE Feature Visualisation

**Setup:**
- Extract 768-dim features for all 188 test clips
- backbone(video).last_hidden_state.mean(dim=1) → [1, 768]
- Full feature array: shape (188, 768)

**t-SNE Parameters:**
- n_components=2 (reduce 768 → 2 dimensions)
- perplexity=20
- random_state=42
- Converged at 1000 iterations: KL divergence = 0.382533

**Results:**
- Well-separated clusters: golf, handstand, ride_bike (visually distinct)
- Overlapping clusters: climb vs climb_stairs (same body motion)
- Overlapping clusters: draw_sword vs fencing (similar weapon use)
- Overlaps EXACTLY MATCH the classification errors from Part A!

**File saved:** `checkpoints/tsne_features.png`

---

## Method 3 — Error Analysis

**Setup:**
- Collected 5 correctly classified samples
- Collected 5 incorrectly classified samples
- For each: showed 4 video frames + top-5 confidence bar chart

**Common correct predictions (high confidence):**
- golf → golf (confidence ~98%)
- situp → situp (confidence ~95%)
- ride_bike → ride_bike (confidence ~93%)

**Common incorrect predictions (confusions):**
- pick → throw or catch (both involve reaching arm motion)
- shoot_gun → shoot_bow (both involve aiming at target)
- flic_flac → cartwheel (both are acrobatic rotations)

**Key insight:** ALL errors are between SEMANTICALLY SIMILAR classes.
The model makes INTELLIGENT confusions — never confuses "golf" with "kiss".
This confirms the model has learned meaningful action representations.

**Files saved:**
- `checkpoints/error_correct.png`
- `checkpoints/error_incorrect.png`

---

## Gradio Live Demo

Built an interactive web app where users upload any video and the model
predicts the top-5 action classes with confidence scores.

- URL: https://16818750b34065c59e.gradio.live (expires 1 week)
- Input: Any video file (.mp4, .avi, .mov)
- Output: Top-5 predicted actions with % confidence
- Model used: VideoMAE (best performing)

---

## Final Results Summary — All Parts Complete

| Part | Task | Result |
|------|------|--------|
| A | TimeSFormer classification | Top-1: 81.91% Top-5: 95.21% |
| A | VideoMAE classification | Top-1: 93.09% Top-5: 98.94% |
| B | Best ablation (partial FT) | 90.91% val accuracy |
| C | Robustness (3 seeds) | 95.92% ± 2.05% |
| D | DETR localisation | Mean IoU: 0.9687 |
| E | Interpretability | Attn maps + t-SNE + Error analysis |

## All Files Saved to Google Drive
- `checkpoints/timesformer_best.pt`
- `checkpoints/videomae_best.pt`
- `checkpoints/all_models_saved.pt` (830.3 MB)
- `checkpoints/localiser_best.pt`
- `checkpoints/final_results.json`
- `checkpoints/confusion_timesformer.png`
- `checkpoints/confusion_videomae.png`
- `checkpoints/curves_timesformer.png`
- `checkpoints/curves_videomae.png`
- `checkpoints/ablation_results.png`
- `checkpoints/robustness_results.png`
- `checkpoints/localisation_results.png`
- `checkpoints/attention_maps.png`
- `checkpoints/tsne_features.png`
- `checkpoints/error_correct.png`
- `checkpoints/error_incorrect.png`

## Next Steps
- Write technical report (IEEE double column, 5 pages)
- Prepare oral exam presentation (5 minutes)
- Practice Q&A for all parts A B C D E
