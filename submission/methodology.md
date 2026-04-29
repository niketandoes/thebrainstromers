# EuroSAT Land Use Classification — Methodology

**Team:** The Brainstromers | **Hackathon:** KaggleHacX '26

---

## 1. Overview

The goal of this project is to build an accurate, robust image classification model for the **EuroSAT dataset**, containing 27,000 Sentinel-2 satellite images categorized into 10 distinct land use and land cover classes.

Our solution prioritizes efficiency and high accuracy, adhering strictly to the local-execution hackathon constraints without relying on any external cloud APIs or services.

**Final Result: 99% Validation Accuracy | Macro F1-Score: 0.99**

---

## 2. Data Preparation

The dataset is structured using CSV mapping files (`train.csv`, `validation.csv`, `test.csv`). We designed a **Custom PyTorch Dataset class (`EuroSATDataset`)** to handle this non-standard format efficiently.

- **Image Loading:** Images loaded from a local base directory using filenames from the CSVs.
- **Label Mapping:** String class names dynamically mapped to 0–9 integer indices at runtime.
- **Training Augmentations (applied only to training set):**
  - `RandomHorizontalFlip`
  - `RandomRotation(15°)`
  - `ColorJitter` (brightness & contrast)
  - Standard ImageNet Normalization (`mean=[0.485, 0.456, 0.406]`)

---

## 3. Model Architecture

**Chosen Model: EfficientNet-B0** (PyTorch / TorchVision)

| Property | Value |
|----------|-------|
| Base Architecture | EfficientNet-B0 |
| Pre-training | ImageNet-1k |
| Output Classes | 10 (EuroSAT) |
| Input Size | 224 × 224 × 3 |
| Total Epochs | 15 |

**Why EfficientNet-B0?**
It delivers an optimal trade-off between computational efficiency and accuracy using compound scaling. This makes it ideal for long CPU-based training runs while still capturing the fine-grained geospatial textures that differentiate classes like `HerbaceousVegetation` from `PermanentCrop`.

**Head Modification:** The final `nn.Linear` classifier layer was replaced with a new 10-class output layer.

---

## 4. Two-Phase Fine-Tuning Strategy

To avoid destroying valuable pre-trained ImageNet features early in training, we adopted a proven two-phase approach.

### Phase 1: Feature Extraction — Epochs 1–5
- **Backbone:** Frozen (all `requires_grad = False`)
- **Optimizer:** Adam, LR = `1e-3`
- **Goal:** Train only the new 10-class classifier head.
- **Outcome:** Achieved a strong **~90% validation accuracy baseline** within 5 epochs, confirming that ImageNet features transfer well to satellite imagery.

### Phase 2: Full Fine-Tuning — Epochs 6–15
- **Backbone:** Unfrozen (all parameters trainable)
- **Optimizer:** Adam, LR = `1e-4` (10× lower to prevent catastrophic forgetting)
- **Scheduler:** `CosineAnnealingLR` for smooth LR decay
- **Goal:** Allow the deep convolutional filters to subtly adapt to satellite-specific textures.
- **Outcome:** Accuracy rapidly converged to **99%**, with the final training loss settling at ~0.022.

---

## 5. Training Results

| Epoch | Phase | Avg. Loss | Val. Accuracy |
|-------|-------|-----------|---------------|
| 1 | Feature Extraction | 0.677 | 90% |
| 5 | Feature Extraction | ~0.30 | ~95% |
| 6 | Fine-Tuning Start | ~0.25 | ~96% |
| 11 | Fine-Tuning | 0.040 | 99% |
| 15 | Fine-Tuning (Final) | 0.022 | **99%** |

### Final Per-Class Performance (Epoch 15)

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Annual Crop | 0.98 | 0.99 | 0.99 |
| Forest | 0.99 | 1.00 | **1.00** |
| Herbaceous Vegetation | 0.98 | 0.96 | 0.97 |
| Highway | 0.99 | 0.98 | 0.99 |
| Industrial | 0.99 | 0.99 | 0.99 |
| Pasture | 0.98 | 0.96 | 0.97 |
| Permanent Crop | 0.96 | 0.99 | 0.98 |
| Residential | 0.99 | 0.99 | 0.99 |
| River | 0.98 | 1.00 | 0.99 |
| Sea/Lake | 1.00 | 0.99 | **1.00** |
| **Weighted Avg** | **0.99** | **0.99** | **0.99** |

---

## 6. Inference Application

The trained weights (`model.pth`, 16MB) are saved locally. We built a fully offline **Streamlit web application** (`main/app.py`) that:

1. Loads the EfficientNet-B0 model from disk using `torch.load`.
2. Applies the identical normalization transforms used during training.
3. Displays the **predicted land cover class** and a **top-5 confidence bar chart**.

The app runs entirely on the local machine with no external API calls.

---

## 7. Dependencies

```
torch
torchvision
streamlit
pandas
scikit-learn
Pillow
numpy
```
