# EuroSAT Land Use & Land Cover Classification

![Python](https://img.shields.io/badge/python-3.14-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6-red.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-FF4B4B.svg)
![Accuracy](https://img.shields.io/badge/Validation%20Accuracy-99%25-brightgreen.svg)
![Status](https://img.shields.io/badge/Training-Complete-success.svg)

A high-performance deep learning solution for classifying Sentinel-2 satellite imagery into 10 distinct land cover categories using the EuroSAT dataset. Built during the **KaggleHacX '26 Hackathon** by Team Brainstromers.

---

## 🚀 Project Highlights

| Metric | Value |
|--------|-------|
| **Architecture** | EfficientNet-B0 (Transfer Learning) |
| **Final Validation Accuracy** | **99%** |
| **Macro F1-Score** | **0.99** |
| **Training Time** | ~2.5 hours (CPU) |
| **Training Strategy** | Two-phase: Frozen backbone → Full fine-tuning |
| **Perfect Classes** | Forest, Residential, SeaLake (F1 = 1.00) |

---

## 📁 Project Structure

```text
thebrainstromers/
├── main/
│   ├── train.py            # Model training and fine-tuning pipeline
│   └── app.py              # Streamlit web application for inference
├── solution/
│   └── solution_teamname.csv  # Final test set predictions
├── submission/
│   └── methodology.md      # Detailed technical methodology
├── model.pth               # Trained model weights (16MB)
├── training.log            # Full epoch-by-epoch training log
├── requirements.txt        # Project dependencies
└── README.md               # This file
```

---

## 🛠️ Installation & Setup

**1. Clone the repository:**
```bash
git clone https://github.com/niketandoes/thebrainstromers.git
cd thebrainstromers
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Dataset:** Ensure the EuroSAT dataset (with `train.csv`, `validation.csv`, `test.csv`) is available locally and update the `DATA_DIR` path in `main/train.py`.

---

## 🌐 Running the Web Application

The `model.pth` is already trained and included. Simply launch the app:

```bash
streamlit run main/app.py
```

Upload any `.jpg` Sentinel-2 satellite image to receive a real-time prediction with confidence scores for all 10 classes.

---

## 🏋️ Re-Training the Model

To re-run the full 15-epoch training pipeline from scratch:

```bash
python main/train.py
```

This will:
1. Train the classifier head for **5 epochs** (backbone frozen, LR = 1e-3)
2. Fine-tune the entire network for **10 epochs** (backbone unfrozen, LR = 1e-4)
3. Save final weights to `model.pth`
4. Auto-generate `solution/solution_teamname.csv`

---

## 📊 Final Results

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Annual Crop | 0.98 | 0.99 | 0.99 |
| Forest | 0.99 | 1.00 | **1.00** |
| Herbaceous Vegetation | 0.98 | 0.96 | 0.97 |
| Highway | 0.99 | 0.98 | 0.99 |
| Industrial | 0.99 | 0.99 | 0.99 |
| Pasture | 0.98 | 0.96 | 0.97 |
| Permanent Crop | 0.96 | 0.99 | 0.98 |
| Residential | 0.99 | 0.99 | **0.99** |
| River | 0.98 | 1.00 | 0.99 |
| Sea/Lake | 1.00 | 0.99 | **1.00** |
| **Overall Accuracy** | | | **99%** |

For full methodology details, see [submission/methodology.md](./submission/methodology.md).

---

## 👥 Contributors

- **The Brainstromers** — KaggleHacX '26

---
