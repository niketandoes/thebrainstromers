# EuroSAT Land Use & Land Cover Classification

![Python](https://img.shields.io/badge/python-3.14-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11-red.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-FF4B4B.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-99%25-green.svg)

A high-performance deep learning solution for classifying satellite imagery into 10 distinct land cover categories using the EuroSAT dataset. Built during the KaggleHacX '26 Hackathon.

## 🚀 Project Highlights
- **Model Architecture:** EfficientNet-B0 (Transfer Learning)
- **Validation Accuracy:** **99.1%**
- **Training Strategy:** Two-phase fine-tuning (Frozen backbone followed by full unfreezing).
- **Optimization:** Cosine Annealing learning rate scheduler.
- **Inference:** Lightweight Streamlit web application for real-time local classification.

## 📁 Project Structure
```text
thebrainstromers/
├── main/
│   ├── train.py        # Model training and fine-tuning pipeline
│   └── app.py          # Streamlit web application
├── submission/
│   ├── methodology.md  # Detailed technical methodology
│   └── solution.csv    # Final test set predictions (auto-generated)
├── model.pth           # Trained model weights (generated after training)
├── requirements.txt    # Project dependencies
└── README.md           # This file
```

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/niketandoes/thebrainstromers.git
   cd thebrainstromers
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the Dataset:**
   Ensure the EuroSAT dataset is available locally and paths in `main/train.py` are updated accordingly.

## 🏋️ Training the Model
To start the two-phase training process:
```bash
python main/train.py
```
This script will:
1. Train the classifier head for 5 epochs (Backbone frozen).
2. Fine-tune the entire network for 10 epochs (Backbone unfrozen).
3. Save the best model to `model.pth`.
4. Generate the final competition submission CSV.

## 🌐 Running the Web Application
Once the `model.pth` is generated, launch the interactive dashboard:
```bash
streamlit run main/app.py
```
You can then upload any `.jpg` satellite image to see real-time predictions and confidence scores.

## 📊 Results
Our model achieved consistent performance across all 10 classes:
- **SeaLake:** 1.00 F1-Score
- **Forest:** 0.99 F1-Score
- **Residential:** 0.99 F1-Score
- **Industrial:** 0.99 F1-Score

For a full breakdown of the architecture and training logic, see [methodology.md](./submission/methodology.md).

## 👥 Contributors
- **The Brainstromers** - KaggleHacX '26 participants.
