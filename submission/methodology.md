# EuroSAT Land Use Classification Methodology

## 1. Overview
The goal of this project is to build an accurate, robust image classification model for the **EuroSAT dataset**, which contains 27,000 Sentinel-2 satellite images categorized into 10 distinct land use and land cover classes. 

Our solution prioritizes speed, efficiency, and high accuracy, adhering strictly to the local-execution hackathon constraints without relying on external cloud APIs.

## 2. Data Preparation and Loading
The dataset is structured using mapping CSVs (`train.csv`, `validation.csv`, `test.csv`) rather than standard subfolders. To efficiently feed this data into the model, we designed a **Custom PyTorch Dataset Class**.
- **Images:** Loaded locally from the base directory using the filename column.
- **Labels:** Mapped from string class names to 0-9 integers using the `label_map.json`.
- **Augmentation:** For the training set, we implemented dynamic data augmentations including Random Horizontal Flips, Random Vertical Flips, and Random Rotations. This artificially expands the dataset, helping the model generalize to satellite imagery regardless of orientation.

## 3. Model Architecture
We selected **EfficientNet-B0** as our core architecture.
- **Why EfficientNet-B0?** It provides an optimal balance between parameter efficiency and high accuracy. It drastically reduces training time while capturing complex geospatial textures, which is critical for a 6-hour hackathon.
- **Transfer Learning:** We initialized the model with weights pre-trained on ImageNet. We replaced the final fully-connected layer with a new custom classifier head designed specifically to output predictions for the 10 EuroSAT classes.

## 4. Two-Phase Fine-Tuning Strategy
To avoid destroying the highly valuable pre-trained ImageNet weights during the early stages of training, we utilized a Two-Phase Strategy:

### Phase 1: Feature Extraction (Epochs 1-5)
- **Frozen Backbone:** We froze all layers of the EfficientNet-B0 feature extractor.
- **Head Training:** Only the newly attached 10-class classifier head was trained using a standard Learning Rate. 
- **Result:** Within 5 epochs, the model successfully mapped the frozen ImageNet features to the 10 EuroSAT categories, achieving a baseline **90% validation accuracy**.

### Phase 2: Full Fine-Tuning (Epochs 6-15)
- **Unfrozen Backbone:** We unfroze the entire network.
- **Micro-Adjustments:** We implemented a `CosineAnnealingLR` scheduler to smoothly decay the learning rate. This prevents catastrophic forgetting and allows the model to subtly adjust its deep convolutional filters specifically for satellite textures (e.g., distinguishing between 'Forest' and 'PermanentCrop').
- **Result:** Accuracy rapidly climbed to **98%**, with perfect (1.00) F1 scores on distinctive classes like `SeaLake`.

## 5. Inference and Application
The trained weights (`model.pth`) are saved locally. We built a fully offline **Streamlit Application** that allows users to upload any `.jpg` image. The app loads the custom EfficientNet model, applies the exact same normalization transforms used during training, and outputs the predicted land cover class alongside a confidence probability bar.

## 6. Performance Summary
- **Peak Validation Accuracy:** 98%
- **Loss:** Converged smoothly from 1.32 down to ~0.02.
- **Inference Time:** < 0.5 seconds per image on CPU.
