import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import os
from PIL import Image

# Dataset Path
DATA_DIR = r"C:\Users\niket\Documents\Hackathon-103\EuroSAT\EuroSAT"

class EuroSATDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
        # Get unique classes and create a mapping to indices
        self.classes = sorted(self.data_frame['ClassName'].unique())
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 1]) # Column 1 is Filename
        image = Image.open(img_name).convert("RGB")
        
        # We can use the Label column directly, but let's map it securely
        class_name = self.data_frame.iloc[idx, 3] # Column 3 is ClassName
        label = self.class_to_idx[class_name]

        if self.transform:
            image = self.transform(image)

        # For test set, we might also need the image id (filename) for the submission
        return image, label, os.path.basename(img_name)

def get_dataloaders(batch_size=32):
    print("Loading data...")
    if not os.path.exists(DATA_DIR):
        print(f"Data directory not found at {DATA_DIR}")
        return None, None, None

    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = EuroSATDataset(csv_file=os.path.join(DATA_DIR, "train.csv"),
                                   root_dir=DATA_DIR, transform=train_transforms)
    
    val_dataset = EuroSATDataset(csv_file=os.path.join(DATA_DIR, "validation.csv"),
                                 root_dir=DATA_DIR, transform=val_transforms)
                                 
    test_dataset = EuroSATDataset(csv_file=os.path.join(DATA_DIR, "test.csv"),
                                  root_dir=DATA_DIR, transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def build_model(num_classes):
    print("Building model (EfficientNet-B0)...")
    model = torchvision.models.efficientnet_b0(pretrained=True)
    
    # Phase 1: Freeze backbone, replace classifier head
    for param in model.parameters():
        param.requires_grad = False
        
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    return model

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    # 1. Dataloaders
    train_loader, val_loader, test_loader = get_dataloaders()
    if not train_loader:
        return
    
    num_classes = len(train_loader.dataset.classes)
    
    # 2. Model
    model = build_model(num_classes).to(device)
    
    # 3. Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-3)
    epochs = 15 # Phase 1 + Phase 2
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    # 4. Training Loop
    for epoch in range(epochs):
        if epoch == 5:
            print("--- Unfreezing all layers for fine-tuning ---")
            for param in model.parameters():
                param.requires_grad = True
            optimizer = Adam(model.parameters(), lr=1e-4) # Lower LR for fine-tuning
            scheduler = CosineAnnealingLR(optimizer, T_max=epochs-5)

        model.train()
        running_loss = 0.0
        for i, (images, labels, _) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            
            if (i+1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
            
        scheduler.step()
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")

        # Validation
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        print(f"Validation for Epoch {epoch+1}:")
        print(classification_report(all_labels, all_preds, target_names=train_loader.dataset.classes, zero_division=0))
        
    # Save the model
    torch.save(model, "model.pth")
    print("Model saved to model.pth")

    # Generate Predictions & Format Submission
    print("Generating predictions for test set...")
    model.eval()
    results = []
    with torch.no_grad():
        for images, labels, img_names in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            
            for pred, img_name in zip(preds, img_names):
                results.append({"image_id": img_name, "label": train_loader.dataset.classes[pred]})

    df = pd.DataFrame(results)
    os.makedirs("../solution", exist_ok=True)
    df.to_csv("../solution/solution_teamname.csv", index=False)
    print("Submission saved to ../solution/solution_teamname.csv")

if __name__ == "__main__":
    train()
