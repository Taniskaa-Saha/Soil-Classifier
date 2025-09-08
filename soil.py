# main.py  -- corrected for local/VS Code runs (no Colab)

import os
import zipfile
import random
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import timm
from PIL import Image
import json

if __name__ == "__main__":
    # ========== CONFIG ==========
    project_root = os.path.dirname(os.path.abspath(__file__))
    zip_path = os.path.join(project_root, "soil.zip")   # if you have a zip
    data_root = os.path.join(project_root, "soil")      # fixed folder
    images_subfolder = "Soil types"
    images_root = os.path.join(data_root, images_subfolder)

    # Extract ZIP if needed
    if os.path.exists(zip_path) and not os.path.exists(data_root):
        print("Extracting", zip_path, "->", data_root)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(data_root)

    if not os.path.exists(images_root):
        raise FileNotFoundError(f"Images folder not found: {images_root}")

    # ========== Transforms ==========
    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    val_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # ========== Dataset & Split ==========
    full_dataset = datasets.ImageFolder(images_root)
    n_total = len(full_dataset)
    print("Total images found:", n_total)
    print("Classes:", full_dataset.classes)

    indices = list(range(n_total))
    random.seed(42)
    random.shuffle(indices)
    split = int(0.8 * n_total)
    train_idx, val_idx = indices[:split], indices[split:]

    train_dataset = datasets.ImageFolder(images_root, transform=train_tfms)
    val_dataset = datasets.ImageFolder(images_root, transform=val_tfms)

    train_ds = Subset(train_dataset, train_idx)
    val_ds = Subset(val_dataset, val_idx)

    batch_size = 32
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)  # Windows safe
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    num_classes = len(train_dataset.classes)
    print("Num classes:", num_classes)
    print("Train samples:", len(train_ds), "Val samples:", len(val_ds))

    # ========== Model ==========
    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # ========== Training ==========
    epochs = 2  # reduced for quick testing
    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        avg_loss = train_loss / total
        acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}/{epochs}  Train Loss: {avg_loss:.4f}  Train Acc: {acc:.2f}%")

    # ========== Save model ==========
    save_path = os.path.join(project_root, "soil_model.pth")
    torch.save(model.state_dict(), save_path)
    print("✅ Model saved as soil_model.pth")

    # ========== Validation ==========
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    val_acc = 100.0 * val_correct / val_total if val_total > 0 else 0.0
    print(f"Validation Accuracy: {val_acc:.2f}%")

    # ========== Save classes ==========
    with open(os.path.join(project_root, "classes.json"), "w") as f:
        json.dump(train_dataset.classes, f)
    print("✅ classes.json saved")

    # ========== Single-image test (optional) ==========
    test_img_path = os.path.join(project_root, "Screenshot.png")
    if os.path.exists(test_img_path):
        img = Image.open(test_img_path).convert("RGB")
        img_tensor = val_tfms(img).unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            out = model(img_tensor)
            _, pred = out.max(1)
        print("Predicted class:", train_dataset.classes[pred.item()])
    else:
        print("Test image not found — skipping single-image test")
