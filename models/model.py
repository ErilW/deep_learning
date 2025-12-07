# swinv2_full_pipeline.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import f1_score, accuracy_score, classification_report
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from timm import create_model

# ==============================
# CONFIGURATION
# ==============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
IMG_SIZE = 224
EPOCHS = 30
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.0005
MOMENTUM = 0.937

# Model options: tiny, small, base
MODEL_NAME = "swinv2_tiny_patch4_window8_256"
DATA_DIR = r"..\Dataset HAM1000\segmentation_masks"  # ganti sesuai dataset
SAVE_DIR = "saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)

# ==============================
# AUGMENTATIONS (replicate notebook)
# ==============================
train_transforms = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomRotate90(p=0.3),
    A.Transpose(p=0.2),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3),
    A.CLAHE(clip_limit=4.0, p=0.3),
    A.Blur(blur_limit=3, p=0.1),
    A.MotionBlur(blur_limit=3, p=0.1),
    A.CoarseDropout(max_holes=8, max_height=IMG_SIZE//10, max_width=IMG_SIZE//10, p=0.2),
    A.Cutout(num_holes=8, max_h_size=IMG_SIZE//10, max_w_size=IMG_SIZE//10, p=0.2),
    ToTensorV2()
])

val_transforms = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    ToTensorV2()
])

# ==============================
# CUSTOM DATASET
# ==============================
class CustomImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = np.array(Image.open(img_path).convert("RGB"))
        if self.transform:
            image = self.transform(image=image)['image']
        return image, label

# ==============================
# LOAD DATA
# ==============================
def load_dataset(data_dir):
    classes = sorted(os.listdir(os.path.join(data_dir, "train")))
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    def get_paths_and_labels(split):
        paths, labels = [], []
        split_dir = os.path.join(data_dir, split)
        for cls in classes:
            cls_dir = os.path.join(split_dir, cls)
            for fname in os.listdir(cls_dir):
                paths.append(os.path.join(cls_dir, fname))
                labels.append(class_to_idx[cls])
        return paths, labels

    train_paths, train_labels = get_paths_and_labels("train")
    val_paths, val_labels = get_paths_and_labels("val")

    train_dataset = CustomImageDataset(train_paths, train_labels, transform=train_transforms)
    val_dataset = CustomImageDataset(val_paths, val_labels, transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    return train_loader, val_loader, len(classes)

train_loader, val_loader, num_classes = load_dataset(DATA_DIR)
print(f"Number of classes: {num_classes}")

# ==============================
# MODEL
# ==============================
model = create_model(MODEL_NAME, pretrained=True, num_classes=num_classes)
model = model.to(DEVICE)

# ==============================
# LOSS, OPTIMIZER, SCHEDULER
# ==============================
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ==============================
# TRAINING LOOP
# ==============================
best_f1 = 0.0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Training"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {epoch_loss:.4f}")

    # ==============================
    # VALIDATION
    # ==============================
    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Validation"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(val_labels, val_preds)
    f1 = f1_score(val_labels, val_preds, average="macro")
    print(f"Validation Accuracy: {acc:.4f}, F1 Macro: {f1:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"{MODEL_NAME}_best.pth"))
        print("Saved best model!")

    scheduler.step()

# ==============================
# SAVE FINAL MODEL
# ==============================
torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"{MODEL_NAME}_final.pth"))
print("Training complete. Final model saved!")
