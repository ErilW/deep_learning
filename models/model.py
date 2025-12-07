import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import classification_report, f1_score, accuracy_score
from timm import create_model
import cv2
from tqdm import tqdm

# ==============================
# CONFIG
# ==============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224
BATCH_SIZE = 16
MODEL_NAME = "swin_base_patch4_window7_224"
NUM_CLASSES = 7  # sesuaikan
DATA_DIR = r"C:\Users\User\Documents\deep_learning\root\preprocessed_datasets\test"  # folder test set
MODEL_PATH = r"C:\Users\User\Downloads\best_model_state_dict.pth"

# ==============================
# HAIR REMOVAL
# ==============================
def remove_hair(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17,17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    inpainted = cv2.inpaint(img, hair_mask, 1, cv2.INPAINT_TELEA)
    return inpainted

# ==============================
# TEST TRANSFORMS
# ==============================
test_transforms = A.Compose([
    A.Lambda(image=remove_hair),
    A.Resize(IMG_SIZE, IMG_SIZE),
    ToTensorV2()
])

# ==============================
# DATASET
# ==============================
class TestDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.paths = []
        self.labels = []
        self.transform = transform
        self.class_names = sorted(os.listdir(folder))
        self.class_to_idx = {cls:i for i, cls in enumerate(self.class_names)}

        for cls in self.class_names:
            cls_dir = os.path.join(folder, cls)
            for fname in os.listdir(cls_dir):
                self.paths.append(os.path.join(cls_dir, fname))
                self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        label = self.labels[idx]
        image = np.array(Image.open(img_path).convert("RGB"))
        if self.transform:
            image = self.transform(image=image)['image']
        return image, label

# ==============================
# LOAD TEST DATA
# ==============================
test_dataset = TestDataset(DATA_DIR, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# ==============================
# LOAD MODEL
# ==============================
model = create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# ==============================
# PREDICT
# ==============================
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# ==============================
# METRICS
# ==============================
f1_macro = f1_score(all_labels, all_preds, average='macro')
f1_weighted = f1_score(all_labels, all_preds, average='weighted')
acc = accuracy_score(all_labels, all_preds)

print(f"Test Accuracy: {acc:.4f}")
print(f"F1 Macro: {f1_macro:.4f}")
print(f"F1 Weighted: {f1_weighted:.4f}")

print("\nClassification Report:")
class_names = test_dataset.class_names
print(classification_report(all_labels, all_preds, target_names=class_names))
