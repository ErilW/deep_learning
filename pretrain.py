import datetime
import os
import random
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from sklearn.metrics import f1_score, confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image
import warnings

from utils import notif

import albumentations as A
from albumentations.pytorch import ToTensorV2

warnings.filterwarnings("ignore")

# reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# ================================================================
# Loss
# ================================================================
LabelSmoothingLoss = lambda weight=None, smoothing=0.1: nn.CrossEntropyLoss(weight=weight, label_smoothing=smoothing)

# ================================================================
# Model Factory
# ================================================================
class ModelFactory:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def create(self, name):
        name = name.lower()
        if name == "convnext":
            model = models.convnext_tiny(weights="IMAGENET1K_V1")
            in_feat = model.classifier[2].in_features
            model.classifier[2] = nn.Linear(in_feat, self.num_classes)
        elif name == "efficientnet_v2_m":
            model = models.efficientnet_v2_m(weights="IMAGENET1K_V1")
            in_feat = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_feat, self.num_classes)
        elif name == "densenet":
            model = models.densenet121(weights="IMAGENET1K_V1")
            in_feat = model.classifier.in_features
            model.classifier = nn.Linear(in_feat, self.num_classes)
        else:
            raise ValueError("Unknown model: " + name)
        return model

# ================================================================
# Transforms
# ================================================================
IMG_SIZE = 224
transform_asymmetry = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.Rotate(limit=20, p=0.5),
    A.CoarseDropout(max_holes=1, max_height=30, max_width=30, p=0.4),
    A.GaussNoise(var_limit=(5.0, 20.0), p=0.2),
    A.Normalize(),
    ToTensorV2(),
])
transform_border = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.RandomBrightnessContrast(brightness_limit=0.08, contrast_limit=0.12, p=0.6),
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8,8), p=0.4),
    A.ElasticTransform(alpha=10, sigma=5, p=0.3),
    A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.06, rotate_limit=15, p=0.4),
    A.Normalize(),
    ToTensorV2(),
])
transform_color = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.RandomBrightnessContrast(p=0.6),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.6),
    A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.3),
    A.Normalize(),
    ToTensorV2(),
])
transform_val = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(),
    ToTensorV2(),
])

# ================================================================
# Class-wise augmentation
# ================================================================
def classwise_augment(train_csv, img_root, save_root, target_per_class=2500):
    df = pd.read_csv(train_csv)
    os.makedirs(save_root, exist_ok=True)
    transforms_dict = {
        "asymmetry": transform_asymmetry,
        "border": transform_border,
        "color": transform_color
    }

    done_file = os.path.join(save_root, ".done")
    if os.path.exists(done_file):
        print("Augmentation already done, skipping...")
        return
    print("=== START AUGMENTATION PER CLASS ===")
    for label in sorted(df['label_idx'].unique()):
        df_class = df[df['label_idx'] == label]
        n_current = len(df_class)
        n_needed = max(0, target_per_class - n_current)
        print(f"Label {label}: {n_current} -> target {target_per_class} | Augment needed: {n_needed}")
        if n_needed > 0:
            save_dir = os.path.join(save_root, str(label))
            os.makedirs(save_dir, exist_ok=True)
            for i in range(n_needed):
                row = df_class.sample(1).iloc[0]
                img_id = str(row['image_id'])
                img_path = os.path.join(img_root, img_id + ".jpg")
                if not os.path.exists(img_path):
                    continue
                image = np.array(Image.open(img_path).convert("RGB"))
                aug_name, aug_transform = random.choice(list(transforms_dict.items()))
                augmented = aug_transform(image=image)
                aug_img = Image.fromarray(augmented['image'])
                aug_img.save(os.path.join(save_dir, f"{img_id}_aug{i}_{aug_name}.jpg"))
    with open(done_file, "w") as f:
        f.write("done")
    print("=== AUGMENTATION DONE ===")

# ================================================================
# Dataset class
# ================================================================
class HAM10000Dataset(Dataset):
    def __init__(self, csv_path, img_root, augment_root=None, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_root = img_root
        self.augment_root = augment_root
        self.transform = transform

        self.augmented_files = []
        if augment_root and os.path.exists(augment_root):
            for label_dir in os.listdir(augment_root):
                label_path = os.path.join(augment_root, label_dir)
                if os.path.isdir(label_path):
                    for f in os.listdir(label_path):
                        if f.lower().endswith((".jpg",".png")):
                            self.augmented_files.append((os.path.join(label_path,f), int(label_dir)))
        if self.augmented_files:
            aug_df = pd.DataFrame(self.augmented_files, columns=["img_path","label_idx"])
            self.df = pd.concat([self.df, aug_df], ignore_index=True)
            self.df.reset_index(drop=True, inplace=True)

    def __len__(self):
        return len(self.df)

    def _find_image_path(self, image_id):
        if isinstance(image_id, str) and os.path.exists(image_id):
            return image_id
        exts = ['.jpg', '.jpeg', '.png', '.bmp']
        for ext in exts:
            p = os.path.join(self.img_root, image_id + ext)
            if os.path.exists(p):
                return p
        for root, _, files in os.walk(self.img_root):
            for f in files:
                name, _ = os.path.splitext(f)
                if name == image_id or name.startswith(image_id) or image_id in name:
                    return os.path.join(root, f)
        raise FileNotFoundError(f"Image for id {image_id} not found under {self.img_root}")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row.get("image_id", None)
        img_path = row.get("img_path", None)
        if img_path is None and image_id is not None:
            img_path = self._find_image_path(str(image_id))
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            augmented = self.transform(image=np.array(image))
            img = augmented['image']
            img_tensor = transforms.ToTensor()(Image.fromarray(img)) if isinstance(img, np.ndarray) else img
        else:
            img_tensor = transforms.ToTensor()(image)
        label = int(row['label_idx'])
        return img_tensor, label

# ================================================================
# Utility: compute class weights
# ================================================================
def compute_class_weights_from_csv(train_csv):
    df = pd.read_csv(train_csv)
    counts = df['label_idx'].value_counts().sort_index()
    counts = counts.reindex(range(counts.index.min(), counts.index.max()+1), fill_value=0)
    counts = counts.values
    counts = np.where(counts == 0, 1, counts)
    weights = 1.0 / counts.astype(np.float32)
    weights = weights / weights.sum() * len(weights)
    return torch.tensor(weights, dtype=torch.float32)

# ================================================================
# MAIN PIPELINE
# ================================================================
if __name__ == "__main__":
    train_csv = r"./Dataset HAM1000/train.csv"
    val_csv = r"./Dataset HAM1000/val_public.csv"
    test_csv = r"./Dataset HAM1000/test_hidden.csv"
    img_root = r"./root/preprocessed_datasets"
    augment_root = r"./augmented_dataset"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # jalankan augmentasi aman
    classwise_augment(train_csv, img_root, augment_root, target_per_class=2500)

    # siapkan dataset dan loader
    df_train = pd.read_csv(train_csv)
    unique_labels = sorted(df_train['label_idx'].unique().tolist())
    class_names = [f"class_{int(l)}" for l in unique_labels]
    num_classes = len(unique_labels)
    factory = ModelFactory(num_classes=num_classes)
    weights = compute_class_weights_from_csv(train_csv)

    model_names = ["convnext", "efficientnet_v2_m", "densenet"]
    model_configs = {
        "convnext": {"transform": transform_color, "batch_size": 32, "epochs": 50, "optimizer_cfg": {"name": "adamw", "lr": 1e-4, "weight_decay": 1e-2}, "scheduler_cfg": {"type": "cosine", "T_max": 20}},
        "efficientnet_v2_m": {"transform": transform_asymmetry, "batch_size": 32, "epochs": 50, "optimizer_cfg": {"name": "adamw", "lr": 1e-4, "weight_decay": 1e-3}, "scheduler_cfg": {"type": "cosine", "T_max": 18}},
        "densenet": {"transform": transform_border, "batch_size": 32, "epochs": 50, "optimizer_cfg": {"name": "adamw", "lr": 1e-4, "weight_decay": 1e-3}, "scheduler_cfg": {"type": "cosine", "T_max": 16}},
    }

    summary_results = []

    from trainer import Trainer  # pakai trainer lama yang sudah ada

    for model_name in model_names:
        cfg = model_configs[model_name]
        print(f"\n========== TRAINING {model_name.upper()} ==========")
        timer = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = f"./results_{timer}_{model_name}"
        os.makedirs(save_dir, exist_ok=True)

        train_set = HAM10000Dataset(train_csv, img_root, augment_root=augment_root, transform=cfg["transform"])
        val_set = HAM10000Dataset(val_csv, img_root, transform=transform_val)
        test_set = HAM10000Dataset(test_csv, img_root, transform=transform_val)

        batch_size = cfg["batch_size"]
        num_workers = min(8, os.cpu_count() or 4)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

        cnn = factory.create(model_name)
        trainer = Trainer(model=cnn, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, device=device,
                          class_weights=weights, save_dir=save_dir, class_names=class_names,
                          optimizer_cfg=cfg["optimizer_cfg"], scheduler_cfg=cfg["scheduler_cfg"], use_amp=True)

        history = trainer.train(epochs=cfg["epochs"])
        summary_results.append([model_name, history["f1_macro"][-1], history["val_loss"][-1]])
        trainer.evaluate()

    print("\n===== FINAL SUMMARY TABLE =====")
    print("Model | F1 Macro | Val Loss")
    for row in summary_results:
        print(f"{row[0]:15s} | {row[1]:.4f} | {row[2]:.4f}")

    notif("DONE", str(summary_results))
