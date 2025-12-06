"""
FULL SWIN-ONLY PIPELINE — TRAIN + EVAL PER VARIANT
Menggunakan format code pipeline kamu, tetapi hanya fokus Swin timm.
Loop otomatis dari model terkecil → terbesar.
"""

import os
import random
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2

from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    f1_score,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    classification_report
)

import timm
warnings.filterwarnings("ignore")

# ============================================================
# SEED
# ============================================================
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# ============================================================
# Label Smoothing Loss
# ============================================================
LabelSmoothingLoss = lambda weight=None, smoothing=0.1: nn.CrossEntropyLoss(
    weight=weight,
    label_smoothing=smoothing
)

# ============================================================
# Dataset
# ============================================================
class HAM10000Dataset(Dataset):
    def __init__(self, csv_path, img_root, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_root = img_root
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def _find_image_path(self, image_id):
        exts = ['.jpg', '.jpeg', '.png']
        for ext in exts:
            p = os.path.join(self.img_root, image_id + ext)
            if os.path.exists(p):
                return p
        # fallback OS walk
        for root, _, files in os.walk(self.img_root):
            for f in files:
                name, _ = os.path.splitext(f)
                if name == image_id or image_id in name:
                    return os.path.join(root, f)
        raise FileNotFoundError(f"{image_id} not found")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = str(row['image_id'])
        label = int(row['label_idx'])

        path = self._find_image_path(image_id)
        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(image=np.array(img))["image"]
        else:
            img = transforms.ToTensor()(img)

        return img, label

# ============================================================
# Trainer
# ============================================================
class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader,
                 device, class_weights, save_dir, class_names):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.class_names = class_names
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.criterion = LabelSmoothingLoss(class_weights.to(device), 0.1)

        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(params, lr=1e-4, weight_decay=1e-4)

        self.scaler = torch.cuda.amp.GradScaler()

        self.best_f1 = -1

    # --------------- TRAIN LOOP ----------------------
    def train(self, epochs=3):
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            loop = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

            for img, label in loop:
                img, label = img.to(self.device), label.to(self.device)
                self.optimizer.zero_grad()

                with torch.cuda.amp.autocast():
                    out = self.model(img)
                    loss = self.criterion(out, label)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                total_loss += loss.item()
                loop.set_postfix(loss=total_loss / (len(loop)+1))

            # validation
            val_loss, f1_macro = self.evaluate()

            print(f"Epoch {epoch+1} → TrainLoss={total_loss:.3f} | ValLoss={val_loss:.3f} | F1={f1_macro:.3f}")

            # simpan best
            if f1_macro > self.best_f1:
                self.best_f1 = f1_macro
                torch.save(self.model.state_dict(), f"{self.save_dir}/best.pt")
                print(">> BEST MODEL UPDATED")

        # TEST pakai best model
        self.model.load_state_dict(torch.load(f"{self.save_dir}/best.pt", map_location=self.device))
        self.test()

    # --------------- EVAL ----------------------
    def evaluate(self):
        self.model.eval()
        preds, trues = [], []
        total_loss = 0

        with torch.no_grad():
            for img, label in self.val_loader:
                img, label = img.to(self.device), label.to(self.device)
                out = self.model(img)
                loss = self.criterion(out, label)
                total_loss += loss.item()

                preds += torch.argmax(out, 1).cpu().numpy().tolist()
                trues += label.cpu().numpy().tolist()

        f1_macro = f1_score(trues, preds, average="macro")
        return total_loss / max(1, len(self.val_loader)), f1_macro

    # --------------- TEST ----------------------
    def test(self):
        self.model.eval()
        preds, trues = [], []

        with torch.no_grad():
            for img, label in self.test_loader:
                img, label = img.to(self.device), label.to(self.device)
                out = self.model(img)

                preds += torch.argmax(out, 1).cpu().numpy().tolist()
                trues += label.cpu().numpy().tolist()

        print("\n===== FINAL TEST =====")
        print(classification_report(trues, preds, target_names=self.class_names))
        print("MACRO F1:", f1_score(trues, preds, average="macro"))


# ============================================================
# HELPER UNTUK SWIN VARIANTS
# ============================================================
def get_swin_variants_sorted():
    variants = timm.list_models("swin*")

    # Urutkan berdasarkan ukuran model (timm sudah mengurutkan dari kecil → besar)
    variants = sorted(variants)

    # Buang variant yang bukan classifier (patch4/patch2)
    clean = []
    for v in variants:
        if "in21k" in v or "in22k" in v:
            continue
        clean.append(v)

    return clean


# ============================================================
# MAIN
# ============================================================
def main():

    # =====================================================
    # EDIT PATH SESUAI KAMU
    # =====================================================
    TRAIN_CSV =  r"./Dataset HAM1000/train.csv"
    VAL_CSV   =  r"./Dataset HAM1000/val_public.csv"
    TEST_CSV  =  r"./Dataset HAM1000/test_hidden.csv"
    IMG_ROOT  =  r"./root/preprocessed_datasets"

    class_names = ["akiec","bcc","bkl","df","mel","nv","vasc"]
    num_classes = len(class_names)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Albumentations
    train_tf = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        ToTensorV2()
    ])

    test_tf = A.Compose([
        A.Resize(224, 224),
        ToTensorV2()
    ])

    # Dataset
    train_ds = HAM10000Dataset(TRAIN_CSV, IMG_ROOT, train_tf)
    val_ds   = HAM10000Dataset(VAL_CSV, IMG_ROOT, test_tf)
    test_ds  = HAM10000Dataset(TEST_CSV, IMG_ROOT, test_tf)

    # class weight
    counts = train_ds.df["label_idx"].value_counts().sort_index().values
    class_weights = torch.tensor((1 / (counts + 1e-6)), dtype=torch.float32)

    # Loader
    train_ld = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2)
    val_ld   = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2)
    test_ld  = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=2)

    # =====================================================
    # LOOP SWIN VARIANT
    # =====================================================
    swin_list = get_swin_variants_sorted()

    print("\n=========== LIST SWIN ===========")
    for v in swin_list:
        print(v)

    print("\n=========== TRAINING EACH MODEL ===========")

    for variant in swin_list:
        print(f"\n\n==============================")
        print(f"   TRAINING MODEL: {variant}")
        print("==============================")

        save_dir = f"./results_swin/{variant}"
        os.makedirs(save_dir, exist_ok=True)

        # CREATE MODEL
        model = timm.create_model(
            variant,
            pretrained=True,
            num_classes=num_classes
        )

        trainer = Trainer(
            model, train_ld, val_ld, test_ld,
            device=device,
            class_weights=class_weights,
            save_dir=save_dir,
            class_names=class_names
        )

        trainer.train(epochs=3)     # <-- ubah sesuai kebutuhan

        print(f">>> DONE: {variant}")


if __name__ == "__main__":
    main()
