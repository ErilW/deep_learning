"""
FULL SWIN-ONLY PIPELINE (224×224 fixed)
Semua model Swin yang kompatibel akan training otomatis.
Model yang tidak support 224 akan di-skip otomatis.
"""

import os
import random
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.metrics import (
    f1_score, classification_report
)

warnings.filterwarnings("ignore")

# ============================================================
# SEED
# ============================================================
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


# ============================================================
# Label Smoothing Loss
# ============================================================
LabelSmoothingLoss = lambda weight=None, smoothing=0.1: nn.CrossEntropyLoss(
    weight=weight,
    label_smoothing=smoothing
)


# ============================================================
# DATASET
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

        for root, _, files in os.walk(self.img_root):
            for f in files:
                if f.startswith(image_id):
                    return os.path.join(root, f)

        raise FileNotFoundError(f"{image_id} not found")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = str(row["image_id"])
        label = int(row["label_idx"])

        path = self._find_image_path(image_id)
        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(image=np.array(img))["image"]
        else:
            img = ToTensorV2()(image=np.array(img))["image"]

        return img, label


# ============================================================
# TRAINER
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

        self.best_f1 = -1

    # -------------------------------------
    def train(self, epochs=3):
        for ep in range(epochs):
            self.model.train()
            total_loss = 0

            loop = tqdm(self.train_loader, desc=f"Epoch {ep+1}/{epochs}")

            for img, label in loop:
                img, label = img.to(self.device), label.to(self.device)
                self.optimizer.zero_grad()

                out = self.model(img)
                loss = self.criterion(out, label)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                loop.set_postfix(loss=total_loss / (len(loop) + 1))

            val_loss, val_f1 = self.evaluate()
            print(f"[EPOCH {ep+1}] TrainLoss={total_loss:.3f} | ValLoss={val_loss:.3f} | F1={val_f1:.3f}")

            if val_f1 > self.best_f1:
                self.best_f1 = val_f1
                torch.save(self.model.state_dict(), f"{self.save_dir}/best.pt")
                print(">> BEST UPDATED")

        self.model.load_state_dict(torch.load(f"{self.save_dir}/best.pt"))
        self.test()

    # -------------------------------------
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

        return total_loss / max(1, len(self.val_loader)), f1_score(trues, preds, average="macro")

    # -------------------------------------
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
# LIST SWIN FIXED 224
# ============================================================
def get_swin_224_only():

    all_models = timm.list_models("swin*")

    allowed = []
    for m in all_models:
        # Hanya Swin-S / Swin-T / Swin-B (V1) yg compatible 224
        if any(x in m for x in ["tiny", "small", "base"]):
            allowed.append(m)

        # Skip pretrained 384 models (large/huge)
        if "large" in m or "huge" in m:
            continue

    allowed = sorted(list(set(allowed)))
    return allowed


# ============================================================
# MAIN
# ============================================================
def main():

    # ========= PATH KAMU =========
    TRAIN_CSV = r"./Dataset HAM1000/train.csv"
    VAL_CSV   = r"./Dataset HAM1000/val_public.csv"
    TEST_CSV  = r"./Dataset HAM1000/test_hidden.csv"
    IMG_ROOT  = r"./root/preprocessed_datasets"

    class_names = ["akiec","bcc","bkl","df","mel","nv","vasc"]
    num_classes = len(class_names)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_tf = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(0.5),
        A.VerticalFlip(0.5),
        A.RandomBrightnessContrast(0.5),
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

    # class weights
    counts = train_ds.df["label_idx"].value_counts().sort_index().values
    class_weights = torch.tensor((1 / (counts + 1e-6)), dtype=torch.float32)

    # Loaders
    train_ld = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2)
    val_ld   = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2)
    test_ld  = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=2)

    # Ambil Swin yg fix 224
    swin_list = get_swin_224_only()

    print("\n=========== SWIN LIST (224×224) ===========")
    for m in swin_list:
        print(m)

    print("\n=========== TRAINING ALL MODEL ===========")

    for variant in swin_list:
        print(f"\n\n==============================")
        print(f"   TRAINING MODEL: {variant}")
        print("==============================")

        save_dir = f"./results_swin/{variant}"
        os.makedirs(save_dir, exist_ok=True)

        try:
            # forcing 224×224
            model = timm.create_model(
                variant,
                pretrained=True,
                num_classes=num_classes,
                img_size=224
            )

            # override config
            model.default_cfg["input_size"] = (3, 224, 224)

        except Exception as e:
            print(f"SKIPPED MODEL: {variant} (reason: {e})")
            continue

        trainer = Trainer(
            model, train_ld, val_ld, test_ld,
            device=device,
            class_weights=class_weights,
            save_dir=save_dir,
            class_names=class_names
        )

        trainer.train(epochs=3)

        print(f">>> DONE: {variant}")


if __name__ == "__main__":
    main()
