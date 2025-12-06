# edited_pipeline_no_freeze_no_multimodal.py
import datetime
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.metrics import f1_score, confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image
import warnings

from utils import notif

warnings.filterwarnings("ignore")

# albumentations
import albumentations as A
from albumentations.pytorch import ToTensorV2

# reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# ================================================================
#  Loss: Class-weighted CrossEntropy with Label Smoothing
#  Reason: stable, well-understood, easier to tune than focal for
#  whole-network fine-tuning. See comments below.
# ================================================================
# we'll use nn.CrossEntropyLoss with weight and label_smoothing param (PyTorch >=1.10)
# If your torch doesn't support label_smoothing, replace with CrossEntropyLoss(weight=...) only.
# label_smoothing chosen = 0.1 (empirically common starting point)
LabelSmoothingLoss = lambda weight=None, smoothing=0.1: nn.CrossEntropyLoss(weight=weight, label_smoothing=smoothing)

# ================================================================
#  Model Factory (unchanged behavior except we'll always train full model)
# ================================================================
class ModelFactory:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def create(self, name):
        name = name.lower()

        if name == "convnext":
            model = models.convnext_tiny(weights="IMAGENET1K_V1")
            # replace head
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

        elif name == "resnet50":
            model = models.resnet50(weights="IMAGENET1K_V2")
            in_feat = model.fc.in_features
            model.fc = nn.Linear(in_feat, self.num_classes)

        else:
            raise ValueError("Unknown model: " + name)

        return model

# ================================================================
#  Simple image-only dataset that accepts albumentations or torchvision transforms
# ================================================================
class HAM10000Dataset(Dataset):
    def __init__(self, csv_path, img_root, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_root = img_root
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def _find_image_path(self, image_id):
        exts = ['.jpg', '.jpeg', '.png', '.bmp']
        # quick direct checks first
        for ext in exts:
            p = os.path.join(self.img_root, image_id + ext)
            if os.path.exists(p):
                return p
        # fallback: scan root (could be slow if huge)
        for root, _, files in os.walk(self.img_root):
            for f in files:
                name, _ = os.path.splitext(f)
                if name == image_id or name.startswith(image_id) or image_id in name:
                    return os.path.join(root, f)
        raise FileNotFoundError(f"Image for id {image_id} not found under {self.img_root}")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = str(row['image_id'])
        img_path = self._find_image_path(image_id)
        image = Image.open(img_path).convert("RGB")
        # apply transform: support albumentations and torchvision
        if self.transform is None:
            img_tensor = transforms.ToTensor()(image)
        else:
            try:
                augmented = self.transform(image=np.array(image))
                img = augmented['image']
                if isinstance(img, np.ndarray):
                    img_tensor = transforms.ToTensor()(Image.fromarray(img))
                else:
                    img_tensor = img
            except Exception:
                # assume torchvision transform callable
                img_tensor = self.transform(image)
        label = int(row['label_idx'])
        return img_tensor, label

# ================================================================
#  Trainer: simplified for image-only models, no freeze/unfreeze
# ================================================================
class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, device, class_weights, save_dir, class_names, optimizer_cfg=None, scheduler_cfg=None, use_amp=True):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.save_dir = save_dir
        self.class_names = class_names
        # use label smoothing + class weights
        self.criterion = LabelSmoothingLoss(weight=class_weights.to(device), smoothing=0.1)
        self.best_f1 = -1
        os.makedirs(save_dir, exist_ok=True)

        # optimizer
        params = [p for p in self.model.parameters() if p.requires_grad]
        if optimizer_cfg is None:
            self.optimizer = AdamW(params, lr=1e-4, weight_decay=1e-5)
        else:
            opt_name = optimizer_cfg.get("name", "adamw").lower()
            lr = optimizer_cfg.get("lr", 1e-4)
            wd = optimizer_cfg.get("weight_decay", 1e-5)
            if opt_name == "adamw":
                self.optimizer = AdamW(params, lr=lr, weight_decay=wd)
            else:
                self.optimizer = optim.Adam(params, lr=lr, weight_decay=wd)

        # scheduler
        if scheduler_cfg is None:
            self.scheduler = None
        else:
            stype = scheduler_cfg.get("type")
            if stype == "cosine":
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=scheduler_cfg.get("T_max", 10))
            elif stype == "step":
                self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=scheduler_cfg.get("step_size", 5), gamma=scheduler_cfg.get("gamma", 0.1))
            else:
                self.scheduler = None

        self.use_amp = use_amp
        self.scaler = torch.cuda.amp.GradScaler() if (use_amp and torch.cuda.is_available()) else None

    def train(self, epochs=3):
        print(f"\nDevice: {self.device}")
        history = {"train_loss": [], "val_loss": [], "f1_macro": []}
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            for imgs, labels in tqdm(self.train_loader, desc=f"Train Epoch {epoch+1}/{epochs}"):
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(imgs)
                        loss = self.criterion(outputs, labels)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(imgs)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
                total_loss += loss.item()
            if self.scheduler is not None:
                try:
                    self.scheduler.step()
                except Exception:
                    pass
            val_loss, f1_macro = self.evaluate()
            history["train_loss"].append(total_loss / len(self.train_loader))
            history["val_loss"].append(val_loss)
            history["f1_macro"].append(f1_macro)
            print(f"Train Loss: {total_loss:.4f} | Val Loss: {val_loss:.4f} | F1 Macro: {f1_macro:.4f}")
            if f1_macro > self.best_f1:
                self.best_f1 = f1_macro
                torch.save(self.model.state_dict(), f"{self.save_dir}/best_model.pt")
                print("Saved BEST MODEL")
        self.save_history(history)
        self.test()
        return history

    def evaluate(self):
        self.model.eval()
        preds, trues = [], []
        total_loss = 0.0
        with torch.no_grad():
            for imgs, labels in tqdm(self.val_loader, desc="Validation"):
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                outputs = self.model(imgs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                trues.extend(labels.cpu().numpy())
        f1_macro = f1_score(trues, preds, average="macro")
        self.save_confusion(trues, preds, "val")
        return total_loss / len(self.val_loader), f1_macro

    def test(self):
        print("Running TEST evaluation")
        self.model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for imgs, labels in tqdm(self.test_loader, desc="Testing"):
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                outputs = self.model(imgs)
                preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                trues.extend(labels.cpu().numpy())
        acc = accuracy_score(trues, preds)
        prec, rec, f1, support = precision_recall_fscore_support(trues, preds, zero_division=0)
        macro_f1 = f1_score(trues, preds, average="macro")
        report = classification_report(trues, preds, target_names=self.class_names)
        print("\n===== TEST METRICS =====")
        print(report)
        print(f"Accuracy: {acc:.4f}  Macro F1: {macro_f1:.4f}")
        with open(f"{self.save_dir}/classification_report.txt", "w") as f:
            f.write(report)
        self.save_confusion(trues, preds, "test")

    def save_confusion(self, y_true, y_pred, stage):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(7, 6))
        sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
        plt.title(f"Confusion Matrix ({stage})")
        plt.savefig(f"{self.save_dir}/confusion_matrix_{stage}.png")
        plt.close()

    def save_history(self, history):
        plt.figure(figsize=(7,5))
        plt.plot(history["train_loss"], label="Train Loss")
        plt.plot(history["val_loss"], label="Val Loss")
        plt.legend()
        plt.title("Training History")
        plt.savefig(f"{self.save_dir}/history.png")
        plt.close()

# ================================================================
#  Transforms (unchanged)
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
#  Utility: compute class weights from train csv (pytorch tensor)
# ================================================================
def compute_class_weights_from_csv(train_csv):
    df = pd.read_csv(train_csv)
    counts = df['label_idx'].value_counts().sort_index()
    counts = counts.reindex(range(counts.index.min(), counts.index.max()+1), fill_value=0)
    counts = counts.values
    counts = np.where(counts == 0, 1, counts)
    weights = 1.0 / counts.astype(np.float32)
    weights = weights / weights.sum() * len(weights)  # normalize to roughly num_classes scale
    return torch.tensor(weights, dtype=torch.float32)

# ================================================================
#  MAIN
# ================================================================
if __name__ == "__main__":
    # Paths
    train_csv = r"./Dataset HAM1000/train.csv"
    val_csv   = r"./Dataset HAM1000/val_public.csv"
    test_csv  = r"./Dataset HAM1000/test_hidden.csv"
    img_root  = r"./root/preprocessed_datasets"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_names = [
        "convnext",
        "efficientnet_v2_m",
        "densenet"
    ]

    model_configs = {
        "convnext": {
            "transform": transform_color,
            "batch_size": 32,
            "epochs": 50,
            "optimizer_cfg": {"name": "adamw", "lr": 1e-4, "weight_decay": 1e-2},
            "scheduler_cfg": {"type": "cosine", "T_max": 20}
        },
        "efficientnet_v2_m": {
            "transform": transform_asymmetry,
            "batch_size": 32,
            "epochs": 50,
            "optimizer_cfg": {"name": "adamw", "lr": 1e-4, "weight_decay": 1e-3},
            "scheduler_cfg": {"type": "cosine", "T_max": 18}
        },
        "densenet": {
            "transform": transform_border,
            "batch_size": 32,
            "epochs": 50,
            "optimizer_cfg": {"name": "adamw", "lr": 1e-4, "weight_decay": 1e-3},
            "scheduler_cfg": {"type": "cosine", "T_max": 16}
        }
    }

    # prepare
    df_train = pd.read_csv(train_csv)
    unique_labels = sorted(df_train['label_idx'].unique().tolist())
    class_names = [f"class_{int(l)}" for l in unique_labels]
    num_classes = len(unique_labels)
    factory = ModelFactory(num_classes=num_classes)

    summary_results = []

    for model_name in model_names:
        cfg = model_configs[model_name]
        print(f"\n========== TRAINING {model_name.upper()} ==========")
        timer = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = f"./results_{timer}_{model_name}"
        os.makedirs(save_dir, exist_ok=True)

        train_set = HAM10000Dataset(train_csv, img_root, transform=cfg["transform"])
        val_set   = HAM10000Dataset(val_csv, img_root, transform=transform_val)
        test_set  = HAM10000Dataset(test_csv, img_root, transform=transform_val)

        # class weights computed from CSV
        weights = compute_class_weights_from_csv(train_csv)

        batch_size = cfg["batch_size"]
        num_workers = min(8, os.cpu_count() or 4)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

        cnn = factory.create(model_name)
        model = cnn  # image-only model
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            class_weights=weights,
            save_dir=save_dir,
            class_names=class_names,
            optimizer_cfg=cfg["optimizer_cfg"],
            scheduler_cfg=cfg["scheduler_cfg"],
            use_amp=True
        )

        history = trainer.train(epochs=cfg["epochs"])
        summary_results.append([model_name, history["f1_macro"][-1], history["val_loss"][-1]])
        trainer.evaluate()

    print("\n===== FINAL SUMMARY TABLE =====")
    print("Model | F1 Macro | Val Loss")
    for row in summary_results:
        print(f"{row[0]:15s} | {row[1]:.4f} | {row[2]:.4f}")

    notif("DONE", str(summary_results))
