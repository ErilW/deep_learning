# edited_pipeline_ham10000.py
import datetime
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
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
#  Focal Loss (same as yours)
# ================================================================
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ce = nn.CrossEntropyLoss(weight=weight)

    def forward(self, logits, targets):
        logpt = -self.ce(logits, targets)
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * logpt
        return loss.mean()


# ================================================================
#  Model Factory (unchanged)
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

        elif name == "efficientnet_b3":
            model = models.efficientnet_b3(weights="IMAGENET1K_V1")
            in_feat = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_feat, self.num_classes)

        elif name == "efficientnet_b7":
            model = models.efficientnet_b7(weights="IMAGENET1K_V1")
            in_feat = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_feat, self.num_classes)

        elif name == "efficientnet_v2_s":
            model = models.efficientnet_v2_s(weights="IMAGENET1K_V1")
            in_feat = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_feat, self.num_classes)

        elif name == "efficientnet_v2_m":
            model = models.efficientnet_v2_m(weights="IMAGENET1K_V1")
            in_feat = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_feat, self.num_classes)

        elif name == "efficientnet_v2_l":
            model = models.efficientnet_v2_l(weights="IMAGENET1K_V1")
            in_feat = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_feat, self.num_classes)

        elif name == "resnet50":
            model = models.resnet50(weights="IMAGENET1K_V2")
            in_feat = model.fc.in_features
            model.fc = nn.Linear(in_feat, self.num_classes)

        elif name == "densenet":
            model = models.densenet121(weights="IMAGENET1K_V1")
            in_feat = model.classifier.in_features
            model.classifier = nn.Linear(in_feat, self.num_classes)

        else:
            raise ValueError("Unknown model: " + name)

        return model


# ================================================================
#  Tabular helpers & Multimodal Dataset (modified to accept albumentations)
# ================================================================
def build_tabular_maps(train_csv_path):
    df = pd.read_csv(train_csv_path)
    unique_sex = df['sex'].fillna('unknown').unique().tolist()
    sex_map = {s: i for i, s in enumerate(sorted(unique_sex))}
    unique_loc = df['localization'].fillna('unknown').unique().tolist()
    loc_map = {l: i for i, l in enumerate(sorted(unique_loc))}
    ages = df['age'].dropna().astype(float)
    age_mean = ages.mean() if len(ages) > 0 else 50.0
    age_std = ages.std() if len(ages) > 0 else 20.0
    if age_std == 0:
        age_std = 1.0
    return sex_map, loc_map, age_mean, age_std


class HAM10000MultimodalDataset(torch.utils.data.Dataset):
    """
    Returns: image_tensor, tab_tensor, label
    Accepts albumentations.Compose or torchvision.transforms
    """
    def __init__(self, csv_path, img_root, sex_map, loc_map, age_mean, age_std, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_root = img_root
        self.transform = transform
        self.sex_map = sex_map
        self.loc_map = loc_map
        self.age_mean = age_mean
        self.age_std = age_std
        self.sex_dim = len(self.sex_map)
        self.loc_dim = len(self.loc_map)
        self.tab_dim = 1 + self.sex_dim + self.loc_dim

    def __len__(self):
        return len(self.df)

    def _find_image_path(self, image_id):
        exts = ['.jpg', '.jpeg', '.png', '.bmp']
        possible = []
        for root, _, files in os.walk(self.img_root):
            for f in files:
                name, ext = os.path.splitext(f)
                if name == image_id or name.startswith(image_id):
                    return os.path.join(root, f)
                if image_id in name:
                    possible.append(os.path.join(root, f))
        if possible:
            return possible[0]
        for ext in exts:
            p = os.path.join(self.img_root, image_id + ext)
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"Image for id {image_id} not found under {self.img_root}")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = str(row['image_id'])
        img_path = self._find_image_path(image_id)
        image = Image.open(img_path).convert("RGB")
        # ---> apply transform: support albumentations and torchvision
        if self.transform is None:
            img_tensor = transforms.ToTensor()(image)
        else:
            try:
                # albumentations expects numpy array and returns dict with 'image' key
                augmented = self.transform(image=np.array(image))
                img = augmented['image']
                # img is torch tensor if ToTensorV2 used
                if isinstance(img, np.ndarray):
                    # fallback: convert to tensor
                    img_tensor = transforms.ToTensor()(Image.fromarray(img))
                else:
                    img_tensor = img
            except Exception:
                # assume torchvision transform
                img_tensor = self.transform(image)

        # TABULAR
        try:
            age = float(row['age'])
            if np.isnan(age):
                age = self.age_mean
        except Exception:
            age = self.age_mean
        age_norm = (age - self.age_mean) / self.age_std
        sex = str(row.get('sex', 'unknown'))
        sex_idx = self.sex_map.get(sex, self.sex_map.get('unknown', 0))
        sex_oh = np.zeros(self.sex_dim, dtype=np.float32); sex_oh[sex_idx] = 1.0
        loc = str(row.get('localization', 'unknown'))
        loc_idx = self.loc_map.get(loc, self.loc_map.get('unknown', 0))
        loc_oh = np.zeros(self.loc_dim, dtype=np.float32); loc_oh[loc_idx] = 1.0
        tab_vec = np.concatenate([[age_norm], sex_oh, loc_oh]).astype(np.float32)
        tab_tensor = torch.tensor(tab_vec)
        label = int(row['label_idx'])
        return img_tensor, tab_tensor, label


# ================================================================
#  Tabular encoder, prior, fusion (unchanged)
# ================================================================
class TabularEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, out_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU()
        )
    def forward(self, x):
        return self.net(x)

class PriorModule(nn.Module):
    def __init__(self, tab_input_dim, num_classes, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(tab_input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes)
        )
    def forward(self, tab):
        return self.net(tab)

class MultiModalFusion(nn.Module):
    def __init__(self, cnn_model, num_classes, tabular_dim=10, tabular_emb_dim=32, use_prior=False):
        super().__init__()
        self.cnn = cnn_model
        feat_dim = None
        if hasattr(self.cnn, "classifier"):
            if isinstance(self.cnn.classifier, nn.Sequential):
                last = None
                for m in reversed(self.cnn.classifier):
                    if isinstance(m, nn.Linear):
                        last = m; break
                if last is not None:
                    feat_dim = last.in_features
        if feat_dim is None and hasattr(self.cnn, "fc"):
            feat_dim = self.cnn.fc.in_features
        if feat_dim is None:
            raise RuntimeError("Cannot determine CNN feature dim.")
        # replace classifier/fc with identity to get features
        if hasattr(self.cnn, "classifier"):
            try:
                self.cnn.classifier = nn.Identity()
            except Exception:
                if isinstance(self.cnn.classifier, nn.Sequential):
                    modules = list(self.cnn.classifier.children())[:-1]
                    self.cnn.classifier = nn.Sequential(*modules)
        elif hasattr(self.cnn, "fc"):
            self.cnn.fc = nn.Identity()

        self.tab_encoder = TabularEncoder(input_dim=tabular_dim, hidden_dim=tabular_emb_dim, out_dim=tabular_emb_dim)
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim + tabular_emb_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        self.use_prior = use_prior
        if self.use_prior:
            self.prior = PriorModule(tabular_dim, num_classes)
            self.prior_alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, img, tab):
        img_feat = self.cnn(img)
        if img_feat.dim() == 4:
            img_feat = torch.flatten(img_feat, 1)
        tab_emb = self.tab_encoder(tab)
        fused = torch.cat([img_feat, tab_emb], dim=1)
        logits = self.classifier(fused)
        if self.use_prior:
            prior_bias = self.prior(tab)
            logits = logits + self.prior_alpha * prior_bias
        return logits


# ================================================================
#  Trainer (modified slightly: accepts optimizer/scheduler configs)
# ================================================================
class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, device, class_weights, save_dir, class_names, multimodal=False, optimizer_cfg=None, scheduler_cfg=None, use_amp=True):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.save_dir = save_dir
        self.class_names = class_names
        self.multimodal = multimodal
        self.criterion = FocalLoss(weight=class_weights.to(device))
        self.best_f1 = -1
        os.makedirs(save_dir, exist_ok=True)

        self.freeze_backbone()
        # build optimizer from cfg or default
        if optimizer_cfg is None:
            trainable = filter(lambda p: p.requires_grad, self.model.parameters())
            self.optimizer = AdamW(trainable, lr=1e-4, weight_decay=1e-5)
        else:
            params = [p for p in self.model.parameters() if p.requires_grad]
            opt_name = optimizer_cfg.get("name", "adamw").lower()
            if opt_name == "adamw":
                self.optimizer = AdamW(params, lr=optimizer_cfg.get("lr", 1e-4), weight_decay=optimizer_cfg.get("weight_decay", 1e-5))
            else:
                self.optimizer = optim.Adam(params, lr=optimizer_cfg.get("lr", 1e-4), weight_decay=optimizer_cfg.get("weight_decay", 1e-5))

        # scheduler
        if scheduler_cfg is None:
            self.scheduler = None
        else:
            if scheduler_cfg.get("type") == "cosine":
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=scheduler_cfg.get("T_max", 10))
            elif scheduler_cfg.get("type") == "step":
                self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=scheduler_cfg.get("step_size", 5), gamma=scheduler_cfg.get("gamma", 0.1))
            else:
                self.scheduler = None

        self.use_amp = use_amp
        self.scaler = torch.cuda.amp.GradScaler() if (use_amp and torch.cuda.is_available()) else None

    def freeze_backbone(self):
        keywords = ("classifier", "fc", "head", "linear")
        for name, param in self.model.named_parameters():
            if any(k in name.lower() for k in keywords):
                param.requires_grad = True
            else:
                param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def train(self, epochs=3):
        print(f"\n🚀 Device digunakan: {self.device.upper()}")
        history = {"train_loss": [], "val_loss": [], "f1_macro": []}
        for epoch in range(epochs):
            # unfreeze after 5 epochs if desired
            if epoch == 5:
                print("🔓 Unfreezing backbone & lowering LR...")
                self.unfreeze_backbone()
                # reconfigure optimizer to include all params with smaller LR
                self.optimizer = AdamW(self.model.parameters(), lr=1e-5, weight_decay=1e-5)
                if self.scheduler is not None:
                    # reconstruct scheduler if cosine
                    try:
                        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs - epoch)
                    except Exception:
                        self.scheduler = None

            self.model.train()
            total_loss = 0.0
            if self.multimodal:
                iterator = self.train_loader
                desc = "Training (multimodal)"
                for imgs, tabs, labels in tqdm(iterator, desc=desc):
                    imgs, tabs, labels = imgs.to(self.device), tabs.to(self.device), labels.to(self.device)
                    self.optimizer.zero_grad()
                    if self.scaler:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(imgs, tabs)
                            loss = self.criterion(outputs, labels)
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        outputs = self.model(imgs, tabs)
                        loss = self.criterion(outputs, labels)
                        loss.backward()
                        self.optimizer.step()
                    total_loss += loss.item()
            else:
                iterator = self.train_loader
                desc = "Training"
                for imgs, labels in tqdm(iterator, desc=desc):
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
            print(f"📊 Train Loss: {total_loss:.4f} | Val Loss: {val_loss:.4f} | F1 Macro: {f1_macro:.4f}")

            if f1_macro > self.best_f1:
                self.best_f1 = f1_macro
                torch.save(self.model.state_dict(), f"{self.save_dir}/best_model.pt")
                print("💾 Model disave (BEST MODEL)")

        self.save_history(history)
        self.test()
        return history

    def evaluate(self):
        self.model.eval()
        preds, trues = [], []
        total_loss = 0.0
        with torch.no_grad():
            if self.multimodal:
                for imgs, tabs, labels in tqdm(self.val_loader, desc="Validation (multimodal)"):
                    imgs, tabs, labels = imgs.to(self.device), tabs.to(self.device), labels.to(self.device)
                    outputs = self.model(imgs, tabs)
                    loss = self.criterion(outputs, labels)
                    total_loss += loss.item()
                    preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                    trues.extend(labels.cpu().numpy())
            else:
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
        print("\n🧪 Running TEST evaluation…")
        self.model.eval()
        preds, trues = [], []
        with torch.no_grad():
            if self.multimodal:
                for imgs, tabs, labels in tqdm(self.test_loader, desc="Testing (multimodal)"):
                    imgs, tabs, labels = imgs.to(self.device), tabs.to(self.device), labels.to(self.device)
                    outputs = self.model(imgs, tabs)
                    preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                    trues.extend(labels.cpu().numpy())
            else:
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
#  Utility: compute class weights
# ================================================================
def compute_class_weights_from_labels(labels_tensor):
    count = torch.bincount(labels_tensor)
    count = torch.where(count==0, torch.ones_like(count), count)
    return 1.0 / count.float()


# ================================================================
#  ALBUMENTATIONS TRANSFORMS (3 variants)
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

# simple val/test transform (deterministic)
transform_val = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(),
    ToTensorV2(),
])


# ================================================================
#  MAIN: configuration per model (hyperparameters and which transform to use)
# ================================================================
if __name__ == "__main__":
    # Paths (sesuaikan)
    train_csv = r"./Dataset HAM1000/train.csv"
    val_csv   = r"./Dataset HAM1000/val_public.csv"
    test_csv  = r"./Dataset HAM1000/test_hidden.csv"
    img_root  = r"./root/preprocessed_datasets"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # model list you wanted to run
    model_names = [
        "convnext",         # will use 'color' transform (multimodal)
        "efficientnet_v2_m",# will use 'asymmetry' transform
        "densenet"          # will use 'border' transform
    ]

    # map model -> hyperparams and transform type
    model_configs = {
        "convnext": {
            "transform": transform_color,
            "batch_size": 32,
            "epochs": 50,
            "optimizer_cfg": {"name": "adamw", "lr": 2e-4, "weight_decay": 1e-2},
            "scheduler_cfg": {"type": "cosine", "T_max": 20},
            "use_prior": True,
            "multimodal": True
        },
        "efficientnet_v2_m": {
            "transform": transform_asymmetry,
            "batch_size": 32,
            "epochs": 50,
            "optimizer_cfg": {"name": "adamw", "lr": 2e-4, "weight_decay": 1e-3},
            "scheduler_cfg": {"type": "cosine", "T_max": 18},
            "use_prior": False,
            "multimodal": False
        },
        "densenet": {
            "transform": transform_border,
            "batch_size": 32,
            "epochs": 50,
            "optimizer_cfg": {"name": "adamw", "lr": 1e-4, "weight_decay": 1e-3},
            "scheduler_cfg": {"type": "cosine", "T_max": 16},
            "use_prior": False,
            "multimodal": False
        }
    }

    # build tabular maps
    sex_map, loc_map, age_mean, age_std = build_tabular_maps(train_csv)
    tab_dim = 1 + len(sex_map) + len(loc_map)

    # read labels to build class_names and label mapping
    df_labels = pd.read_csv(train_csv)
    unique_labels = sorted(df_labels['label_idx'].unique().tolist())
    class_names = [f"class_{int(l)}" for l in unique_labels]

    summary_results = []
    factory = ModelFactory(num_classes=len(unique_labels))

    for model_name in model_names:
        cfg = model_configs[model_name]
        print(f"\n========== TRAINING {model_name.upper()} ==========")
        timer = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = f"./results_{timer}_{model_name}"
        os.makedirs(save_dir, exist_ok=True)

        # dataset and loaders
        print("LOADER DATASETS")
        train_set = HAM10000MultimodalDataset(train_csv, img_root, sex_map, loc_map, age_mean, age_std, transform=cfg["transform"])
        val_set   = HAM10000MultimodalDataset(val_csv, img_root, sex_map, loc_map, age_mean, age_std, transform=transform_val)
        test_set  = HAM10000MultimodalDataset(test_csv, img_root, sex_map, loc_map, age_mean, age_std, transform=transform_val)

        # compute class weights once from train labels
        labels = torch.tensor([label for _, _, label in train_set])
        weights = compute_class_weights_from_labels(labels)

        batch_size = cfg["batch_size"]
        num_workers = min(8, os.cpu_count() or 4)

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

        cnn = factory.create(model_name)
        model = MultiModalFusion(cnn_model=cnn, num_classes=len(unique_labels), tabular_dim=tab_dim, tabular_emb_dim=32, use_prior=cfg.get("use_prior", False))
        multimodal_flag = cfg.get("multimodal", False)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            class_weights=weights,
            save_dir=save_dir,
            class_names=class_names,
            multimodal=multimodal_flag,
            optimizer_cfg=cfg["optimizer_cfg"],
            scheduler_cfg=cfg["scheduler_cfg"],
            use_amp=True
        )
        # dataset and loaderss

        print("TRAINMODEL DATASETS")
        history = trainer.train(epochs=cfg["epochs"])

        summary_results.append([model_name, history["f1_macro"][-1], history["val_loss"][-1]])
        trainer.evaluate()

    print("\n===== FINAL SUMMARY TABLE =====")
    print("Model | F1 Macro | Val Loss")
    for row in summary_results:
        print(f"{row[0]:15s} | {row[1]:.4f} | {row[2]:.4f}")

    notif("DONE", str(summary_results))