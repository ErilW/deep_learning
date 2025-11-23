import datetime
import os
import torch
import torch.nn as nn
import torch.optim as optim
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
warnings.filterwarnings("ignore")

# ================================================================
#  Focal Loss
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
#  Model Factory
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
#  --- NEW: Tabular helpers & Multimodal Dataset / Model / Prior
# ================================================================
def build_tabular_maps(train_csv_path):
    """
    Build mappings for sex and localization from training CSV.
    Also compute age mean/std for normalization.
    """
    df = pd.read_csv(train_csv_path)
    # sex mapping: male, female, unknown (fallback)
    unique_sex = df['sex'].fillna('unknown').unique().tolist()
    sex_map = {s: i for i, s in enumerate(sorted(unique_sex))}
    # localization mapping: take unique
    unique_loc = df['localization'].fillna('unknown').unique().tolist()
    loc_map = {l: i for i, l in enumerate(sorted(unique_loc))}
    # age stats
    ages = df['age'].dropna().astype(float)
    age_mean = ages.mean() if len(ages) > 0 else 50.0
    age_std = ages.std() if len(ages) > 0 else 20.0
    if age_std == 0:
        age_std = 1.0
    return sex_map, loc_map, age_mean, age_std


class HAM10000MultimodalDataset(torch.utils.data.Dataset):
    """
    Return tuples:
      - image tensor
      - tabular tensor (age_normalized, sex_onehot..., loc_onehot...)
      - label (int)
    """

    def __init__(self, csv_path, img_root, sex_map, loc_map, age_mean, age_std, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_root = img_root
        self.transform = transform

        self.sex_map = sex_map
        self.loc_map = loc_map
        self.age_mean = age_mean
        self.age_std = age_std

        # Precompute tabular vector size
        self.sex_dim = len(self.sex_map)
        self.loc_dim = len(self.loc_map)
        self.tab_dim = 1 + self.sex_dim + self.loc_dim  # age + sex_onehot + loc_onehot

    def __len__(self):
        return len(self.df)

    def _find_image_path(self, image_id):
        """
        Search recursively for a file that starts with image_id in img_root.
        Returns first match or raises FileNotFoundError.
        """
        # Try usual image extensions
        exts = ['.jpg', '.jpeg', '.png', '.bmp']
        # quick try common locations
        possible = []
        for root, _, files in os.walk(self.img_root):
            for f in files:
                name, ext = os.path.splitext(f)
                if name == image_id or name.startswith(image_id):
                    return os.path.join(root, f)
                # sometimes filenames include image_id in longer name, try contains
                if image_id in name:
                    possible.append(os.path.join(root, f))
        if possible:
            return possible[0]
        # last resort: try image_id + ext in root
        for ext in exts:
            p = os.path.join(self.img_root, image_id + ext)
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"Image for id {image_id} not found under {self.img_root}")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = str(row['image_id'])
        # --- IMAGE ---
        img_path = self._find_image_path(image_id)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # --- TABULAR ---
        # age normalized
        try:
            age = float(row['age'])
            if np.isnan(age):
                age = self.age_mean
        except Exception:
            age = self.age_mean
        age_norm = (age - self.age_mean) / self.age_std

        # sex one-hot
        sex = str(row.get('sex', 'unknown'))
        sex_idx = self.sex_map.get(sex, None)
        if sex_idx is None:
            # fallback to 'unknown' mapping if present else 0
            sex_idx = self.sex_map.get('unknown', 0)
        sex_oh = np.zeros(self.sex_dim, dtype=np.float32)
        sex_oh[sex_idx] = 1.0

        # localization one-hot
        loc = str(row.get('localization', 'unknown'))
        loc_idx = self.loc_map.get(loc, None)
        if loc_idx is None:
            loc_idx = self.loc_map.get('unknown', 0)
        loc_oh = np.zeros(self.loc_dim, dtype=np.float32)
        loc_oh[loc_idx] = 1.0

        tab_vec = np.concatenate([[age_norm], sex_oh, loc_oh]).astype(np.float32)
        tab_tensor = torch.tensor(tab_vec)

        # --- LABEL ---
        label = int(row['label_idx'])

        return image, tab_tensor, label


class TabularEncoder(nn.Module):
    """
    Simple MLP to encode tabular vector into embedding.
    """
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
    """
    Produce a per-class bias from tabular features (lightweight).
    logits' = logits + alpha * prior_bias(tab_features)
    """
    def __init__(self, tab_input_dim, num_classes, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(tab_input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, tab):
        return self.net(tab)  # (batch, num_classes)


class MultiModalFusion(nn.Module):
    def __init__(self, cnn_model, num_classes, tabular_dim=10, tabular_emb_dim=32, use_prior=False):
        super().__init__()
        self.cnn = cnn_model
        # Extract feature dim from cnn by looking at classifier/fc
        feat_dim = None
        # convnext: classifier is Sequential with last linear at index 2
        if hasattr(self.cnn, "classifier"):
            if isinstance(self.cnn.classifier, nn.Sequential):
                # get last linear in classifier
                last = None
                for m in reversed(self.cnn.classifier):
                    if isinstance(m, nn.Linear):
                        last = m
                        break
                if last is not None:
                    feat_dim = last.in_features
            else:
                if hasattr(self.cnn.classifier, "in_features"):
                    feat_dim = self.cnn.classifier.in_features
        if feat_dim is None and hasattr(self.cnn, "fc"):
            feat_dim = self.cnn.fc.in_features
        if feat_dim is None:
            raise RuntimeError("Cannot determine CNN feature dim. Please check model factory.")

        # replace classifier/fc with identity to get features
        if hasattr(self.cnn, "classifier"):
            try:
                # set whole classifier to Identity (works for many torchvision nets)
                self.cnn.classifier = nn.Identity()
            except Exception:
                # fallback: if classifier is sequential, remove last layer instead
                if isinstance(self.cnn.classifier, nn.Sequential):
                    modules = list(self.cnn.classifier.children())[:-1]
                    self.cnn.classifier = nn.Sequential(*modules)
        elif hasattr(self.cnn, "fc"):
            self.cnn.fc = nn.Identity()

        # Tabular encoder
        self.tab_encoder = TabularEncoder(input_dim=tabular_dim, hidden_dim=tabular_emb_dim, out_dim=tabular_emb_dim)

        # Fusion classifier
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim + tabular_emb_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        # Prior
        self.use_prior = use_prior
        if self.use_prior:
            self.prior = PriorModule(tabular_dim, num_classes)
            self.prior_alpha = nn.Parameter(torch.tensor(0.5))  # learnable scalar weight

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
#  Trainer
# ================================================================
class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, device, class_weights, save_dir, class_names, multimodal=False):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.save_dir = save_dir
        self.class_names = class_names
        self.multimodal = multimodal

        self.criterion = FocalLoss(weight=class_weights.to(device))
        # initial optimizer will be set after freezing backbone (only classifier params)
        self.optimizer = None

        self.best_f1 = -1

        os.makedirs(save_dir, exist_ok=True)
        self.freeze_backbone()

        # optimizer hanya melatih classifier dulu
        trainable = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = optim.Adam(trainable, lr=1e-4, weight_decay=1e-5)

    # -----------------------------------------------

    def freeze_backbone(self):
        # freeze all except classifier/fc/head layers (robust name check)
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
            # Unfreeze & lower LR at epoch == 5 (i.e. after finishing epoch 0..4)
            if epoch == 5:
                print("🔓 Unfreezing backbone & lowering LR...")
                self.unfreeze_backbone()
                # reset optimizer to include all parameters with a smaller LR
                self.optimizer = optim.Adam(self.model.parameters(), lr=1e-5, weight_decay=1e-5)

            self.model.train()
            total_loss = 0

            print(f"\n📘 Epoch {epoch+1}/{epochs}")
            if self.multimodal:
                iterator = self.train_loader
                desc = "Training (multimodal)"
                for imgs, tabs, labels in tqdm(iterator, desc=desc):
                    imgs, tabs, labels = imgs.to(self.device), tabs.to(self.device), labels.to(self.device)
                    self.optimizer.zero_grad()
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
                    outputs = self.model(imgs)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item()

            val_loss, f1_macro = self.evaluate()

            history["train_loss"].append(total_loss / len(self.train_loader))
            history["val_loss"].append(val_loss)
            history["f1_macro"].append(f1_macro)

            print(f"📊 Train Loss: {total_loss:.4f} | Val Loss: {val_loss:.4f} | F1 Macro: {f1_macro:.4f}")

            # === Save best model ===
            if f1_macro > self.best_f1:
                self.best_f1 = f1_macro
                torch.save(self.model.state_dict(), f"{self.save_dir}/best_model.pt")
                print("💾 Model disave (BEST MODEL)")

        self.save_history(history)
        self.test()

        return history

    # -----------------------------------------------
    def evaluate(self):
        self.model.eval()
        preds, trues = [], []
        total_loss = 0

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

    # -----------------------------------------------
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

        # metrics
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

    # -----------------------------------------------
    def save_confusion(self, y_true, y_pred, stage):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(7, 6))
        sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
        plt.title(f"Confusion Matrix ({stage})")
        plt.savefig(f"{self.save_dir}/confusion_matrix_{stage}.png")
        plt.close()

    # -----------------------------------------------
    def save_history(self, history):
        plt.figure(figsize=(7,5))
        plt.plot(history["train_loss"], label="Train Loss")
        plt.plot(history["val_loss"], label="Val Loss")
        plt.legend()
        plt.title("Training History")
        plt.savefig(f"{self.save_dir}/history.png")
        plt.close()


# ================================================================
#  DATA & MAIN
# ================================================================
def compute_class_weights_from_labels(labels_tensor):
    count = torch.bincount(labels_tensor)
    # avoid zero
    count = torch.where(count==0, torch.ones_like(count), count)
    return 1.0 / count.float()


def make_image_only_datasets(root, transform):
    """
    Keep original ImageFolder-based datasets for models other than convnext.
    """
    train = datasets.ImageFolder(os.path.join(root, "train"), transform=transform)
    val   = datasets.ImageFolder(os.path.join(root, "val"), transform=transform)
    test  = datasets.ImageFolder(os.path.join(root, "test"), transform=transform)
    return train, val, test


if __name__ == "__main__":
    # Paths (gunakan path yang kamu berikan)
    train_csv = r"C:\Users\User\Documents\deep_learning\Dataset HAM1000\train.csv"
    val_csv   = r"C:\Users\User\Documents\deep_learning\Dataset HAM1000\val_public.csv"
    test_csv  = r"C:\Users\User\Documents\deep_learning\Dataset HAM1000\test_hidden.csv"
    img_root  = r"C:\Users\User\Documents\deep_learning\Dataset HAM1000"  # akan dicari recursive di dalam folder ini

    # Note: kalau struktur gambar kamu berada di ./segmentation_masks/... ubah img_root ke folder itu.
    # root untuk image-only ImageFolder (original script) - penyesuaian
    imagefolder_root = os.path.join(img_root, "segmentation_masks")  # asumsikan sama seperti sebelumnya

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Transforms (kamu bisa tambahin augmentasi jika mau)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # ================================================================
    # Build tabular maps from train CSV (for one-hot encoding)
    sex_map, loc_map, age_mean, age_std = build_tabular_maps(train_csv)
    tab_dim = 1 + len(sex_map) + len(loc_map)

    # ================================================================
    # Decide which models to run
    model_names = [
        "convnext",
        # "efficientnet_b3",
        # "efficientnet_v2_s",
        # "efficientnet_v2_m",
        # "efficientnet_v2_l",
    ]

    factory = ModelFactory(num_classes=7)  # ham10000 typically 7 classes; but we'll read class names from CSV instead

    # Build class_names from training CSV label_idx mapping (assuming 0..6)
    # If you have mapping dx -> label_idx separately, insert here. For now read unique label_idx and sort.
    df_labels = pd.read_csv(train_csv)
    unique_labels = sorted(df_labels['label_idx'].unique().tolist())
    # create placeholder class names
    class_names = [f"class_{int(l)}" for l in unique_labels]

    summary_results = []

    for model_name in model_names:
        print(f"\n========== TRAINING {model_name.upper()} ==========")
        timer = datetime.datetime.now()
        # === prepare datasets/loaders ===
        if model_name.lower() == "convnext":
            # multimodal + prior enabled
            print("=> Using multimodal pipeline (image + tabular) with prior correction for ConvNeXt")
            train_set = HAM10000MultimodalDataset(train_csv, img_root, sex_map, loc_map, age_mean, age_std, transform=transform)
            val_set   = HAM10000MultimodalDataset(val_csv, img_root, sex_map, loc_map, age_mean, age_std, transform=transform)
            test_set  = HAM10000MultimodalDataset(test_csv, img_root, sex_map, loc_map, age_mean, age_std, transform=transform)

            # compute class weights from labels in train_set
            labels = torch.tensor([label for _, _, label in train_set])
            weights = compute_class_weights_from_labels(labels)
            # DataLoaders
            train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)
            val_loader   = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=4)
            test_loader  = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=4)

            # build model
            cnn = factory.create(model_name)
            model = MultiModalFusion(cnn_model=cnn, num_classes=len(unique_labels), tabular_dim=tab_dim, tabular_emb_dim=32, use_prior=True)
            multimodal_flag = True

        else:
            # original image-only pipeline
            train_img, val_img, test_img = make_image_only_datasets(imagefolder_root, transform)
            # compute class weights from imagefolder labels
            labels = torch.tensor([y for _, y in train_img])
            weights = compute_class_weights_from_labels(labels)
            train_loader = DataLoader(train_img, batch_size=32, shuffle=True, num_workers=4)
            val_loader   = DataLoader(val_img, batch_size=32, shuffle=False, num_workers=4)
            test_loader  = DataLoader(test_img, batch_size=32, shuffle=False, num_workers=4)

            model = factory.create(model_name)
            multimodal_flag = False

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            class_weights=weights,
            save_dir=f"./results__{model_name}",
            class_names=class_names,
            multimodal=multimodal_flag
        )

        history = trainer.train(epochs=20)

        summary_results.append([
            model_name,
            history["f1_macro"][-1],
            history["val_loss"][-1]
        ])

    print("\n===== FINAL SUMMARY TABLE =====")
    print("Model | F1 Macro | Val Loss")
    for row in summary_results:
        print(f"{row[0]:15s} | {row[1]:.4f} | {row[2]:.4f}")

# 5 freeze, 5unfreeze
# ===== FINAL SUMMARY TABLE =====
# convnext        | 0.6871 | 0.1904
# efficientnet_b3 | 0.6231 | 0.2525
# efficientnet_b7 | 0.5817 | 0.2895
# efficientnet_v2_s | 0.6294 | 0.2227
# efficientnet_v2_m | 0.6515 | 0.1801
# resnet50        | 0.6005 | 0.2783
# densenet        | 0.6102 | 0.2094

# 5 freeze, 15 epo unfreeze
# convnext        | 0.7259 | 0.1991
# efficientnet_b3 | 0.6271 | 0.3175
# efficientnet_b7 | 0.6197 | 0.2697
# efficientnet_v2_s | 0.6580 | 0.1878
# efficientnet_v2_m | 0.6925 | 0.1734
# resnet50        | 0.6656 | 0.2896
# densenet        | 0.6412 | 0.1592

# ===== FINAL SUMMARY TABLE =====
# convnext        | 0.6804 | 0.2281
