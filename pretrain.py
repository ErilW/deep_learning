import datetime
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import f1_score, confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np

# ================================================================
#  Focal Loss (fixed implementation)
# ================================================================
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        """
        weight : torch.Tensor or None -> class weights (will be passed to cross_entropy)
        gamma : focal gamma
        reduction: 'mean' or 'sum' or 'none'
        """
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits, targets):
        # compute per-sample cross entropy (no reduction) to get logpt per sample
        ce_loss = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')  # shape (N,)
        pt = torch.exp(-ce_loss)  # pt = exp(-CE) = model prob of true class
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss  # no reduction


# ================================================================
#  Model Factory (robust classifier replacement + typo fixes)
# ================================================================
class ModelFactory:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def _safe_replace_linear(self, parent, attr_name, in_features):
        """Helper: set parent.attr_name = nn.Linear(in_features, num_classes) if exists."""
        try:
            setattr(parent, attr_name, nn.Linear(in_features, self.num_classes))
            return True
        except Exception:
            return False

    def create(self, name):
        name = name.lower()

        # ---------------- CNN / EfficientNet / ResNet / DenseNet ----------------
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
            # fixed: previously branch typo mapped v2_m -> v2_l; now explicit
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

        # ---------------- Vision Transformer (robust replacement) ----------------
        elif name in ("vit_b_16", "vit_b16"):
            try:
                model = models.vit_b_16(weights="IMAGENET1K_V1")
            except Exception:
                # fallback for older torchvision where API differs
                model = models.vit_b_16(pretrained=True)

            # try common head locations
            replaced = False
            if hasattr(model, "heads") and hasattr(model.heads, "head"):
                in_feat = model.heads.head.in_features
                model.heads.head = nn.Linear(in_feat, self.num_classes)
                replaced = True
            elif hasattr(model, "heads") and isinstance(model.heads, nn.Linear):
                in_feat = model.heads.in_features
                model.heads = nn.Linear(in_feat, self.num_classes)
                replaced = True
            elif hasattr(model, "head") and isinstance(model.head, nn.Linear):
                in_feat = model.head.in_features
                model.head = nn.Linear(in_feat, self.num_classes)
                replaced = True

            # fallback: replace last linear found
            if not replaced:
                for n, m in reversed(list(model.named_modules())):
                    if isinstance(m, nn.Linear):
                        # locate parent module and attr
                        parent_name = n.rsplit(".", 1)[0] if "." in n else ""
                        try:
                            if parent_name:
                                parent = dict(model.named_modules())[parent_name]
                                # find attribute name on parent that references m
                                for attr in dir(parent):
                                    if getattr(parent, attr) is m:
                                        setattr(parent, attr, nn.Linear(m.in_features, self.num_classes))
                                        replaced = True
                                        break
                                if replaced:
                                    break
                        except Exception:
                            pass
                if not replaced:
                    raise RuntimeError("Couldn't replace ViT classifier head (unexpected layout).")

        elif name in ("vit_b_32", "vit_b32"):
            try:
                model = models.vit_b_32(weights="IMAGENET1K_V1")
            except Exception:
                model = models.vit_b_32(pretrained=True)

            replaced = False
            if hasattr(model, "heads") and hasattr(model.heads, "head"):
                in_feat = model.heads.head.in_features
                model.heads.head = nn.Linear(in_feat, self.num_classes)
                replaced = True
            elif hasattr(model, "heads") and isinstance(model.heads, nn.Linear):
                in_feat = model.heads.in_features
                model.heads = nn.Linear(in_feat, self.num_classes)
                replaced = True
            elif hasattr(model, "head") and isinstance(model.head, nn.Linear):
                in_feat = model.head.in_features
                model.head = nn.Linear(in_feat, self.num_classes)
                replaced = True

            if not replaced:
                for n, m in reversed(list(model.named_modules())):
                    if isinstance(m, nn.Linear):
                        parent_name = n.rsplit(".", 1)[0] if "." in n else ""
                        try:
                            if parent_name:
                                parent = dict(model.named_modules())[parent_name]
                                for attr in dir(parent):
                                    if getattr(parent, attr) is m:
                                        setattr(parent, attr, nn.Linear(m.in_features, self.num_classes))
                                        replaced = True
                                        break
                                if replaced:
                                    break
                        except Exception:
                            pass
                if not replaced:
                    raise RuntimeError("Couldn't replace ViT classifier head (unexpected layout).")

        else:
            raise ValueError("Unknown model: " + name)

        return model


# ================================================================
#  Trainer (logic preserved; improved freeze/unfreeze & AMP + scheduler)
# ================================================================
class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, device, class_weights, save_dir, class_names):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.save_dir = save_dir
        self.class_names = class_names

        # criterion uses weight as torch tensor on device
        w = class_weights.to(device) if class_weights is not None else None
        self.criterion = FocalLoss(weight=w, gamma=2.0)
        self.optimizer = None
        self.scheduler = None
        self.scaler = torch.cuda.amp.GradScaler() if (device.startswith("cuda") and torch.cuda.is_available()) else None

        self.best_f1 = -1

        os.makedirs(save_dir, exist_ok=True)
        self.freeze_backbone()

        # optimizer hanya melatih classifier dulu (params with requires_grad True)
        trainable = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = optim.Adam(trainable, lr=1e-4, weight_decay=1e-5)
        # scheduler that reduces LR on plateau (will be re-created when optimizer changes)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=3)

    # -----------------------------------------------
    def freeze_backbone(self):
        """
        Robust freezing: set all params requires_grad=False, then try to unfreeze only
        the classifier/head modules (common attribute names). If not found, fallback to
        unfreezing the last Linear layer (safer than wildcard 'linear' matching).
        """
        for param in self.model.parameters():
            param.requires_grad = False

        # common candidate attribute names for classifier heads
        candidate_attrs = [
            "classifier", "fc", "head", "heads", "proj", "mlp_head", "classifier_head", "head_fc"
        ]

        unfreezed = False
        # try to walk attributes on top-level model
        for attr in candidate_attrs:
            if hasattr(self.model, attr):
                module = getattr(self.model, attr)
                # unfreeze all params in this module
                for p in module.parameters():
                    p.requires_grad = True
                unfreezed = True

        # Special-case: some models (e.g., convnext) have classifier as ModuleList/Sequential
        # try to find submodules with name containing 'classifier' in named_modules
        if not unfreezed:
            for name, module in self.model.named_modules():
                lname = name.lower()
                if lname.endswith("classifier") or lname.endswith("head") or lname.endswith("fc"):
                    for p in module.parameters():
                        p.requires_grad = True
                    unfreezed = True
                    break

        # Fallback: unfreeze last linear layer encountered (safe fallback)
        if not unfreezed:
            for n, m in reversed(list(self.model.named_modules())):
                if isinstance(m, nn.Linear):
                    # set requires_grad True for that module's params by locating parent
                    parent_name = n.rsplit(".", 1)[0] if "." in n else ""
                    try:
                        if parent_name:
                            parent = dict(self.model.named_modules())[parent_name]
                            for attr in dir(parent):
                                try:
                                    if getattr(parent, attr) is m:
                                        # found the attribute, replace/unwrap if necessary
                                        for p in m.parameters():
                                            p.requires_grad = True
                                        unfreezed = True
                                        break
                                except Exception:
                                    pass
                            if unfreezed:
                                break
                        else:
                            # module at top-level is linear
                            for p in m.parameters():
                                p.requires_grad = True
                            unfreezed = True
                            break
                    except Exception:
                        pass

        # As a last resort, if nothing got unfreezed (very unlikely), unfreeze all params
        if not unfreezed:
            for p in self.model.parameters():
                p.requires_grad = True

    def unfreeze_backbone(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def _recreate_optimizer_and_scheduler(self, lr=1e-5):
        # recreate optimizer with all trainable params (call after unfreeze)
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=3)

    def train(self, epochs=3):
        print(f"\n🚀 Device digunakan: {self.device.upper()}")
        history = {"train_loss": [], "val_loss": [], "f1_macro": []}

        for epoch in range(epochs):
            # Unfreeze & lower LR at epoch == 5 (preserve your original logic)
            if epoch == 5:
                print("🔓 Unfreezing backbone & lowering LR...")
                self.unfreeze_backbone()
                # reset optimizer to include all parameters with a smaller LR
                self._recreate_optimizer_and_scheduler(lr=1e-5)

            self.model.train()
            total_loss = 0.0

            print(f"\n📘 Epoch {epoch+1}/{epochs}")
            for imgs, labels in tqdm(self.train_loader, desc="Training"):
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()

                # mixed precision if available
                if self.scaler is not None:
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

            val_loss, f1_macro = self.evaluate()

            # step scheduler if present (ReduceLROnPlateau uses val_loss)
            if self.scheduler is not None:
                try:
                    self.scheduler.step(val_loss)
                except Exception:
                    pass

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

    # -----------------------------------------------
    def test(self):
        print("\n🧪 Running TEST evaluation…")
        self.model.eval()
        preds, trues = [], []

        with torch.no_grad():
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
#  DATA & MAIN (augmentations + normalization)
# ================================================================
def load_datasets(root, image_size=224, batch_size=64, num_workers=8):
    # ImageNet normalization (important for pretrained backbones)
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((int(image_size*1.15), int(image_size*1.15))),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    train = datasets.ImageFolder(os.path.join(root, "train"), transform=train_transform)
    val   = datasets.ImageFolder(os.path.join(root, "val"), transform=val_transform)
    test  = datasets.ImageFolder(os.path.join(root, "test"), transform=val_transform)
    return train, val, test


def compute_class_weights(dataset):
    # uses sklearn compute_class_weight balanced
    labels = np.array([y for _, y in dataset])
    classes = np.unique(labels)
    # compute_class_weight returns array aligned with classes
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=labels)
    # map to full class index range (in case classes are not 0..C-1 contiguous)
    # here assuming ImageFolder gives contiguous 0..C-1
    weights_tensor = torch.tensor(weights, dtype=torch.float)
    return weights_tensor


if __name__ == "__main__":
    root = "./root/segmentation_masks"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # adapt num_workers safer
    num_workers = min(8, os.cpu_count() or 4)

    train_set, val_set, test_set = load_datasets(root, image_size=224, batch_size=64, num_workers=num_workers)
    class_names = train_set.classes

    weights = compute_class_weights(train_set)

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=num_workers, pin_memory=True)

    model_names = [
        "convnext",
        # you may switch to vit_b_16 / vit_b_32 here if wanted
        "vit_b_16",
        # "vit_b_32",
    ]

    factory = ModelFactory(num_classes=len(class_names))
    summary_results = []

    for model_name in model_names:
        print(f"\n========== TRAINING {model_name.upper()} ==========")
        timer = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        model = factory.create(model_name)
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            class_weights=weights,
            save_dir=f"./results_{timer}_{model_name}",
            class_names=class_names
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
# Model | F1 Macro | Val Loss
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
