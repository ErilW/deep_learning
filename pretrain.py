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

        elif name == "efficientnet_v2_m":
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
#  Trainer
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
            for imgs, labels in tqdm(self.train_loader, desc="Training"):
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
#  DATA & MAIN
# ================================================================
def load_datasets(root):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    train = datasets.ImageFolder(os.path.join(root, "train"), transform=transform)
    val   = datasets.ImageFolder(os.path.join(root, "val"), transform=transform)
    test  = datasets.ImageFolder(os.path.join(root, "test"), transform=transform)
    return train, val, test


def compute_class_weights(dataset):
    labels = [y for _, y in dataset]
    labels = torch.tensor(labels)
    count = torch.bincount(labels)
    return 1.0 / count.float()


if __name__ == "__main__":
    root = "./root/segmentation_masks"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_set, val_set, test_set = load_datasets(root)
    class_names = train_set.classes

    weights = compute_class_weights(train_set)

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=8)
    val_loader   = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=8)
    test_loader  = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=8)

    model_names = [
        "convnext",
        # "efficientnet_b3",
        # "efficientnet_v2_s",
        # "efficientnet_v2_m",
        "efficientnet_v2_l",
    ]

    factory = ModelFactory(num_classes=len(class_names))
    summary_results = []

    for model_name in model_names:
        print(f"\n========== TRAINING {model_name.upper()} ==========")

        model = factory.create(model_name)
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            class_weights=weights,
            save_dir=f"./results_{model_name}",
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