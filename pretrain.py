import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

# -----------------------------
# Focal Loss
# -----------------------------
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


# -----------------------------
# Model Factory
# -----------------------------
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

        elif name == "inception":
            model = models.inception_v3(weights="IMAGENET1K_V1")
            in_feat = model.fc.in_features
            model.fc = nn.Linear(in_feat, self.num_classes)

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


# -----------------------------
# Trainer Class
# -----------------------------
class Trainer:
    def __init__(self, model, train_loader, val_loader, device, class_weights, save_dir):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = save_dir

        self.criterion = FocalLoss(weight=class_weights.to(device))
        self.optimizer = optim.Adam(model.parameters(), lr=1e-4)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def train(self, epochs=3):
        history = {"train_loss": [], "val_loss": [], "f1_macro": []}

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0

            for imgs, labels in self.train_loader:
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

            print(f"Epoch {epoch+1}/{epochs} | Train Loss {total_loss:.4f} | Val F1 {f1_macro:.4f}")

        self.save_history(history)
        return history

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        preds = []
        trues = []

        with torch.no_grad():
            for imgs, labels in self.val_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                outputs = self.model(imgs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                trues.extend(labels.cpu().numpy())

        f1_macro = f1_score(trues, preds, average="macro")
        self.save_confusion(trues, preds)
        return total_loss / len(self.val_loader), f1_macro

    def save_confusion(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
        plt.title("Confusion Matrix")
        plt.savefig(f"{self.save_dir}/confusion_matrix.png")
        plt.close()

    def save_history(self, history):
        plt.figure(figsize=(7,5))
        plt.plot(history["train_loss"], label="Train Loss")
        plt.plot(history["val_loss"], label="Val Loss")
        plt.legend()
        plt.title("Training History")
        plt.savefig(f"{self.save_dir}/history.png")
        plt.close()


# -----------------------------
# MAIN RUNNER
# -----------------------------
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
    class_count = torch.bincount(labels)
    weights = 1.0 / class_count.float()
    return weights


if __name__ == "__main__":
    root = "./root/segmentation_masks"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_set, val_set, test_set = load_datasets(root)
    weights = compute_class_weights(train_set)

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False)

    model_names = [
        "convnext",
        "efficientnet_b3",
        "efficientnet_b7",
        "efficientnet_v2_s",
        "efficientnet_v2_m",
        "inception",
        "resnet50",
        "densenet"
    ]

    factory = ModelFactory(num_classes=len(train_set.classes))

    for model_name in model_names:
        print(f"\n===== TRAINING {model_name.upper()} =====")
        model = factory.create(model_name)
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            class_weights=weights,
            save_dir=f"./results_{model_name}"
        )
        trainer.train(epochs=3)

