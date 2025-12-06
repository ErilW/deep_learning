import os
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm


# ==========================================================
#  PATHS (sesuai permintaan)
# ==========================================================
TRAIN_CSV = r"./Dataset HAM1000/train.csv"
VAL_CSV   = r"./Dataset HAM1000/val_public.csv"
TEST_CSV  = r"./Dataset HAM1000/test_hidden.csv"
IMG_ROOT  = r"./root/preprocessed_datasets"


# ==========================================================
#  DATASET CLASS
# ==========================================================
class SkinDataset(Dataset):
    def __init__(self, csv_file, img_root, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_root = img_root
        self.transform = transform

        # wajib ada kolom 'image' dan 'label'
        assert "image" in self.df.columns
        assert "label" in self.df.columns

        # mapping label → index
        self.label_to_idx = {label: i for i, label in enumerate(sorted(self.df["label"].unique()))}
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_root, row["label"], row["image"])

        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Image not found: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(img)

        label = self.label_to_idx[row["label"]]
        return img, label


# ==========================================================
#  TRANSFORMS (ImageNet + 224)
# ==========================================================
train_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

val_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])


# ==========================================================
#  DATALOADERS
# ==========================================================
train_ds = SkinDataset(TRAIN_CSV, IMG_ROOT, train_tf)
val_ds   = SkinDataset(VAL_CSV, IMG_ROOT, val_tf)
test_ds  = SkinDataset(TEST_CSV, IMG_ROOT, val_tf)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=2)

num_classes = len(train_ds.label_to_idx)
print("Classes:", train_ds.label_to_idx)


# ==========================================================
#  MODEL LOADER — dari kecil → besar
# ==========================================================
def load_model(name):
    if name == "resnet18":
        model = models.resnet18(weights="IMAGENET1K_V1")
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        return model

    if name == "efficientnet_b0":
        model = models.efficientnet_b0(weights="IMAGENET1K_V1")
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model

    raise ValueError("Unknown model name")


# ==========================================================
#  TRAINER CLASS
# ==========================================================
class Trainer:
    def __init__(self, model):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)

        self.loss_fn = nn.CrossEntropyLoss()
        self.opt = optim.Adam(model.parameters(), lr=1e-4)

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0

        for imgs, labels in tqdm(loader, desc="Train"):
            imgs, labels = imgs.to(self.device), labels.to(self.device)

            self.opt.zero_grad()
            preds = self.model(imgs)
            loss = self.loss_fn(preds, labels)
            loss.backward()
            self.opt.step()

            total_loss += loss.item()
        return total_loss / len(loader)

    def eval_epoch(self, loader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for imgs, labels in tqdm(loader, desc="Val"):
                imgs, labels = imgs.to(self.device), labels.to(self.device)

                preds = self.model(imgs)
                loss = self.loss_fn(preds, labels)

                total_loss += loss.item()
                correct += (preds.argmax(1) == labels).sum().item()
                total += labels.size(0)

        return total_loss / len(loader), correct / total

    def evaluate_test(self, loader, idx_to_label):
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for imgs, labels in tqdm(loader, desc="Test"):
                imgs = imgs.to(self.device)
                preds = self.model(imgs).argmax(1).cpu()

                all_preds.extend(preds.numpy())
                all_labels.extend(labels.numpy())

        print("\n=== CLASSIFICATION REPORT ===")
        print(classification_report(all_labels, all_preds, target_names=list(idx_to_label.values())))

        print("\n=== CONFUSION MATRIX ===")
        print(confusion_matrix(all_labels, all_preds))


# ==========================================================
#  MAIN TRAINING
# ==========================================================
if __name__ == "__main__":
    model_name = "resnet18"   # ganti: "mobilenet_v3_small", "efficientnet_b0"
    model = load_model(model_name)

    t = Trainer(model)

    for epoch in range(3):
        print(f"\n================== Epoch {epoch+1} ==================")
        loss = t.train_epoch(train_loader)
        val_loss, val_acc = t.eval_epoch(val_loader)

        print(f"Train Loss: {loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")
        print(f"Val Acc:    {val_acc:.4f}")

    print("\n================== TEST ==================")
    t.evaluate_test(test_loader, test_ds.idx_to_label)
