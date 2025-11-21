import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
import numpy as np
import itertools

# ============================================================
# CONFIG
# ============================================================
DATA_ROOT = "./root/segmentation_masks"
BATCH_SIZE = 16
EPOCHS = 3
LR = 1e-4
INPUT_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# TRANSFORMS (NO AUGMENTATION)
# ============================================================
tfm = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

# ============================================================
# LOAD DATASET
# ============================================================
train_ds = datasets.ImageFolder(os.path.join(DATA_ROOT, "train"), transform=tfm)
val_ds   = datasets.ImageFolder(os.path.join(DATA_ROOT, "val"), transform=tfm)
test_ds  = datasets.ImageFolder(os.path.join(DATA_ROOT, "test"), transform=tfm)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

num_classes = len(train_ds.classes)

# ============================================================
# CLASS WEIGHTS
# ============================================================
from collections import Counter
counts = Counter([y for _, y in train_ds.samples])
total = sum(counts.values())
class_weights = torch.tensor(
    [total / (counts[i] * num_classes) for i in range(num_classes)],
    dtype=torch.float
).to(DEVICE)

# ============================================================
# FOCAL LOSS
# ============================================================
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()

criterion = FocalLoss(weight=class_weights)

# ============================================================
# MODEL LIST
# ============================================================
model_zoo = {
    "resnet50": models.resnet50,
    "densenet121": models.densenet121,
    "convnext_tiny": models.convnext_tiny,
    "efficientnet_b3": models.efficientnet_b3,
    "efficientnet_b4": models.efficientnet_b4,
    "efficientnet_b5": models.efficientnet_b5,
    "efficientnet_b6": models.efficientnet_b6,
    "efficientnet_b7": models.efficientnet_b7,
    "efficientnet_v2_s": models.efficientnet_v2_s,
    "efficientnet_v2_m": models.efficientnet_v2_m,
    "inception_v3": models.inception_v3
}

# ============================================================
# HELPER FOR CONFUSION MATRIX SAVE
# ============================================================
def save_confusion_matrix(cm, classes, save_path, title):
    plt.figure(figsize=(8,8))
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.colorbar()

    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes, rotation=45)
    plt.yticks(ticks, classes)

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, str(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ============================================================
# TRAIN LOOP
# ============================================================
def train_one_model(model_name, builder):
    print(f"\n===== TRAINING: {model_name} =====")

    model = builder(pretrained=True)

    # replace classifier/fc
    if hasattr(model, "fc"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif "inception" in model_name:
        model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_classes)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    history = {"train_loss": [], "val_loss": [], "val_f1": []}

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            if model_name == "inception_v3":
                out, aux = out
                loss = criterion(out, y) + 0.4 * criterion(aux, y)
            else:
                loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)

        train_loss = running_loss / len(train_ds)

        # VALIDATION
        model.eval()
        all_y = []
        all_p = []
        val_loss = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                if model_name == "inception_v3":
                    out = out[0]
                loss = criterion(out, y).item()
                val_loss += loss * x.size(0)

                preds = out.argmax(1).cpu().numpy()
                all_p.extend(preds)
                all_y.extend(y.cpu().numpy())

        val_loss /= len(val_ds)
        f1 = f1_score(all_y, all_p, average="macro")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(f1)

        print(f"[{model_name}] Epoch {epoch+1}/{EPOCHS}  "
              f"TrainLoss={train_loss:.4f}  ValLoss={val_loss:.4f}  F1={f1:.4f}")

    # ============================================================
    # EVALUATE ON TEST SET + SAVE OUTPUTS
    # ============================================================
    model.eval()
    all_y = []
    all_p = []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            if model_name == "inception_v3":
                out = out[0]
            preds = out.argmax(1).cpu().numpy()
            all_p.extend(preds)
            all_y.extend(y.cpu().numpy())

    cm = confusion_matrix(all_y, all_p)

    # output folder
    out_dir = f"./results_{model_name}"
    os.makedirs(out_dir, exist_ok=True)

    # save cm
    save_confusion_matrix(cm, train_ds.classes,
                          os.path.join(out_dir, "confusion_matrix.png"),
                          f"CM {model_name}")

    # save history plot
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.plot(history["val_f1"], label="val_f1")
    plt.legend()
    plt.xlabel("Epoch")
    plt.savefig(os.path.join(out_dir, "history.png"))
    plt.close()

    torch.save(model.state_dict(), os.path.join(out_dir, "model.pth"))

    return history


# ============================================================
# MAIN LOOP: RUN ALL MODELS
# ============================================================
for name, builder in model_zoo.items():
    train_one_model(name, builder)
