import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import timm

# ==========================================================
# DATASET (identik dengan notebook)
# ==========================================================
class ImageFolderDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.files = []
        self.targets = []
        self.transform = transform

        classes = sorted(os.listdir(root))
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

        for c in classes:
            class_dir = os.path.join(root, c)
            for f in os.listdir(class_dir):
                if f.lower().endswith((".jpg", ".png", ".jpeg")):
                    self.files.append(os.path.join(class_dir, f))
                    self.targets.append(self.class_to_idx[c])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        target = self.targets[idx]

        if self.transform:
            img = self.transform(img)

        return img, target


# ==========================================================
# TRANSFORM (100% sama dengan notebook)
# ==========================================================
train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # ALWAYS float32 (0–1)
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

test_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# ==========================================================
# TRAINER (identik dengan notebook)
# ==========================================================
class Trainer:
    def __init__(self, model, trainloader, testloader, device):
        self.model = model.to(device)
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = device

        self.crit = nn.CrossEntropyLoss()
        self.opt = optim.Adam(self.model.parameters(), lr=1e-4)

    def train(self, epochs):
        self.model.train()
        for e in range(epochs):
            loop = tqdm(self.trainloader, total=len(self.trainloader),
                        desc=f"Epoch {e+1}/{epochs}")

            for img, label in loop:

                # =============================
                # SAFETY PATCH (fix uint8 bug)
                # =============================
                img = img.to(self.device)
                label = label.to(self.device)

                if img.dtype != torch.float32:
                    img = img.float() / 255.0

                self.opt.zero_grad()
                out = self.model(img)
                loss = self.crit(out, label)
                loss.backward()
                self.opt.step()

                loop.set_postfix(loss=loss.item())

    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for img, label in self.testloader:

                img = img.to(self.device)
                label = label.to(self.device)

                if img.dtype != torch.float32:
                    img = img.float() / 255.0

                out = self.model(img)
                pred = out.argmax(dim=1)
                correct += (pred == label).sum().item()
                total += label.size(0)

        acc = correct / total
        return acc


# ==========================================================
# MODEL BUILDER (Swin 224 semua)
# ==========================================================
def build_swin(model_name, num_classes):
    return timm.create_model(
        model_name,
        pretrained=True,
        img_size=224,     # Force 224 even for models that default 384
        num_classes=num_classes
    )


# ==========================================================
# MAIN PIPELINE
# ==========================================================
def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # SESUAIKAN PATH
    train_dir = "/workspace/dataset/train"
    test_dir  = "/workspace/dataset/test"

    trainset = ImageFolderDataset(train_dir, train_tf)
    testset  = ImageFolderDataset(test_dir,  test_tf)

    trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4)
    testloader  = DataLoader(testset,  batch_size=32, shuffle=False, num_workers=4)

    # LIST MODEL YANG MAU DITEST (dari kecil → besar)
    swin_models = [
        "swin_tiny_patch4_window7_224",
        "swin_small_patch4_window7_224",
        "swin_base_patch4_window7_224",
        # tambah kalau mau
    ]

    print("\n=== Evaluating Swin Family ===")

    for name in swin_models:
        print(f"\n🔥 Model: {name}")

        model = build_swin(name, num_classes=len(trainset.class_to_idx))

        trainer = Trainer(model, trainloader, testloader, device)
        trainer.train(epochs=3)

        acc = trainer.evaluate()
        print(f"➡️ FINAL ACC: {acc:.4f}")


if __name__ == "__main__":
    main()
