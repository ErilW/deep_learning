import os
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from transformers import ViTForImageClassification, ViTImageProcessor


class ViTTrainer:
    def __init__(
        self,
        model_name="google/vit-base-patch16-224",
        train_dir="dataset/train",
        val_dir="dataset/val",
        batch_size=16,
        lr=2e-5,
        num_epochs=5
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs

        # === Load processor (normalize/settings sesuai ViT) ===
        self.processor = ViTImageProcessor.from_pretrained(model_name)

        # === Dataset transforms ===
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.processor.image_mean,
                std=self.processor.image_std
            )
        ])

        # === Load datasets ===
        self._load_datasets()

        # === Load model ===
        self._load_model()

        # === Optimizer and loss ===
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()

    # ---------------------------------------------------------
    # Load Dataset
    # ---------------------------------------------------------
    def _load_datasets(self):
        print("Loading dataset...")

        self.train_ds = datasets.ImageFolder(self.train_dir, transform=self.transform)
        self.val_ds = datasets.ImageFolder(self.val_dir, transform=self.transform)

        self.train_loader = DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_ds, batch_size=self.batch_size)

        self.num_classes = len(self.train_ds.classes)
        print(f"Classes: {self.train_ds.classes}")
        print(f"Num classes: {self.num_classes}")

    # ---------------------------------------------------------
    # Load ViT pretrained
    # ---------------------------------------------------------
    def _load_model(self):
        print("Loading model...")

        self.model = ViTForImageClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_classes,
            ignore_mismatched_sizes=True
        ).to(self.device)

    # ---------------------------------------------------------
    # Training Loop
    # ---------------------------------------------------------
    def train(self):
        print("\n=== Training Start ===")

        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0

            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
            for imgs, labels in pbar:
                imgs, labels = imgs.to(self.device), labels.to(self.device)

                outputs = self.model(pixel_values=imgs)
                loss = self.criterion(outputs.logits, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            print(f"Epoch {epoch+1} Average Loss: {total_loss/len(self.train_loader):.4f}")

            self.evaluate()

    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------
    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for imgs, labels in self.val_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)

                outputs = self.model(pixel_values=imgs)
                pred = outputs.logits.argmax(-1)

                correct += (pred == labels).sum().item()
                total += labels.size(0)

        acc = correct / total
        print(f"Validation Accuracy: {acc:.4f}")
        return acc

    # ---------------------------------------------------------
    # Save Model
    # ---------------------------------------------------------
    def save(self, out_dir="vit_finetuned"):
        os.makedirs(out_dir, exist_ok=True)
        self.model.save_pretrained(out_dir)
        self.processor.save_pretrained(out_dir)
        print(f"Model saved to: {out_dir}")

    # ---------------------------------------------------------
    # Inference Single Image
    # ---------------------------------------------------------
    def predict(self, img_path):
        self.model.eval()
        image = Image.open(img_path).convert("RGB")

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        pred = outputs.logits.argmax(-1).item()
        label = self.model.config.id2label[pred]
        return label

if __name__ == "__main__":
    trainer = ViTTrainer(
        model_name="google/vit-base-patch16-224",
        train_dir="./root/augmented_balanced_dataset5/train",
        val_dir="./root/augmented_balanced_dataset5/val",
        batch_size=16,
        lr=2e-5,
        num_epochs=20,
        augment=Fakse
    )

    trainer.train()
    trainer.save("vit_custom_model")

    label = trainer.predict("./root/augmented_balanced_dataset5/test/df/")
    print("Prediksi:", label)

