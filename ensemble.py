import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm

from pretrain import ModelFactory, load_datasets


# =====================================================
#  ENSEMBLE CLASS
# =====================================================
class ModelEnsembler:
    def __init__(self, model_names, factory, device, class_names):
        self.model_names = model_names
        self.factory = factory
        self.device = device
        self.class_names = class_names
        self.models = []

    def load_models(self):
        print("\n🔄 Loading all models for ensemble...\n")

        for name in self.model_names:
            model = self.factory.create(name)
            path = f"./results_{name}/best_model.pt"

            if not os.path.exists(path):
                print(f"❌ WARNING: model weight not found: {path}")
                continue

            model.load_state_dict(torch.load(path, map_location=self.device))
            model.to(self.device)
            model.eval()
            self.models.append(model)

            print(f"✅ Loaded: {name} | {path}")

        print(f"\n📦 Total model loaded: {len(self.models)}\n")

    # ---------------------------
    def predict_batch(self, imgs):
        """Soft voting: average probabilities"""
        all_probs = []

        with torch.no_grad():
            for model in self.models:
                logits = model(imgs)
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs)

        avg_probs = torch.stack(all_probs, dim=0).mean(dim=0)
        preds = torch.argmax(avg_probs, dim=1)
        return preds.cpu().numpy()

    # ---------------------------
    def evaluate(self, loader, save_dir="./results_ensemble"):
        os.makedirs(save_dir, exist_ok=True)

        y_true, y_pred = [], []

        print("🧪 Running ENSEMBLE evaluation...\n")

        for imgs, labels in tqdm(loader):
            imgs, labels = imgs.to(self.device), labels.to(self.device)

            preds = self.predict_batch(imgs)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds)

        # ---- METRICS ----
        macro_f1 = f1_score(y_true, y_pred, average="macro")
        report = classification_report(y_true, y_pred, target_names=self.class_names)

        print("\n===== ENSEMBLE REPORT =====")
        print(report)
        print(f"Macro F1: {macro_f1:.4f}")

        # save report
        with open(f"{save_dir}/ensemble_report.txt", "w") as f:
            f.write(report)

        # ---- CONFUSION MATRIX ----
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
        plt.title("Confusion Matrix (Ensemble)")
        plt.savefig(f"{save_dir}/ensemble_confusion_matrix.png")
        plt.close()

        return macro_f1

if "__main__" == __name__:
    # ================== ENSEMBLE ==================
    root = "./root/segmentation_masks"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_set, val_set, test_set = load_datasets(root)
    class_names = train_set.classes

    # weights = compute_class_weights(train_set)

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=8)

    model_names = [
        "convnext",
        # "efficientnet_b3",
        # "efficientnet_v2_s",
        # "efficientnet_v2_m",
        "efficientnet_v2_l",
    ]

    factory = ModelFactory(num_classes=len(class_names))
    summary_results = []


    ensembler = ModelEnsembler(
        model_names=model_names,
        factory=factory,
        device=device,
        class_names=class_names
    )

    ensembler.load_models()

    ensemble_f1 = ensembler.evaluate(test_loader)
    print(f"\n🎉 ENSEMBLE Macro F1: {ensemble_f1:.4f}")
