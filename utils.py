import json
import re
import time
from datetime import datetime

import cv2
import pandas as pd
import shutil

import requests
# from keras.src.utils import to_categorical
from tabulate import tabulate
import random
from config import CLASS_NAMES
# import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
import seaborn as sns


# ============================
# YOLO AUG PIPELINE (AS REQUESTED)
# ============================
import albumentations as A

yolo_aug = A.Compose([
    A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=70, val_shift_limit=40, p=0.8),
    A.ShiftScaleRotate(shift_limit=0.10, scale_limit=0.50, rotate_limit=0, border_mode=0, p=0.9),
    A.Affine(shear=(-10, 10), p=0.7),
    A.HorizontalFlip(p=1.0),
    A.VerticalFlip(p=1.0),
    # A.Mosaic(p=1.0, metadata_key=),
    # A.RandAugment(num_ops=2, magnitude=7, p=1.0),
    # A.CoarseDropout(max_holes=8, max_height=0.3, max_width=0.3, min_holes=1, p=0.4),
])


# ============================
# CREATE BALANCED YOLO DATASET
# ============================
def create_yolo_balanced_dataset(
    input_root,               # dataset format train/class/*.jpg
    output_root,              # dataset baru
    augmentations_per_image=1 # berapa augmentasi untuk oversampling
):
    os.makedirs(output_root, exist_ok=True)

    splits = ["train", "val", "test"]
    class_counts = {}

    print("\n🔍 Menghitung jumlah dataset per class...")
    # Hitung jumlah per kelas (di train saja)
    train_path = os.path.join(input_root, "train")
    for cls in os.listdir(train_path):
        cls_dir = os.path.join(train_path, cls)
        if not os.path.isdir(cls_dir):
            continue
        n = len([f for f in os.listdir(cls_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))])
        class_counts[cls] = n

    print("📊 Jumlah per class:", class_counts)

    max_class = max(class_counts.values())  # target balance
    print(f"\n🎯 Target per kelas (balanced 1:1): {max_class}")

    # Process all splits
    final_output_paths = {}

    for split in splits:
        src_split_dir = os.path.join(input_root, split)
        dst_split_dir = os.path.join(output_root, split)
        os.makedirs(dst_split_dir, exist_ok=True)

        if split == "train":
            print("\n🚀 Membuat TRAIN dataset balanced + augmentasi...")

            for cls, n in class_counts.items():
                src_cls_dir = os.path.join(src_split_dir, cls)
                dst_cls_dir = os.path.join(dst_split_dir, cls)
                os.makedirs(dst_cls_dir, exist_ok=True)

                images = [f for f in os.listdir(src_cls_dir) if f.lower().endswith((".jpg",".png",".jpeg"))]

                # Copy original images
                for img_file in images:
                    shutil.copy(os.path.join(src_cls_dir, img_file), os.path.join(dst_cls_dir, img_file))

                # Hitung kebutuhan oversampling
                need = max_class - n
                if need <= 0:
                    continue

                print(f"  ➕ Oversampling class '{cls}' sebanyak {need} images")

                for i in range(need):
                    img_file = random.choice(images)
                    img_path = os.path.join(src_cls_dir, img_file)

                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    # ----------------
                    # APPLY AUGMENTATION (FULL PIPELINE)
                    # ----------------
                    aug = yolo_aug(image=img)['image']

                    # save augment
                    save_name = f"{os.path.splitext(img_file)[0]}_aug_{i}.jpg"
                    save_path = os.path.join(dst_cls_dir, save_name)
                    cv2.imwrite(save_path, cv2.cvtColor(aug, cv2.COLOR_RGB2BGR))

        else:
            # VAL & TEST: copy only
            print(f"\n📁 Menyalin split '{split}' tanpa augmentasi...")
            for cls in os.listdir(src_split_dir):
                src_cls_dir = os.path.join(src_split_dir, cls)
                dst_cls_dir = os.path.join(dst_split_dir, cls)
                os.makedirs(dst_cls_dir, exist_ok=True)

                for f in os.listdir(src_cls_dir):
                    if f.lower().endswith((".jpg",".jpeg",".png")):
                        shutil.copy(os.path.join(src_cls_dir, f), os.path.join(dst_cls_dir, f))

        final_output_paths[split] = dst_split_dir

    print("\n✅ Dataset YOLO balanced berhasil dibuat!")
    print(final_output_paths)
    return final_output_paths

def create_dataset_by_dx(csv_path, image_root, output_folder, split_type="train", label_col="dx"):
    """
    Membuat dataset dari CSV metadata dengan struktur folder berdasarkan kolom dx (misalnya 'bkl', 'nv', 'mel', dll).
    Bisa mencari file gambar hingga ke seluruh subfolder image_root.
    """
    # 1. Load metadata
    df = pd.read_csv(csv_path)

    # 2. Buat folder dasar output seperti /dataset/train/
    target_base = os.path.join(output_folder, split_type)
    os.makedirs(target_base, exist_ok=True)

    # 3. Scan semua gambar dari image_root (termasuk subfolder)
    image_paths = {}
    for root, _, files in os.walk(image_root):
        for file in files:
            file_id, ext = os.path.splitext(file)
            if ext.lower() in ['.jpg', '.jpeg', '.png']:
                image_paths[file_id] = os.path.join(root, file)

    # 4. Proses setiap baris metadata dan copy ke folder berdasarkan dx
    total_copied = 0  # counter jumlah file yang berhasil disalin
    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_id = str(row["image_id"])
        label = str(row[label_col])

        if img_id not in image_paths:
            continue

        # buat folder sesuai label dx
        dst_dir = os.path.join(target_base, label)
        os.makedirs(dst_dir, exist_ok=True)

        # copy gambar
        src = image_paths[img_id]
        dst = os.path.join(dst_dir, os.path.basename(src))
        shutil.copy(src, dst)
        total_copied += 1
    print(f"📊 Total gambar berhasil disalin: {total_copied}")
    print(f"✅ Dataset berhasil dibuat di: {target_base}")
    
# =========================================
# 1. AUGMENTATION PIPELINE
# =========================================
def build_augmentation_layer(img, rotate_angle=15, flip_prob=0.5, zoom=0.1):
    h, w = img.shape[:2]

    # Horizontal flip
    if random.random() < flip_prob:
        img = cv2.flip(img, 1)

    # Vertical flip
    if random.random() < flip_prob / 2:
        img = cv2.flip(img, 0)

    # Random rotation
    angle = random.uniform(-rotate_angle, rotate_angle)
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # Random zoom
    if zoom > 0:
        zx, zy = random.uniform(1 - zoom, 1 + zoom), random.uniform(1 - zoom, 1 + zoom)
        img = cv2.resize(img, (int(w * zx), int(h * zy)))
        # crop or pad to original
        img = cv2.resize(img, (w, h))

    return img


def load_ham10000_tensor(base_dir="dataset", img_size=(224, 224), batch_size=32,
                     augment=True, balance=False, undersample=False, ratio=1.0,
                     class_names=None, shuffle=True):
    import tensorflow as tf
    if class_names is None:
        raise ValueError("class_names harus diisi (list nama kelas).")

    def load_split(split):
        images, labels = [], []
        for idx, cls in enumerate(class_names):
            cls_dir = os.path.join(base_dir, split, cls)
            if not os.path.isdir(cls_dir):
                continue
            for f in os.listdir(cls_dir):
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    img_path = os.path.join(cls_dir, f)
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, img_size)
                    images.append(img)
                    labels.append(idx)
        images = np.array(images, dtype=np.float32) / 255.0
        labels = to_categorical(labels, num_classes=len(class_names))
        return images, labels

    # Load train/val/test
    train_images, train_labels = load_split("train")
    val_images, val_labels = load_split("val")
    test_images, test_labels = load_split("test")

    # ====================
    # Balance dataset
    # ====================
    if balance:
        counts = train_labels.sum(axis=0)
        target = int(max(counts) * ratio)
        new_imgs, new_lbls = [], []
        for cls_idx in range(len(class_names)):
            cls_mask = np.argmax(train_labels, axis=1) == cls_idx
            cls_imgs = train_images[cls_mask]
            cls_lbls = train_labels[cls_mask]
            n = len(cls_imgs)
            if undersample and n > target:
                cls_imgs = cls_imgs[:target]
                cls_lbls = cls_lbls[:target]
            elif not undersample and n < target:
                reps = target // n
                rem = target % n
                cls_imgs = np.concatenate([cls_imgs] * reps + [cls_imgs[:rem]], axis=0)
                cls_lbls = np.concatenate([cls_lbls] * reps + [cls_lbls[:rem]], axis=0)
            new_imgs.append(cls_imgs)
            new_lbls.append(cls_lbls)
        train_images = np.concatenate(new_imgs, axis=0)
        train_labels = np.concatenate(new_lbls, axis=0)

    # ====================
    # Apply augmentation
    # ====================
    if augment:
        for i in range(len(train_images)):
            train_images[i] = build_augmentation_layer(train_images[i])

    # ====================
    # Convert ke tf.data.Dataset
    # ====================
    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
    test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

    if shuffle:
        train_ds = train_ds.shuffle(4096)

    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    print(f"Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}")
    return train_ds, val_ds, test_ds



# def show_augment_per_class(base_dir="dataset_ham_seg", output_dir="augment_output", img_size=(224, 224), samples_per_class=3):
#
#     os.makedirs(output_dir, exist_ok=True)
#
#     class_names = sorted(os.listdir(os.path.join(base_dir, "train")))
#     aug_layer = build_augmentation_layer(img_size)
#
#     for class_name in class_names:
#
#         class_input_path = os.path.join(base_dir, "train", class_name)
#         class_output_path = os.path.join(output_dir, class_name)
#         os.makedirs(class_output_path, exist_ok=True)
#
#         img_files = [f for f in os.listdir(class_input_path)
#                      if f.lower().endswith((".jpg", ".jpeg", ".png"))]
#
#         if len(img_files) == 0:
#             print(f"[WARN] No images in class {class_name}")
#             continue
#
#         # Random sample
#         img_files = random.sample(img_files, min(samples_per_class, len(img_files)))
#
#         print(f"Saving augmentation samples for class: {class_name}")
#
#         for img_file in img_files:
#             img_path = os.path.join(class_input_path, img_file)
#
#             # Load
#             img = tf.keras.preprocessing.image.load_img(img_path, target_size=img_size)
#             img = tf.keras.preprocessing.image.img_to_array(img)
#             img = tf.expand_dims(img, 0) / 255.0
#
#             # Augment 3x
#             aug_images = [
#                 img[0],
#                 aug_layer(img, training=True)[0],
#                 aug_layer(img, training=True)[0],
#                 aug_layer(img, training=True)[0],
#             ]
#
#             # Save 4 images: original + 3 aug
#             labels = ["original", "aug1", "aug2", "aug3"]
#
#             for i in range(4):
#                 out_path = os.path.join(
#                     class_output_path,
#                     f"{os.path.splitext(img_file)[0]}_{labels[i]}.png"
#                 )
#                 plt.imsave(out_path, aug_images[i].numpy())
#




def focal_loss(gamma=2., alpha=0.25):
    import tensorflow as tf
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)

        # ubah label ke one-hot
        y_true_onehot = tf.one_hot(y_true, depth=y_pred.shape[-1])

        ce = tf.keras.losses.categorical_crossentropy(y_true_onehot, y_pred)
        pt = tf.reduce_sum(y_true_onehot * y_pred, axis=-1)

        loss = alpha * tf.pow((1 - pt), gamma) * ce
        return loss

    return focal_loss_fixed

def custom_model_summary(model):
    rows = []
    for i, layer in enumerate(model.layers):
        rows.append([
            i,
            layer.name,
            layer.__class__.__name__,
            str(layer.output_shape if hasattr(layer, "output_shape") else "-"),
            layer.count_params(),
            layer.trainable
        ])

    print("\n========== FULL MODEL SUMMARY ==========")
    print(tabulate(
        rows,
        headers=["Index", "Layer Name", "Type", "Output Shape", "Params", "Trainable"],
        tablefmt="grid"
    ))

def safe_unfreeze_blocks(backbone, n_blocks=2):
    """
    Unfreeze berdasarkan JUMLAH BLOCK terakhir.
    BatchNorm tetap frozen.
    """

    block_dict = {}
    block_pattern = re.compile(r"(block\d+[a-z]*)")

    for layer in backbone.layers:
        match = block_pattern.search(layer.name)
        if match:
            block_name = match.group(1)
        else:
            block_name = "other"   # stem / top

        if block_name not in block_dict:
            block_dict[block_name] = []
        block_dict[block_name].append(layer)

    # 2. Urutkan block berdasarkan angkanya
    sorted_blocks = sorted(
        [b for b in block_dict.keys() if b != "other"],
        key=lambda x: int(re.findall(r'\d+', x)[0])
    )

    # 3. Ambil N block terakhir
    blocks_to_unfreeze = sorted_blocks[-n_blocks:]
    print("\n🔓 Unfreezing blocks:", blocks_to_unfreeze)

    # 4. Set trainable=True untuk block tersebut (kecuali BN)
    for block_name in blocks_to_unfreeze:
        for layer in block_dict[block_name]:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = True

    # 5. Semua block lain → trainable=False
    for block_name in sorted_blocks:
        if block_name not in blocks_to_unfreeze:
            for layer in block_dict[block_name]:
                layer.trainable = False

    # 6. Tampilkan tabel status trainable
    table = []
    idx = 0

    for block_name in sorted_blocks:
        for layer in block_dict[block_name]:
            table.append([
                idx,
                layer.name,
                block_name,
                layer.trainable
            ])
            idx += 1

    print("\n========== BLOCK TRAINABLE TABLE ==========")
    print(tabulate(
        table,
        headers=["Index", "Layer Name", "Block", "Trainable"],
        tablefmt="grid"
    ))

    return blocks_to_unfreeze


def notif():
    url = "http://38.134.41.59:8080/message?token=AaekWDGvjiGO49P"

    try:
        payload = {
            "title": "Train model DARI VAST AI, done!",  # judul notifikasi
            "message": f"Time Training: {time.perf_counter() - timer}, Score macro = {macro_f1:.3f}, \nAccuracy {overall_acc:.3f}\n Model Name {hyperparams}, \n Model Report {report}",
            # isi pesan
            "priority": 0  # prioritas (1-10)
        }
    except Exception as e:
        payload = {
            "title": "ERROR, Train Done but not working (dont know the error)!",  # judul notifikasi
            "message": f"Training Done , Error Type: {e}",  # isi pesan
            "priority": 0  # prioritas (1-10)
        }

    response = requests.post(url, data=payload)

    if response.status_code == 200:
        print("Pesan berhasil dikirim!")
    else:
        print("Gagal mengirim pesan:", response.text)

def save_experiments(model, history, test_ds, class_names, hyperparams, save_dir=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if save_dir is None:
        save_dir = f"/kaggle/working/experiments_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    # ---- Save Hyperparameters ----
    with open(os.path.join(save_dir, "hyperparameters.json"), "w") as f:
        json.dump(hyperparams, f, indent=4)

    # ---- Save Model ----
    model.save(os.path.join(save_dir, "model_best.h5"))

    # ---- Save Summary ----
    with open(os.path.join(save_dir, "model_summary.txt"), "w") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))

    # ---- Plot Accuracy ----
    if history is not None:
        plt.figure()
        plt.plot(history.history['accuracy'], label='Train Acc')
        plt.plot(history.history['val_accuracy'], label='Val Acc')
        plt.xlabel("Epoch"); plt.ylabel("Accuracy")
        plt.legend(); plt.title("Train vs Validation Accuracy")
        plt.savefig(os.path.join(save_dir, "train_val_accuracy.jpg")); plt.close()

        # ---- Plot Loss ----
        plt.figure()
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.xlabel("Epoch"); plt.ylabel("Loss")
        plt.legend(); plt.title("Train vs Validation Loss")
        plt.savefig(os.path.join(save_dir, "train_val_loss.jpg")); plt.close()

    # ---- Collect Predictions ----
    y_true, y_pred, y_probs = [], [], []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_probs.extend(preds)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(labels.numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)

    # ---- Confusion Matrix (Raw) ----
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix (Raw)")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.savefig(os.path.join(save_dir, "confusion_matrix_raw.jpg"))
    plt.close()

    # ---- Confusion Matrix (Normalized) ----
    cm_norm = confusion_matrix(y_true, y_pred, normalize='true')
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix (Normalized)")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.savefig(os.path.join(save_dir, "confusion_matrix_normalized.jpg"))
    plt.close()

    # ---- Classification Report ----
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    with open(os.path.join(save_dir, "classification_report.json"), "w") as f:
        json.dump(report, f, indent=4)

    # ---- ROC-AUC ----
    try:
        y_true_bin = to_categorical(y_true, num_classes=len(class_names))
        auc_scores = {}

        plt.figure(figsize=(8, 6))
        for i, cls in enumerate(class_names):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
            score = auc(fpr, tpr)
            auc_scores[cls] = score
            plt.plot(fpr, tpr, label=f"{cls} (AUC={score:.2f})")

        macro_auc = roc_auc_score(y_true_bin, y_probs, average='macro')
        plt.plot([0,1],[0,1],'k--')
        plt.title(f"ROC Curve (macro AUC = {macro_auc:.3f})")
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend()
        plt.savefig(os.path.join(save_dir, "roc_auc_curve.jpg")); plt.close()

        with open(os.path.join(save_dir, "roc_auc_scores.json"), "w") as f:
            json.dump({"per_class": auc_scores, "macro_auc": macro_auc}, f, indent=4)

    except Exception as e:
        print("ROC AUC skipped:", e)

    # ---- SIMPAN HASIL PREDIKSI KE CSV ----
    results = pd.DataFrame({
        "true_label": [class_names[i] for i in y_true],
        "pred_label": [class_names[i] for i in y_pred],
    })
    prob_df = pd.DataFrame(y_probs, columns=[f"prob_{c}" for c in class_names])
    results = pd.concat([results, prob_df], axis=1)
    results.to_csv(os.path.join(save_dir, "predictions_test.csv"), index=True)

    print(f"✅ Semua hasil disimpan di: {save_dir}")



if __name__ == "__main__":
    notif()