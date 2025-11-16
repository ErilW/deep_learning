import pandas as pd
import shutil
from config import CLASS_NAMES
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os
import tqdm
import matplotlib.pyplot as plt


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

def build_augmentation_layer(
    max_angle=10,
    max_shift=0.05,
    contrast_strength=0.05,
    noise_std=0.03,
):
    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(max_angle / 180.0),
            layers.RandomTranslation(max_shift, max_shift),
            layers.RandomContrast(contrast_strength),
            layers.GaussianNoise(noise_std),
        ],
        name="augmentation",
    )


# =========================================
# 2. OVERSAMPLING / UNDERSAMPLING
# =========================================
def balance_dataset(raw_train, class_counts, ratio=1.0, undersample=False):
    """
    Mengembalikan dataset balanced (oversample atau undersample).
    """
    if undersample:
        target = int(min(class_counts.values()) * ratio)
    else:
        target = int(max(class_counts.values()) * ratio)

    print("Target per-class =", target)

    datasets = []

    for cls, count in class_counts.items():

        # Filter dataset untuk 1 kelas
        ds_cls = raw_train.filter(lambda x, y: tf.equal(y, cls))

        if undersample:
            # Batasi jumlah
            ds_cls = ds_cls.shuffle(4096).take(target)
        else:
            # Perbanyak jumlah
            repeat_factor = max(1, target // count)
            ds_cls = ds_cls.repeat(repeat_factor + 1)

        datasets.append(ds_cls)

    # Gabungkan semua kelas
    final = datasets[0]
    for ds in datasets[1:]:
        final = final.concatenate(ds)

    return final.shuffle(4096)
    
def show_batch(dataset, aug_layer=None, max_images=9):
    plt.figure(figsize=(12, 12))

    for images, labels in dataset.take(1):
        images = tf.image.convert_image_dtype(images, tf.float32)

        if aug_layer is not None:
            images = aug_layer(images, training=True)

        imgs = images.numpy()
        imgs = (imgs * 255).astype("uint8")

        labels = labels.numpy()
        count = min(max_images, imgs.shape[0])
        rows = int(np.ceil(count / 3))

        for i in range(count):
            plt.subplot(rows, 3, i + 1)
            plt.imshow(imgs[i])
            plt.title(str(labels[i]))
            plt.axis("off")

    plt.tight_layout()
    plt.show()

# =========================================
# 3. DATASET LOADER
# =========================================

def load_ham10000(
    base_dir="dataset",
    img_size=(224,224),
    batch_size=32,
    augment=True,
    balance=False,
    undersample=False,
    ratio=1.0,
    class_names=None,
):

    AUTOTUNE = tf.data.AUTOTUNE
    aug = build_augmentation_layer()

    if class_names is None:
        raise ValueError("class_names harus diisi (list nama kelas).")

    # ------------------------------------
    # A. LOAD RAW TRAIN (batch=1 jika balance=True)
    # ------------------------------------
    train_raw = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(base_dir, "train"),
        label_mode="int",
        class_names=class_names,
        image_size=img_size,
        shuffle=True,
        batch_size=1 if balance else batch_size
    ).map(lambda x,y: (tf.image.convert_image_dtype(x, tf.float32), y))

    # Hitung distribusi
    class_counts = None

    if balance:
        class_counts = {i:0 for i in range(len(class_names))}
        for _, lbl in train_raw:
            class_counts[int(lbl.numpy()[0])] += 1

        print("\nClass Counts:", class_counts)

        # Balance dataset
        train_raw = balance_dataset(train_raw, class_counts, ratio, undersample)

        # Resize batch kembali
        train_ds = train_raw.batch(batch_size)

    else:
        # No balancing
        train_ds = train_raw

    # ------------------------------------
    # B. Tambah augmentasi
    # ------------------------------------
    if augment:
        train_ds = train_ds.map(
            lambda x,y: (aug(x, training=True), y),
            num_parallel_calls=AUTOTUNE
        )

    train_ds = train_ds.prefetch(AUTOTUNE)

    # ------------------------------------
    # C. VAL
    # ------------------------------------
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(base_dir, "val"),
        label_mode="int",
        class_names=class_names,
        image_size=img_size,
        shuffle=False,
        batch_size=batch_size
    ).map(lambda x,y: (tf.image.convert_image_dtype(x, tf.float32), y)
    ).prefetch(AUTOTUNE)

    # ------------------------------------
    # D. TEST
    # ------------------------------------
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(base_dir, "test"),
        label_mode="int",
        class_names=class_names,
        image_size=img_size,
        shuffle=False,
        batch_size=batch_size
    ).map(lambda x,y: (tf.image.convert_image_dtype(x, tf.float32), y)
    ).prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds


def focal_loss(gamma=2., alpha=0.25):

    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)

        # ubah label ke one-hot
        y_true_onehot = tf.one_hot(y_true, depth=y_pred.shape[-1])

        ce = tf.keras.losses.categorical_crossentropy(y_true_onehot, y_pred)
        pt = tf.reduce_sum(y_true_onehot * y_pred, axis=-1)

        loss = alpha * tf.pow((1 - pt), gamma) * ce
        return loss

    return focal_loss_fixed
