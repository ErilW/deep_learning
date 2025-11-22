from tqdm import tqdm
import pandas as pd
import numpy as np
import gdown
import matplotlib.pyplot as plt
from utils import create_dataset_by_dx
import os
from PIL import Image
import albumentations as A
import cv2

def augment_dataset(
    input_root,
    output_root,
    classes_no_aug=["NV", "mel", "bkl", "bcc"],
    augment_factor=5
):

    os.makedirs(output_root, exist_ok=True)

    aug = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.25),
        A.GaussianBlur(p=0.1)
    ])

    # Loop setiap kelas
    for cls in os.listdir(input_root):
        cls_input = os.path.join(input_root, cls)
        cls_output = os.path.join(output_root, cls)
        os.makedirs(cls_output, exist_ok=True)

        # kelas yang tidak diaugment
        if cls in classes_no_aug:
            print(f"❌ Skip augmentasi untuk kelas: {cls}")
            # copy semua file apa adanya
            for img_file in os.listdir(cls_input):
                src = os.path.join(cls_input, img_file)
                dst = os.path.join(cls_output, img_file)
                os.system(f"cp '{src}' '{dst}'")
            continue

        print(f"🚀 Augmentasi kelas: {cls}")

        for img_file in os.listdir(cls_input):
            img_path = os.path.join(cls_input, img_file)

            image = cv2.imread(img_path)
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Simpan original
            cv2.imwrite(os.path.join(cls_output, img_file), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            # Bikin 5 augmentasi
            for i in range(augment_factor):
                augmented = aug(image=image)["image"]
                out_name = img_file.replace(".jpg", f"_aug{i}.jpg")
                cv2.imwrite(
                    os.path.join(cls_output, out_name),
                    cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
                )

    print("\n🎉 DONE! Dataset baru siap dipakai:")
    print(output_root)

class SkinDatasetPreprocessor:
    def __init__(self,  
                 dataset_dir,
                 base_image_dir="./Dataset HAM1000",
                 output_root="./preprocessed_datasets",
                 output_root_segment="./preprocessed_datasets_segment",
                 segmentations_dir=None):

        self.base_image_dir = base_image_dir
        self.dataset_dir = dataset_dir
        self.output_root = output_root
        self.output_root_segment = output_root_segment
        self.segmentations_dir = segmentations_dir


        gdown.download_folder(
            url="https://drive.google.com/drive/folders/17jgvIeKQnUvk6DmdUJWCIA_fcMxbZ_h_",
            output=self.base_image_dir,
            quiet=False,
            use_cookies=False,
        )

        self.dict_datasets = {
            "test": "test_hidden.csv",
            "train": "train.csv",
            "val": "val_public.csv",
        }
        self.dfs = {}

    # ============================================================
    def load_csv(self):
        for split, filename in self.dict_datasets.items():
            path = os.path.join(self.base_image_dir, filename)
            self.dfs[split] = pd.read_csv(path)
            print(f"Loaded {split:<5}: {self.dfs[split].shape[0]} rows")

    # ============================================================
    def safe_create_dataset(self, csv_file, split_type):
        split_path = os.path.join(self.output_root, split_type)
        if os.path.exists(split_path) and len(os.listdir(split_path)) > 0:
            print(f"[SKIP] Folder '{split_type}' sudah ada dan tidak kosong")
            return

        print(f"[CREATE] Dataset '{split_type}'...")
        create_dataset_by_dx(
            csv_path=csv_file,
            image_root=self.dataset_dir,
            split_type=split_type,
            label_col="dx",
            output_folder=self.output_root,
        )
        print(f"[DONE] '{split_type}'")

    # ============================================================
    def create_all_datasets(self):
        for mode, filename in self.dict_datasets.items():
            csv_path = os.path.join(self.base_image_dir, filename)
            self.safe_create_dataset(csv_path, mode)

    # ============================================================
    def plot_train_distribution(self):
        if "train" not in self.dfs:
            print("Train CSV not loaded. Call load_csv() first.")
            return

        counts = self.dfs["train"]["dx"].value_counts()

        plt.figure(figsize=(10, 5))
        ax = counts.plot(kind="bar")

        for i, count in enumerate(counts):
            ax.text(i, count + 1, str(count), ha="center", fontsize=10)

        ax.set_title("Jumlah Data per Kelas - TRAIN")
        ax.set_ylabel("Jumlah")
        ax.set_xlabel("Kelas Diagnosis (dx)")
        plt.tight_layout()
        plt.show()

    # ============================================================
    #       SAFE & TQDM — APPLY SEGMENTATION MASKS
    # ============================================================
    def apply_segmentation_masks(self, do_crop=True, size=(224, 224)):

        if self.segmentations_dir is None:
            raise ValueError("ERROR: segmentations_dir belum diisi!")

        print("\n=== APPLY SEGMENTATION MASKS ===")
        print(f"Segmentation folder: {self.segmentations_dir}\n")

        # Skip entire process if already created
        if os.path.exists(self.output_root_segment):
            if any(os.scandir(self.output_root_segment)):
                print("[SKIP] Folder segmentasi sudah ada dan berisi file.")
                return

        for split in ["train", "val", "test"]:
            src_split_path = os.path.join(self.output_root, split)
            dst_split_path = os.path.join(self.output_root_segment, split)

            if not os.path.isdir(src_split_path):
                print(f"[SKIP] Folder '{split}' tidak ditemukan.")
                continue

            print(f"[PROCESS] {split}...")
            os.makedirs(dst_split_path, exist_ok=True)

            classes = [d for d in os.listdir(src_split_path)
                       if os.path.isdir(os.path.join(src_split_path, d))]

            for cls in classes:
                cls_src = os.path.join(src_split_path, cls)
                cls_dst = os.path.join(dst_split_path, cls)
                os.makedirs(cls_dst, exist_ok=True)

                images = [f for f in os.listdir(cls_src)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

                for fname in tqdm(images, desc=f"{split}/{cls}", ncols=80):

                    img_path = os.path.join(cls_src, fname)
                    img = Image.open(img_path).convert("RGB")
                    img_np = np.array(img)

                    # find mask
                    base, _ = os.path.splitext(fname)
                    mask_name = f"{base}_segmentation.png"
                    mask_path = os.path.join(self.segmentations_dir, mask_name)

                    if not os.path.isfile(mask_path):
                        continue

                    # load mask
                    mask = Image.open(mask_path).convert("L")
                    mask_np = np.array(mask)
                    mask_bool = mask_np > 0

                    mask3 = np.repeat(mask_bool[:, :, None], 3, axis=2)
                    result = img_np * mask3
                    segmented = Image.fromarray(result)

                    # crop bounding box
                    if do_crop:
                        ys, xs = np.where(mask_bool)
                        if len(xs) > 0:
                            x1, x2 = xs.min(), xs.max()
                            y1, y2 = ys.min(), ys.max()
                            segmented = segmented.crop((x1, y1, x2, y2))

                    # 🔥 **RESIZE DI SINI**
                    if size is not None:
                        segmented = segmented.resize(size, Image.Resampling.LANCZOS)

                    # save
                    segmented.save(os.path.join(cls_dst, fname))

            print(f"[DONE] {split}")

        print("\n=== SEGMENTATION COMPLETE ===\n")

if __name__ == "__main__":
    augment_dataset(
        input_root="./root/segmentation_masks/train",
        output_root="./root/segmentation_masks2/train",
        classes_no_aug=["NV", "mel", "bkl", "bcc"],
        augment_factor=6
    )
