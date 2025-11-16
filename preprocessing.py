import os
import pandas as pd
import numpy as np
from PIL import Image
import gdown
import matplotlib.pyplot as plt
from utils import create_dataset_by_dx

class SkinDatasetPreprocessor:
    def __init__(self,  
                 dataset_dir,
                 base_image_dir="./Dataset HAM1000",
                 output_root="./preprocessed_datasets",
                 output_root_segment="./preprocessed_datasets_segment",
                 segmentations_dir=None):
        """
        segmentations_dir → lokasi folder mask segmentasi,
        contoh:
        /kaggle/input/ham10000-lesion-segmentations/HAM10000_segmentations_lesion_tschandl
        """

        self.base_image_dir = base_image_dir
        self.dataset_dir = dataset_dir
        self.output_root = output_root
        self.output_root_segment = output_root_segment
        self.segmentations_dir = segmentations_dir  # <--- input new

        # ---- Download original HAM10000 ----
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

    # -----------------------------------------------------------
    def load_csv(self):
        for split, filename in self.dict_datasets.items():
            path = os.path.join(self.base_image_dir, filename)
            self.dfs[split] = pd.read_csv(path)
            print(f"Loaded {split:<5}: {self.dfs[split].shape[0]} rows")

    # -----------------------------------------------------------
    def safe_create_dataset(self, csv_file, split_type):
        split_path = os.path.join(self.output_root, split_type)
        if os.path.exists(split_path) and len(os.listdir(split_path)) > 0:
            print(f"Folder '{split_type}' already exists — SKIP")
            return

        print(f"Creating dataset for '{split_type}'...")
        create_dataset_by_dx(
            csv_path=csv_file,
            image_root=self.dataset_dir,
            split_type=split_type,
            label_col="dx",
            output_folder=self.output_root,
        )
        print(f"Done creating '{split_type}'")

    # -----------------------------------------------------------
    def create_all_datasets(self):
        for mode, filename in self.dict_datasets.items():
            csv_path = os.path.join(self.base_image_dir, filename)
            self.safe_create_dataset(csv_path, mode)

    # -----------------------------------------------------------
    def plot_train_distribution(self):
        if "train" not in self.dfs:
            print("Train CSV not loaded. Call load_csv() first.")
            return

        counts = self.dfs["train"]["dx"].value_counts()

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        counts.plot(kind="bar", ax=ax)

        for i, count in enumerate(counts):
            ax.text(i, count + 1, str(count), ha="center", va="bottom", fontsize=10)

        ax.set_title("Jumlah Data per Kelas (dx) - TRAIN")
        ax.set_ylabel("Jumlah")
        ax.set_xlabel("Kelas Diagnosis (dx)")
        plt.tight_layout()
        plt.show()

    # -----------------------------------------------------------
    def apply_segmentation_masks(self, do_crop=True):

        if self.segmentations_dir is None:
            raise ValueError("ERROR: segmentations_dir belum di-set. Masukkan path lewat constructor!")

        print("\n=== APPLY SEGMENTATION MASKS ===\n")
        print(f"Using segmentation folder: {self.segmentations_dir}\n")

        for split in ["train", "val", "test"]:
            in_split_dir = os.path.join(self.output_root, split)
            out_split_dir = os.path.join(self.output_root_segment, split)

            if not os.path.isdir(in_split_dir):
                print(f"[SKIP] Split '{split}' tidak ditemukan.")
                continue

            print(f"Processing {split}...")

            for cls in os.listdir(in_split_dir):
                cls_in_dir = os.path.join(in_split_dir, cls)
                if not os.path.isdir(cls_in_dir):
                    continue

                cls_out_dir = os.path.join(out_split_dir, cls)
                os.makedirs(cls_out_dir, exist_ok=True)

                for fname in os.listdir(cls_in_dir):
                    if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                        continue

                    img_path = os.path.join(cls_in_dir, fname)
                    img = Image.open(img_path).convert("RGB")
                    img_np = np.array(img)

                    # ---- Mask name ----
                    base, _ = os.path.splitext(fname)
                    mask_name = f"{base}_segmentation.png"
                    mask_path = os.path.join(self.segmentations_dir, mask_name)

                    if not os.path.isfile(mask_path):
                        print(f"[WARNING] Mask missing: {mask_name}")
                        continue

                    mask = Image.open(mask_path).convert("L")
                    mask_np = np.array(mask)
                    mask_bool = mask_np > 0
                    mask_bool_3c = np.expand_dims(mask_bool, axis=-1)

                    segmented = img_np * mask_bool_3c
                    segmented_img = Image.fromarray(segmented)

                    # ---- CROP ----
                    if do_crop:
                        ys, xs = np.where(mask_bool)
                        if len(xs) == 0 or len(ys) == 0:
                            print(f"[WARNING] Empty mask: {mask_name}")
                            continue

                        x_min, x_max = xs.min(), xs.max()
                        y_min, y_max = ys.min(), ys.max()

                        segmented_img = segmented_img.crop((x_min, y_min, x_max, y_max))

                    # Save output
                    segmented_img.save(os.path.join(cls_out_dir, fname))

            print(f"[DONE] {split}\n")

        print("\n=== ALL SEGMENTATION DONE! ===\n")
        
