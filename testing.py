import os
from datetime import datetime

import kagglehub
from ultralytics import YOLO
import argparse

from main import preprocessing


def create_datasets():
    path_ham10000 = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")
    path_segmentations = kagglehub.dataset_download("tschandl/ham10000-lesion-segmentations")
    path_segmentations = f"{path_segmentations}/HAM10000_segmentations_lesion_tschandl"

    datasets_output = "./root/preprocessed_datasets5"
    datasets_segmentation = "./root/segmentation_masks5"
    output_experiments = f"experiments_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_experiments, exist_ok=True)

    preprocessing(
        ham_path=path_ham10000,
        segmentations_path=path_segmentations,
        output_dir=datasets_output,
        output_segmentations_dir=datasets_segmentation,
        size=None
    )


params=dict(
    data=output_root,
    patience=20,
    epochs=100,
    batch=16,
    imgsz=640,
    lr0=0.01,
    momentum=0.9,
    weight_decay=0.0001,
    device=0,
    optimizer="SGD",
    augment=False,
    mosaic=0.0,
    mixup=0.0,
    copy_paste=0.0,
    auto_augment=None,
    erasing=0.0,
    hsv_h=0.0,
    hsv_s=0.0,
    hsv_v=0.0,
    translate=0.0,
    scale=0.0,
    fliplr=0.0,
    flipud=0.0,
)

model = YOLO(**params)

k_fold = 5

if "__main__" == __name__:
    args =
