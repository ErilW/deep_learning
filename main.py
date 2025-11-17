from config import CLASS_NAMES, HYPERPARAMS
# from models.base_model import ModelBuilder
# from models.model_trainer import ModelTrainer
from preprocessing import SkinDatasetPreprocessor
import kagglehub
import os

from test import evaluate_yolo_classification
from utils import  notif, \
    create_yolo_balanced_dataset
from datetime import datetime
from ultralytics import YOLO


def preprocessing(ham_path, segmentations_path, output_dir="./preprocessed_datasets", output_segmentations_dir="./preprocessed_datasets_segment", size=None):
    """
    Proses preprocessing dataset HAM10000 + segmentation.
    """
    preprocessor = SkinDatasetPreprocessor(
        dataset_dir=ham_path,
        segmentations_dir=segmentations_path,
        output_root=output_dir,
        output_root_segment=output_segmentations_dir
    )

    preprocessor.load_csv()
    preprocessor.create_all_datasets()
    # preprocessor.plot_train_distribution()
    preprocessor.apply_segmentation_masks(True, size=size)


def main():
    path_ham10000 = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")
    path_segmentations = kagglehub.dataset_download("tschandl/ham10000-lesion-segmentations")
    path_segmentations = f"{path_segmentations}/HAM10000_segmentations_lesion_tschandl"

    datasets_output = "./root/preprocessed_datasets5"
    datasets_segmentation = "./root/segmentation_masks5"
    # output_experiments = f"experiments_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    # os.makedirs(output_experiments, exist_ok=True)

    preprocessing(
        ham_path=path_ham10000,
        segmentations_path=path_segmentations,
        output_dir=datasets_output,
        output_segmentations_dir=datasets_segmentation,
        size=HYPERPARAMS["input_shape"][:2]
    )
    output_root="./root/augmented_balanced_dataset5"
    path = create_yolo_balanced_dataset(input_root=datasets_segmentation, output_root=output_root, ratio=2 )
    model = YOLO("yolov8x-cls.pt")

    data = model.train(
        data=output_root,
        patience=5,
        epochs=30,
        batch=32,
        imgsz=224,
        lr0=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        optimizer="SGD",
        augment=False,
        # patience=10,
        # MATIKAN SEMUA AUGMENT
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

    notif()
    # evaluate_yolo_classification()


if __name__ == "__main__":
    main()