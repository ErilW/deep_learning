from config import CLASS_NAMES, HYPERPARAMS
# from models.base_model import ModelBuilder
# from models.model_trainer import ModelTrainer
from preprocessing import SkinDatasetPreprocessor
import kagglehub
import os

# from test import evaluate_yolo_classification
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
    )

    preprocessor.load_csv()
    preprocessor.create_all_datasets()
    # preprocessor.plot_train_distribution()
    preprocessor.apply_segmentation_masks(output_segmentations_dir, True, size=size, )


def main():
    path_ham10000 = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")
    path_segmentations = kagglehub.dataset_download("tschandl/ham10000-lesion-segmentations")
    path_segmentations = f"{path_segmentations}/HAM10000_segmentations_lesion_tschandl"

    datasets_output = "./root/preprocessed_datasets"
    datasets_segmentation = "./root/segmentation_masks"
    output_experiments = f"experiments_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_experiments, exist_ok=True)

    preprocessing(
        ham_path=path_ham10000,
        segmentations_path=path_segmentations,
        output_dir=datasets_output,
        output_segmentations_dir=datasets_segmentation,
        size=None
    )

    notif(None, None, None, None, None)


if __name__ == "__main__":
    main()