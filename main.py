from config import CLASS_NAMES
from preprocessing import SkinDatasetPreprocessor
import kagglehub
import os
from utils import load_ham10000

def preprocessing(ham_path, segmentations_path, output_dir="./preprocessed_datasets", output_segmentations_dir="./preprocessed_datasets_segment"):
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
    preprocessor.plot_train_distribution()
    preprocessing.apply_segmentation_masks(True)


def main():
    path_ham10000 = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")
    path_segmentations = kagglehub.dataset_download("tschandl/ham10000-lesion-segmentations")

    datasets_output = "./root/preprocessed_datasets"
    datasets_segmentation = "./root/segmentation_masks"

    preprocessing(
        ham_path=path_ham10000,
        segmentations_path=path_segmentations,
        output_dir=datasets_output,
        output_segmentations_dir=datasets_segmentation
    )

    # test, val, train = load_ham10000(
    #     base_dir=segment_datasets,
    #     img_size=HYPERPARAMS["input_shape"][:2],
    #     batch_size=HYPERPARAMS["batch_size"],
    #     augment=True
    # )

    # change based on folder you need
    load_ham10000(
            base_dir=datasets_segmentation,
            img_size=(224, 224),
            batch_size=32,
            augment=True,
            balance=False,
            undersample=False,
            ratio=1.0,
            class_names=CLASS_NAMES,
    )


if __name__ == "__main__":
    main()