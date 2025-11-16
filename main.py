from config import CLASS_NAMES, HYPERPARAMS
# from models.base_model import ModelBuilder
# from models.model_trainer import ModelTrainer
from preprocessing import SkinDatasetPreprocessor
import kagglehub
import os
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

    datasets_output = "/root/preprocessed_datasets3"
    datasets_segmentation = "/root/segmentation_masks3"
    output_experiments = f"experiments_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_experiments, exist_ok=True)

    preprocessing(
        ham_path=path_ham10000,
        segmentations_path=path_segmentations,
        output_dir=datasets_output,
        output_segmentations_dir=datasets_segmentation,
        size=HYPERPARAMS["input_shape"][:2]
    )
    output_root="/root/augmented_balanced_dataset3"
    path = create_yolo_balanced_dataset(input_root=datasets_segmentation, output_root=output_root, )
    model = YOLO("yolo11m-cls.pt")

    data = model.train(
        data=output_root,
        epochs=HYPERPARAMS["epochs"],
        batch=HYPERPARAMS["batch_size"],
        imgsz=HYPERPARAMS["input_shape"][:2],
        name="yolov11n-ham10000-segmentation",
        patience=HYPERPARAMS["patience"],
    )

    print(data)
    notif()



    # show_augment_per_class(base_dir=datasets_segmentation, output_dir=output_experiments)

    # change based on folder you need
    # train, val, test = load_ham10000_tensor(
    #         base_dir=datasets_segmentation,
    #         img_size=HYPERPARAMS["input_shape"][:2],
    #         batch_size=HYPERPARAMS["batch_size"],
    #         augment=True,
    #         balance=True,
    #         undersample=False,
    #         ratio=1.0,
    #         class_names=CLASS_NAMES,
    # )


    # print(f"total val: {len(val)}")
    # print(f"total test: {len(test)}")
    #
    # builder = ModelBuilder(
    #     HYPERPARAMS["input_shape"],
    #     HYPERPARAMS["num_classes"]
    # )
    #
    # backbone = builder.build_efficientnet()
    # model = builder.build_model(backbone)
    #
    # loss = focal_loss(gamma=2.0, alpha=0.25)
    # trainer = ModelTrainer(model, loss, HYPERPARAMS)
    #
    # trainer.compile_model()
    # print("=> Starting Stage 1 training")
    # try:
    #     history_stage1 = trainer.train_stage1(train, val)
    #
    # # # 6) Optional fine-tune
    #     if HYPERPARAMS.get("ft_epochs", 0) > 0:
    #         print("=> Starting Fine-tuning (Stage 2)")
    #         history_ft = trainer.train_stage2(train, val, backbone)
    # except KeyboardInterrupt:
    #     print("Training interrupted. Proceeding to evaluation...")
    # # 7) Evaluate on test set
    # print("=> Evaluating on test set")
    #
    # # out_dir = "./saved_models"
    # # out_path = os.path.join(out_dir, f"{HYPERPARAMS['model_name']}_final.h5")
    # # model.save(out_path)
    # # print(f"Model saved to {out_path}")
    #
    # save_experiments(
    #     trainer.model,
    #     None if 'HISTORY_STAGE1' not in locals() else history_stage1,
    #     test,
    #     CLASS_NAMES,
    #     HYPERPARAMS,
    #     save_dir=output_experiments
    # )

    notif()


if __name__ == "__main__":
    main()