CLASS_NAMES = class_names = class_name = [
    "akiec",  # Actinic keratoses and intraepithelial carcinoma / Bowen's disease
    "bcc",    # Basal cell carcinoma
    "bkl",    # Benign keratosis-like lesions
    "df",     # Dermatofibroma
    "mel",    # Melanoma
    "nv",     # Melanocytic nevi
    "vasc"    # Vascular lesions
]

HYPERPARAMS = {
    "model_name": "YOLOV11m-cls",
    "input_shape": (224, 224, 3),
    "num_classes": len(CLASS_NAMES),

    "optimizer": "Adam",
    "learning_rate": 1e-3,
    "batch_size": 32,
    "epochs": 25,
    "patience": 5,

    "ft_lr": 1e-5,
    "ft_epochs": 50,
    "ft_unfreeze_blocks": 18,
    "ft_weight_decay": 1e-5,

    "dropout": 0.3,
    "l2_reg": 1e-5
}


FIT_CONFIG = {
    "epochs": HYPERPARAMS["epochs"],
    "shuffle": False,
    "verbose": 1,
    "validation_data": None,
    "callbacks": None
}
