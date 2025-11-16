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
    "model_name": "efficientnetb3-finetuned-300",
    "input_shape": (224, 224, 3),
    "num_classes": 7,

    "optimizer": "Adam",
    "learning_rate": 1e-3,
    "batch_size": 8,
    "epochs": 5,
    "patience": 2
    
}

FIT_CONFIG = {
    "epochs": HYPERPARAMS["epochs"],
    "validation_data": None, 
    "shuffle": True,
    "verbose": 1,
    "callbacks": None, 
}