import os
import random
import time
from datetime import datetime

import numpy as np
import requests
from ultralytics import YOLO
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from ultralytics.utils.loss import FocalLoss


def on_setup(trainer):
    m = getattr(trainer.model, "model", trainer.model)
    m.criterion = FocalLoss(gamma=2.0, alpha=0.25)
    trainer.loss_names = ["fl"]

def notif(timer, macro_f1,  hyperparams,):
    url = "http://38.134.41.59:8080/message?token=AaekWDGvjiGO49P"

    try:
        elapsed = time.perf_counter() - timer
        payload = {
            "title": "Train model DARI VAST AI, done!",
            "message": f"Time Training: {elapsed}s\n"
                       f"Macro F1: {macro_f1}\n"
                       f"Hyperparams: {hyperparams}\n",
            "priority": 10
        }
    except Exception as e:
        payload = {
            "title": "ERROR, Train Done but not working!",
            "message": f"Training Done, Error Type: {e}",
            "priority": 10
        }

    try:
        response = requests.post(url, data=payload)
        if response.status_code == 200:
            print("Pesan berhasil dikirim!")
        else:
            print("Gagal mengirim pesan:", response.text)
    except Exception as e:
        print("Gagal mengirim notifikasi:", e)


def evaluate_yolo_classification(model_path, test_dir: str):
    model = model_path
    class_names = model.names
    class_list = [class_names[i] for i in range(len(class_names))]

    y_true = []
    y_pred = []

    for class_name in class_list:
        class_folder = os.path.join(test_dir, class_name)

        if not os.path.isdir(class_folder):
            print(f"[WARNING] Missing test folder for class: {class_name}")
            continue

        for filename in os.listdir(class_folder):
            img_path = os.path.join(class_folder, filename)

            res = model.predict(img_path, verbose=False)[0]
            pred_class_idx = res.probs.top1

            y_true.append(class_list.index(class_name))
            y_pred.append(pred_class_idx)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    cm = confusion_matrix(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    return cm, macro_f1, class_list


SAVE_ROOT = "../runs"
os.makedirs(SAVE_ROOT, exist_ok=True)


# FIXED hyperparams you researched
HYPERPARAM_SPACE = {
    "lr0": [0.01],
    "momentum": [0.90],
    "weight_decay": [0.0001],
    "optimizer": ["SGD"]
}



def random_sample_hyperparams():
    return {k: random.choice(v) for k, v in HYPERPARAM_SPACE.items()}


def fine_tune_model(base_model_path, train_data_path, test_dir, trial_id, imgsz):
    hp = random_sample_hyperparams()
    print(f"Trial {trial_id} hyperparams: {hp}")

    timer = time.perf_counter()
    model = YOLO(base_model_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trial_folder = os.path.join(SAVE_ROOT, f"trial_{trial_id}_{timestamp}")
    os.makedirs(trial_folder, exist_ok=True)

    model.add_callback("on_train_start",on_setup)

    model.train(
        data=train_data_path,
        epochs=50,
        device=0,
        batch=32,
        patience=10,
        lr0=hp["lr0"],
        momentum=hp["momentum"],
        weight_decay=hp["weight_decay"],
        optimizer=hp["optimizer"],
        imgsz=imgsz,
        # ALL AUGMENT OFF
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
        project=trial_folder,
        name="finetune"
    )

    trained_model_path = os.path.join(trial_folder, "finetune", "weights", "last.pt")
    trained_model_path2 = os.path.join(trial_folder, "finetune", "weights", "best.pt")
    trained_model = YOLO(trained_model_path)
    trained_model_best = YOLO(trained_model_path2)

    cm, macro_f1, cls_list = evaluate_yolo_classification(trained_model, test_dir)
    cm2, macro_f12, cls_list2 = evaluate_yolo_classification(trained_model, test_dir)

    print(f"\n=== Trial {trial_id} Completed ===")
    print("Macro F1:", macro_f1)
    print("Confusion Matrix:\n", cm)
    print("Class List:\n", cls_list)

    notif(timer=timer, macro_f1=macro_f1, hyperparams=cm)
    notif(timer=timer, macro_f1=macro_f12, hyperparams=cm2)
    return macro_f1, trial_folder


def main():
    base_model_path = r"yolo11x-cls.pt"
    train_data_path = "../root/augmented_balanced_dataset2"
    test_dir = r"../root/augmented_balanced_dataset2/test"

    best_f1 = -1
    best_folder = None

    # RANDOMIZED BUT SAFE IMGSZ (224–256)
    imgsz = [640]

    try:
        for trial, data in enumerate(imgsz):
            f1, folder = fine_tune_model(
                base_model_path,
                train_data_path,
                test_dir,
                trial,
                data
            )
            if f1 > best_f1:
                best_f1 = f1
                best_folder = folder
    except Exception as e:
        print("ERROR:", e)

    print("=" * 60)
    print(f"BEST MACRO F1 = {best_f1:.4f}")
    print(f"BEST MODEL FOLDER = {best_folder}")


if __name__ == "__main__":
    main()
