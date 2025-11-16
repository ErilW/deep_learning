import os
import argparse
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
from ultralytics import YOLO


def evaluate_yolo_classification(model_path: str, test_dir: str):
    """
    Evaluasi YOLO Classification:
    - Confusion Matrix
    - Macro F1 Score
    """
    model = YOLO(model_path)
    class_names = model.names
    class_list = [class_names[i] for i in range(len(class_names))]

    y_true = []
    y_pred = []

    # Loop through test dataset
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


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO Classification Model")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path ke model YOLO .pt"
    )

    parser.add_argument(
        "--test_dir",
        type=str,
        required=True,
        help="Path ke folder test dataset (harus berisi subfolder kelas)"
    )

    args = parser.parse_args()

    cm, macro_f1, class_list = evaluate_yolo_classification(
        model_path=args.model,
        test_dir=args.test_dir
    )

    print("\n===== CONFUSION MATRIX =====")
    print(cm)

    print("\n===== MACRO F1 SCORE =====")
    print(macro_f1)

    print("\n===== CLASS ORDER =====")
    print(class_list)


if __name__ == "__main__":
    main()
