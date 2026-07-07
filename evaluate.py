import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from torch.utils.data import DataLoader

from configs.config import *
from models import build_model
from data.dataset import RiceDiseaseDataset
from data.transforms import get_valid_transforms
from utils.split_data import create_train_valid_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import numpy as np

def evaluate():

    print("=" * 60)
    print("Evaluating Model")
    print("=" * 60)

    # ------------------------------------
    # Load Dataset
    # ------------------------------------

    df = pd.read_csv(TRAIN_CSV)

    classes = sorted(df["label"].unique())

    class_to_idx = {
        cls: idx
        for idx, cls in enumerate(classes)
    }

    idx_to_class = {
        idx: cls
        for cls, idx in class_to_idx.items()
    }

    _, valid_df = create_train_valid_split(df)

    valid_dataset = RiceDiseaseDataset(
        dataframe=valid_df,
        image_dir=TRAIN_DIR,
        class_to_idx=class_to_idx,
        transform=get_valid_transforms()
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    # ------------------------------------
    # Load Model
    # ------------------------------------

    model = build_model(
        MODEL,
        NUM_CLASSES
    ).to(DEVICE)

    model.load_state_dict(
        torch.load(
            CHECKPOINT_NAME,
            map_location=DEVICE
        )
    )

    model.eval()

    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():

        for images, labels in valid_loader:

            images = images.to(DEVICE)

            outputs = model(images)

            probabilities = torch.softmax(outputs, dim=1)

            predictions = probabilities.argmax(1).cpu()

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

            all_predictions.extend(
                predictions.numpy()
            )

            all_labels.extend(
                labels.numpy()
            )

    # ------------------------------------
    # Metrics
    # ------------------------------------

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        average="weighted"
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        average="weighted"
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        average="weighted"
    )

    print(f"\nAccuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    # ------------------------------------
    # Save Metrics
    # ------------------------------------

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(
        OUTPUT_DIR / "metrics.txt",
        "w"
    ) as f:

        f.write(f"Accuracy : {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall   : {recall:.4f}\n")
        f.write(f"F1 Score : {f1:.4f}\n")

    # ------------------------------------
    # Classification Report
    # ------------------------------------

    report = classification_report(
        all_labels,
        all_predictions,
        target_names=classes
    )

    with open(
        OUTPUT_DIR / "classification_report.txt",
        "w"
    ) as f:

        f.write(report)

    print("\nClassification Report Saved")

    # ------------------------------------
    # Confusion Matrix
    # ------------------------------------

    cm = confusion_matrix(
        all_labels,
        all_predictions
    )

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes
    )

    plt.xlabel("Predicted")

    plt.ylabel("True")

    plt.title("Confusion Matrix")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "confusion_matrix.png",
        dpi=300
    )

    plt.close()

    print("Confusion Matrix Saved")

    # ------------------------------------
    # Prediction CSV
    # ------------------------------------

    predictions = pd.DataFrame({

        "Actual":
            [idx_to_class[x] for x in all_labels],

        "Predicted":
            [idx_to_class[x] for x in all_predictions]

    })

    predictions.to_csv(
        OUTPUT_DIR / "predictions.csv",
        index=False
    )

    print("Predictions Saved")

    # =====================================================
# ROC Curve
# =====================================================

    classes = list(range(NUM_CLASSES))

    labels_bin = label_binarize(
        all_labels,
        classes=classes
    )

    probabilities = np.array(all_probabilities)

    plt.figure(figsize=(8,8))

    for i in range(NUM_CLASSES):

        fpr, tpr, _ = roc_curve(
            labels_bin[:, i],
            probabilities[:, i]
        )

        roc_auc = auc(fpr, tpr)

        plt.plot(
            fpr,
            tpr,
            label=f"{idx_to_class[i]} (AUC={roc_auc:.3f})"
        )

    plt.plot(
        [0,1],
        [0,1],
        linestyle="--"
    )

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.title("Multi-class ROC Curve")

    plt.legend(
        fontsize=7,
        loc="lower right"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR/"roc_curve.png",
        dpi=300
    )

    plt.close()

    print("ROC Curve Saved")

    print("\nEvaluation Completed Successfully!")


if __name__ == "__main__":

    evaluate()