"""
Lightweight utility helpers shared across the MekaNet package.
"""

from typing import Dict, Iterable, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def calculate_metrics(y_true: Iterable[int], y_pred: Iterable[int]) -> Dict[str, float]:
    """Return common classification metrics as floats."""
    y_true = np.asarray(list(y_true))
    y_pred = np.asarray(list(y_pred))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def plot_confusion_matrix(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    class_names: Optional[Iterable[str]] = None,
    save_path: Optional[str] = None,
):
    """Create a simple confusion-matrix figure and optionally save it."""
    cm = confusion_matrix(list(y_true), list(y_pred))
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)

    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]

    class_names = list(class_names)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def visualize_detections(image: np.ndarray, predictions, color=(0, 255, 0), thickness: int = 2) -> np.ndarray:
    """Draw bounding boxes from standardized prediction dictionaries."""
    output = image.copy()
    for prediction in predictions:
        bbox = prediction.get("bbox", {})
        x1 = int(bbox.get("x1", 0))
        y1 = int(bbox.get("y1", 0))
        x2 = int(bbox.get("x2", 0))
        y2 = int(bbox.get("y2", 0))
        score = prediction.get("score")

        cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)

        if score is not None:
            label = f"{score:.2f}"
            cv2.putText(
                output,
                label,
                (x1, max(y1 - 8, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

    return output
