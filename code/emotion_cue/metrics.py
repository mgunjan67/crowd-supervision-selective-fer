"""Named, JSON-serializable evaluation metrics."""

from __future__ import annotations

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def classification_metrics(y_true, y_pred, class_names: tuple[str, ...]):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(range(len(class_names))), average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=list(range(len(class_names))), average="weighted", zero_division=0)),
        "per_class": classification_report(
            y_true,
            y_pred,
            labels=list(range(len(class_names))),
            target_names=list(class_names),
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=list(range(len(class_names))),
        ).tolist(),
    }
