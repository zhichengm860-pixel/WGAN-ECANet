import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    classification_report, top_k_accuracy_score
)


def compute_metrics(labels, predictions, probabilities=None, num_classes=24):

    accuracy = accuracy_score(labels, predictions)

    if probabilities is not None:
        all_labels = list(range(num_classes))
        top3_accuracy = top_k_accuracy_score(labels, probabilities, k=3, labels=all_labels)
        top5_accuracy = top_k_accuracy_score(labels, probabilities, k=5, labels=all_labels)
    else:
        top3_accuracy = None
        top5_accuracy = None

    f1_macro = f1_score(labels, predictions, average='macro')
    f1_weighted = f1_score(labels, predictions, average='weighted')
    f1_per_class = f1_score(labels, predictions, average=None)

    conf_matrix = confusion_matrix(labels, predictions)

    metrics = {
        'accuracy': float(accuracy),
        'top3_accuracy': float(top3_accuracy) if top3_accuracy else None,
        'top5_accuracy': float(top5_accuracy) if top5_accuracy else None,
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'f1_per_class': f1_per_class.tolist(),
        'confusion_matrix': conf_matrix.tolist(),
        'total_samples': len(labels)
    }

    return metrics


def print_classification_report(labels, predictions, target_names=None):

    if target_names is None:
        target_names = [f'Class_{i}' for i in range(max(labels) + 1)]

    report = classification_report(
        labels, predictions,
        target_names=target_names,
        digits=4
    )

    print("\nClassification report:")
    print(report)


def compute_class_accuracy(labels, predictions):

    classes = np.unique(labels)
    class_acc = {}

    for c in classes:
        mask = (labels == c)
        class_mask = predictions[mask]
        acc = np.mean(class_mask == c)
        class_acc[c] = acc

    return class_acc
