import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    classification_report, top_k_accuracy_score
)
from typing import Dict, List, Optional, Tuple


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


def compute_snr_accuracy(
    snr_values: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    snr_bins: Optional[List[int]] = None
) -> Dict[str, float]:
    """
    Compute accuracy under different SNR conditions
    
    Args:
        snr_values: SNR values array, shape (N,)
        labels: True labels
        predictions: Predicted labels
        snr_bins: SNR groups, default [-20, -10, 0, 10, 20, 30]
    
    Returns:
        Dictionary containing accuracy for each SNR range
    """
    if snr_bins is None:
        snr_bins = [-20, -10, 0, 10, 20, 30]
    
    snr_accuracy = {}
    
    for i in range(len(snr_bins) - 1):
        low, high = snr_bins[i], snr_bins[i+1]
        mask = (snr_values >= low) & (snr_values < high)
        
        if mask.sum() > 0:
            acc = accuracy_score(labels[mask], predictions[mask])
            key = f'snr_{low}_to_{high}'
            snr_accuracy[key] = float(acc)
            snr_accuracy[f'{key}_count'] = int(mask.sum())
    
    for snr in snr_bins:
        mask = (snr_values == snr)
        if mask.sum() > 0:
            acc = accuracy_score(labels[mask], predictions[mask])
            key = f'snr_{snr}'
            snr_accuracy[key] = float(acc)
            snr_accuracy[f'{key}_count'] = int(mask.sum())
    
    return snr_accuracy


def compute_snr_class_accuracy(
    snr_values: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    num_classes: int = 24
) -> Dict[int, Dict[int, float]]:
    """
    Compute accuracy for each class under different SNR conditions
    
    Args:
        snr_values: SNR values array
        labels: True labels
        predictions: Predicted labels
        num_classes: Number of classes
    
    Returns:
        Nested dictionary: {snr: {class_id: accuracy}}
    """
    unique_snrs = np.unique(snr_values)
    snr_class_acc = {}
    
    for snr in unique_snrs:
        mask = (snr_values == snr)
        snr_labels = labels[mask]
        snr_preds = predictions[mask]
        
        class_acc = {}
        for c in range(num_classes):
            class_mask = (snr_labels == c)
            if class_mask.sum() > 0:
                acc = np.mean(snr_preds[class_mask] == c)
                class_acc[c] = float(acc)
        
        snr_class_acc[int(snr)] = class_acc
    
    return snr_class_acc


def print_snr_report(
    snr_values: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    snr_bins: Optional[List[int]] = None
):
    """
    Print SNR dimension evaluation report
    
    Args:
        snr_values: SNR values array
        labels: True labels
        predictions: Predicted labels
        snr_bins: SNR groups
    """
    if snr_bins is None:
        snr_bins = [-20, -10, 0, 10, 20, 30]
    
    print("\n" + "=" * 60)
    print("SNR Robustness Analysis")
    print("=" * 60)
    
    print(f"\n{'SNR Range':<15} {'Accuracy':>10} {'Samples':>10}")
    print("-" * 40)
    
    for i in range(len(snr_bins) - 1):
        low, high = snr_bins[i], snr_bins[i+1]
        mask = (snr_values >= low) & (snr_values < high)
        
        if mask.sum() > 0:
            acc = accuracy_score(labels[mask], predictions[mask])
            print(f"[{low:3d}, {high:3d})     {acc*100:>8.2f}% {mask.sum():>10}")
    
    print("-" * 40)
    
    print(f"\n{'Specific SNR':<15} {'Accuracy':>10} {'Samples':>10}")
    print("-" * 40)
    
    unique_snrs = sorted(np.unique(snr_values))
    for snr in unique_snrs:
        mask = (snr_values == snr)
        if mask.sum() > 0:
            acc = accuracy_score(labels[mask], predictions[mask])
            print(f"{snr:>6} dB       {acc*100:>8.2f}% {mask.sum():>10}")
    
    print("-" * 40)
    
    overall_acc = accuracy_score(labels, predictions)
    print(f"\n{'Overall':<15} {overall_acc*100:>8.2f}% {len(labels):>10}")
    print("=" * 60)


def compute_confusion_matrix_by_snr(
    snr_values: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    snr: int
) -> np.ndarray:
    """
    Compute confusion matrix for specific SNR
    
    Args:
        snr_values: SNR values array
        labels: True labels
        predictions: Predicted labels
        snr: Target SNR
    
    Returns:
        Confusion matrix
    """
    mask = (snr_values == snr)
    if mask.sum() == 0:
        return np.array([])
    
    return confusion_matrix(labels[mask], predictions[mask])
