"""
Evaluation metrics for EEG-BCI classification.

Implements metrics following CBraMod paper conventions:
- Binary: Balanced Accuracy, AUROC, AUC-PR
- Multi-class: Balanced Accuracy, Cohen's Kappa, Weighted F1
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Union, Tuple
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    classification_report,
)


def balanced_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    """
    Calculate balanced accuracy.

    Balanced accuracy is the average of recall for each class.
    Handles class imbalance well.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        Balanced accuracy score [0, 1]
    """
    return balanced_accuracy_score(y_true, y_pred)


def cohens_kappa(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    """
    Calculate Cohen's Kappa coefficient.

    Measures agreement between predictions and ground truth,
    accounting for chance agreement.

    Interpretation:
    - < 0: Less than chance
    - 0.01-0.20: Slight
    - 0.21-0.40: Fair
    - 0.41-0.60: Moderate
    - 0.61-0.80: Substantial
    - 0.81-1.00: Almost perfect

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        Cohen's Kappa coefficient [-1, 1]
    """
    return cohen_kappa_score(y_true, y_pred)


def weighted_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    """
    Calculate weighted F1 score.

    Weights by class frequency, suitable for imbalanced data.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        Weighted F1 score [0, 1]
    """
    return f1_score(y_true, y_pred, average='weighted')


def auroc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    multi_class: str = 'ovr'
) -> float:
    """
    Calculate Area Under ROC Curve.

    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities [n_samples, n_classes]
        multi_class: Multi-class strategy ('ovr' or 'ovo')

    Returns:
        AUROC score [0, 1]
    """
    n_classes = y_prob.shape[1] if y_prob.ndim > 1 else 2

    if n_classes == 2:
        # Binary case
        probs = y_prob[:, 1] if y_prob.ndim > 1 else y_prob
        return roc_auc_score(y_true, probs)
    else:
        # Multi-class
        return roc_auc_score(
            y_true, y_prob,
            multi_class=multi_class,
            average='weighted'
        )


def auc_pr(
    y_true: np.ndarray,
    y_prob: np.ndarray
) -> float:
    """
    Calculate Area Under Precision-Recall Curve.

    More informative than AUROC for imbalanced datasets.

    Args:
        y_true: Ground truth labels (binary)
        y_prob: Predicted probabilities for positive class

    Returns:
        AUC-PR score [0, 1]
    """
    probs = y_prob[:, 1] if y_prob.ndim > 1 else y_prob
    return average_precision_score(y_true, probs)


def confusion_matrix_stats(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List] = None
) -> Dict:
    """
    Calculate confusion matrix and derived statistics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: Optional list of label names

    Returns:
        Dict with confusion matrix and per-class metrics
    """
    cm = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]

    # Per-class metrics
    per_class = {}
    for i in range(n_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        class_name = labels[i] if labels else f"class_{i}"
        per_class[class_name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': int(cm[i, :].sum()),
        }

    return {
        'confusion_matrix': cm.tolist(),
        'per_class': per_class,
    }


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    task_type: str = 'multi_class',
    labels: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Calculate all relevant metrics for a classification task.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional)
        task_type: 'binary' or 'multi_class'
        labels: Optional list of label names

    Returns:
        Dict of metric names to values
    """
    results = {}

    # Common metrics
    results['balanced_accuracy'] = balanced_accuracy(y_true, y_pred)
    results['cohens_kappa'] = cohens_kappa(y_true, y_pred)
    results['weighted_f1'] = weighted_f1(y_true, y_pred)
    results['accuracy'] = (y_true == y_pred).mean()

    # Probability-based metrics
    if y_prob is not None:
        try:
            results['auroc'] = auroc(y_true, y_prob)
        except Exception as e:
            results['auroc'] = None

        if task_type == 'binary':
            try:
                results['auc_pr'] = auc_pr(y_true, y_prob)
            except Exception as e:
                results['auc_pr'] = None

    # Confusion matrix stats
    cm_stats = confusion_matrix_stats(y_true, y_pred, labels)
    results['confusion_matrix'] = cm_stats['confusion_matrix']
    results['per_class'] = cm_stats['per_class']

    return results


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = 'cuda',
    task_type: str = 'multi_class'
) -> Dict:
    """
    Evaluate a model on a dataloader.

    Args:
        model: PyTorch model
        dataloader: Test dataloader
        device: Device to run on
        task_type: 'binary' or 'multi_class'

    Returns:
        Dict of evaluation metrics
    """
    model.eval()
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    all_preds = []
    all_probs = []
    all_labels = []

    for data, target in dataloader:
        data = data.to(device)

        output = model(data)

        # Get predictions and probabilities
        if task_type == 'binary':
            probs = torch.sigmoid(output)
            preds = (probs > 0.5).long().squeeze()
        else:
            probs = torch.softmax(output, dim=1)
            preds = output.argmax(dim=1)

        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        all_labels.append(target.numpy())

    # Concatenate
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels)

    # Calculate metrics
    return calculate_metrics(y_true, y_pred, y_prob, task_type)


def format_metrics_report(metrics: Dict) -> str:
    """
    Format metrics as a readable report.

    Args:
        metrics: Dict of metrics from calculate_metrics

    Returns:
        Formatted string report
    """
    lines = [
        "=" * 50,
        "Classification Report",
        "=" * 50,
        "",
        f"Accuracy:           {metrics['accuracy']:.4f}",
        f"Balanced Accuracy:  {metrics['balanced_accuracy']:.4f}",
        f"Cohen's Kappa:      {metrics['cohens_kappa']:.4f}",
        f"Weighted F1:        {metrics['weighted_f1']:.4f}",
    ]

    if metrics.get('auroc') is not None:
        lines.append(f"AUROC:              {metrics['auroc']:.4f}")

    if metrics.get('auc_pr') is not None:
        lines.append(f"AUC-PR:             {metrics['auc_pr']:.4f}")

    lines.append("")
    lines.append("Per-Class Metrics:")
    lines.append("-" * 50)

    for class_name, class_metrics in metrics.get('per_class', {}).items():
        lines.append(
            f"{class_name:15s} | "
            f"P: {class_metrics['precision']:.3f} | "
            f"R: {class_metrics['recall']:.3f} | "
            f"F1: {class_metrics['f1']:.3f} | "
            f"N: {class_metrics['support']}"
        )

    lines.append("=" * 50)

    return "\n".join(lines)


if __name__ == '__main__':
    # Test metrics
    np.random.seed(42)

    # Binary classification test
    print("Binary Classification Test")
    print("-" * 40)

    n_samples = 100
    y_true_binary = np.random.randint(0, 2, n_samples)
    y_pred_binary = np.random.randint(0, 2, n_samples)
    y_prob_binary = np.random.rand(n_samples, 2)
    y_prob_binary = y_prob_binary / y_prob_binary.sum(axis=1, keepdims=True)

    metrics_binary = calculate_metrics(
        y_true_binary, y_pred_binary, y_prob_binary,
        task_type='binary',
        labels=['Thumb', 'Pinky']
    )
    print(format_metrics_report(metrics_binary))

    # Multi-class test
    print("\nMulti-Class Classification Test")
    print("-" * 40)

    y_true_multi = np.random.randint(0, 4, n_samples)
    y_pred_multi = np.random.randint(0, 4, n_samples)
    y_prob_multi = np.random.rand(n_samples, 4)
    y_prob_multi = y_prob_multi / y_prob_multi.sum(axis=1, keepdims=True)

    metrics_multi = calculate_metrics(
        y_true_multi, y_pred_multi, y_prob_multi,
        task_type='multi_class',
        labels=['Thumb', 'Index', 'Middle', 'Pinky']
    )
    print(format_metrics_report(metrics_multi))
