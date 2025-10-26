"""
Model evaluation utilities for the Asteroid Hazard Classification System.

This module provides functions to evaluate a model with classification metrics,
plot and save confusion matrices, compare multiple models, save classification
reports, and an evaluation pipeline that orchestrates these steps.
"""
from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, model_name: str) -> Dict[str, float]:
    """Evaluate a model on the test set and compute classification metrics.

    Parameters
    ----------
    model : object
        Trained model instance with a predict method.
    X_test : np.ndarray
        Test feature matrix.
    y_test : np.ndarray
        Test target vector.
    model_name : str
        Name of the model (for logging).

    Returns
    -------
    Dict[str, float]
        Dictionary with accuracy, precision, recall, and f1-score.
    """
    print(f"[Eval] Evaluating model: {model_name}")
    y_pred = model.predict(X_test)

    # Ensure binary metrics use the correct average
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='binary')
    rec = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')

    metrics = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
    }
    print(f"[Eval] Metrics for {model_name}: {metrics}")
    return metrics


def plot_confusion_matrix(y_test: np.ndarray, y_pred: np.ndarray, save_path: str) -> None:
    """Plot confusion matrix as a seaborn heatmap and save to file.

    Parameters
    ----------
    y_test : np.ndarray
        Ground truth labels.
    y_pred : np.ndarray
        Predicted labels.
    save_path : str
        Destination file path (PNG) under 'results/'.
    """
    print(f"[Eval] Saving confusion matrix to: {save_path}")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print("[Eval] Confusion matrix saved.")


def save_classification_report(y_test: np.ndarray, y_pred: np.ndarray, save_path: str) -> None:
    """Save detailed classification report to a text file.

    Parameters
    ----------
    y_test : np.ndarray
        Ground truth labels.
    y_pred : np.ndarray
        Predicted labels.
    save_path : str
        Destination path under 'results/'.
    """
    print(f"[Eval] Saving classification report to: {save_path}")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    report = classification_report(y_test, y_pred)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print("[Eval] Classification report saved.")


def compare_models(models: Dict[str, object], X_test: np.ndarray, y_test: np.ndarray) -> Tuple[str, Dict[str, Dict[str, float]]]:
    """Compare all models and identify the best one by accuracy.

    Parameters
    ----------
    models : Dict[str, object]
        Mapping of model names to trained model instances.
    X_test : np.ndarray
        Test feature matrix.
    y_test : np.ndarray
        Test target vector.

    Returns
    -------
    Tuple[str, Dict[str, Dict[str, float]]]
        Best model name and a dictionary of metrics for all models.
    """
    print("[Eval] Comparing models...")
    comparison = {}
    best_name = None
    best_acc = -1.0

    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test, name)
        comparison[name] = metrics
        if metrics['accuracy'] > best_acc:
            best_acc = metrics['accuracy']
            best_name = name

    # Print formatted comparison table
    print("\n[Eval] Model Comparison:")
    print("Model                | Accuracy  | Precision | Recall   | F1")
    print("---------------------+-----------+-----------+----------+----------")
    for name, m in comparison.items():
        print(f"{name:21} | {m['accuracy']:.4f}   | {m['precision']:.4f}   | {m['recall']:.4f}  | {m['f1']:.4f}")

    print(f"\n[Eval] Best model: {best_name} (accuracy={best_acc:.4f})")
    return best_name, comparison


def evaluation_pipeline(models: Dict[str, object], X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, object]:
    """Run the complete evaluation pipeline.

    Steps:
    1) Compare models and select best
    2) Generate and save confusion matrix for best model
    3) Save classification report for best model

    Parameters
    ----------
    models : Dict[str, object]
        Mapping of model names to trained model instances.
    X_test : np.ndarray
        Test feature matrix.
    y_test : np.ndarray
        Test target vector.

    Returns
    -------
    Dict[str, object]
        Dictionary containing:
        - 'best_model_name': str
        - 'metrics': Dict of metrics per model
        - 'confusion_matrix_path': str
        - 'classification_report_path': str
        - 'best_accuracy': float
    """
    try:
        best_name, metrics = compare_models(models, X_test, y_test)
        best_model = models[best_name]
        y_pred_best = best_model.predict(X_test)

        cm_path = os.path.join('results', 'confusion_matrix.png')
        report_path = os.path.join('results', 'classification_report.txt')

        plot_confusion_matrix(y_test, y_pred_best, cm_path)
        save_classification_report(y_test, y_pred_best, report_path)

        result = {
            'best_model_name': best_name,
            'metrics': metrics,
            'confusion_matrix_path': cm_path,
            'classification_report_path': report_path,
            'best_accuracy': metrics[best_name]['accuracy'],
        }
        print("[Pipeline] Evaluation pipeline completed successfully.")
        return result
    except Exception as e:
        print(f"[Pipeline][Error] Evaluation pipeline failed: {e}")
        raise