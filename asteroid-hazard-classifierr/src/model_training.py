"""
Model training utilities for the Asteroid Hazard Classification System.

This module provides functions to split data, scale features, train multiple
models, save trained artifacts, and an orchestration pipeline that returns the
trained models, test sets, and scaler.
"""
from __future__ import annotations

import os
import pickle
from typing import Dict, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def split_data(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data into train and test sets with stratification (80-20 split).

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target vector.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        X_train, X_test, y_train, y_test
    """
    print("[Train] Splitting data (stratified 80/20, random_state=42)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"[Train] Split complete. Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def scale_features(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Scale features using StandardScaler.

    Parameters
    ----------
    X_train : np.ndarray
        Training features.
    X_test : np.ndarray
        Test features.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, StandardScaler]
        Scaled X_train, scaled X_test, and the fitted scaler.
    """
    print("[Train] Scaling features (StandardScaler)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("[Train] Feature scaling complete.")
    return X_train_scaled, X_test_scaled, scaler


def train_models(X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, object]:
    """Train Logistic Regression, Decision Tree, and Random Forest models.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix.
    y_train : np.ndarray
        Training target vector.

    Returns
    -------
    Dict[str, object]
        Dictionary mapping model names to trained model instances.
    """
    print("[Train] Training models...")

    models = {}

    # Logistic Regression
    print("[Train] -> Logistic Regression (max_iter=1000)")
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    models['Logistic Regression'] = lr

    # Decision Tree
    print("[Train] -> Decision Tree (max_depth=10)")
    dt = DecisionTreeClassifier(max_depth=10)
    dt.fit(X_train, y_train)
    models['Decision Tree'] = dt

    # Random Forest
    print("[Train] -> Random Forest (n_estimators=100, max_depth=15, min_samples_split=5)")
    rf = RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_split=5, random_state=42)
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf

    print("[Train] Model training completed.")
    return models


def save_model(model: object, scaler: StandardScaler, filepath: str) -> None:
    """Save the model and scaler to disk using pickle.

    Parameters
    ----------
    model : object
        Trained model instance.
    scaler : StandardScaler
        Fitted scaler instance.
    filepath : str
        Destination file path under the 'models/' directory.
    """
    print(f"[Persist] Saving model to: {filepath}")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump({"model": model, "scaler": scaler}, f)
    print("[Persist] Save complete.")


def training_pipeline(X: np.ndarray, y: np.ndarray) -> Dict[str, object]:
    """Run the complete training pipeline.

    Steps:
    1) Split data (80/20 stratified)
    2) Scale features (StandardScaler)
    3) Train models (LR, DT, RF)
    4) Persist models to 'models/' directory

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target vector.

    Returns
    -------
    Dict[str, object]
        Dictionary containing:
        - 'models': dict of trained models
        - 'X_test': scaled test features
        - 'y_test': test targets
        - 'scaler': fitted StandardScaler
    """
    try:
        X_train, X_test, y_train, y_test = split_data(X, y)
        X_train_s, X_test_s, scaler = scale_features(X_train, X_test)
        models = train_models(X_train_s, y_train)

        # Save each model
        save_model(models['Logistic Regression'], scaler, os.path.join('models', 'logistic_regression.pkl'))
        save_model(models['Decision Tree'], scaler, os.path.join('models', 'decision_tree.pkl'))
        save_model(models['Random Forest'], scaler, os.path.join('models', 'random_forest.pkl'))

        print("[Pipeline] Training pipeline completed successfully.")
        return {
            'models': models,
            'X_test': X_test_s,
            'y_test': y_test,
            'scaler': scaler,
        }
    except Exception as e:
        print(f"[Pipeline][Error] Training pipeline failed: {e}")
        raise