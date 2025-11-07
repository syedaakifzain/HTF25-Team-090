"""
Data preprocessing utilities for the Asteroid Hazard Classification System.

This module provides functions to load data, handle missing values,
encode categorical features, and select the final feature set used for
model training. It also exposes a high-level preprocess_pipeline that
returns feature matrix X, target vector y, and the list of feature names.
"""
from __future__ import annotations

import os
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_data(filepath: str) -> pd.DataFrame:
    """Load the asteroid dataset from a CSV file.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file cannot be parsed as CSV.
    """
    print(f"[Data] Attempting to load data from: {filepath}")
    if not os.path.exists(filepath):
        msg = f"Dataset not found at '{filepath}'. Please place 'dataset.csv' in the data/ folder."
        print(f"[Error] {msg}")
        raise FileNotFoundError(msg)
    try:
        df = pd.read_csv(filepath)
        print(f"[Data] Loaded dataset with shape: {df.shape}")
        return df
    except Exception as e:
        print(f"[Error] Failed to read CSV due to: {e}")
        raise ValueError("Failed to read CSV. Ensure the file is a valid CSV.") from e


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values in numeric and categorical columns.

    Numeric columns are filled with the median, categorical columns with the mode.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with missing values handled.
    """
    print("[Preprocess] Handling missing values...")
    df = df.copy()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # Fill numeric columns with median
    for col in numeric_cols:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
    print(f"[Preprocess] Filled numeric columns ({len(numeric_cols)}) with median values.")

    # Fill categorical columns with mode
    for col in categorical_cols:
        if df[col].isna().any():
            mode_val = df[col].mode(dropna=True)
            fill_val = mode_val.iloc[0] if not mode_val.empty else "Unknown"
            df[col] = df[col].fillna(fill_val)
    print(f"[Preprocess] Filled categorical columns ({len(categorical_cols)}) with mode values.")

    return df


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode selected categorical features using LabelEncoder.

    The following columns will be label-encoded if present and of object dtype:
    - 'Relative Velocity km per sec'
    - 'Orbital Period'
    - 'Orbit Uncertainity'

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with selected categorical columns encoded.
    """
    print("[Preprocess] Encoding categorical features (LabelEncoder)...")
    df = df.copy()

    target_columns = [
        'Relative Velocity km per sec',
        'Orbital Period',
        'Orbit Uncertainity',
    ]

    for col in target_columns:
        if col in df.columns:
            if df[col].dtype == 'object':
                print(f"[Preprocess] Label-encoding column: {col}")
                le = LabelEncoder()
                try:
                    df[col] = le.fit_transform(df[col].astype(str))
                except Exception as e:
                    print(f"[Warning] Failed to encode '{col}' due to: {e}. Leaving column as-is.")
            else:
                print(f"[Preprocess] Skipping '{col}' (dtype={df[col].dtype}); not object dtype.")
        else:
            print(f"[Preprocess] Column '{col}' not found; skipping.")

    return df


def select_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Select required features and target variable.

    Selected features:
    - 'Relative Velocity km per hr'
    - 'Miles per hour'
    - 'Miss Dist.(Astronomical)'
    - 'Miss Dist.(lunar)'
    - 'Miss Dist.(kilometers)'
    - 'Semi Major Axis'
    - 'Aphelion Dist'
    - 'Mean Motion'
    - 'Orbital Period'

    Target variable:
    - 'Hazardous' (True/False)

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame after preprocessing.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series, List[str]]
        X (features DataFrame), y (target Series), and feature names list.

    Raises
    ------
    KeyError
        If target variable 'Hazardous' is missing.
    ValueError
        If none of the requested features exist in the dataset.
    """
    print("[Preprocess] Selecting features and target...")
    required_features = [
        'Relative Velocity km per hr',
        'Miles per hour',
        'Miss Dist.(Astronomical)',
        'Miss Dist.(lunar)',
        'Miss Dist.(kilometers)',
        'Semi Major Axis',
        'Aphelion Dist',
        'Mean Motion',
        'Orbital Period',
    ]

    if 'Hazardous' not in df.columns:
        msg = "Target column 'Hazardous' not found in dataset."
        print(f"[Error] {msg}")
        raise KeyError(msg)

    available = [c for c in required_features if c in df.columns]
    missing = [c for c in required_features if c not in df.columns]

    if missing:
        print(f"[Warning] Missing feature columns: {missing}")
    if not available:
        msg = "None of the required feature columns are present."
        print(f"[Error] {msg}")
        raise ValueError(msg)

    X = df[available].copy()
    y = df['Hazardous'].copy()

    # If target is boolean or string, map to integers for consistency
    if y.dtype == 'bool':
        y = y.astype(int)
    elif y.dtype == 'object':
        y = y.astype(str).str.lower().map({'true': 1, 'false': 0})
        if y.isna().any():
            print("[Warning] Non-standard target values detected; filling unknowns with 0.")
            y = y.fillna(0).astype(int)

    print(f"[Preprocess] Selected {len(available)} features. Feature names: {available}")
    print(f"[Preprocess] Target distribution: {y.value_counts().to_dict()}")

    return X, y, available


def preprocess_pipeline(filepath: str) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Run the full preprocessing pipeline and return X, y, and feature names.

    Steps:
    1) Load data
    2) Handle missing values
    3) Encode categorical features (selected columns)
    4) Select feature set and target

    Parameters
    ----------
    filepath : str
        Path to the CSV dataset.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series, List[str]]
        Processed X, y, and feature names.

    Raises
    ------
    Exception
        Any error encountered will be logged and re-raised for the caller.
    """
    print("[Pipeline] Starting preprocessing pipeline...")
    try:
        df = load_data(filepath)
        df = handle_missing_values(df)
        df = encode_categorical_features(df)
        X, y, feature_names = select_features(df)
        print("[Pipeline] Preprocessing completed successfully.")
        return X, y, feature_names
    except Exception as e:
        print(f"[Pipeline][Error] Preprocessing failed: {e}")
        raise