"""
Data preprocessing module for TrustFed-RL client CSVs.

This module handles loading, preprocessing, and label preparation for client CSV data.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional


def load_client_data(csv_path: str) -> pd.DataFrame:
    """
    Load and return a client's CSV data.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        DataFrame with loaded data
    """
    df = pd.read_csv(csv_path)
    return df


def prepare_labels(df: pd.DataFrame, label_column: str = 'label') -> pd.DataFrame:
    """
    Convert labels to binary format (0 for BENIGN, 1 for attacks).
    Automatically detects if labels are already binary (0/1) and preserves them.
    
    Args:
        df: DataFrame with label column
        label_column: Name of the label column (case-insensitive)
        
    Returns:
        DataFrame with binary 'label' column added
    """
    df = df.copy()
    
    # Find the label column (case-insensitive)
    label_col = None
    for col in df.columns:
        if col.lower() == label_column.lower():
            label_col = col
            break
    
    if label_col is None:
        raise ValueError(f"Label column '{label_column}' not found in DataFrame")
    
    # Check if labels are already binary (0/1)
    unique_labels = df[label_col].unique()
    
    # Check if all values are numeric and in {0, 1}
    is_binary = False
    try:
        numeric_labels = pd.to_numeric(df[label_col], errors='coerce')
        if numeric_labels.notna().all():
            unique_numeric = set(numeric_labels.unique())
            if unique_numeric.issubset({0, 1}):
                is_binary = True
    except:
        pass
    
    if is_binary:
        # Labels are already binary (0/1), preserve them
        df['label'] = numeric_labels.astype(int)
    else:
        # Convert text labels to binary: BENIGN -> 0, anything else -> 1
        df['label'] = df[label_col].apply(
            lambda x: 0 if str(x).upper() == 'BENIGN' else 1
        )
    
    return df


def split_data(
    X: pd.DataFrame, 
    y: pd.Series, 
    test_size: float = 0.2, 
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into training and validation sets.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        test_size: Proportion of data for validation
        random_state: Random seed for reproducibility
        stratify: Whether to stratify by class labels
        
    Returns:
        X_train, X_val, y_train, y_val
    """
    stratify_param = y if stratify else None
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=stratify_param
    )
    return X_train, X_val, y_train, y_val


def prepare_features(df: pd.DataFrame, exclude_columns: Optional[list] = None) -> pd.DataFrame:
    """
    Prepare feature matrix by excluding non-feature columns.
    
    Args:
        df: DataFrame with all columns
        exclude_columns: List of column names to exclude (default: common non-feature columns)
        
    Returns:
        DataFrame with only feature columns
    """
    if exclude_columns is None:
        # Default columns to exclude
        exclude_columns = [
            'flow_id', 'timestamp', 'src_ip', 'src_port', 
            'dst_ip', 'dst_port', 'protocol', 'Label', 'label'
        ]
    
    # Get columns to exclude that actually exist
    cols_to_exclude = [col for col in exclude_columns if col in df.columns]
    
    # Select feature columns
    feature_cols = [col for col in df.columns if col not in cols_to_exclude]
    X = df[feature_cols].copy()
    
    # Handle any remaining non-numeric columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X = X[numeric_cols]
    
    # Fill any NaN values with 0
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Defensive normalization:
    # Client telemetry can contain extremely large magnitudes (or heavy-tailed counts) which can
    # trigger hard floating-point exceptions inside low-level BLAS during dot products.
    X = X.astype(float)
    X = X.clip(lower=-1e6, upper=1e6)
    std = X.std(axis=0, ddof=0).replace(0.0, 1.0)
    mean = X.mean(axis=0)
    X = (X - mean) / (std + 1e-8)
    
    return X
