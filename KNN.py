import pandas as pd
import numpy as np


def get_data(filename: str) -> tuple[np.ndarray, np.ndarray]:
    """Takes .csv filename and reads it. Return tuple of np.ndarrays with features and labels."""

    data = pd.read_csv(filename)

    features = data.iloc[:, :-1].values

    labels = data.iloc[:, -1].values

    return features, labels


def ensure_numeric(features: np.ndarray) -> None:
    """Ensures that all collumns of passed np.ndarray are all numeric."""

    if not np.issubdtype(features.dtype, np.number):
        print("Error: All features must be numeric.")
        exit(1)
    return None


def normalize_data(features: np.ndarray) -> np.ndarray:
    """Normalizes the dataset features, scaling values to [0,1]."""

    min_vals = np.min(features, axis=0)
    max_vals = np.max(features, axis=0)

    ranges = max_vals - min_vals

    ranges[ranges == 0] = 1
    normalised_features = (features - min_vals) / ranges
    return normalised_features
