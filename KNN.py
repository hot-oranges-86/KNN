import pandas as pd
import numpy as np


def get_data(filename: str) -> tuple[np.ndarray, np.ndarray]:
    """Takes .csv filename and reads it. Return tuple of np.ndarrays with features and labels."""

    data = pd.read_csv(filename)

    features = data.iloc[:, :-1].values

    labels = data.iloc[:, -1].values

    return features, labels
