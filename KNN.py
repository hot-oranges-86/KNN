import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import math
from collections import Counter


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


def split_data(features: np.ndarray, labels: np.ndarray, test_size: float = 0.4) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Splits the dataset into training and testing sets."""
    num_samples = features.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    test_set_size = int(num_samples * test_size)
    test_indices = indices[:test_set_size]
    train_indices = indices[test_set_size:]

    train_features = features[train_indices]
    train_labels = labels[train_indices]
    test_features = features[test_indices]
    test_labels = labels[test_indices]

    return train_features, train_labels, test_features, test_labels


def euclidean_distance(point1: np.ndarray, point2: np.ndarray) -> float:
    """Calculates the Euclidean distance between two points."""
    distance = np.sqrt(np.sum((point1 - point2) ** 2))
    return distance


def classify0(inX: np.ndarray, dataSet: np.ndarray, labels: np.ndarray, k: int) -> any:
    """Classifies a test point inX using the k nearest neighbors in the dataset."""
    distances = []
    for i in range(dataSet.shape[0]):
        dist = euclidean_distance(inX, dataSet[i])
        distances.append((dist, labels[i]))

    distances.sort(key=lambda x: x[0])
    k_nearest_labels = [distances[i][1] for i in range(k)]

    most_common_label = Counter(k_nearest_labels).most_common(1)[0][0]
    return most_common_label


def knn_classifier(train_features: np.ndarray, train_labels: np.ndarray, test_features: np.ndarray, k: int) -> list:
    """Classifies a set of test points using the k nearest neighbors algorithm."""
    predictions = []
    for test_point in test_features:
        label = classify0(test_point, train_features, train_labels, k)
        predictions.append(label)
    return predictions


def calculate_accuracy(predictions: list, true_labels: np.ndarray) -> float:
    """Calculates the accuracy of predictions against true labels."""
    correct_predictions = sum(pred == true for pred,
                              true in zip(predictions, true_labels))
    accuracy = correct_predictions / \
        len(true_labels) * 100
    return accuracy


def visualize_classification(test_features: np.ndarray, test_labels: np.ndarray, predictions: list) -> None:
    """Visualizes test data points and compares true labels with predicted labels."""

    unique_classes = np.unique(test_labels)

    color_map = plt.get_cmap('tab20')
    colors = [color_map(i / len(unique_classes))
              for i in range(len(unique_classes))]

    for idx, cls in enumerate(unique_classes):

        correct_indices = np.where(
            (test_labels == cls) & (predictions == cls))[0]
        correct_points = test_features[correct_indices]
        plt.scatter(correct_points[:, 0], correct_points[:, 1], color=colors[idx],
                    marker='o', label=f'Correct Class {cls}', alpha=0.6)

        incorrect_indices = np.where(
            (test_labels == cls) & (predictions != cls))[0]
        incorrect_points = test_features[incorrect_indices]
        plt.scatter(incorrect_points[:, 0], incorrect_points[:, 1],
                    color=colors[idx], marker='x', label=f'Incorrect Class {cls}')

    plt.title('Correct vs Incorrect Predictions')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

    plt.legend(loc='upper right')
    plt.show()


if __name__ == "__main__":
    train_features, train_labels = get_data("cesfał.csv")
    ensure_numeric(train_features)

    train_features = normalize_data(train_features)

    train_features, train_labels, test_features, test_labels = split_data(
        train_features, train_labels)

    k = 3
    predictions = knn_classifier(
        train_features, train_labels, test_features, k)

    # print(f"Predictions: {predictions}")
    # print(f"True labels: {test_labels}")

    accuracy = calculate_accuracy(predictions, test_labels)
    print(f"Accuracy: {accuracy:.2f}%")

    visualize_classification(test_features, test_labels, predictions)
