import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D  # Import 3D plotting toolkit
from sklearn.datasets import load_iris
from sklearn.datasets import fetch_openml


# def get_data(filename: str) -> tuple[np.ndarray, np.ndarray]:
def get_data() -> tuple[np.ndarray, np.ndarray]:
    """Takes .csv filename and reads it. Returns tuple of np.ndarrays with features and labels."""

    # data = pd.read_csv(filename)

    # features = data.iloc[:, :-1].values

    # labels = data.iloc[:, -1].values

    # iris = load_iris()
    # features = iris.data
    # labels = iris.target

    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    features = mnist.data.astype(int)
    labels = mnist.target.astype(int)

    features = features[:2000]
    labels = labels[:2000]

    return features, labels


def ensure_numeric(features: np.ndarray) -> None:
    """Ensures that all columns of passed np.ndarray are all numeric."""

    if not np.issubdtype(features.dtype, np.number):
        print("Error: All features must be numeric.")
        exit(1)

    return None


def apply_pca(features: np.ndarray, n_components: int = 3) -> np.ndarray:
    """Reduces the dimensionality of the dataset to n_components using PCA."""
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(features)
    return reduced_features


def normalize_data(features: np.ndarray) -> np.ndarray:
    """Normalizes the dataset features, scaling values to [0, 1]."""

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


'''Old visualization'''
# def visualize_classification(test_features: np.ndarray, test_labels: np.ndarray, predictions: list) -> None:
#     """Visualizes test data points and compares true labels with predicted labels."""
#     # I assumed F1 and F2 are most important and didn't want to change visualization as this is not my focus.

#     unique_classes = np.unique(test_labels)

#     color_map = plt.get_cmap('tab20')
#     colors = [color_map(i / len(unique_classes))
#               for i in range(len(unique_classes))]

#     for idx, cls in enumerate(unique_classes):

#         correct_indices = np.where(
#             (test_labels == cls) & (predictions == cls))[0]
#         correct_points = test_features[correct_indices]
#         plt.scatter(correct_points[:, 0], correct_points[:, 1], color=colors[idx],
#                     marker='o', label=f'Correct Class {cls}', alpha=0.6)

#         incorrect_indices = np.where(
#             (test_labels == cls) & (predictions != cls))[0]
#         incorrect_points = test_features[incorrect_indices]
#         plt.scatter(incorrect_points[:, 0], incorrect_points[:, 1],
#                     color=colors[idx], marker='x', label=f'Incorrect Class {cls}')

#     plt.title('Correct vs Incorrect Predictions')
#     plt.xlabel('Feature 1')
#     plt.ylabel('Feature 2')

#     plt.legend(loc='upper right')
#     plt.show()


def visualize_classification(test_features: np.ndarray, test_labels: np.ndarray, predictions: list) -> None:
    """Visualizes test data points in 3D and compares true labels with predicted labels."""
    unique_classes = np.unique(test_labels)
    color_map = plt.get_cmap('tab20')
    colors = [color_map(i / len(unique_classes))
              for i in range(len(unique_classes))]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.view_init(elev=20, azim=30)

    for idx, cls in enumerate(unique_classes):
        correct_indices = np.where(
            (test_labels == cls) & (predictions == cls))[0]
        correct_points = test_features[correct_indices]
        ax.scatter(correct_points[:, 0], correct_points[:, 1], correct_points[:, 2],
                   color=colors[idx], marker='o', s=60, alpha=0.5, label=f'Correct Class {cls}')

        incorrect_indices = np.where(
            (test_labels == cls) & (predictions != cls))[0]
        incorrect_points = test_features[incorrect_indices]
        ax.scatter(incorrect_points[:, 0], incorrect_points[:, 1], incorrect_points[:, 2],
                   color=colors[idx], marker='x', s=80, label=f'Incorrect Class {cls}', alpha=0.8)

    ax.set_title('Correct vs Incorrect Predictions (3D)', fontsize=15)
    ax.set_xlabel('Feature 1', fontsize=12)
    ax.set_ylabel('Feature 2', fontsize=12)
    ax.set_zlabel('Feature 3', fontsize=12)
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
    ax.grid(True)

    plt.legend(loc='upper right', fontsize=10)
    plt.show()


if __name__ == "__main__":
    # features, labels = get_data("cesfał.csv")
    features, labels = get_data()
    ensure_numeric(features)

    features = apply_pca(features)

    features = normalize_data(features)

    train_features, train_labels, test_features, test_labels = split_data(
        features, labels)

    k = 3

    predictions = knn_classifier(
        train_features, train_labels, test_features, k)

    accuracy = calculate_accuracy(predictions, test_labels)
    print(f"Accuracy: {accuracy:.2f}%")

    visualize_classification(test_features, test_labels, predictions)
