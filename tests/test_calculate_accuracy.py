import numpy as np
import pytest
from KNN import calculate_accuracy


def test_calculate_accuracy_all_correct():
    predictions = [1, 0, 1, 1, 0]
    true_labels = np.array([1, 0, 1, 1, 0])
    assert calculate_accuracy(predictions, true_labels) == 100.0


def test_calculate_accuracy_all_incorrect():
    predictions = [0, 1, 0, 0, 1]
    true_labels = np.array([1, 0, 1, 1, 0])
    assert calculate_accuracy(predictions, true_labels) == 0.0


def test_calculate_accuracy_half_correct():
    predictions = [1, 0, 1, 0, 1]
    true_labels = np.array([1, 0, 0, 1, 0])
    assert calculate_accuracy(predictions, true_labels) == 40.0


def test_calculate_accuracy_mixed():
    predictions = [1, 1, 0, 0, 1, 0, 1, 0, 1, 0]
    true_labels = np.array([1, 0, 0, 1, 1, 0, 1, 1, 0, 0])
    assert calculate_accuracy(predictions, true_labels) == 60.0


if __name__ == "__main__":
    pytest.main()
