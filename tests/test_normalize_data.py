import numpy as np
import pytest

from KNN import normalize_data


def test_normalize_data_basic():
    features = np.array([[1, 2], [3, 4], [5, 6]])
    expected = np.array([[0, 0], [0.5, 0.5], [1, 1]])
    np.testing.assert_array_almost_equal(normalize_data(features), expected)


def test_normalize_data_single_value():
    features = np.array([[1, 1], [1, 1], [1, 1]])
    expected = np.array([[0, 0], [0, 0], [0, 0]])
    np.testing.assert_array_almost_equal(normalize_data(features), expected)


def test_normalize_data_negative_values():
    features = np.array([[-1, -2], [-3, -4], [-5, -6]])
    expected = np.array([[1, 1], [0.5, 0.5], [0, 0]])
    np.testing.assert_array_almost_equal(normalize_data(features), expected)


def test_normalize_data_mixed_values():
    features = np.array([[1, -2], [3, 0], [5, 2]])
    expected = np.array([[0, 0], [0.5, 0.5], [1, 1]])
    np.testing.assert_array_almost_equal(normalize_data(features), expected)


def test_normalize_data_large_range():
    features = np.array([[1, 1000], [2, 2000], [3, 3000]])
    expected = np.array([[0, 0], [0.5, 0.5], [1, 1]])
    np.testing.assert_array_almost_equal(normalize_data(features), expected)


def test_normalize_data_zero_range():
    features = np.array([[1, 2], [1, 2], [1, 2]])
    expected = np.array([[0, 0], [0, 0], [0, 0]])
    np.testing.assert_array_almost_equal(normalize_data(features), expected)


if __name__ == "__main__":
    pytest.main()
