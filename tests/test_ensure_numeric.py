import numpy as np
import pytest
from KNN import ensure_numeric


def test_ensure_numeric_all_numeric():
    features = np.array([[1, 2, 3], [4, 5, 6]])
    assert ensure_numeric(features) is None


def test_ensure_numeric_non_numeric():
    features = np.array([[1, 2, 'a'], [4, 5, 6]])
    with pytest.raises(SystemExit):
        ensure_numeric(features)


def test_ensure_numeric_empty_array():
    features = np.array([])
    assert ensure_numeric(features) is None


def test_ensure_numeric_mixed_types():
    features = np.array([[1, 2, 3], [4, 5, 'a']])
    with pytest.raises(SystemExit):
        ensure_numeric(features)


def test_ensure_numeric_float():
    features = np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]])
    assert ensure_numeric(features) is None
