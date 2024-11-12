import numpy as np
from collections import Counter
from KNN import classify0


def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))


def test_classify0_single_point():
    dataSet = np.array([[1.0, 2.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = np.array(['A', 'A', 'B', 'B'])
    inX = np.array([0, 0])
    k = 3
    result = classify0(inX, dataSet, labels, k)
    assert result == 'B'


def test_classify0_tie():
    dataSet = np.array([[1.0, 2.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = np.array(['A', 'A', 'B', 'B'])
    inX = np.array([0.1, 0.1])
    k = 4
    result = classify0(inX, dataSet, labels, k)
    assert result in ['A', 'B']


def test_classify0_different_k():
    dataSet = np.array([[1.0, 2.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = np.array(['A', 'A', 'B', 'B'])
    inX = np.array([0.1, 0.1])
    k = 2
    result = classify0(inX, dataSet, labels, k)
    assert result == 'B'
