import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.cross_decomposition import CCA
import numpy as np

def get_train_data(one_hot=False):
    """ Fetches and returns training data and labels from MNIST data set.

    Args:
        one_hot: whether to generate one-hot labels
    Returns:
        train_data: (55000, 784) matrix
        train_labels: (55000, 9) matrix (labels are one-hot)
    """
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=one_hot)
    train_data = mnist.train.images
    train_labels = mnist.train.labels
    return train_data, train_labels

def CCA_reduction():
    """ Reduces dimensions of MNIST data using CCA

    Returns:
        X_scores: training data reduced to 16 dimenional vectors
    """
    data, labels = get_train_data(one_hot=True)
    X_scores, _ = CCA(n_components=16).fit_transform(data, labels)
    return X_scores

def normalize_row(row):
    """ Normalizes data between 0 and 1.

    Args:
        row: list of values
    Returns:
        lst: list of values normalized between 0 and 1
    """
    max_val = max(row)
    min_val = min(row)
    diff = max_val - min_val
    return [np.float(x - min_val)/np.float(diff) for x in row]

def normalize_scores(data):
    """ Normalizes matrix.

    Args:
        data: matrix to normalize
    Returns:
        lst: matrix where each row is normalized between 0 and 1
    """
    return [normalize_row(row) for row in data]

def test_normalize_scores():
    """ Tests that rows are normalized correctly.
    """
    X_scores = CCA_reduction()
    normalized = normalize_scores(X_scores)
    for row in normalized:
        assert max(row) <= 1.0
        assert min(row) >= 0.0
    print("Rows normalized correctly")
