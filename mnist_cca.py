import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.cross_decomposition import CCA
import numpy as np
import os
import pickle

cca_vectors_base_path = "data/cca_reduced_vectors_normalized"

def get_train_data(one_hot=False):
    """ Fetches and returns training data and labels from MNIST data set.

    Args:
        one_hot: whether to generate one-hot labels
    Returns:
        train_data: (55000, 784) matrix
        train_labels: (55000, 9) matrix (labels are one-hot)
    """
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=one_hot)


def get_data(data_set, one_hot=False):
    """ Fetches and returns data and labels from MNIST data set.

    Args:
        data_set: one of train, test, validation
        one_hot: whether to generate one-hot labels
    Returns:
        train_data: (55000, 784) matrix
        train_labels: (55000, 9) matrix (labels are one-hot)
    """
    options = ["train", "test", "validation"]
    if data_set not in options:
        print "data set must be one of train, test, or validation"
        return
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=one_hot)
    if data_set == "train":
        return mnist.train.images, mnist.train.labels
    elif data_set == "test":
        return mnist.test.images, mnist.test.labels
    elif data_set == "validation":
        return mnist.validation.images, mnist.validation.labels


def CCA_reduction(data_set="train"):
    """ Reduces dimensions of MNIST data using CCA

    Returns:
        X_scores: training data reduced to 16 dimenional vectors
        labels: data labels
    """
    data, labels = get_data(data_set, one_hot=True)
    X_scores, _ = CCA(n_components=16).fit_transform(data, labels)
    return X_scores, labels

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

def get_cca_vectors(data_set="train", load_files=True):
    """ Retrieves pickled data for

    Args:
        data_set: one of train, test, validation
        load_files: whether to load the pkl files or generate new ones
    Returns:
        data: dictionary with two keys, features and labels. Features contains
              the normalized vector and labels is a one-hot list.
    """
    full_path = cca_vectors_base_path + data_set + ".pkl"
    if os.path.isfile(full_path) and load_files:
        with open(full_path, "rb") as fp:
            data = pickle.load(fp)
        return data
    else:
        X_scores, labels = CCA_reduction(data_set=data_set)
        normalized = normalize_scores(X_scores)
        data = []
        assert len(labels) == len(normalized)
        for idx, row in enumerate(normalized):
            data.append({
                "features": row,
                "labels": labels[idx]
            })
        with open(full_path, "wb") as fp:
            pickle.dump(data, fp)
        return data
