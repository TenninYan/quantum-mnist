import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def get_train_data():
    """ Fetches and returns training data and labels from MNIST data set.

    Returns:
        train_data: (50000, 784) matrix
        train_labels: (50000, 9) matrix (labels are one-hot)
    """
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    train_data = mnist.train.images
    train_labels = mnist.train.labels
    print(train_data[0])
    print(train_labels[0])
    return train_data, train_labels
