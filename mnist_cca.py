import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.cross_decomposition import CCA

def get_train_data(one_hot=False):
    """ Fetches and returns training data and labels from MNIST data set.

    Args:
        one_hot: whether to generate one-hot labels
    Returns:
        train_data: (50000, 784) matrix
        train_labels: (50000, 9) matrix (labels are one-hot)
    """
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=one_hot)
    train_data = mnist.train.images
    train_labels = mnist.train.labels
    return train_data, train_labels

def CCA_reduction():
    data, labels = get_train_data(one_hot=True)
    X_scores, Y_scores = CCA(n_components=16).fit_transform(data, labels)

if __name__ == "__main__":
    CCA_reduction()
