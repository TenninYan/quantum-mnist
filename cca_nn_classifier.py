import tensorflow as tf
import numpy as np
from mnist_cca import get_cca_data_as_matrices
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import os
import pickle
from tensorflow.examples.tutorials.mnist import input_data
from progress.bar import Bar
import matplotlib.pyplot as plt
import itertools

def get_mnist_counts():
    """ Prints number of examples in each class in the training data.
    """
    x = [i for i in range(10)]
    _, y = np.unique(decode_one_hot(mnist.train.labels), return_counts=True)
    print(y)


def sklearn_neural_classifier(data, h1_size=28, h2_size=39, verbose=False):
    """ Trains and tests a neural classifier with sklearn
    Args:
        data: dict with four keys: train_data, train_labels, dev_data, dev_labels
        h1_size: number of neurons in first hidden layer
        h2_size: number of neurons in second hidden layer
    Returns:
        accuracy: recall on test data
    """
    train_data = data.get("train_data")
    train_labels = data.get("train_labels")
    validation_data = data.get("validation_data")
    validation_labels = data.get("validation_labels")
    # make sure data is correctly formatted
    assert len(train_data) == len(train_labels)
    assert len(validation_data) == len(validation_labels)

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(h1_size, h2_size), random_state=1, verbose=False)
    clf.fit(train_data, train_labels)
    predictions = clf.predict(validation_data)

    accuracy = accuracy_score(validation_labels, predictions)
    if verbose == True:
        print("-------------------")
        print("First hidden layer size: {} second hidden layer size: {}".format(h1_size, h2_size))
        print(accuracy)
        print(classification_report(validation_labels, predictions))
        print("-------------------")
        cf = confusion_matrix(validation_labels, predictions)

        plt.figure()
        plot_confusion_matrix(cf)
        plt.show()
    return accuracy


def tune_params(data):
    """ Runs the neural model and experiments with the size of the hidden layers.

    Args:
        data: dict with four keys: train_data, train_labels, dev_data, dev_labels
    """
    max_acc = 0.0
    max_h1 = None
    max_h2 = None
    total = (80.0 - 15.0) * (30.0 - 15.0)
    bar = Bar("Processing", max=total)
    for h1 in range(15, 80):
        for h2 in range(15, 30):
            acc = sklearn_neural_classifier(data, h1_size=h1, h2_size=h2)
            if acc > max_acc:
                max_acc = acc
                max_h1 = h1
                max_h2 = h2
        bar.next()
    bar.finish()
    print("Accuracy:                    {}".format(max_acc))
    print("First hidden layer size      {}".format(max_h1))
    print("Second hidden layer size:    {}".format(max_h2))
    sklearn_neural_classifier(data, h1_size=max_h1, h2_size=max_h2, verbose=True)


def plot_confusion_matrix(cm, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], \
    title="Classification Confusion Matrix", cmap=plt.cm.Blues):
    """ Plots a confusion matrix of the data.

    Args:
        cm: confusion matrix generated with sklearn.metrics
        classes: list of all classes
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def sklearn_nn_cca():
    """ Runs the cca reduced vectors with sklearn neural model.
    """
    train_data, train_labels = get_cca_data_as_matrices(data_set="train")
    validation_data, validation_labels = get_cca_data_as_matrices(data_set="validation")
    data = {
        "train_data": train_data,
        "train_labels": train_labels,
        "validation_data": validation_data,
        "validation_labels": validation_labels
    }
    sklearn_neural_classifier(data)


def test_sklearn_neural_classifier():
    """ Runs the mnist data set with the sklearn model as a sanity check.
    """
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    data = {
        "train_data": mnist.train.images,
        "train_labels": mnist.train.labels,
        "validation_data": mnist.validation.images,
        "validation_labels": mnist.validation.labels,
    }
    sklearn_neural_classifier(data)


class Config:
    def __init__(self, lr=0.01, epochs=8, hidden_layer_size=28, \
        hidden_layer_2_size=39 ,output_size=10, input_size=16, activation="relu"):
        """
        Args:
            lr: learning rate
            epochs: number of epochs to run training for
            hidden_layer_size: number of neurons in hidden layer
            output_size: number of classes
            input_size: dimensionality of input vector
            activation: one of relu or sigmoid
        """
        self.lr = lr
        self.epochs = epochs
        self.hidden_layer_size = hidden_layer_size
        self.hidden_layer_2_size = hidden_layer_2_size
        self.output_size = output_size
        self.input_size = input_size
        self.activation = activation


class NeuralClassifier:
    def __init__(self, data, config):
        """
        Args:
            data: dict with four keys: train_data, train_labels, dev_data, dev_labels
            lr: learning rate
            epochs: number of epochs to run training for
            hidden_layer_size: number of neurons in hidden layer
            output_size: number of classes
            input_size: dimensionality of input vector
        """
        # make sure data is correctly formatted
        self.train_data = data.get("train_data")
        self.train_labels = data.get("train_labels")
        self.validation_data = data.get("validation_data")
        self.validation_labels = data.get("validation_labels")
        self.config = config


    def build_and_train(self):
        # make sure data is correctly formatted
        assert len(self.train_data) == len(self.train_labels)
        assert len(self.validation_data) == len(self.validation_labels)
        # placeholders
        inputs = tf.placeholder(tf.float32, shape=[None, self.config.input_size], name="inputs")
        labels = tf.placeholder(tf.int32, shape=[None, self.config.output_size], name="labels")

        # initialize vars
        # hidden layer
        w1 = tf.get_variable(name="hidden_weights", shape=[self.config.input_size, self.config.hidden_layer_size], \
            initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.get_variable(name="hidden_bias", shape=[self.config.hidden_layer_size], \
            initializer=tf.zeros_initializer())

        # h2
        w2 = tf.get_variable(name="output_weights", shape=[self.config.hidden_layer_size, \
            self.config.hidden_layer_2_size], initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable(name="outpus_bias", shape=[self.config.hidden_layer_2_size], \
            initializer=tf.zeros_initializer())

        # output layer
        w3 = tf.get_variable(name="output_weights", shape=[self.config.hidden_layer_2_size, \
            self.config.output_size], initializer=tf.contrib.layers.xavier_initializer())
        b3 = tf.get_variable(name="outpus_bias", shape=[self.config.output_size], \
            initializer=tf.zeros_initializer())

        # build network
        hidden_layer = tf.add(tf.matmul(inputs, w1), b1)
        if self.config.activation == "sigmoid":
            hidden_layer = tf.nn.sigmoid(hidden_layer)
        else:
            hidden_layer = tf.nn.relu(hidden_layer)

        hidden_layer_2 = tf.add(tf.matmul(hidden_layer, w2), b2)
        if self.config.activation == "sigmoid":
            hidden_layer_2 = tf.nn.sigmoid(hidden_layer_2)
        else:
            hidden_layer_2 = tf.nn.relu(hidden_layer_2)

        output_layer = tf.add(tf.matmul(hidden_layer, w3), b3)


        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=labels))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.config.lr).minimize(cost)

        # train
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(self.config.epochs):
                for row, label in zip(self.train_data, self.train_labels):
                    tensor_row = (np.expand_dims(np.array(row), axis=1)).T
                    tensor_label = (np.expand_dims(np.array(label), axis=1)).T
                    _, c = sess.run([optimizer, cost], feed_dict={
                        inputs: tensor_row,
                        labels: tensor_label
                    })
                print("Completed epoch: {}".format((epoch + 1)))

            pred_temp = tf.equal(tf.argmax(output_layer, 1), tf.argmax(labels, 1))
            accuracy = tf.reduce_mean(tf.cast(pred_temp, "float"))
            print ("Validation accuracy: {}".format(accuracy.eval({
                inputs: self.validation_data,
                labels: self.validation_labels
            })))


def format_data(data, labels):
    """ Formats rows so that they can be fed into a nn.

    Args:
        data: data to transform
        labels: labels to transform
    returns
        data_transformed: transformed data
        labels_transformed: transformed labels
    """
    data_transformed = [(np.expand_dims(np.array(row), axis=1)).T for row in data]
    labels_transformed = [(np.expand_dims(np.array(label), axis=1)).T for label in labels]
    return data_transformed, labels_transformed


def classify_with_cca_data():
    """ Runs the neural classifier on vectors reduced with cca.
    """
    train_data, train_labels = get_cca_data_as_matrices(data_set="train")
    validation_data, validation_labels = get_cca_data_as_matrices(data_set="validation")
    data = {
        "train_data": train_data,
        "train_labels": train_labels,
        "validation_data": validation_data,
        "validation_labels": validation_labels
    }
    config = Config(epochs=1)
    classifier = NeuralClassifier(data, config)
    classifier.build_and_train()


def convert_to_one_hot(labels, num_classes):
    """ Convert array of labels into matrix of one-hot vectors

    Args:
        labels: 1D list of labels
        num_classes: number of classes
    """
    one_hots = np.zeros((len(labels), num_classes))
    for idx, label in enumerate(labels):
        one_hots[idx][(label - 1)] = 1
    return one_hots


def test_one_hots():
    x = convert_to_one_hot([3, 4, 5], 5)
    print(x)


def get_ptrace_data():
    """ Fetches ptrace data.

    Returns:
        data: dict with four keys: train_data, train_labels, validation_data, validation_labels
    """
    train_path = "data/test_pca_reservoir_output_concatenated_samples0thru300_reducedNonlocal_0_4_8_12_FIXED_LABELS.pickle"
    test_path = "data/test_pca_reservoir_output_concatenated_samples0thru300_reducedNonlocal_0_4_8_12_FIXED_LABELS.pickle"
    if os.path.isfile(train_path):
        with open(train_path, "rb") as fp:
            train = pickle.load(fp)
    if os.path.isfile(test_path):
        with open(test_path, "rb") as fp:
            test = pickle.load(fp)
    train_data = train[0]
    train_labels = train[1]
    validation_data = test[0]
    validation_labels = test[1]

    print("train data len: {} label len: {}".format(len(train[0]), len(train[1])))
    return {
        "train_data": train_data,
        "train_labels": train_labels,
        "validation_data": validation_data,
        "validation_labels": validation_labels
    }


def test_neural_classifier():
    """ Sanity check for the tensorflow classifier with MNIST data.
    """
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    data = {
        "train_data": mnist.train.images,
        "train_labels": mnist.train.labels,
        "validation_data": mnist.validation.images,
        "validation_labels": mnist.validation.labels,
    }
    config = Config(epochs=5, input_size=784)
    classifier = NeuralClassifier(data, config)
    classifier.build_and_train()

def ptrace_neural_classifier():
    config = Config(epochs=5)
    data = get_ptrace_data()
    classifier = NeuralClassifier(data, config)
    classifier.build_and_train()

def classify_best_ptrace():
    """ Runs the neural classifier with the best achieving hyperparameters.
    """
    data = get_ptrace_data()
    sklearn_neural_classifier(data, h1_size=28, h2_size=39, verbose=True)


def tune_ptrace():
    """ Trains the neural model with different parameters.
    """
    data = get_ptrace_data()
    tune_params(data)


def decode_one_hot(one_hots):
    """ Decodes one hot vector into labels.
    """
    return [(np.argmax(row, axis=0) + 1) for row in one_hots]


def test_decode_one_hot():
    x = convert_to_one_hot([3, 4, 5], 5)
    y = decode_one_hot(x)
    assert y == [3, 4, 5]
    print("Decoder works")

if __name__ == "__main__":
    ptrace_neural_classifier()
