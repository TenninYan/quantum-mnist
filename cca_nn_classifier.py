import tensorflow as tf
import numpy as np
from mnist_cca import get_cca_data_as_matrices
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

def test_sklearn():
    train_data, train_labels = get_cca_data_as_matrices(data_set="train")
    validation_data, validation_labels = get_cca_data_as_matrices(data_set="validation")
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    clf.fit(train_data, train_labels)
    predictions = clf.predict(validation_data)
    print(classification_report(validation_labels, predictions))

class Config:
    def __init__(self, lr=0.01, epochs=8, hidden_layer_size=100, \
        output_size=10, input_size=16):
        """
        Args:
            lr: learning rate
            epochs: number of epochs to run training for
            hidden_layer_size: number of neurons in hidden layer
            output_size: number of classes
            input_size: dimensionality of input vector
        """
        self.lr = lr
        self.epochs = epochs
        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size
        self.input_size = input_size

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
        self.train_data = data.get("train_data")
        self.train_labels = data.get("train_labels")
        self.validation_data = data.get("validation_data")
        self.validation_labels = data.get("validation_labels")
        self.config = config

    def build_and_train(self):
        # placeholders
        inputs = tf.placeholder(tf.float32, shape=[None, self.config.input_size], name="inputs")
        labels = tf.placeholder(tf.int32, shape=[None, self.config.output_size], name="labels")

        # initialize vars
        # hidden layer
        w1 = tf.get_variable(name="hidden_weights", shape=[self.config.input_size, self.config.hidden_layer_size], \
            initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.get_variable(name="hidder_bias", shape=[self.config.hidden_layer_size], \
            initializer=tf.zeros_initializer())

        # output layer
        w2 = tf.get_variable(name="output_weights", shape=[self.config.hidden_layer_size, \
            self.config.output_size], initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable(name="outpus_bias", shape=[self.config.output_size], \
            initializer=tf.zeros_initializer())

        # build network
        hidden_layer = tf.add(tf.matmul(inputs, w1), b1)
        hidden_layer = tf.nn.relu(hidden_layer)
        output_layer = tf.add(tf.matmul(hidden_layer, w2), b2)

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

if __name__ == "__main__":
    classify_with_cca_data()
