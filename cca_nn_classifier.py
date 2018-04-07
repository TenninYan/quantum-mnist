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

def build_classifier():
    # params
    lr = 0.01
    epochs = 8
    hidden_layer_size = 100
    output_size = 10
    input_size = 16

    # placeholders
    # TODO: add minibatching
    inputs = tf.placeholder(tf.float32, shape=[None, input_size], name="inputs")
    labels = tf.placeholder(tf.int32, shape=[None, output_size], name="labels")

    # initialize vars
    # hidden layer
    w1 = tf.get_variable(name="hidden_weights", shape=[input_size, hidden_layer_size], \
        initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable(name="hidder_bias", shape=[hidden_layer_size], \
        initializer=tf.zeros_initializer())

    # output layer
    w2 = tf.get_variable(name="output_weights", shape=[hidden_layer_size, \
        output_size], initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable(name="outpus_bias", shape=[output_size], \
        initializer=tf.zeros_initializer())

    # build network
    hidden_layer = tf.add(tf.matmul(inputs, w1), b1)
    hidden_layer = tf.nn.relu(hidden_layer)
    output_layer = tf.add(tf.matmul(hidden_layer, w2), b2)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=labels))
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)

    train_data, train_labels = get_cca_data_as_matrices(data_set="train")
    validation_data, validation_labels = get_cca_data_as_matrices(data_set="validation")

    # train
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            for row, label in zip(train_data, train_labels):
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
            inputs: validation_data,
            labels: validation_labels
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

if __name__ == "__main__":
    test_sklearn()
    # build_classifier()
