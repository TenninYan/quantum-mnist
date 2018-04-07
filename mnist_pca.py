import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn import svm, metrics
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.decomposition import PCA
mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

def normalize(x, axis = None):
    xmean = x.mean(axis=axis, keepdims=True)
    xstd  = np.std(x, axis=axis, keepdims=True)
    norm = (x-xmean)/6*xstd + 0.5
    clipped = np.clip(norm, 0.0, 1.0)
    return clipped

X_train = np.vstack([img.reshape(-1,) for img in mnist.train.images])
y_train = mnist.train.labels

X_test = np.vstack([img.reshape(-1,) for img in mnist.test.images])
y_test = mnist.test.labels

pca = PCA(n_components = 16)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

train_pca = [normalize(X_train_pca), y_train]
test_pca = [normalize(X_test_pca), y_test]

with open('train_pca.pickle', 'wb') as f:
    pickle.dump(train_pca, f)

with open('test_pca.pickle', 'wb') as f:
    pickle.dump(test_pca, f)
# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
# n_samples = len(digits.images)
# data = digits.images.reshape((n_samples, -1))
#
# # Create a classifier: a support vector classifier
# classifier = svm.SVC(gamma=0.001)
#
# # We learn the digits on the first half of the digits
# classifier.fit(data[:n_samples // 2], digits.target[:n_samples // 2])
#
# # Now predict the value of the digit on the second half:
# expected = digits.target[n_samples // 2:]
# predicted = classifier.predict(data[n_samples // 2:])
#
# print("Classification report for classifier %s:\n%s\n"
#       % (classifier, metrics.classification_report(expected, predicted)))
# print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
