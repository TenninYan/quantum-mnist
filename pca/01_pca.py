import matplotlib.pyplot as plt
import numpy as np
import pickle
from tools import normalize
from sklearn import svm, metrics
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.decomposition import PCA
mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)


X_train = np.vstack([img.reshape(-1,) for img in mnist.train.images])
y_train = mnist.train.labels

X_test = np.vstack([img.reshape(-1,) for img in mnist.test.images])
y_test = mnist.test.labels

zero_one_train = np.where(y_train<2)
zero_one_test = np.where(y_test<2)

X_train = X_train[zero_one_train]
y_train = y_train[zero_one_train]
X_test = X_test[zero_one_test]
y_test = y_test[zero_one_test]

pca = PCA(n_components = 6)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

train_pca = [normalize.normalize(X_train_pca), y_train]
test_pca = [normalize.normalize(X_test_pca), y_test]

with open('01_train_pca.pickle', 'wb') as f:
    pickle.dump(train_pca, f)

with open('01_test_pca.pickle', 'wb') as f:
    pickle.dump(test_pca, f)
