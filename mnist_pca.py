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
