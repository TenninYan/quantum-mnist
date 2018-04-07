import pickle
from sklearn.neural_network import MLPClassifier

train = pickle.load(open('train_pca.pickle','rb'))
test = pickle.load(open('test_pca.pickle','rb'))

mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=50, alpha=1e-4,
                    solver='sgd', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.1)

mlp.fit(train[0], train[1])
print("Training set score: %f" % mlp.score(train[0], train[1]) )
print("Test set score: %f" % mlp.score(test[0], test[1]))
