import pickle
from sklearn.neural_network import MLPClassifier

train = pickle.load(open('train_pca.pickle','rb'))
test = pickle.load(open('test_pca.pickle','rb'))

train_num = 200
test_num = 50

X_train = train[0][:train_num]
y_train = train[1][:train_num]

X_test = test[0][:test_num]
y_test = test[1][:test_num]

mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=50, alpha=1e-4,
                    solver='sgd', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.1)

mlp.fit(X_train, y_train)
print("Training set score: %f" % mlp.score(X_train, y_train) )
print("Test set score: %f" % mlp.score(X_test, y_test))
