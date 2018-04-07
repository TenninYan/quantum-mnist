import pickle
from sklearn.neural_network import MLPClassifier

train = pickle.load(open('train_pca_reservoir_output_200samples.pickle','rb'))
test = pickle.load(open('test_pca_reservoir_output_50samples.pickle','rb'))

train_num = 200
test_num = 50

mlp = MLPClassifier(hidden_layer_sizes=(2000,), max_iter=100, alpha=1e-5,
                    solver='sgd', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.1, batch_size= 20)

mlp.fit(train[0], train[1][:train_num])
print("Training set score: %f" % mlp.score(train[0], train[1][:train_num]))
print("Test set score: %f" % mlp.score(test[0], test[1][:test_num]))
