# quantum-mnist
Classify MNIST using Quantum Machine Learning. This project was built during the Rigetti Hackathon 2018.

The model reduces MNIST data to 16 dimensions using PCA, applies reservoir computing to the output, and finally classifies the data using a neural network.

## Running the model
First, make sure to get an API key for the rigetti forest API. After installing the remaining packages, run

`$ python cca_nn_classifier.py`

This will run the model end to end.
