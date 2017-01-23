import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.metrics import make_scorer
from sklearn import preprocessing
import time

def sigmoid(x):
	return 1. / (1 + np.exp(-x))

def sigmoidGradient(x):
	return sigmoid(x) * (1. - sigmoid(x))

def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, n_labels, X, y, lam):
	Theta1 = np.reshape(nn_params[0: hidden_layer_size * (input_layer_size + 1)],
					 [hidden_layer_size, input_layer_size + 1]) # 30-by-785
	Theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):], # 10-by-31
					 [n_labels, hidden_layer_size + 1])
	n_samples = X.shape[0]
	Theta1_grad = np.zeros(Theta1.shape)
	Theta2_grad = np.zeros(Theta2.shape)
	# Perform Feedforward
	X = np.append(X.T, np.ones((1, n_samples)),axis=0)
	z2 = np.dot(Theta1, X)
	a2 = sigmoid(z2)
	a2 = np.append(a2, np.ones((1, n_samples)), axis=0)
	z3 = np.dot(Theta2, a2)
	h = sigmoid(z3).T # Hypothesis function
	J = 0
	#print y.shape, h.shape
	for i, y_train in enumerate(y):
		J = J -np.dot(y_train, np.log2(h[i]))-np.dot((1-y_train), np.log2(1-h[i]))
	J = J/n_samples
	# Add regularized term
	J_regularized = J + lam * 0.5 / n_samples * (sum(sum(Theta1**2)) + sum(sum(Theta2**2)))

	Theta2_drop = np.delete(Theta2.T, 30, 0).T # Drop last entry
	for i, y_train in enumerate(y):
		err_3 = y_train - h[i] # 1-by-10
		err_2 = np.dot((Theta2_drop).T, err_3.T).T * z2.T[i]
	return J_regularized
# Prepare data
train_data = np.asarray(pd.read_csv('train.csv'))
X_train = train_data[:, 1:] / 255.
n_features = X_train.shape[1]
n_samples = X_train.shape[0]
y_train = train_data[:, 0]
X_test = np.asarray(pd.read_csv('test.csv')) / 255.
n_test = X_test.shape[0]

# Prepare for NN hidden layer.
# 1 hidden layer with 30 units and 1 bias unit.
# Init parameters theta1 and theta2.
# theta1 = np.random.random((30, 785))
# theta2 = np.random.random((10, 31))

# Preprocess y_train, transform to vectors containing only 1 and 0's.
enc = preprocessing.OneHotEncoder()
enc = enc.fit([[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]])
y_train_array = np.reshape(y_train, [n_samples, 1])
y_train_enc = enc.transform(y_train_array).toarray()
epsilon_init = 0.0731 # sqrt(6) / (#in + #out)
epsilon_init_2 = 0.2806
nn_params = np.zeros(30 * 785 + 10 * 31)
nn_params[0: 30*785] = np.random.random(30 * 785) * 2 * epsilon_init - epsilon_init
nn_params[30*785: 30 * 785 + 10 * 31] = np.random.random(10 * 31) * 2 * epsilon_init_2 - epsilon_init_2
J = nnCostFunction(nn_params=nn_params, 
			    	input_layer_size=784, 
			    	hidden_layer_size=30,
			    	n_labels=10,
			    	X=X_train,
			    	y=y_train_enc,
			   	 	lam = 1)
print J 