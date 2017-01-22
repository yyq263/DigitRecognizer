import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from collections import Counter
import time

def show_image(data): 
	'''
	data: an array of 784 elements (28-by-28)
	'''
	im = np.reshape(data, [28, 28])
	plt.imshow(im, cmap = 'Greys')
	plt.show()
def transform_to_pca(X, n_components=28):
	nsample = X.shape[0]
	pca_result = np.zeros([nsample, n_components*28])
	for i, sample in enumerate(X):
		image = np.reshape(sample, [28, 28])
		pca = PCA(n_components=n_components).fit(image)
		pca_result[i] = np.reshape(pca.transform(image), [1, n_components*28])[0] # n_components * nsample
	return pca_result
def calculate_distance(test_pca, train_pca):
	nsample = train_pca.shape[0]
	distance = np.zeros(nsample)
	for i in range(nsample):
		sub = test_pca - train_pca[i]
		distance[i] = np.dot(sub, sub)
	return distance
def predict_process(X_test_pca, X_train_pca, n=1):
	nsample = X_train_pca.shape[0]
	ntest = X_test_pca.shape[0]
	y_predict = zeros(ntest)
	for i in range(ntest):
		print "Calculating case"+" "+"["+str(i)+"]"+"..."
		distance = calculate_distance(X_test_pca[i], X_train_pca)
		idx = np.argsort(distance)
		nn = y_train[idx[0:n]]
		c = Counter(nn)
		y_predict[i] = c.most_common(1)[0][0]
		print "case"+" "+"["+str(i)+"]"+" "+"done!"
	return y_predict
# Read csv file
data_train = np.asarray(pd.read_csv('train.csv'))
data_test = np.asarray(pd.read_csv('test.csv'))
# Read X_train and y_train from data_train
X_train = data_train[:, 1:]
y_train = data_train[:, 0]
X_test = data_test
nsample = X_train.shape[0]
ntest = X_test.shape[0]
# Preprocess substract mean and do PCA for each image
pick = 453
n_components = 20
k = 5 
train_pca = transform_to_pca(X_train, n_components=n_components)
test_pca = transform_to_pca(X_test, n_components=n_components)
tic = time.time()
y_predict = predict_process(test_pca, train_pca, n=k)
toc = time.time()
print("Complete Predicting 5 test cases in " + str(toc-tic) + " Secs.")
print y_predit
# Plot to verify
#show_image(X_test[pick])
# Write predictions to y_predict.csv
out_file = open("y_predict.csv", "w")
out_file.write("ImageId,Label\n")
for i in range(ntest):
    out_file.write(str(i+1) + "," + str(int(y_predict[i])) + "\n")
out_file.close()