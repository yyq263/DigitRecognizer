import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn import svm
from sklearn.metrics import make_scorer
import time

def show_image(data): 
	'''
	data: an array of 784 elements (28-by-28)
	'''
	im = np.reshape(data, [28, 28])
	plt.imshow(im, cmap = 'Greys')
	plt.show()

def performance_metric(y_true, y_pred):
	return float(sum(y_true == y_pred)) / float(len(y_true)) 
# Prepare data
train_data = np.asarray(pd.read_csv('train.csv'))
X_train = train_data[:, 1:]
n_features = X_train.shape[1]
n_samples = X_train.shape[0]
y_train = train_data[:, 0]
X_test = np.asarray(pd.read_csv('test.csv'))

# Do PCA
n_components = 400
pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(X_train)
print("Projecting the train and test data on the eigenvector's orthonormal basis")
t0 = time.time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test) # Project the test data on the basis of train data
print("done in %0.3fs" % (time.time() - t0))

# Do SVM
print("Fitting the classifier to the training set")
t0 = time.time()
cv_sets = ShuffleSplit(n_splits = 3, test_size = 0.25, train_size = None, random_state=42)
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5, 1e2, 5e2],
              'gamma': [0.00004, 0.00008, 0.0001, 0.00025, 0.0005, 0.001, 0.005], }
scorer = make_scorer(performance_metric)
clf = GridSearchCV(svm.SVC(kernel='rbf', class_weight='balanced'),  
					param_grid, 
					cv = cv_sets,
					scoring = scorer, 
					verbose = 3)
clf = clf.fit(X_train_pca, y_train)
print("done in %0.3fs" % (time.time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)
