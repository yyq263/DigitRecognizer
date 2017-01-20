import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from sklearn import svm
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.metrics import make_scorer

def performance_metric(y_true, y_pred):
	return float(sum(y_true == y_pred)) / float(len(y_true)) 

# load csv
data = np.asarray((pd.read_csv('train.csv')))
test = np.asarray((pd.read_csv('test.csv')))
X = data[:, 1:]
y = data[:, 0]
X = X / 255.
nSamples, nFeatures = X.shape
test = test / 255.

#print nSamples, nFeatures
#print min(y), max(y)
#print y

cv_sets = ShuffleSplit(n_splits = 3, test_size = 0.2, train_size = None, random_state = 42)
clf = svm.SVC(max_iter = -1, random_state = 0)
params = {'C': [0.01, 0.1, 1.,2.,3.,4.,5.], 'gamma': [.1, .01, .001, .0001]}
scorer = make_scorer(performance_metric)
grid = GridSearchCV(clf, params, cv = cv_sets, scoring = scorer, verbose = 3)
grid = grid.fit(X, y)

print grid.best_estimator_