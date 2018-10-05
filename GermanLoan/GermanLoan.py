import csv
from sklearn.cluster import KMeans
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics.pairwise import rbf_kernel
import experiments
import sys

# read dataset
print("Reading dataset...")
with open("german.data-numeric") as tsv:
	all_data = np.array([[int(x) for x in line[0].split()] for line in csv.reader(tsv, dialect="excel-tab")])
# print(all_data.shape)

n = all_data.shape[0]
X = all_data[:,:-1]
y = all_data[:,-1:].reshape((n,)) 
# print(X.shape,y.shape)

# Set 1=good, -1=bad in y
y[y==2] = -1


# partition dataset on the basis of age, threshold = 25
X_young = X[X[:,9]<=25,:]
X_old = X[X[:,9]>25,:]
y_young = y[X[:,9]<=25]
y_old = y[X[:,9]>25]
# print(X_young.shape, X_old.shape)


# Remove age as a feature
X_young = np.delete(X_young, 9, 1)
X_old = np.delete(X_old, 9, 1)
# print(X_young.shape, X_old.shape)


# Random arrays for partitioning into A,B,C datasets
np.random.seed(123)
young = np.random.permutation(np.arange(X_young.shape[0]))
old = np.random.permutation(np.arange(X_old.shape[0]))

# --------------------------------------------------------------
print("Creating datasets A, B and C...")
# Create dataset A (trusted dataset)
X_A = np.concatenate((X_young[young[:20],:],X_old[old[:20],:]))
y_A = np.concatenate((y_young[young[:20]],y_old[old[:20]]))
# print(X_A.shape,y_A.shape)

# Create dataset B (buggy dataset)
X_B = np.concatenate((X_young[young[20:190],:],X_old[old[20:190],:]))
y_B = np.concatenate((y_young[young[20:190]],y_old[old[20:190]]))
# print(X_B.shape,y_B.shape)

# Create dataset C (ground truth)
X_C = X_old[old[190:],:]
y_C = y_old[old[190:]]
# print(X_C.shape,y_C.shape)


# --------------------------------------------------------------
print("Training Model on dataset C...")

# the learner is hard coded to be logistic regression
lam = 5e-3	# L2 regularization weight of learner
# Training model f* on dataset C
clf = LogisticRegression(solver='lbfgs', C=lam)

clf = AdaBoostClassifier()


clf.fit(X_C, y_C)
y_C_pred = clf.predict(X_C)

print("Creating trusted labels for dataset A...")
y_A_pred = clf.predict(X_A)

print("Number of label changes made in Dataset A to make it trusted: %d"%np.count_nonzero(y_A-y_A_pred)),"\n"
y_A = y_A_pred


# --------------------------------------------------------------
# Experimenting with clustering
max_changes = 100

if len(sys.argv)==1:
	experiments.cluster_data(X_B, y_B, X_A, y_A, clf, max_changes=max_changes,try_all_combos=False)
elif len(sys.argv)==2:
	if sys.argv[1]=="all": 
		experiments.cluster_data(X_B, y_B, X_A, y_A, clf, max_changes=max_changes,try_all_combos=True)
	elif sys.argv[1]=="plot": 
		experiments.cluster_data(X_B, y_B, X_A, y_A, clf, max_changes=max_changes,try_all_combos=False, plot=True)
elif len(sys.argv)==3:
	if sys.argv[1]=="all" and sys.argv[2]=="plot":
		experiments.cluster_data(X_B, y_B, X_A, y_A, clf, max_changes=max_changes,try_all_combos=True, plot=True)
else:
	print "Invalid"
