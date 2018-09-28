from sklearn.cluster import KMeans
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import rbf_kernel
import plotting
from itertools import product
import matplotlib.pyplot as plt

def evaluate_model(y_train,y_train_mod, y_mod_pred):
	print "Number of label changes made in dataset:",np.count_nonzero(y_train - y_train_mod)
	print "Model Accuracy w.r.t. Original dataset:",np.count_nonzero((y_train - y_mod_pred)==0)*100.0/y_train.shape[0]
	print "Model Accuracy w.r.t. Modified dataset:",np.count_nonzero((y_train_mod - y_mod_pred)==0)*100.0/y_train.shape[0]
	print
	return 


def check_trusted_items(X_train,y_train,X_trust,y_trust, sigma, lam, combo=None, 
						train_cluster_labels=None, trust_cluster_labels=None, plot=False):
	clf = LogisticRegression(solver='lbfgs', C=lam)
	K = rbf_kernel(X_train, X_train, gamma=(-1/(2.0*sigma**2)))
	clf.fit(K, y_train)
	y_train_pred = clf.predict(K)

	K_trust = rbf_kernel(X_trust, X_train, gamma=(-1/(2.0*sigma**2)))
	y_trust_pred = clf.predict(K_trust)

	# print "Plotting..."

	if np.array_equal(y_trust, y_trust_pred):
		print "All trusted items predicted correctly by model"
		plotting.plot_model(X_train, y_train, clf, sigma, X_trust, y_trust, str(combo)+" Good", 
													train_cluster_labels, trust_cluster_labels)
		check = True
	else:
		print "Some trusted items were incorrectly predicted. Continuing experiment..."
		if plot:
			plotting.plot_model(X_train, y_train, clf, sigma, X_trust, y_trust, str(combo)+" Bad", 
		 											train_cluster_labels, trust_cluster_labels)
		check = False
	return clf, y_train_pred, y_trust_pred, check


def modify_labels(K, y_train, train_cluster_labels, combo, max_changes=np.inf):
	
	y_train_mod = np.copy(y_train)
	for i in range(K):
		y_train_mod[train_cluster_labels==i] = combo[i]
	
	num_changes = np.count_nonzero(y_train - y_train_mod)
	if num_changes>max_changes:
		# print "Too many changes to dataset:",num_changes,"\n"
		return None
	else:
		return y_train_mod

def cluster_data(X_train, y_train, X_trust, y_trust, sigma, lam, try_all_combos=False):
	
	label_set = list(set(y_train))

	# Visualize original dataset
	print "Running Kernel Logistic Regression without any modifications to dataset..."

	clf, y_pred, y_trust_pred, check = check_trusted_items(X_train,y_train,X_trust,y_trust, sigma, lam, plot=True)
	evaluate_model(y_train,y_train, y_pred)

	if check:
		print "All trusted items classified correctly using original dataset! \n"
		return

	print
	# try_all_combos = True

	min_K = 5
	max_K = 10
	if try_all_combos:
		print "Trying all labeling of clusters, from K =%d to %d\n"%(min_K,max_K)
	else:
		print "Modifying only labels of clusters containing mislabeled trusted items, from K =%d to %d\n"%(min_K,max_K)

	for K in range(min_K,max_K+1):
		kmeans = KMeans(n_clusters=K)
		kmeans.fit(X_train)

		train_cluster_labels = kmeans.labels_
		trust_cluster_labels = kmeans.predict(X_trust)

		if try_all_combos:
			all_combos = list(product(label_set, repeat=K)) 
			# this list contains every possible combination of labels for each cluster
			
			for combo in all_combos:
				if len(set(combo))<=1:
					continue

				y_train_mod = modify_labels(K, y_train, train_cluster_labels, combo, max_changes=20)
				if y_train_mod is None:
					continue

				print combo,":",

				clf, y_mod_pred, _, check = check_trusted_items(X_train,y_train_mod,X_trust,y_trust, sigma, lam, combo, 
																train_cluster_labels, trust_cluster_labels)
				
				evaluate_model(y_train,y_train_mod, y_mod_pred)
		else:
			y_train_mod = np.copy(y_train)
			print "Trusted Clustering for K=%d"%K
			for i in range(len(y_trust_pred)):
				if y_trust_pred[i]!=y_trust[i]:
					y_train_mod[train_cluster_labels==trust_cluster_labels[i]] = y_trust[i]

			print(y_trust_pred,y_trust)
			print(y_train)
			print(y_train_mod)
			print(train_cluster_labels,trust_cluster_labels)


			clf, y_mod_pred, _, check = check_trusted_items(X_train,y_train_mod,X_trust,y_trust, sigma, lam, "Trusted clustering: %d"%K, 
															train_cluster_labels, trust_cluster_labels, plot=True)
			
			evaluate_model(y_train,y_train_mod, y_mod_pred)

	plt.show()
	return