from sklearn.cluster import KMeans
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import rbf_kernel
import plotting
from itertools import product
import matplotlib.pyplot as plt

def evaluate_model(y_train,y_train_mod, y_mod_pred):
	print "Number of label changes made in dataset: %d" % np.count_nonzero(y_train - y_train_mod)
	print "Model Accuracy w.r.t. Original dataset: ", np.count_nonzero((y_train - y_mod_pred)==0)*100.0/y_train.shape[0]
	print "Model Accuracy w.r.t. Modified dataset: ", np.count_nonzero((y_train_mod - y_mod_pred)==0)*100.0/y_train.shape[0],"\n"

def check_trusted_items(X_train,y_train,X_trust,y_trust, clf, combo=None, 
						train_cluster_labels=None, trust_cluster_labels=None, plot=False):

	clf.fit(X_train, y_train)
	y_train_pred = clf.predict(X_train)

	y_trust_pred = clf.predict(X_trust)

	# print "Plotting..."

	if np.array_equal(y_trust, y_trust_pred):
		print "All trusted items predicted correctly by model, combo:",combo
		if plot:
			plotting.plot_model(X_train, y_train, X_trust, y_trust, str(combo)+" Good", 
													train_cluster_labels, trust_cluster_labels)
		check = True
	else:
		print "%d trusted items were incorrectly predicted. Continuing experiment..."%np.count_nonzero(y_trust-y_trust_pred)
		if plot:
			plotting.plot_model(X_train, y_train, clf, X_trust, y_trust, str(combo)+" Bad", 
		 											train_cluster_labels, trust_cluster_labels)
		check = False
	return clf, y_train_pred, y_trust_pred, check


def modify_labels(K, y_train, train_cluster_labels, combo, max_changes=np.inf):
	
	y_train_mod = np.copy(y_train)
	for i in range(K):
		y_train_mod[train_cluster_labels==i] = combo[i]
	
	num_changes = np.count_nonzero(y_train - y_train_mod)
	if num_changes>max_changes:
		# print combo,"Too many changes to dataset:",num_changes,"\n"
		return None
	else:
		print combo,"Changes to dataset:",num_changes,"\n"
		return y_train_mod

def modify_labels_trust(y_train,y_trust,y_trust_pred,train_cluster_labels,trust_cluster_labels,max_changes=np.inf):
	y_train_mod = np.copy(y_train)
	count = 0
	for i in range(len(y_trust_pred)):
		if y_trust_pred[i]!=y_trust[i]:
			print "Set Cluster",trust_cluster_labels[i],"to",y_trust[i]
			y_train_mod[train_cluster_labels==trust_cluster_labels[i]] = y_trust[i]
	num_changes = np.count_nonzero(y_train - y_train_mod)
	if num_changes>max_changes:
		print "Too many changes to dataset:",num_changes,"\n"
		return None
	else:
		return y_train_mod


def cluster_data(X_train, y_train, X_trust, y_trust, clf, max_changes=np.inf, try_all_combos=False, plot=False):
	
	label_set = list(set(y_train))

	# Visualize original dataset
	print "Running Model without any modifications to dataset..."

	clf, y_pred, y_trust_pred, check = check_trusted_items(X_train,y_train,X_trust,y_trust,clf, plot=plot)
	print "Number of bugs: %d" % np.count_nonzero(y_train - y_pred)
	evaluate_model(y_train,y_train, y_pred)

	if check:
		print "All trusted items classified correctly using original dataset! \n"
		return

	print

	min_K = 3
	max_K = 10
	if try_all_combos:
		print "Trying all labeling of clusters, from K = %d to %d\n"%(min_K,max_K)
	else:
		print "Modifying only labels of clusters containing mislabeled trusted items, from K = %d to %d\n"%(min_K,max_K)

	for K in range(min_K,max_K+1):
		kmeans = KMeans(n_clusters=K)
		kmeans.fit(X_train)

		train_cluster_labels = kmeans.labels_
		trust_cluster_labels = kmeans.predict(X_trust)

		if try_all_combos:
			print "Brute Force Clustering for K=%d"%K
			all_combos = list(product(label_set, repeat=K)) 
			# this list contains every possible combination of labels for each cluster
			
			for combo in all_combos:
				if len(set(combo))<=1:
					continue

				
				y_train_mod = modify_labels(K, y_train, train_cluster_labels, combo, max_changes=max_changes)
				if y_train_mod is None:
					continue

				print combo,":",
				clf, y_mod_pred, _, check = check_trusted_items(X_train,y_train_mod,X_trust,y_trust, clf, combo, 
																train_cluster_labels, trust_cluster_labels)
				
				evaluate_model(y_train,y_train_mod, y_mod_pred)
		else:
			print "Trusted Clustering for K=%d"%K
			y_train_mod = modify_labels_trust(y_train,y_trust,y_trust_pred,train_cluster_labels,trust_cluster_labels,max_changes=max_changes)
			if y_train_mod is None:
				continue

			# print(y_trust_pred,y_trust)
			# print "List of points whose labels were changed:",[i for i in range(len(y_train)) if y_train[i]!=y_train_mod[i]]
			# print(train_cluster_labels,trust_cluster_labels)


			clf, y_mod_pred, _, check = check_trusted_items(X_train,y_train_mod,X_trust,y_trust, clf, 
														"Trusted clustering: %d"%K, train_cluster_labels, 
														trust_cluster_labels, plot=plot)
			
			evaluate_model(y_train,y_train_mod,y_mod_pred)

	plt.show()