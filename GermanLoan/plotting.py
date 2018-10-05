import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

# ----------------------------------------
# For plotting data and decision boundary
def plot_model(X_train, y_train, clf, X_trust=None, y_trust=None, title=None, 
													train_cluster_labels=None, trust_cluster_labels=None):
	
	pca = PCA(n_components=2)
	X_train_2d = pca.fit_transform(X_train)
	X_trust_2d = pca.transform(X_trust)

	# X = np.zeros((10000,2))
	# a = np.linspace(0,1,100)
	# b = np.linspace(0,1,100)
	# e, d = np.meshgrid(a, b)
	# X[:,0] = np.reshape(e,(10000,))
	# X[:,1] = np.reshape(d,(10000,))

	# Z = clf.predict(X)
	# probs = clf.predict_proba(K)[:, 1].reshape(e.shape)

	plt.figure()

	# Put the result into a color plot
	# Z = Z.reshape(e.shape)
	# plt.contour(e, d, probs, levels=[0.5])

	# Plot clusters
	if train_cluster_labels is not None:
		if trust_cluster_labels is not None:
			cluster_labels = np.concatenate([train_cluster_labels,trust_cluster_labels])
			X = np.vstack([X_train_2d, X_trust_2d])
			plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, marker='o', s=80)
		else:
			plt.scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=train_cluster_labels, marker='o', s=80)

	# Plot the training points
	plt.scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=y_train, cmap=plt.cm.Paired, marker='.')
	
	# Plot the trusted points
	if X_trust_2d is not None:
		plt.scatter(X_trust_2d[:, 0], X_trust_2d[:, 1], c=y_trust, cmap=plt.cm.Paired, marker='X')

	if title is not None:
		plt.title(str(title))

	plt.xlabel('x1')
	plt.ylabel('x2')
	plt.draw()