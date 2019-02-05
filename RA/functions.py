import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.utils.fixes import signature
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
import tensorflow as tf


def custom_log_loss(y_true, y_proba):
    y_proba[y_proba==0] = 0.00000001
    y_proba[y_proba==1] = 1 - 0.00000001
    return -(np.multiply(y_true, np.log(y_proba))+np.multiply(1-y_true, np.log(1-y_proba)))

def calculate_influences_on_trust_remove(clf, X_train, y_train, X_trust, y_trust, all=True):
    
    clf.fit(X_train, y_train)
    
    pred = clf.predict(X_train)
    
    y_trust_pred = clf.predict(X_trust)
    
    l = []
    for i in range(len(y_trust)):
        if all or y_trust_pred[i] != y_trust[i]:
            l.append(i)
    
    
    
    if len(l)==0:
        print("No trusted points misclassified!")
        return np.zeros((X_train.shape[0], X_trust.shape[0])).T
    
    X_trust_misclassified = X_trust[l]
    y_trust_misclassified = y_trust[l]
    X_trust_misclassified = X_trust[l]
    trust_misclassified_proba = clf.predict_proba(X_trust_misclassified)[:,1]
    
    influences = np.zeros((X_train.shape[0], X_trust_misclassified.shape[0]))

    
#     print(X_trust_misclassified.shape,y_trust_misclassified.shape,trust_misclassified_proba.shape)
    orig_loss = custom_log_loss(y_trust_misclassified, trust_misclassified_proba)
    
    for i in tqdm_notebook(range(X_train.shape[0])):
        ind = np.ones((X_train.shape[0],))
        ind[i] = 0
        ind = ind==1
        
        X_i = X_train[ind]
        y_i = y_train[ind]
        
        clf.fit(X_i, y_i)
        
        trust_misclassified_proba = clf.predict_proba(X_trust_misclassified)[:,1]
        mod_loss = custom_log_loss(y_trust_misclassified, trust_misclassified_proba)
                
        influences[i,:]= orig_loss - mod_loss
                
    return influences.T



def subset_indices(n, m=None):
    
    # Samples m numbers uniformly at random with replacement from 0,...,n-1 : bootstrap of size m
    if m is not None:
        indices = np.floor(np.random.rand(m)*n).astype(int)
    # If m not specified, random (non-empty) subset is returned
    else:
        indices = np.nonzero(np.round(np.random.rand(n)))[0]
        if indices.shape[0] == 0:
            indices = subset_indices(n)
        
    return indices


def sample_data(X, y, bootstrap_size=None, random_features=False):
    # if bootstrap_size is None, then random subset returned
    data_indices = subset_indices(X.shape[0], bootstrap_size)
    if random_features:
        feature_indices = subset_indices(X.shape[1])
        X_sample = X[data_indices, feature_indices]
        X_sample = np.reshape(X_sample, (data_indices.shape[0],feature_indices.shape[0]))
    else:
        X_sample = X[data_indices, :]
        feature_indices = np.array(range(X.shape[1]))
    y_sample = y[data_indices]
    
    return X_sample, y_sample, data_indices, feature_indices



def bagging(X_train, y_train, X_trust, y_trust, clf, ensemble_size, bootstrap_size=None, random_features=False):
    
    clf_list = []
    counts = np.zeros((X_trust.shape[0], X_train.shape[0]))
    total_counts = np.zeros((X_trust.shape[0], X_train.shape[0]))
    
    data_indices_X = np.zeros((ensemble_size, X_train.shape[0]))
    outputs_y = np.zeros((ensemble_size, X_trust.shape[0]))

    
    for i in tqdm(range(ensemble_size)):
        X_sample, y_sample, data_indices, feature_indices = sample_data(X_train, y_train, bootstrap_size, random_features)
        clf.fit(X_sample, y_sample)
        y_trust_pred = clf.predict(X_trust)
#         print(y_trust_pred.shape)
        clf_list.append(copy.copy(clf))
        
        for k in range(data_indices.shape[0]):
            data_indices_X[i][data_indices[k]] += 1
        
        for j in range(X_trust.shape[0]):
            # misclassified trusted item
            for k in range(data_indices.shape[0]):
                total_counts[j][data_indices[k]] += 1
            
            if(y_trust[j] != y_trust_pred[j]): 
                outputs_y[i][j] = 1
                for k in range(data_indices.shape[0]):
                    counts[j][data_indices[k]] += 1
                    
    return np.divide(counts,total_counts), data_indices_X, outputs_y         

def ovr_classifier(data_indices_X, outputs_y):
    
    est_clf = LogisticRegression(solver='lbfgs')
#     est_clf = MLPClassifier()

    ovr_clf = OneVsRestClassifier(est_clf)
    
    ovr_clf.fit(data_indices_X, outputs_y)
    
#     print(nn_clf.coefs_)
    
    X = np.identity(data_indices_X.shape[1])
    preds = ovr_clf.predict_proba(X)
    
    return preds

def plot_pr_curves(all_influences, true_bugs, PR=True, title=None):
    
    for i in range(all_influences.shape[0]):
        metric = all_influences[i].reshape((1,all_influences.shape[1]))
        tit = " %d "%i
        if title:
            tit = title + tit
        plot_pr_curve(metric, 0, true_bugs, PR=PR, title=tit)
        
    metric = np.max(all_influences, axis=0).reshape((1,all_influences.shape[1]))
    tit = "Max "
    if title:
        tit = title + tit
    plot_pr_curve(metric, 0, true_bugs, PR=PR, title=tit)
        

def plot_pr_curve(metric, ind, bugs, PR=True, title=None):
    # precision = tp/(tp+fp)
    # recall = tp/(tp + fn)
    precision = []
    recall = []
    tp_list = []
    fp_list = []
    found = [[],[]]
    num_bugs = np.sum(bugs)
    
    tp = 0
    fp = 0
    
    a = metric[ind]
    duo = [(i,a[i]) for i in range(a.shape[0])]
    duo.sort(key=lambda x: x[1], reverse=True)
#     print("(Index of bug, TP, FP)")

    for ind,_ in duo:
        
        if(bugs[ind]==1):
            tp += 1
        else:
            fp += 1
        tp_list.append(tp)
        fp_list.append(fp)
        
        pr = tp/(tp+fp)
        re = tp/num_bugs
#         print("(%d,%d,%d)"%(ind,tp,fp), end=", ")
        precision.append(pr)
        recall.append(re)
        
        if(bugs[ind]==1):
            if PR:
                found[0].append(re)
                found[1].append(pr)
            else:
                found[0].append(fp)
                found[1].append(tp)
        
    plt.figure()
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    
    if PR:
        x = recall
        y = precision
        t = 'PR Curve, AUC:%f'
        x_label = 'Recall'
        y_label = 'Precision'
        total_area = 1
    else:
        x = fp_list
        y = tp_list
        t = 'TP vs FP Curve, AUC:%f'
        x_label = 'FP'
        y_label = 'TP'
        total_area = tp*fp
    
    area = metrics.auc(x, y)/total_area
    plt.step(x, y, color='b', alpha=0.2, where='post')
    plt.fill_between(x, y, alpha=0.2, color='b', **step_kwargs)
    
    plt.scatter(found[0], found[1])
    
    t = t%area
    if title:
        t = title + t
    plt.title(t)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()
        



class kernelLR_model:

    
    def __init__(self): # g is a tensorflow graph
        self.n = 0
        self.sigma = 0.4 # RBF kernel bandwidth
        self.gamma = (1/(2.0*self.sigma**2))
        
        
    def create_graph(self,bootstrap_size):
        self.n = bootstrap_size
        self.g = tf.Graph()
        with self.g.as_default():
            self.lam_tf = tf.placeholder(tf.float32)
            self.K_tf = tf.placeholder(tf.float32, [bootstrap_size, bootstrap_size])
            self.y_tf = tf.placeholder(tf.float32, [bootstrap_size,1])

            self.alpha_tf = tf.Variable(tf.zeros((bootstrap_size,1)))

            self.logistic_loss = tf.reduce_mean(-tf.log_sigmoid(self.y_tf*tf.matmul(self.K_tf, self.alpha_tf)))
            self.reg_loss = 0.5*self.lam_tf*tf.matmul(tf.matmul(self.alpha_tf, self.K_tf, transpose_a = True),self.alpha_tf)
            self.loss = self.logistic_loss + self.reg_loss
            self.opt = tf.train.AdamOptimizer().minimize(self.loss)
            self.init = tf.initializers.global_variables()
    
    def fit(self,X_train,y_train,lam=1e-3,verbose=False):
        
        if self.n != X_train.shape[0]:
            self.create_graph(X_train.shape[0])
        
        self.X_train = X_train
        
        K_train = rbf_kernel(X_train, X_train, self.gamma)
       
        if len(y_train.shape)==1:
            y_train = y_train.reshape((y_train.shape[0],1))
                    
        n = K_train.shape[0]

        tf.reset_default_graph()

        with tf.Session(graph=self.g) as sess:
            sess.run(self.init)

            prevloss = np.inf
            c = 0
            while True:
                _,self.alpha,ll,rl, l = sess.run([self.opt, self.alpha_tf, self.logistic_loss, self.reg_loss, self.loss], 
                            feed_dict={self.K_tf: K_train, self.y_tf:y_train, self.lam_tf: lam})
                c += 1
                if abs((prevloss - l)/l) < 5e-4:
                    if verbose:
                        print("Iterations for convergence:",c, "Loss: ", l)
                    break

                prevloss = l


    def predict(self, X):
        K = rbf_kernel(X, self.X_train, self.gamma)
        pred = np.dot(K,self.alpha)
        pred_proba = 1/(1+np.exp(-pred))
        pred = np.sign(pred).reshape((X.shape[0],))
        return pred

    def predict_proba(self, X):
        ans = np.zeros((X.shape[0], 2))
        K = rbf_kernel(X, self.X_train, self.gamma)
        pred = np.dot(K,self.alpha).reshape((X.shape[0],))
        pred_proba = 1/(1+np.exp(-pred))
        ans[:,1] = pred_proba
        ans[:,0] = 1 - pred_proba
        return ans
        


    