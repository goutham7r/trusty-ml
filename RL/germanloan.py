import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
pd.options.mode.chained_assignment = None

class germanloan():
    
    def __init__(self, clf = None, verbose=False):
        self.clf = clf
        self.verbose = verbose
    
    def get_data(self, random_seed=123):
    
        np.random.seed(random_seed)

        # read dataset
        col_names = ['Checking Status', 'Duration','Credit History','Purpose','Credit Amt', 
              'Saving acc', 'Present emp since', 'Installment Rate',
             'Personal Status', 'Other debtors', 'Present Residence since', 'Property', 'Age', 
              'Other installment plans', 'Housing', 'Existing credits',
             'Job', 'Num People', 'Telephone', 'Foreign Worker','Approval Status']

        numerical = ['Duration','Credit Amt','Installment Rate','Age','Existing credits','Present Residence since', 'Num People']
        categorical = [l for l in col_names if l not in numerical and l!="Approval Status"]
#         print(numerical, categorical)
        
        all_data = pd.read_csv("german.csv", names=col_names)
#         print(all_data.shape)

        if self.verbose:
            print("Reading dataset...")
            print(all_data.head(5))

        n = all_data.shape[0]
        X = all_data[all_data.columns.difference(['Approval Status'])]
        y = all_data['Approval Status']
#         print(X.shape,y.shape)
        
        y[y==2] = -1
        
        feature_names_raw = list(X)
        
        X_raw = np.copy(X)
        y_raw = np.copy(y)
        
        # fit scaler on the numerical features
        # fit scaler on the numerical features
        scaler = StandardScaler()
        scaler.fit(X.loc[:,numerical].astype('float64'))

        # one-hot encoding the categorical features
        X = pd.get_dummies(X,prefix=categorical)
        feature_names = list(X)
        y = pd.get_dummies(y)
        
        # partition dataset on the basis of age, threshold = 25
        lt = X['Age']<=25
        gt = X['Age']>25


        #transform numerical features
        X.loc[:,numerical] = scaler.transform(X.loc[:,numerical].astype('float64'))
        
        data = {}
#         data['scaler'] = scaler
#         data['feature_names'] = feature_names
        data['X'] = X.values
        data['y'] = y.values
        data['X_raw'] = X_raw
        data['y_raw'] = y_raw
        
        data['feature_data_raw'] = {}
        data['feature_data_raw'] = {}
        for feature in feature_names_raw:
            data['feature_data_raw'][feature] = {}
            data['feature_data_raw'][feature]['values'] = np.unique(X_raw[:,feature_names_raw.index(feature)])
            data['feature_data_raw'][feature]['type'] = 'categorical'
        
        for i, feature in enumerate(numerical):
            del data['feature_data_raw'][feature]['values']
            data['feature_data_raw'][feature]['type'] = 'numerical'
            data['feature_data_raw'][feature]['mean'] = scaler.mean_[i]
            data['feature_data_raw'][feature]['std'] = np.sqrt(scaler.var_[i])
        
            
        
        
        if self.clf is not None:
            X_y = X[lt]
            X_o = X[gt]
            y_y = y[lt]
            y_o = y[gt]

            # Remove age as a feature
            del X_y['Age']
            del X_o['Age']
    #         print(X_y.shape, X_o.shape)


            # Converting to numpy arrays
            X_young = X_y.values
            X_old = X_o.values
            y_young = y_y.values
            y_old = y_o.values

            young = np.random.permutation(np.arange(X_young.shape[0]))
            old = np.random.permutation(np.arange(X_old.shape[0]))

            # --------------------------------------------------------------
            print("Creating datasets A, B and C...")
            # Create dataset A (trusted dataset)
            X_A = np.concatenate((X_young[young[:20],:],X_old[old[:20],:]))
            y_A = np.concatenate((y_young[young[:20]],y_old[old[:20]]))
    #         print(X_A.shape,y_A.shape)

            # Create dataset B (buggy dataset)
            X_B = np.concatenate((X_young[young[20:190],:],X_old[old[20:190],:]))
            y_B = np.concatenate((y_young[young[20:190]],y_old[old[20:190]]))
    #         print(X_B.shape,y_B.shape)

            # Create dataset C (ground truth)
            X_C = X_old[old[190:],:]
            y_C = y_old[old[190:]]
    #         print(X_C.shape,y_C.shape)


            # --------------------------------------------------------------
            print("Training Model on dataset C...")

            clf = self.clf

            clf.fit(X_C, y_C)
            y_C_pred = clf.predict(X_C)
            print("Accuracy of model on dataset C:", 1.0-np.count_nonzero(y_C_pred-y_C)/y_C.shape[0])

            print("Creating trusted labels for dataset A...")
            y_A_pred = clf.predict(X_A)

            print("Number of label changes made in Dataset A to make it trusted: %d"%np.count_nonzero(y_A-y_A_pred),"\n")
            y_A = y_A_pred

            X_train = X_B
            y_train = y_B
            X_trust = X_A
            y_trust = y_A

            y_train_pred = clf.predict(X_train)
            true_bugs = y_train-y_train_pred
            true_bugs[true_bugs!=0] = 1
            print("Number of bugs in training set:",np.sum(true_bugs))


            data['A,B,C'] = {'X_A':X_A, 'y_A':y_A,
                             'X_B':X_B, 'y_B':y_B,
                             'X_C':X_C, 'y_C':y_C}
            data['train, trust, true_bugs'] = {'X_train':X_train, 'y_train':y_train, 
                                               'X_trust':X_trust, 'y_trust':y_trust, 
                                               'true_bugs':true_bugs} 
        
        
        self.data = data
        return data
    
    
    