import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class harrypotter:
    
    def get_data(self,n=100, random_seed=123):
        
        np.random.seed(random_seed)
        # ----------------------------------------
        # Generate "dirty" training data. 
        # that is, we will plant some "historical bias" 
        # in the form of labels: the Ministry of Magic refused to hire
        # muggle-born graduates with high edcuation.

        print("Creating training data...\n")

        # data points are on a uniform grid, then dithered with a Gaussian.
        # x_1=magic heritage; x_2=education

        X_train = np.zeros((n,2))
        a = np.linspace(0.05, 0.95, num=int(np.sqrt(n)))
        e, d = np.meshgrid(a, a)
        X_train[:,0] = np.reshape(e,(n,))
        X_train[:,1] = np.reshape(d,(n,))
        X_train = X_train + 0.03*np.random.rand(n,2)

        # the noiseless 'desired' label obeys y = sign(x_2 - 0.5)
        y_clean = np.sign(X_train[:,1]-0.5)
        # print(y_clean)

        y_train = np.copy(y_clean)
        y_train[(X_train[:,1]<(4*(X_train[:,0]-0.5)**2+0.5)) & (X_train[:,0]<0.5)] = -1
        # print(y_train-y_clean)
        
        true_bugs = y_train-y_clean
        true_bugs[true_bugs!=0] = 1
        
        # --------------------------------------------------------------
        # Generate trusted data
        # we manually picked these two trusted items for pedagogical purpose
        print("Creating trusted data...\n")

        X_trust = np.array([[0.3, 0.4],[0.2, 0.6]])
        y_trust = np.sign(X_trust[:,1]-0.5)
        
        return X_train,y_train,X_trust,y_trust,true_bugs
    
    
    # ----------------------------------------
    # For plotting data and decision boundary
    def plot_model(self,X_train, y_train, clf=None, bugs=None, X_trust=None, y_trust=None, title=None):
        if len(y_train.shape)==2:
            y_train = np.reshape(y_train,(y_train.shape[0],))
        
        plt.figure()
        
        if clf:
            X = np.zeros((10000,2))
            a = np.linspace(0,1,100)
            b = np.linspace(0,1,100)
            e, d = np.meshgrid(a, b)
            X[:,0] = np.reshape(e,(10000,))
            X[:,1] = np.reshape(d,(10000,))

            Z = clf.predict(X)
            probs = clf.predict_proba(X)
            if len(probs.shape)==2:
                probs = probs[:,probs.shape[1]-1]
            probs = probs.reshape(e.shape)
            
            # Put the result into a color plot
            Z = Z.reshape(e.shape)
            plt.contour(e, d, probs, levels=[0.5])
            
        

        # Plot the training points
        plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Paired, marker='o')
        
        if bugs is not None:
            buggy = X_train[bugs==1,:]
            plt.scatter(buggy[:, 0], buggy[:, 1], marker='.', c='blue')

        # Plot the trusted points
        if X_trust is not None:
            if len(y_trust.shape)==2:
                y_trust = np.reshape(y_trust,(y_trust.shape[0],))
            plt.scatter(X_trust[:, 0], X_trust[:, 1], c=y_trust, cmap=plt.cm.Paired, marker='*')

        if title is not None:
            plt.title(str(title))

        plt.xlabel('Magical Heritage')
        plt.ylabel('Education')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.draw()
        
        
    def plot_influences(self,X_train, y_train, all_influences, n, top_n=True, X_trust=None, y_trust=None, title=None):
        
        for i in range(all_influences.shape[0]):
            influences = [(j,all_influences[i][j]) for j in range(all_influences.shape[1])]
            
            influences.sort(key=lambda x:x[1], reverse=top_n)
#             print(influences)
            top_influence_indices = [j[0] for j in influences[:n]]
            
            top_influence_X = X_train[top_influence_indices,:]
            top_influence_y = y_train[top_influence_indices]

            plt.figure()
            plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Paired, marker='o')

            plt.scatter(top_influence_X[:, 0], top_influence_X[:, 1], marker='.')

            if X_trust is not None:
                plt.scatter(X_trust[:, 0], X_trust[:, 1], c=y_trust, cmap=plt.cm.Paired, marker='*')
                for i in range(X_trust.shape[0]):
                    plt.annotate(i, (X_trust[i][0], X_trust[i][1]))
            
            tit = " Trusted item %d"%i
            if title:
                tit = title + tit
                
            plt.title(tit)
            plt.xlabel('Magical Heritage')
            plt.ylabel('Education')
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            
            plt.draw()
            plt.show()
        
        

        
        
        
class womenbias:
    
    
    def get_data(self,n=100, random_seed=123):
    
        ################### Bias Against Young Women Toy Example ##################
        # ----------------------------------------
        # Generate "dirty" training data. 
        # that is, we will plant some "historical bias" 
        # in the form of labels: women under the age of 35 are 
        # less likely to be hired than men of similar skill level

        # for reproducibility, fix the random seed 
        # 90,190 works well
        np.random.seed(190)

        print("Creating training data...\n")

        # Training Set size
        n = 200 

        # x_0=age; chosen randomly between 22 and 55
        # x_1=skill level; random integer number between 0 and 100
        # x_2=gender; 0=man, 1=woman


        X_train = np.zeros((n,3))

        X_train[:n//2,2] = np.zeros((n//2,))
        X_train[n//2:n,2] = np.ones((n//2,))

        ages = np.random.randint(22, 56, size=(n//2,))
        X_train[:n//2,0] = ages
        X_train[n//2:n,0] = ages + np.random.randint(-10, 11, size=(n//2,))
        X_train[n//2:n,0] = np.random.randint(22, 56, size=(n//2,))

        skills = np.random.randint(15, 95, size=(n//2,))
        X_train[:n//2,1] = skills
        X_train[n//2:n,1] = skills + np.random.randint(-10, 11, size=(n//2,))
        X_train[n//2:n,1] = np.random.randint(15, 95, size=(n//2,))


        np.random.shuffle(X_train)

        # Optimal Hiring rule: Skill/Age > 2
        y_clean = np.sign(0.7*X_train[:,1] - X_train[:,0] + 1e-7)
        # y_clean = np.reshape(y_clean,(y_clean.shape[0],1))

        # Introducing bugs in dataset
        y_train = np.copy(y_clean)
        y_train[(0.35*X_train[:,1] - X_train[:,0]<0) & (X_train[:,2]==1) & (X_train[:,0]<35)] = -1
        # print(y_train-y_clean)
        
        true_bugs = y_train-y_clean
        true_bugs[true_bugs!=0] = 1
        
        # --------------------------------------------------------------
        # Generate trusted data
        # we manually picked these two trusted items for pedagogical purpose
        print("Creating trusted data...\n")
        X_trust = np.array([[30,60,1],[27,53,0],[38,45,0],[43,55,1]])
        y_trust = np.sign(0.7*X_trust[:,1] - X_trust[:,0] + 1e-7)
        
        return X_train,y_train,X_trust,y_trust,true_bugs
        
        
        
    def plot_model(self,X_train, y_train, clf=None, X_trust=None, y_trust=None, bugs=None, title=None):
    
        fig, ax = plt.subplots(figsize=(10, 10))

        print("Legend:")
        print("Brown: Hired, Blue: Not Hired")
        print("Circle: Men, Cross: Women")
        print("Star: Trusted Men, Plus: Trusted Women")
        print("Dark Blue Dots: Bugs")

        ax.scatter(X_train[X_train[:,2]==0, 0], X_train[X_train[:,2]==0, 1], c=y_train[X_train[:,2]==0], 
                   cmap=plt.cm.Paired, marker='o', s=50)
        ax.scatter(X_train[X_train[:,2]==1, 0], X_train[X_train[:,2]==1, 1], c=y_train[X_train[:,2]==1], 
                   cmap=plt.cm.Paired, marker='X', s=50)


        if X_trust is not None:
            ax.scatter(X_trust[X_trust[:,2]==0, 0], X_trust[X_trust[:,2]==0, 1], c=y_trust[X_trust[:,2]==0], 
                       cmap=plt.cm.Paired, marker='*', s=100)
            ax.scatter(X_trust[X_trust[:,2]==1, 0], X_trust[X_trust[:,2]==1, 1], c=y_trust[X_trust[:,2]==1], 
                       cmap=plt.cm.Paired, marker='P', s=100)
            for i in range(X_trust.shape[0]):
                ax.annotate(i, (X_trust[i][0], X_trust[i][1]))


        if bugs is not None:
            buggy = X_train[bugs==1,:]
            ax.scatter(buggy[:, 0], buggy[:, 1], marker='.', s=30)

        plt.xlabel('Age')
        plt.ylabel('Skill')
        if title:
            plt.title(title)

    #     for i in range(X_train.shape[0]):
    #         ax.annotate(i, (X_train[i][0], X_train[i][1]))
    
    
    def plot_influences(self,X_train, y_train, all_influences, n, top_n=True, X_trust=None, y_trust=None, title=None):
        
        for i in range(all_influences.shape[0]):
            influences = [(j,all_influences[i][j]) for j in range(all_influences.shape[1])]
            
            influences.sort(key=lambda x:x[1], reverse=top_n)
            top_influence_indices = [j[0] for j in influences[:n]]
            bugs = np.zeros((X_train.shape[0],))
            bugs[top_influence_indices] = 1
            
            self.plot_model(X_train, y_train, X_trust=X_trust, y_trust=y_trust, bugs=bugs, title=title)
            
            

            
            
class germanloan():
    
    def __init__(self, clf):
        self.clf = clf
    
    def get_data(self, random_seed=123):
    
        np.random.seed(random_seed)

        # read dataset
        col_names = ['Checking Status', 'Duration','Credit History', 'Credit Amt','Purpose', 
                      'Saving acc', 'Present emp since', 'Installment Rate',
                     'Personal Status', 'Age', 'Other debtors', 'Present Residence since', 'Property',
                      'Other installment plans', 'Housing', 'Existing credits',
                     'Job', 'Num People', 'Telephone', 'Foreign Worker','a','b','c','d','Approval Status']

        numerical = ['Duration','Credit Amt','Age']

        print("Reading dataset...")
        all_data = pd.read_csv("germanloan.csv", names=col_names)
#         print(all_data.shape)

        n = all_data.shape[0]
        X = all_data[all_data.columns.difference(['Approval Status'])]
        y = all_data['Approval Status']
#         print(X.shape,y.shape)
        
        y[y==2] = -1
        
        # partition dataset on the basis of age, threshold = 25
        X_y = X[X['Age']<=25]
        X_o = X[X['Age']>25]
        y_y = y[X['Age']<=25]
        y_o = y[X['Age']>25]
        print(X_y.shape, X_o.shape)
        
        # Remove age as a feature
        del X_y['Age']
        del X_o['Age']
        
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

        
        return X_train,y_train,X_trust,y_trust,true_bugs
        
        
        
    def plot_model(self,X_train, y_train, clf=None, X_trust=None, y_trust=None, bugs=None, title=None):
        pass
    
    
    def plot_influences(self,X_train, y_train, all_influences, n, top_n=True, X_trust=None, y_trust=None, title=None):
        pass
        
        
        