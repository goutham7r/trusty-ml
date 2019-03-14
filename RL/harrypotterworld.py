from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

def_init_feat_vec = np.array([0.45582669, 0.32717371])

POS_LABEL = 1
NEG_LABEL = 0

TERMINAL_REWARD = 50

from world import World

class HarryPotterWorld(World):
    
    def __init__(self, init_feat_vec=def_init_feat_vec, target=POS_LABEL, actions=None):
        super().__init__(init_feat_vec=init_feat_vec, target=target, actions=actions) 

    def describe_world(self):
        print("HarryPotterWorld")
        super().describe_world()
    
    def get_pred(self, state):
        if len(state.shape)==1:
            state = np.reshape(state, (1,state.shape[0]))
        return self.clf.predict(state)[0]
    
    def get_logits(self, state):
        if len(state.shape)==1:
            state = np.reshape(state, (1,state.shape[0]))
        return self.clf.predict_proba(state)[0]
    
    def get_terminal_reward(self):
        self.terminal_reward = TERMINAL_REWARD

#     def reset(self):
#         self.state = np.copy(self.init_state)
# #         print("State reset", self.state)
# #         self.describe_state(self.state)
#         return self.state
    
    
#     def set_actions(self, actions):
#         if actions is None:
#             actions = ['x2']
#         self.actions = actions
#         self.actions_idx = np.array([self.feature_list.index(action) for action in actions])
#         self.action_dim = len(actions)
#         self.state_dim = self.action_dim
#         self.action_high = 1.0
#         self.action_low = -1.0
        
#         self.means = np.zeros((len(actions),))
#         self.stds = np.zeros((len(actions),))
#         for i in range(len(actions)):
#             self.means[i] = 0
#             self.stds[i] = 1
        
#         self.action_step_sizes = np.array([ [0, 0.05]], dtype=float)
#         self.state_bounds = np.array([[0, 0.6]], dtype=float)

#         for i in range(len(actions)):
#             self.state_bounds[i] = np.divide(self.state_bounds[i],self.stds[i])
#             self.action_step_sizes[i] = np.divide(self.action_step_sizes[i],self.stds[i])
            
#     def transform_action(self, action):
#         # from range -1 to 1 to actual change according to bounds defined
        
#         trans_action = []
#         for i in range(self.action_dim):
#             a = ((action[i]-self.action_low)/(self.action_high-self.action_low))
#             a = a*(self.action_step_sizes[i][1] - self.action_step_sizes[i][0]) + self.action_step_sizes[i][0]
#             trans_action.append(a)
#         return np.array(trans_action)
        
#     def clip_state(self, state):
#         for i in range(state.shape[0]):
#             state[i] = max(self.state_bounds[i][0], min(state[i], self.state_bounds[i][1]))
#         return state
    
    
#     def step(self, action):
                
# #         assert 0<=action<len(actions), "Invalid Action"
#         assert len(self.actions)==action.shape[0], "Invalid Action"
# #         assert np.sum(action!=0) == 1, "There must be exactly one action at a time"
        
# #         print("Action:", action)
#         action = self.transform_action(action)
# #         print("Transformed Action:", action)
        
#         new_state = self.clip_state(self.state + action)
# #         print("Old state:", self.state, "New state:", new_state)
        
#         cur_feat_vec = np.copy(self.init_feat_vec)
#         cur_feat_vec[self.actions_idx] += new_state
# #         print("Cur_feat_vec:", cur_feat_vec)
        
#         pred = self.get_pred(cur_feat_vec)
        
#         cost = self.get_cost(self.state, new_state, action)
        
# #         print("Cost:", cost)
        
# #         print("Old state")
# #         self.describe_state(self.state)
# #         print("New state")
# #         self.describe_state(new_state)
        
#         self.state = np.copy(new_state)
        
# #         print("\n\n")

        
#         if np.array_equal(pred, self.target):
#             c = True
#             cost -= TERMINAL_REWARD
#         else:
#             c = False
        
#         return new_state, -cost, c, {}
        
#     def get_cost(self, old_state, new_state, action):
# #         c = 100*np.sqrt(np.sum(np.square(old_state-new_state)))
# #         c = np.log(self.f(new_state)) - np.log(self.f(old_state))
#         c = self.f(new_state) - self.f(old_state)
#         return c     
    
#     def f(self, state):
#         B = 10
# #         f_state = np.exp(np.sum(B*np.abs(state)))
#         f_state = np.sum(np.exp(B*np.abs(state)))
#         return f_state
    
    
#     def describe_action(self, action):
# #         print(action, ": ", end='')
#         action = self.transform_action(action)
#         for i in range(len(self.actions)):
#             ac = (action[i])*self.stds[i]
#             print(self.actions[i], ":", ac, end='    ')
#         print()
            
#     def describe_state(self,state=None, f=None):
#         if state is None:
#             state = self.init_state
        
#         state = np.copy(state)
#         state += self.init_feat_vec[self.actions_idx]
        
#         for i in range(len(self.actions)):
#             st = float(state[i]*self.stds[i] + self.means[i])
#             if f:
#                 f.write("{:}:{:.3f}".format(self.actions[i], st))
#             else:
#                 print("{:}:{:.3f}".format(self.actions[i], st), end = ' | ')
#         if f:
#             f.write("\n")
#         else:
#             print()
    
    
    def get_model(self):
#         clf = LogisticRegression(solver='lbfgs', C=1000)
        clf = SVC(probability=True, gamma='auto', C=5)
        clf.fit(self.X_train, self.y_train)
        self.clf = clf
        
    def get_feature_data(self):
        self.feature_data = {}
        self.feature_data['x1'] = {'mean':0, 'std':1}
        self.feature_data['x2'] = {'mean':0, 'std':1}
        
    def load_data(self):
        # for reproducibility, fix the random seed 
        np.random.seed(123)

        # the learner is hard coded to be kernel logistic regression.
        # learner's parameters:
        sigma = 0.4	# RBF kernel bandwidth
        gamma = (1/(2.0*sigma**2))
        lam = 1e-3
        
        ################### Harry Potter Toy Example ##################
        # ----------------------------------------
        # Generate "dirty" training data. 
        # that is, we will plant some "historical bias" 
        # in the form of labels: the Ministry of Magic refused to hire
        # muggle-born graduates with high edcuation.

        # Training Set size
        n = 100 

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

        y_train = np.copy(y_clean)
        y_train[(X_train[:,1]<(4*(X_train[:,0]-0.5)**2+0.5)) & (X_train[:,0]<0.5)] = -1
        
        # --------------------------------------------------------------
        # Generate trusted data
        # we manually picked these two trusted items for pedagogical purpose
#         print("Creating trusted data...\n")

        X_trust = np.array([[0.3, 0.4],[0.2, 0.6]])
        y_trust = np.sign(X_trust[:,1]-0.5)
#         y_trust = np.reshape(y_trust,(y_trust.shape[0],1))

        self.X_train, self.y_train, self.X_trust, self.y_trust = X_train, y_train, X_trust, y_trust
        
        self.get_feature_data()

        
    def plot_model(self, title=None):

        X_train, y_train, X_trust, y_trust, clf = self.X_train, self.y_train, self.X_trust, self.y_trust, self.clf
        
        X = np.zeros((10000,2))
        a = np.linspace(0,1,100)
        b = np.linspace(0,1,100)
        e, d = np.meshgrid(a, b)
        X[:,0] = np.reshape(e,(10000,))
        X[:,1] = np.reshape(d,(10000,))

        Z = clf.predict(X)
        probs = clf.predict_proba(X)[:, 1].reshape(e.shape)

        plt.figure()

        # Put the result into a color plot
        Z = Z.reshape(e.shape)
        plt.contour(e, d, probs, levels=[0.5])

        # Plot the training points
        plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Paired, marker='.')

        # Plot the trusted points
        if X_trust is not None:
            plt.scatter(X_trust[:, 0], X_trust[:, 1], c=y_trust, cmap=plt.cm.Paired, marker='X')

        if title is not None:
            plt.title(str(title))

        plt.xlabel('Magical Heritage')
        plt.ylabel('Education')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.draw()
        
        
        
