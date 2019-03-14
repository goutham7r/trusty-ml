import numpy as np

class World(object):
    
    def __init__(self, init_feat_vec, target, actions):
        
        self.load_data()
        self.set_actions(actions)
        self.get_model()
        
        self.init_feat_vec = init_feat_vec
        
        self.init_state = np.zeros((self.action_dim,))
        
        self.state = np.copy(self.init_state)
        self.target = target
                
        pred = self.get_pred(self.init_feat_vec)
        
        if np.array_equal(pred, self.target):
            assert False, "Target already achieved"
            
        self.get_terminal_reward()
        self.describe_world()
        
    def describe_world(self):
        print("Feature data:")
        print(self.feature_data)
        
        print("Actions: ")
        print(self.actions_dict)
        
        print("Terminal Reward:", self.terminal_reward)
        
        print("\n\n")

            
    def set_actions(self, actions_dict): 
        
        assert actions_dict is not None, "No actions set!"
        
        self.actions_dict = actions_dict
        self.feature_list = list(self.feature_data)
                
        self.actions = []
        self.action_step_sizes = []
        self.state_bounds = []
        for action in actions_dict:
            self.actions.append(action)
            self.action_step_sizes.append(actions_dict[action]['step_range'])
            self.state_bounds.append(actions_dict[action]['max_change'])
        
        self.action_step_sizes = np.array(self.action_step_sizes, dtype=float)
        self.state_bounds = np.array(self.state_bounds, dtype=float)
        
        self.actions_idx = np.array([self.feature_list.index(action) for action in self.actions])
        self.action_dim = len(self.actions)
        self.state_dim = self.action_dim
        self.action_high = 1.0
        self.action_low = -1.0
        
        self.means = np.zeros((self.action_dim))
        self.stds = np.zeros((self.action_dim))
        for i in range(self.action_dim):
            self.means[i] = self.feature_data[self.actions[i]]['mean']
            self.stds[i] = self.feature_data[self.actions[i]]['std']
            self.state_bounds[i] = np.divide(self.state_bounds[i],self.stds[i])
            self.action_step_sizes[i] = np.divide(self.action_step_sizes[i],self.stds[i])
            
    def reset(self):
        self.state = np.copy(self.init_state)        
        return self.state
    
    def load_data(self):
        gl = germanloan()
        self.data = gl.get_data()
        self.feature_data = self.get_feature_data()
        self.feature_list = list(self.feature_data)
        self.feat_vec_dim = self.data['X'].shape[1]
        
    def describe_data(self):
        print(self.feature_data)

    def transform_action(self, action):
        # from range -1 to 1 to actual change according to bounds defined
        trans_action = []
        for i in range(self.action_dim):
            a = ((action[i]-self.action_low)/(self.action_high-self.action_low))
            a = a*(self.action_step_sizes[i][1] - self.action_step_sizes[i][0]) + self.action_step_sizes[i][0]
            trans_action.append(a)
        return np.array(trans_action)
        
    def clip_state(self, state):
        for i in range(state.shape[0]):
            state[i] = max(self.state_bounds[i][0], min(state[i], self.state_bounds[i][1]))
        return state
    
    def step(self, action):
                
#         assert 0<=action<len(actions), "Invalid Action"
        assert len(self.actions)==action.shape[0], "Invalid Action"
#         assert np.sum(action!=0) == 1, "There must be exactly one action at a time"
        
#         print("Action:", action)
        action = self.transform_action(action)
#         print("Transformed Action:", action)
        
        new_state = self.clip_state(self.state + action)
#         print("Old state:", self.state, "New state:", new_state)
        
        cur_feat_vec = np.copy(self.init_feat_vec)
        cur_feat_vec[self.actions_idx] += new_state
#         print("Cur_feat_vec:", cur_feat_vec)
        
        pred = self.get_pred(cur_feat_vec)
        logits = self.get_logits(cur_feat_vec)
        
        cost = self.get_cost(self.state, new_state, action, logits)
        
#         print("Cost:", cost)
        
#         print("Old state")
#         self.describe_state(self.state)
#         print("New state")
#         self.describe_state(new_state)
        
        self.state = np.copy(new_state)
        
#         print("\n\n")

        if np.array_equal(pred, self.target):
            c = True
            cost -= self.terminal_reward
        else:
            c = False
        
        return new_state, -cost, c, {}
        
    def get_cost(self, old_state, new_state, action, logits):
        
#         c = 100*np.sqrt(np.sum(np.square(old_state-new_state)))
        
#         c = np.log(self.f(new_state)) - np.log(self.f(old_state))
        c = self.f(new_state) - self.f(old_state)

        
#         ind = self.actions_idx[np.nonzero(action)]
#         total_delta = self.init_state[ind] - new_state[ind]        
#         c = A*np.sum(np.exp(B*np.abs(total_delta)))

        return c
    
    def f(self, state):
        B = 3
#         f_state = np.exp(np.sum(B*np.abs(state)))
        f_state = np.sum(np.exp(B*np.abs(state)))

        return f_state
        
    def load_data(self):
        raise NotImplementedError("Must override load_data() method")
        
    def get_model(self):
        raise NotImplementedError("Must override get_model() method")
        
    def get_pred(self):
        raise NotImplementedError("Must override get_pred() method")
        
    def get_feature_data(self):
        raise NotImplementedError("Must override get_feature_data() method")
        
    def get_terminal_reward(self):
        raise NotImplementedError("Must override get_terminal_reward() method")
        
    def describe_action(self, action):
        print(action, ": ", end='')
        action = self.transform_action(action)
        for i in range(len(self.actions)):
            ac = (action[i])*self.stds[i]
            print(self.actions[i], ":", ac, end='    ')
        print()
            
    def describe_state(self, state=None, f=None):
        if state is None:
            state = self.init_state
        
        state = np.copy(state)
        state += self.init_feat_vec[self.actions_idx]
        
        for i in range(len(self.actions)):
            st = float(state[i]*self.stds[i] + self.means[i])
            if f:
                f.write("{:}:{:.3f}".format(self.actions[i], st))
            else:
                print("{:}:{:.3f}".format(self.actions[i], st), end = ' | ')
        if f:
            f.write("\n")
        else:
            print()
    
    