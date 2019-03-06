import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import logging
logging.set_verbosity(logging.ERROR)

from germanloan import germanloan
import numpy as np
POS_LABEL = np.array([0,1])
NEG_LABEL = np.array([1,0])
step_size = 0.01
def_init_state = np.array([-0.57573676,  1.28157579,  1.25257373, -0.704926  ,  0.91847717,
       -0.42828957,  0.14050471,  1.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  1.        ,
        0.        ,  1.        ,  0.        ,  0.        ,  1.        ,
        0.        ,  0.        ,  0.        ,  1.        ,  0.        ,
        1.        ,  0.        ,  0.        ,  0.        ,  1.        ,
        0.        ,  0.        ,  0.        ,  1.        ,  0.        ,
        0.        ,  0.        ,  1.        ,  0.        ,  0.        ,
        0.        ,  1.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  1.        ,  0.        ,  0.        ,  1.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        1.        ])

class GermanLoanWorld():
    
    def __init__(self, init_state=None, target=POS_LABEL, actions=None):
        self.load_data()
        self.set_actions(actions)
        
        if init_state is None:
            init_state = def_init_state
        
        self.init_state = init_state
        self.state = init_state
        self.target = target
        
        pred, logits = self.get_pred_logits(init_state)
        
        if np.array_equal(pred, self.target):
            assert False, "Target already achieved"
        
        
    def reset(self):
        self.state = self.init_state
        return self.state
        
        
    def load_data(self):
        gl = germanloan()
        self.data = gl.get_data()
        self.feature_data = self.data['feature_data_raw']
        self.feature_list = list(self.feature_data)
        self.state_dim = self.data['X'].shape[1]

        
    def set_actions(self, actions):
        if actions is None:
            actions = ['Age', 'Credit Amt', 'Duration']
        self.actions = actions
        self.actions_idx = np.array([self.feature_list.index(action) for action in actions])
        self.action_dim = len(actions)
        self.action_high = 1.0
        self.action_low = -1.0
        
        self.means = np.zeros((len(actions),))
        self.stds = np.zeros((len(actions),))
        for i in range(len(actions)):
            self.means[i] = self.feature_data[actions[i]]['mean']
            self.stds[i] = self.feature_data[actions[i]]['std']
        
        
    def step(self, action):
        
        ## action is an integer from [0,len(actions)]
        
#         assert 0<=action<len(actions), "Invalid Action"
        assert len(self.actions)==action.shape[0], "Invalid Action"
    
#         print(action)
        
        delta = np.zeros(self.state.shape)
        delta[self.actions_idx] += action 
                
        new_state = self.state + delta
        
        pred, logits = self.get_pred_logits(new_state)
        
        cost = self.get_cost(self.state, new_state, logits)
        
        self.state = new_state
        
        if np.array_equal(pred, self.target):
            c = True
        else:
            c = False
        
        return new_state, -cost, c, {}
        
    def get_cost(self, old_state, new_state, logits):
        
        c = 100*np.sqrt(np.sum(np.square(old_state-new_state)))
        
        arg = np.argmax(self.target)
        c1 = logits[arg]
        logits[arg] = 0
        c1 = max(np.max(logits) - c1, 0)
        
        c += c1
        
        return c
    
    def get_pred_logits(self, state):
        if len(state.shape)==1:
            state = np.reshape(state, (1,state.shape[0]))
        
        g_2 = tf.Graph()
        with g_2.as_default():
        
            with tf.Session(graph=g_2) as sess:
                saver = tf.train.import_meta_graph('model_best/germanloan_model.meta')
                saver.restore(sess, tf.train.latest_checkpoint('model_best/'))

                g = tf.get_default_graph()
                self.X = g.get_tensor_by_name("X:0")
                self.Y = g.get_tensor_by_name("Y:0")
                self.keep_prob = g.get_tensor_by_name("keep_prob:0")
                self.accuracy = g.get_tensor_by_name("accuracy:0")
                self.logits = g.get_tensor_by_name("logits:0")
                self.predicted = g.get_tensor_by_name("predicted:0")

                logits, predicted = sess.run([self.logits, self.predicted], feed_dict={self.X: state, self.keep_prob:1.0})
                
        pred = np.zeros(POS_LABEL.shape)
        pred[predicted] = 1
        return pred, logits.ravel()   
    
    def describe_action(self, action):
        print(action, ": ", end='')
        for i in range(len(self.actions)):
            ac = action[i]*self.stds[i]
            print(self.actions[i], ":", ac)
            
        
        
        