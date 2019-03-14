import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import logging
logging.set_verbosity(logging.ERROR)

from germanloan import germanloan
import numpy as np

POS_LABEL = np.array([0,1])
NEG_LABEL = np.array([1,0])

def_init_feat_vec = np.array([-0.57573676,  1.28157579,  1.25257373, -0.704926  ,  0.91847717,
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

from world import World

TERMINAL_REWARD = 100

class GermanLoanWorld(World):
    
    def __init__(self, init_feat_vec=def_init_feat_vec, target=POS_LABEL, actions=None):
        super().__init__(init_feat_vec=init_feat_vec, target=target, actions=actions) 
        
    def describe_world(self):
        print("GermanLoadWorld")
        super().describe_world()
    
    def load_data(self):
        gl = germanloan()
        self.data = gl.get_data()
        self.get_feature_data()
    
    def get_feature_data(self):
        '''Contains means and stds of all possible actions'''
        self.feature_data = self.data['feature_data_raw']
    
    def get_terminal_reward(self):
        self.terminal_reward = TERMINAL_REWARD
    
    def get_model(self):
        self.g = tf.Graph()
        with self.g.as_default(): 
            self.sess = tf.Session(graph=self.g)
            saver = tf.train.import_meta_graph('model_best/germanloan_model.meta')
            saver.restore(self.sess, tf.train.latest_checkpoint('model_best/'))
            g = tf.get_default_graph()
            self.X = g.get_tensor_by_name("X:0")
            self.Y = g.get_tensor_by_name("Y:0")
            self.keep_prob = g.get_tensor_by_name("keep_prob:0")
            self.accuracy = g.get_tensor_by_name("accuracy:0")
            self.logits = g.get_tensor_by_name("logits:0")
            self.predicted = g.get_tensor_by_name("predicted:0")
    
    def get_pred(self, cur_feat_vec):
        if len(cur_feat_vec.shape)==1:
            cur_feat_vec = np.reshape(cur_feat_vec, (1,cur_feat_vec.shape[0]))
        
        predicted = self.sess.run(self.predicted, feed_dict={self.X: cur_feat_vec, self.keep_prob:1.0})
                
        pred = np.zeros(POS_LABEL.shape)
        pred[predicted] = 1
        return pred
    
    def get_logits(self, cur_feat_vec):
        if len(cur_feat_vec.shape)==1:
            cur_feat_vec = np.reshape(cur_feat_vec, (1,cur_feat_vec.shape[0]))
        
        logits = self.sess.run(self.logits, feed_dict={self.X: cur_feat_vec, self.keep_prob:1.0})  
        return logits.ravel() 
    
        