"""
A reproduction of policy gradient by Sherk
"""

import tensorflow as tf
import numpy as np





class PolicyGradient(object):
  def __init__(
      self,
      n_actions,
      n_features,
      learning_rate,
      reward_decay,
      output_graph
      ):
    """
    Initialize the brain
    """
    self.n_actions = n_actions  # number of actions
    self.n_features = n_features  # Q: What is the input features? A: Observation features
    self.lr = learning_rate  # the learning rate used to update the gradient
    self.gamma = reward_decay  # futher the step is, smaller the weight is
    
    self._bulid_net()  # initialize the net
    
    config = tf.ConfigProto(
      allow_soft_placement=True,
      log_device_placement=True
    )
    config.gpu_options.allow_growth = False
    # config.gpu_options.per_process_gpu_memory_fraction = 0.9
    self.sess = tf.Session(config=config)  # initialize the session

    if output_graph:
      tf.summary.FileWriter('logs/', self.sess.graph)

  def _bulid_net():
    pass

  def choose_action(self, observation):
    """
    Choose action from the observations
    """
    pass

  def store_transition():
    """
    Store the transition
    """
    pass

  def learn(self):
    """
    Learn
    """
    pass
  
  def _discount_and_norm_rewards(self):
    """
    reward decay
    """
    pass





if __name__ == '__main__':
  """
  Element test
  """

  pg_brain = PolicyGradient()


