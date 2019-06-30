#%%
import random

import tensorflow as tf

import numpy as np
import numpy.random as npr

#%%
class DeepQNetwork(object):
  def __init__(
    self,
    n_features,
    n_actions,
    learning_rate,
    reward_decay,  # reward decay
    e_greedy,  # epsilon greedy ratio
    replace_target_iter,  # the iterations between the parameters transform
    memory_size,  # the size of memory
    batch_size,  # the batch_size for training
    e_greedy_increment,  # the decay of the epsilon greedy ratio
    output_graph,  # save the tensorflow graph for visualization
  ):
    self.n_features = n_features
    self.n_actions = n_actions
    self.lr = learning_rate
    self.gamma = reward_decay
    self.epsilon_max = e_greedy
    self.replace_target_iter = replace_target_iter
    self.memory_size = memory_size
    self.batch_size = batch_size
    self.epsilon_increment = e_greedy_increment
    self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

    self.learn_step_counter = 0

    # initialize zero memory [s, a, r, s_]: n_features * 2 + 2
    self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

    # build the net
    self._built_net()

    target_pars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
    eval_pars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

    with tf.variable_scope('hard_replacement'):
      self.target_replace_op = [tf.assign(t, e) for t, e in zip(target_pars, eval_pars)]
    
    self.sess = tf.Session()

    if output_graph:
      tf.summary.FileWriter('logs/', self.sess.graph)

    self.sess.run(tf.global_variables_initializer())
    self.cost_his = []
  
  def _built_net(self):
    self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input state
    self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input next state
    self.a = tf.placeholder(tf.int32, [None, ], name='actions')  # actions
    self.r = tf.placeholder(tf.float32, [None, ], name='reward')  # rewards

    w_init = tf.initializers.truncated_normal(stddev=0.1)
    b_init = tf.initializers.constant(0.1)
    # ------------------------ build evaluate net ---------------------------
    with tf.variable_scope('eval_net'):
      hiden_layer1 = tf.layers.dense(
        inputs=self.s,
        units=20,
        activation=tf.nn.relu,
        kernel_initializer=w_init,
        bias_initializer=b_init)
      
      self.q_eval = tf.layers.dense(
        inputs=hiden_layer1,
        units=self.n_actions,
        activation=tf.nn.relu,
        kernel_initializer=w_init,
        bias_initializer=b_init)

    # ------------------------ build target net ---------------------------
    with tf.variable_scope('target_net'):
      hiden_layer1_ = tf.layers.dense(
        inputs=self.s_,
        units=20,
        activation=tf.nn.relu,
        kernel_initializer=w_init,
        bias_initializer=b_init)
        
      self.q_ = tf.layers.dense(
        inputs=hiden_layer1_,
        units=self.n_actions,
        activation=tf.nn.relu,
        kernel_initializer=w_init,
        bias_initializer=b_init)

    with tf.variable_scope('q_target'):
      q_target = self.r + self.gamma * tf.reduce_max(self.q_, axis=1, name='Qmax_s_')
      self.q_target = tf.stop_gradient(q_target)

    # 
    with tf.variable_scope('q_eval'):
      a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
      self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)

    with tf.variable_scope('loss'):
      self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))

    with tf.variable_scope('train'):
      self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

  def store_transition(self, s, a, r, s_):
    """
      store the transition into memory
    """
    if not hasattr(self, 'memory_counter'):
      self.memory_counter = 0
    
    self.memory[self.memory_counter % self.memory_size, :] = np.hstack((s, [a, r], s_))

    self.memory_counter += 1

  def choose_action(self, observation):
    """
      return action from the observation
    """
    observation = observation[np.newaxis, :]

    if npr.rand() > self.epsilon:
      action = npr.randint(self.n_actions)
    else:
      actions = self.sess.run(self.q_eval, feed_dict={self.s: observation})
      action = np.argmax(actions)

    return action

  def learn(self):
    """
      train the network
    """
    # excute the hard replacement operation
    if self.learn_step_counter %  self.replace_target_iter == 0:
      self.sess.run(self.target_replace_op)
      print("target parameters replaced...")

    # choose batch from memory
    batch_index = random.sample(range(min(self.memory_size, self.memory_counter)), self.batch_size)
    batch_memories = self.memory[batch_index, :]


    # excute the train operation
    _, cost = self.sess.run(
      [self._train_op, self.loss],
      feed_dict={
        self.s: batch_memories[:, :self.n_features],
        self.a: batch_memories[:, self.n_features],
        self.r: batch_memories[:, self.n_features+1],
        self.s_: batch_memories[:, -self.n_features:]})


    # update the epsilon max, we may want to choose more accurate action
    self.epsilon = self.epsilon_increment + self.epsilon \
      if self.epsilon < self.epsilon_max else self.epsilon_max


    # append cost to the cost history
    self.cost_his.append(cost)
    self.learn_step_counter += 1


  def show_his(self):
    import matplotlib.pyplot as plt
    
    plt.rcParamsDefault['figure.autolayout'] = True

    plt.plot(np.arange(len(self.cost_his)), self.cost_his)
    plt.ylabel('Cost')
    plt.xlabel('training steps')
    plt.show()

#
if __name__ == '__main__':
  RL = DeepQNetwork(
    n_features=4, 
    n_actions=2, 
    learning_rate=0.01,
    reward_decay=0.9,
    e_greedy=0.9,
    replace_target_iter=300,
    memory_size=500,
    batch_size=32,
    e_greedy_increment=None,
    output_graph=True)
  
  


