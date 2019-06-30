


import tensorflow as tf





def DeepQNetwork(object):
  def __init__(
    self,
    n_features,
    n_actions,
    learning_rate,
  ):
    self.n_features = n_features
    self.n_actions = n_actions
    self.lr = learning_rate

  
  def _built_net(self):
    # ------------------------ build evaluate net ---------------------------
    self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')
    self.actions = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')

    with tf.variable_scope('eval_net'):
      hiden_layer1 = tf.layer.dense(self.s, 12, activation=tf.nn.relu)
      hiden_layer2 = tf.layer.dense(hidden_layer1, 12, activation=tf.nn.relu)
      self.q_eval = tf.nn.softmax(tf.layer.dense(hiden_layer2, 2))

    with tf.variable_scope('loss'):
      self.loss = tf.reduce_sum(tf.squared_difference(self.q_target, self.q_eval))

    with tf.variable_scope('train'):
      self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)


    # ------------------------ build evaluate net ---------------------------
    self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')
    with tf.variable_scope('target_net'):
      hiden_layer1_ = tf.layer.dense(self.s, 12, activation=tf.nn.relu)



  def choose_action(self):
    pass

  def store_transition(self):
    pass
  
  def _replace_target(self):
    pass

  def learn():
    pass

  def plot_cost():
    pass

  



