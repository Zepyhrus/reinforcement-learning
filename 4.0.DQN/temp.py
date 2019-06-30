#%%
import tensorflow as tf

import numpy as np
import numpy.random as npr

#%%
npr.seed(98)

s = tf.placeholder(tf.float32, [None, 4], name='s')
q_target = tf.placeholder(tf.float32, [None, 2], name='Q_target')

with tf.name_scope('eval_net'):
  hidden_layer1 = tf.layers.dense(
    inputs=s,
    units=12,
    activation=tf.nn.relu,
    kernel_initializer=tf.initializers.truncated_normal(stddev=0.1))
  hidden_layer2 = tf.layers.dense(
    inputs=hidden_layer1,
    units=12,
    activation=tf.nn.relu,
    kernel_initializer=tf.initializers.truncated_normal(stddev=0.1))
  q_eval = tf.nn.softmax(tf.layers.dense(hidden_layer2, 2))

with tf.name_scope('loss'):
  loss = tf.reduce_sum(tf.squared_difference(q_target, q_eval))

with tf.name_scope('train'):
  _train_op = tf.train.RMSPropOptimizer(learning_rate=0.01).minimize(loss)

initialize_op = tf.global_variables_initializer()

with tf.Session() as sess:
  sess.run(initialize_op)

  print(sess.run(loss, feed_dict=({s: npr.rand(10, 4), q_target: npr.rand(10, 2)})))


#%%
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
    self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')

    with tf.variable_scope('eval_net'):
      hiden_layer1 = tf.layers.dense(
        inputs=self.s,
        units=12,
        activation=tf.nn.relu,
        kernel_initializer=tf.initializers.truncated_normal(stddev=0.1))
      hiden_layer2 = tf.layers.dense(
        inputs=hidden_layer1,
        units=12,
        activation=tf.nn.relu,
        kernel_initializer=tf.initializers.truncated_normal(stddev=0.1))
      self.q_eval = tf.nn.softmax(tf.layers.dense(hiden_layer2, 2))

    with tf.variable_scope('loss'):
      self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))

    with tf.variable_scope('train'):
      self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

#
if __name__ == '__main__':
  brain = DeepQNetwork(
    n_features=4,
    n_actions=2,
    learning_rate=0.01)

  brain._built_net()


#%%
