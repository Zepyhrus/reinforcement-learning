import numpy as np
import tensorflow as tf
import numpy.random as npr

#%%
class PolicyGradient(object):
  def __init__(
    self,
    n_actions,
    n_features,
    learning_rate,
    reward_decay,  # reward decay
    # e_greedy,  # epsilon greedy ratio
    # replace_target_iter,  # the iterations between the parameters transform
    # memory_size,  # the size of memory
    # batch_size,  # the batch_size for training
    # e_greedy_increment,  # the decay of the epsilon greedy ratio
    output_graph,  # save the tensorflow graph for visualization
  ):
    self.n_actions = n_actions
    self.n_features = n_features
    self.lr = learning_rate
    self.gamma = reward_decay
    # self.epsilon_max = e_greedy
    # self.replace_target_iter = replace_target_iter
    # self.memory_size = memory_size
    # self.batch_size = batch_size
    # self.epsilon_increment = e_greedy_increment
    # self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

    # initialize empty memory 
    self.ep_obs = []
    self.ep_actions = []
    self.ep_rewards = []

    # build the net
    self._build_net()

    # needs no double neural network in policy gradient
    # target_pars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
    # eval_pars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

    # with tf.variable_scope('hard_replacement'):
    #   self.target_replace_op = [tf.assign(t, e) for t, e in zip(target_pars, eval_pars)]
    
    self.sess = tf.Session()

    if output_graph:
      tf.summary.FileWriter('logs/', self.sess.graph)

    self.sess.run(tf.global_variables_initializer())
  
  def _build_net(self):
    self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name='observations')  # input state
    self.tf_acts = tf.placeholder(tf.int32, [None, ], name='actions_num')  # actions
    self.tf_vt = tf.placeholder(tf.float32, [None, ], name='action_value')  # action values

    layer = tf.layers.dense(
      inputs=self.tf_obs,
      units=10,
      activation=tf.nn.tanh,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.3),
      bias_initializer=tf.constant_initializer(0.1),
      name='fc1')
    
    all_act = tf.layers.dense(
      inputs=layer,
      units=self.n_actions,
      activation=None,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.3),
      bias_initializer=tf.constant_initializer(0.1),
      name='fc2'
    )

    self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')  # use softmax to convert to probability

    with tf.variable_scope('loss'):
      neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)
      
      loss = tf.reduce_mean(neg_log_prob * self.tf_vt)
    
    with tf.variable_scope('train'):
      self.train_op = tf.train.RMSPropOptimizer(self.lr).minimize(loss)
  def choose_action(self, observation):
    """
      return action from the observation
    """
    observation = observation[np.newaxis, :]
    
    prob_weights = self.sess.run(
      self.all_act_prob,
      feed_dict={self.tf_obs: observation})

    action = npr.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())

    return action
  def store_transition(self, s, a, r):
    """
      store the transition into memory
    """
    self.ep_obs.append(s)
    self.ep_actions.append(a)
    self.ep_rewards.append(r)

  

  def learn(self):
    """
      train the network
    """
    # Policy Gradient does not have 2 networks so it does not need hard replacement
    # excute the hard replacement operation
    # if self.learn_step_counter %  self.replace_target_iter == 0:
    #   self.sess.run(self.target_replace_op)
    #   print("target parameters replaced...")

    # Policy Gradient use all the memory for training, does not need batch
    # choose batch from memory
    # batch_index = random.sample(range(min(self.memory_size,
    # self.memory_counter)), self.batch_size)
    # batch_memories = self.memory[batch_index, :]

    # 
    discounted_ep_rs_norm = self._discount_and_norm_rewards()

    # excute the train operation
    self.sess.run(
      self.train_op,
      feed_dict={
        self.tf_obs: np.vstack(self.ep_obs),
        self.tf_acts: np.array(self.ep_actions),
        self.tf_vt: discounted_ep_rs_norm
        })
    
    self.ep_obs, self.ep_actions, self.ep_rewards = [], [], []
    return discounted_ep_rs_norm


    # update the epsilon max, we may want to choose more accurate action
    self.epsilon = self.epsilon_increment + self.epsilon \
      if self.epsilon < self.epsilon_max else self.epsilon_max


    # append cost to the cost history
    self.cost_his.append(cost)
    self.learn_step_counter += 1

  def _discount_and_norm_rewards(self):
    # discount episode rewards
    discounted_ep_rs = np.zeros_like(self.ep_rewards)
    running_add = 0

    for t in reversed(range(len(self.ep_rewards))):
      running_add = running_add * self.gamma + self.ep_rewards[t]
      discounted_ep_rs[t] = running_add

    # (0, 1) normalization
    discounted_ep_rs -= np.mean(discounted_ep_rs)
    discounted_ep_rs /= np.std(discounted_ep_rs)

    return discounted_ep_rs


  def show_his(self):
    import matplotlib.pyplot as plt
    
    plt.rcParamsDefault['figure.autolayout'] = True

    plt.plot(np.arange(len(self.tf_vt)), self.tf_vt)
    plt.ylabel('Cost')
    plt.xlabel('training steps')
    plt.show()

#
if __name__ == '__main__':
  RL = PolicyGradient(
    n_features=4, 
    n_actions=2, 
    learning_rate=0.02,
    reward_decay=0.99,
    output_graph=True)
  
  


