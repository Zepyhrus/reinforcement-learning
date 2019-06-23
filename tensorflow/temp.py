"""
This is a demo tensorflow CNN reproduction
"""
#%% 
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# parameters
epochs = 100
learning_rate = 0.001
batch_size = 32



#%%
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])


w1 = tf.Variable(tf.random_normal([784, 300], stddev=0.03), name='w1')
b1 = tf.Variable(tf.random_normal([300]), name='b1')

out1 = tf.add(tf.matmul(x, w1), b1)
out1 = tf.nn.relu(out1)


w2 = tf.Variable(tf.random_normal([300, 10], stddev=0.03), name='w2')
b2 = tf.Variable(tf.random_normal([10]), name='b2')

out2 = tf.add(tf.matmul(out1, w2), b2)
# out2 = tf.nn.relu(out2)

y_ = tf.nn.softmax(out2)
y_clipped = tf.clip_by_value(y_, 1e-10, 0.999999)

# cross entrophy: 1/m * Sigmoid(y * log(y_) + (1-y) * log(1-y_))

loss = - tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped) +\
  (1-y) * tf.log(1-y_clipped), axis=1))
# cross_entrophy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped)
#                          + (1 - y) * tf.log(1 - y_clipped), axis=1))

optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(loss)

# initialize
init_op = tf.global_variables_initializer()

# accuracy
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)

with tf.Session(config=config) as sess:
  sess.run(init_op)

  total_batch = int(len(mnist.train.labels) / batch_size)

  for epoch in range(epochs):
    avg_cost = 0
    for i in range(total_batch):
      batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)

      _, c = sess.run([optimizer, loss], feed_dict={x: batch_x, y: batch_y})
      
      avg_cost += c / total_batch
    print('Epoch: ', (epoch + 1), 'cost = {:.3f}'.format(avg_cost))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))





#%%
