#import
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#import the dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("assignment4/", one_hot=True)
#create a tensor (in this case 2D-array consisting of floats) none=consisting of any length
x = tf.placeholder(tf.float32, [None, 784])


# Apply convolution kernels to input x
def convolution(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME")


# Apply max pooling to outputs
def pooling(max_conv):
    return tf.nn.max_pool(max_conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


# Calculate neuron outputs by applying the activation function
def activation(outputbias):
    return tf.nn.tanh(outputbias)


# first layer of NN
# Initialize kernel weights and biases
initvar = tf.truncated_normal([5, 5, 1, 32], stddev=0.1)
weights1 = tf.Variable(initvar)
initvar = tf.constant(0.1, shape=[32])
bias1 = tf.Variable(initvar)
#reshape x to 4d tensor
x_image = tf.reshape(x, [-1, 28, 28, 1])
h_conv1 = tf.nn.relu(convolution(x_image, weights1))
act_h_conv1 = activation(h_conv1+bias1)
h_pool1 = pooling(act_h_conv1)

# second layer of NN
initvar = tf.truncated_normal([5, 5, 32, 64], stddev=0.1)
weights2 = tf.Variable(initvar)
initvar = tf.constant(0.1, shape=[64])
bias2 = tf.Variable(initvar)
h_conv2 = tf.nn.relu(convolution(h_pool1, weights2))
act_h_conv2 = activation(h_conv2+bias2)
h_pool2 = pooling(act_h_conv2)

# the neural network
initvar = tf.truncated_normal([7*7*64, 1024], stddev=0.1)
weights_nn = tf.Variable(initvar)
initvar = tf.constant(0.1, shape=[1024])
bias_nn = tf.Variable(initvar)

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, weights_nn) + bias_nn)
initvar = tf.truncated_normal([1024, 10])
weigths_finalLayer = tf.Variable(initvar)
initvar = tf.constant(0.1, shape=[10])
bias_finalLayer = tf.Variable(initvar)

y_conv = tf.matmul(h_fc1, weigths_finalLayer) + bias_finalLayer

y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
# train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess = tf.Session()
sess.run(tf.initialize_all_variables())
limit = 200
list_of_numbers = np.empty(limit)
list_of_accuracy = np.empty(limit)
for i in range(limit):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(session=sess, feed_dict={x: batch[0], y_: batch[1]})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(session=sess, feed_dict={x: batch[0], y_: batch[1]})
    list_of_numbers[i] = i
    list_of_accuracy[i] = accuracy.eval(session=sess, feed_dict={x: batch[0], y_: batch[1]})

plt.plot(list_of_numbers, list_of_accuracy)
plt.show()
print("test accuracy %g"%accuracy.eval(session=sess, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
