#import the dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("assignment3/", one_hot=True)
#import
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
#create a tensor (in this case 2D-array consisting of floats) none=consisting of any length
x = tf.placeholder(tf.float32, [None, 784])
#tf.variable = modifiable tensor
#create the weight variables
W = tf.Variable(tf.zeros([784, 10]))
#create the bias variables
b = tf.Variable(tf.zeros([10]))
#multiplies x with W (this way around because x is a tensor with multiple inputs) and add b
#then the softmax regression is applied
y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 10])
'''
First, tf.log computes the logarithm of each element of y. Next, we multiply each element of y_ with the corresponding
element of tf.log(y). Then tf.reduce_sum adds the elements in the second dimension of y, due to the
reduction_indices=[1] parameter. Finally, tf.reduce_mean computes the mean over all the examples in the batch.
'''
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
#backpropagation algorithm implementation
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
#initialize all variables, but don't run them yet
init = tf.initialize_all_variables()
#run the model
sess = tf.Session()
sess.run(init)
# a 1000 times
list_of_numbers = np.empty(1000)
list_of_accuracy = np.empty(1000)
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  list_of_numbers[i] = i
  list_of_accuracy[i] = sess.run(accuracy, feed_dict={x: mnist.validation.images, y_: mnist.validation.labels})
# check for truth value
final_correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
#cast array to float and calculate mean
final_accuracy = tf.reduce_mean(tf.cast(final_correct_prediction, tf.float32))
print(sess.run(final_accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
plt.plot(list_of_numbers, list_of_accuracy)
plt.show()
