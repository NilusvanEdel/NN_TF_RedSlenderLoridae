import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math

### Set Layer Options ###

# Convolutional Layer 1.
filter_size1 = 5  # Convolution filters are 5 x 5 pixels.
num_filters1 = 30  # There are 30 of these filters.

# Convolutional Layer 2.
filter_size2 = 5  # Convolution filters are 5 x 5 pixels.
num_filters2 = 50  # There are 50 of these filters.

# Fully-connected layer.
fc_size = 128  # Number of neurons in fully-connected layer.

### Define data dimensions ###

# Size for each slize
slice_size = 17

# Size for the depth
depth = 9

# (Results in 17x17x9 later)

# Flat slice
slice_flat = slice_size * slice_size

# Number of output results
num_classes = 10

### Load Data ###

# WIP #
# Test data:
data = np.zeros((slice_size, slice_size, depth))
labels = [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]]


### Helper functions ###

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))


### Helper to create new conv layer ###

def new_conv_layer(input,  # The previous layer.
                   num_input_channels,  # Num. channels in prev. layer.
                   filter_size,  # Width, height and depth of each filter.
                   num_filters,  # Number of filters.
                   use_pooling=False):  # Use 2x2 max-pooling.

    # Shape of the filter-weights for the convolution.
    # This format is determined by the TensorFlow API.
    shape = [filter_size, filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights aka. filters with the given shape.
    weights = new_weights(shape=shape)

    # Create new biases, one for each filter.
    biases = new_biases(length=num_filters)

    # create the actual layer
    layer = tf.nn.conv3d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1, 1],
                         padding='SAME')

    # Ass bias
    layer += biases

    # Maybe ususuable right now, had been used for 2d. Thus by default false
    if use_pooling:
        # This is 2x2 max-pooling, which means that we
        # consider 2x2 windows and select the largest value
        # in each window. Then we move 2 pixels to the next window.
        layer = tf.nn.max_pool3d(input=layer,
                                 ksize=[1, 2, 2, 2, 1],
                                 strides=[1, 2, 2, 2, 1],
                                 padding='SAME')

    #Use a RELU at the end
    layer = tf.nn.relu(layer)


    # Return the new layer and the weights
    return layer, weights


### Helper to flatten the last conv layer and send it in a fc layer ###

def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    # The number of features is: height * width * depth * num_channels
    # We can use a function from TensorFlow to calculate this.
    num_features = layer_shape[1:5].num_elements()

    # flatten the layer according to num_fgeatures
    layer_flat = tf.reshape(layer, [-1, num_features])


    # Return both the flattened layer and the number of features.
    return layer_flat, num_features


### Helper for the fc layer ###

def new_fc_layer(input,  # The previous layer.
                 num_inputs,  # Num. inputs from prev. layer.
                 num_outputs,  # Num. outputs.
                 use_relu=True):  # Use Rectified Linear Unit (ReLU)?

    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(input, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer


### Placeholders ###

# INPUT
x = tf.placeholder(tf.float32, shape=[None, slice_flat * depth], name='x')
# tensor has to be 5dim for conv3d
x_tensor = tf.reshape(x, [-1, slice_size, slice_size, depth, 1])

# OUTPUT
y_true = tf.placeholder(tf.float32, shape=[None, 10], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

### Create Layers###
# Conv 1

layer_conv1, weights_conv1 = new_conv_layer(input=x_tensor,
                                            num_input_channels=1,
                                            filter_size=filter_size1,
                                            num_filters=num_filters1,
                                            use_pooling=True)
# conv 2
layer_conv2, weights_conv2 = new_conv_layer(input=layer_conv1,
                                            num_input_channels=num_filters1,
                                            filter_size=filter_size2,
                                            num_filters=num_filters2,
                                            use_pooling=True)

# Flatten the 2nd conv layer
layer_flat, num_features = flatten_layer(layer_conv2)

# Add a fclayer
layer_fc1 = new_fc_layer(input=layer_flat,
                         num_inputs=num_features,
                         num_outputs=fc_size,
                         use_relu=True)

# Output layer

layer_fc2 = new_fc_layer(input=layer_fc1,
                         num_inputs=fc_size,
                         num_outputs=num_classes,
                         use_relu=False)

# use softmax to normalize
y_pred = tf.nn.softmax(layer_fc2)
y_pred_cls = tf.argmax(y_pred, dimension=1)

### Cost function for backprop ####
# Calculate cross-entropy first
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                        labels=y_true)
# This yields the cost:
cost = tf.reduce_mean(cross_entropy)

### Optimization using adamoptimizer
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

### Performace measures ###
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

### start the session ###

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    ### NOW WE NEED ACTUAL DATA AND STUFF - THIS IS SKIPPED FOR NOW
    ### FIRST TEST WITH TEST DATA

    # Needed here is: A loop, in which batches of train data + labels are retrieved (best by a helper script)
    # Format: [data1, data2, ... ] , [label1, label2, ...]. data: 3d array, label: 1d array length 10

    # create a dummy feed dict. Works similarly when using a bigger dataset
    feed_dict_train = {x: [data.flatten()], y_true: labels}
    # run the network
    session.run(optimizer, feed_dict=feed_dict_train)
    # Calculate the accuracy on the training-set.
    acc = session.run(accuracy, feed_dict=feed_dict_train)

    # Message for printing
    msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"

    # Print it
    print(msg.format(1, acc))
