import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import pickle
import os.path
import glob

### Set Layer Options ###

# Convolutional Layer 1.
filter_size1 = 3  # Convolution filters are 3 x 3 pixels.
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
'''
data = np.zeros((slice_size, slice_size, depth))
labels = [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]]
'''


### Helper functions ###

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))


def get_data(state):
    # path needs to be updated
    path = "/media/nilus/INTENSO/pokerData/"
    if (os.path.exists(path + state + "/save0.pickle")):
        files = (glob.glob(path + state + "/save*.pickle"))
        last_number = []
        for l in range(len(files)):
            last_number.append(int(files[l].split('.pic')[0].split('save')[-1]))
        last_number = max(last_number)
        data = []
        print("last_number: ", last_number)
        for i in range(last_number):
            with open(path + state + "/save" + str(last_number) + ".pickle", 'rb') as handle:
                data.append(pickle.load(handle))
    else:
        raise ValueError('Data could not be found')
    return data


def get_results(state):
    # path needs to be updated
    path = "/media/nilus/INTENSO/pokerData/"
    if (os.path.exists(path + state + "/result0.pickle")):
        files = (glob.glob(path + state + "/result*.pickle"))
        last_number = []
        for l in range(len(files)):
            last_number.append(int(files[l].split('.pic')[0].split('result')[-1]))
        last_number = max(last_number)
        print("last_number results: ", last_number)
        result = []
        for i in range(last_number):
            with open(path + state + "/result" + str(i) + ".pickle", 'rb') as handle:
                result.append(pickle.load(handle))
    else:
        raise ValueError('Data could not dbe found')
    return result


def get_batch(data, labels, size):
    batch = []
    batch_labels = []
    for t in range(size):
        no = np.random.randint(0, len(data))
        batch.append(data[no].flatten())
        batch_labels.append(labels[no])
    return batch, batch_labels


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
# test = tf.unpack(y_true)
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

### Cost function for backprop ####
# Calculate cross-entropy first
# cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                        # labels=y_true)
# change name
mean_sq_error = tf.squared_difference(layer_fc2, y_true)

# This yields the cost:
cost = tf.reduce_mean(mean_sq_error)

### Optimization using nesterov momentum
optimizer = tf.train.MomentumOptimizer(learning_rate=0.4, momentum=0.2,
                                   use_locking=False, name='momentum', use_nesterov=True).minimize(cost)

### Performace measures ###
def get_accuracy(output, label):
    correct_prediction = 0
    for i in range(len(output)):
        best_outputs = []
        best_original = []
        max_output = max(output[i])
        max_original = max(label[i])
        for s in range(len(output[i])):
            if output[i][s] == max_output:
                best_outputs.append(s)
            if label[i][s] == max_original:
                best_original.append(s)
        if len(best_outputs) == 1:
            if best_outputs[0] in best_original:
                correct_prediction += 1
        else:
            if best_outputs[:] in best_original:
                correct_prediction += 1
    accuracy = correct_prediction/len(label)
    return accuracy

### start the session ###

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    ### NOW WE NEED ACTUAL DATA AND STUFF - THIS IS SKIPPED FOR NOW
    ### FIRST TEST WITH TEST DATA

    # Needed here is: A loop, in which batches of train data + labels are retrieved (best by a helper script)
    # Format: [data1, data2, ... ] , [label1, label2, ...]. data: 3d array, label: 1d array length 10
    all_data = get_data("preflop")
    all_labels = get_results("preflop")
    data = all_data[0:int(len(all_data)*0.8)]
    labels = all_labels[0:int(len(all_labels)*0.8)]
    test_data = all_data[int(len(all_data)*0.8):len(all_data)]
    test_labels = all_labels[int(len(all_labels)*0.8):len(all_labels)]
    for l in range(50):
        data_batch, labels_batch = get_batch(data, labels, 20)
        # create a dummy feed dict. Works similarly when using a bigger dataset
        feed_dict_train = {x: data_batch, y_true: labels_batch}
        # run the network
        session.run(optimizer, feed_dict=feed_dict_train)
        # Calculate the accuracy on the training-set.
        cost_h = session.run(cost, feed_dict=feed_dict_train)
        # Message for printing
        if l % 10 == 0:
            output = session.run(layer_fc2, feed_dict=feed_dict_train)
            or_labels = session.run(y_true, feed_dict=feed_dict_train)
            print(output[0])
            print(or_labels[0])
            print("________________________________________")
            print(output[1])
            print(or_labels[1])
            acc = get_accuracy(output, or_labels)
            msg = "Optimization Iteration: {0:>6}, cost: {1:>6.4}, accuracy: {2:>2}"
            # Print it
            print(msg.format(l, cost_h, acc))
        else:
            msg = "Optimization Iteration: {0:>6}, cost: {1:>6.4}"
            # Print it
            print(msg.format(l, cost_h))

    data_test, labels_test = get_batch(test_data, test_labels, 100)
    feed_dict_test = {x:data_test, y_true:labels_test}
    or_labels = session.run(y_true, feed_dict=feed_dict_test)
    output = session.run(layer_fc2, feed_dict=feed_dict_test)
    test_accuracy = get_accuracy(output, or_labels)
    print ("Test accuracy: ", test_accuracy)