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
num_filters1 = 16
# Convolutional Layer 2.
filter_size2 = 3  # Convolution filters are 3 x 3 pixels.
num_filters2 = 16  # There are 30 of these filters.
# Convolutional Layer 3.
filter_size3 = 3  # Convolution filters are 3 x 3 Pixels
num_filters3 = 32
# Convolutional Layer 4.
filter_size4 = 3
num_filters4 = 32

# Fully-connected layer.
fc_size = 512# Number of neurons in fully-connected layer.

### Define data dimensions ###

# Size for each slize
slice_size = 17

# Size for the depth
depth = 1

# (Results in 17x17x9 later)

# Flat slice
slice_flat = slice_size * slice_size

# Number of output results
num_classes = 2

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
    path = "/home/nilus/pokerData/"
    if (os.path.exists(path + state + "/save0.pickle")):
        files = (glob.glob(path + state + "/save*.pickle"))
        last_number = []
        for l in range(len(files)):
            last_number.append(int(files[l].split('.pic')[0].split('save')[-1]))
        last_number = max(last_number)
        data = []
        print("last_number: ", last_number)
        for i in range(last_number):
            with open(path + state + "/save" + str(i) + ".pickle", 'rb') as handle:
                or_data = pickle.load(handle)
                data.append(or_data[0])
    else:
        raise ValueError('Data could not be found')
    return data


def get_results(state):
    # path needs to be updated
    path = "/home/nilus/pokerData/"
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


def get_batch(data, results, size):
    batch = []
    batch_labels = []
    batch_results = []
    for t in range(size):
        no = np.random.randint(0, len(data))
        batch.append(data[no].flatten())
        or_results = results[no]
        changed_results = results[no][0:2]
        '''
        if or_results[0] > or_results[1]:
            changed_results[0] = 1
        else:
            changed_results[1] =1
        '''
        '''
        changed_results = [0] * len(or_results)
        index = or_results.index(max(or_results))
        changed_results[index] = 1
        '''
        batch_labels.append(changed_results)
        batch_results.append(or_results)
    return batch, batch_labels, batch_results


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
                 use_softmax=False):  # Use Rectified Linear Unit (ReLU)?

    # Create new weights and biases.
    # weights = new_weights(shape=[num_inputs, num_outputs])
    weights = tf.get_variable("W", shape=[num_inputs, num_outputs],
                              initializer=tf.contrib.layers.xavier_initializer())
    # biases = new_biases(length=num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(input, weights)

    # Use ReLU?
    if use_softmax:
        layer = tf.nn.softmax(layer)

    return layer


### Placeholders ###

# INPUT
x = tf.placeholder(tf.float32, shape=[None, slice_flat * depth], name='x')
# tensor has to be 5dim for conv3d
x_tensor = tf.reshape(x, [-1, slice_size, slice_size, depth, 1])

# OUTPUT
y_true = tf.placeholder(tf.float32, shape=[None, 2], name='y_true')
# test = tf.unpack(y_true)
# y_true_cls = tf.argmax(y_true, dimension=1)

### Create Layers###
# Conv 1

layer_conv1, weights_conv1 = new_conv_layer(input=x_tensor,
                                            num_input_channels=1,
                                            filter_size=filter_size1,
                                            num_filters=num_filters1,
                                            use_pooling=False)
# conv 2
layer_conv2, weights_conv2 = new_conv_layer(input=layer_conv1,
                                            num_input_channels=num_filters1,
                                            filter_size=filter_size2,
                                            num_filters=num_filters2,
                                            use_pooling=True)


# conv 3
layer_conv3, weights_conv3 = new_conv_layer(input=layer_conv2,
                                            num_input_channels=num_filters1,
                                            filter_size=filter_size2,
                                            num_filters=num_filters2,
                                            use_pooling=False)

# conv 4
layer_conv4, weights_conv4 = new_conv_layer(input=layer_conv1,
                                            num_input_channels=num_filters1,
                                            filter_size=filter_size2,
                                            num_filters=num_filters2,
                                            use_pooling=True)
'''
# Flatten the 4nd conv layer
layer_flat, num_features = flatten_layer(layer_conv4)

h_fc1_drop = tf.nn.dropout(layer_flat, 0.75)

# Output layer
layer_fc1 = new_fc_layer(input=h_fc1_drop,
                         num_inputs=num_features,
                         num_outputs=num_classes,
                         use_relu=False)
'''
layer_flat, num_features = flatten_layer(layer_conv4)
dense = tf.layers.dense(inputs=layer_flat, units=512, activation=tf.nn.relu)
h_fc1_drop = tf.nn.dropout(dense, 0.5)
layer_fc1 = new_fc_layer(h_fc1_drop,
                         num_inputs=512,
                         num_outputs=num_classes,
                         use_softmax=False
                         )




### Cost function for backprop ####
# Calculate cross-entropy first
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc1,
                                                        labels=y_true)

# This yields the cost:
# cost = tf.reduce_mean(cross_entropy)
#cost = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y_true, layer_fc1))))


### Optimization
# optimizer = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9,
                                   #use_locking=False, name='momentum', use_nesterov=True).minimize(cost)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cross_entropy)

### Performace measures ###

def get_win_loss(output, label):
    win_or_loss = 0
    for i in range(len(output)):
        ind_output = np.argmax(output[i])
        win_or_loss += label[i][ind_output]
    win_or_loss /= len(output)
    return win_or_loss

def adapt_data(or_data, or_results):
    label = []
    data = []
    for t in range(len(or_data)):
        data.append(or_data[t].flatten())
        changed_results = [0] * 2
        changed_results = or_results[t][0:2]
        '''
        if or_results[t][0] > or_results[t][1]:
            changed_results[0] = 1
        else:
            changed_results[1] = 1
        label.append(changed_results)
        '''
    return data, label

### create saver
saver = tf.train.Saver()
### start the session ###
with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    ### NOW WE NEED ACTUAL DATA AND STUFF - THIS IS SKIPPED FOR NOW
    ### FIRST TEST WITH TEST DATA
    # Needed here is: A loop, in which batches of train data + labels are retrieved (best by a helper script)
    # Format: [data1, data2, ... ] , [label1, label2, ...]. data: 3d array, label: 1d array length 10
    all_data = get_data("preflop")
    all_labels = get_results("preflop")
    data = all_data[0:int(len(all_data)*0.85)]
    labels = all_labels[0:int(len(all_labels)*0.85)]
    test_data = all_data[int(len(all_data)*0.85):len(all_data)]
    test_labels = all_labels[int(len(all_labels)*0.85):len(all_labels)]
    for counter in range(1000):
        data_batch, labels_batch, results_batch = get_batch(data, labels, 250)
        # create a dummy feed dict. Works similarly when using a bigger dataset
        feed_dict_train = {x: data_batch, y_true: labels_batch}
        # run the network
        session.run(optimizer, feed_dict=feed_dict_train)
        output = session.run(layer_fc1, feed_dict=feed_dict_train)
        print(output[0])
        # Calculate the accuracy on the training-set.
        acc = get_win_loss(output, results_batch)
        saver.save(session, "./simple-ffnn.ckpt")
        msg = "Optimization Iteration: {0:>6}, Win_loss: {1:>6.4}"
        # Print it
        print(msg.format(counter, acc))

    print("Validation started")
    data_test, label_test = adapt_data(test_data, test_labels)
    feed_dict_test = {x: data_test, y_true: label_test}
    saver.save(session, "./simple-ffnn.ckpt")
    output_val = session.run(layer_fc1, feed_dict=feed_dict_test)
    print("Accuracy: ", get_win_loss(output_val, test_labels))