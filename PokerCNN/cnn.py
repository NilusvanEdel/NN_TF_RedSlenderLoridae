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

### Set the state
# possibel states: river, preflop, flop,.
state = "river"
### Set Layer Options ###

# Convolutional Layer 1.
filter_size1 = 3
num_filters1 = 50
# Convolutional Layer 2.
filter_size2 = 3
num_filters2 = 50
# Convolutional Layer 3.
filter_size3 = 3
num_filters3 = 32
# Convolutional Layer 4.
filter_size4 = 3
num_filters4 = 32

# Fully-connected layer.
fc_size = 512

### Define data dimensions ###

# Size for each slize
slice_size = 17
# Size for the depth
depth = 9
# (Results in 17x17x9 later)

# Flat slice
slice_flat = slice_size * slice_size
# Number of output results
num_classes = 2

### Helper functions ###

# create new weights
def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

# create new biases
def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

# get data
def get_data(state):
    # path needs to be updated
    path = "pokerData/"
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
                data.append(or_data)
    else:
        raise ValueError('Data could not be found')
    return data

# get labels
def get_results(state):
    # path needs to be updated
    path = "pokerData/"
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

# create a data batch to train/ test
def get_batch(data, results, size):
    batch = []
    batch_labels = []
    batch_results = []
    for t in range(size):
        no = np.random.randint(0, len(data))
        or_results = results[no][0:2]
        changed_results = [0] * 2
        # changed_results = results[no][0:2]
        if or_results[0] >= or_results[1]:
            changed_results[0] = 1
        else:
            changed_results[1] = 1
        '''
        changed_results = [0] * len(or_results)
        index = or_results.index(max(or_results))
        changed_results[index] = 1
        '''
        batch.append(data[no].flatten())
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

    # Add bias
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
    '''
    weights = tf.get_variable("W", shape=[num_inputs, num_outputs],
                                  initializer=tf.contrib.layers.xavier_initializer())
                                  '''

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.

    # Use ReLU?
    if use_relu:
        layer = tf.layers.dense(input, num_outputs, activation=tf.nn.relu)
    else:
        layer = tf.layers.dense(input, num_outputs)

    return layer

### Placeholders ###

# INPUT
x = tf.placeholder(tf.float32, shape=[None, slice_flat * depth], name='x')
# tensor has to be 5dim for conv3d
x_tensor = tf.reshape(x, [-1, slice_size, slice_size, depth, 1])

# OUTPUT
y_true = tf.placeholder(tf.float32, shape=[None, 2], name='y_true')

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

# Flatten the 4th conv layer
layer_flat, num_features = flatten_layer(layer_conv4)

#create dense layer
dense = new_fc_layer(input=layer_flat,
                     num_inputs=num_features,
                     num_outputs=fc_size,
                     use_relu=True)
h_fc2_drop = tf.nn.dropout(dense, 0.5)
dense2 = new_fc_layer(input=h_fc2_drop,
                     num_inputs=num_features,
                     num_outputs=fc_size,
                     use_relu=True)

layer_fc1 = new_fc_layer(dense2,
                         num_inputs=512,
                         num_outputs=num_classes,
                         use_relu=False)

output_layer = tf.nn.softmax(layer_fc1)


### Cost function for backprop ####
# MSE
cost = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y_true, layer_fc1))))


### Optimization
optimizer = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9,
                                   use_locking=False, name='momentum', use_nesterov=True).minimize(cost)

### Performace measures ###
def get_win_loss(output, label):
    win_or_loss = 0
    for i in range(len(output)):
        ind_output = np.argmax(output[i])
        if ind_output == 0: #todo remove later
            win_or_loss += 0.5 #remove later
        else:
            if label[i][ind_output] > 0.5:
                win_or_loss += 1
            else:
                win_or_loss += 0
    win_or_loss /= len(output)
    return win_or_loss

def adapt_data(or_data, or_results):
    label = []
    data = []
    for t in range(len(or_data)):
        data.append(or_data[t].flatten())
        changed_results = [0] * 2
        #changed_results = or_results[t][0:2]
        if or_results[t][0] > or_results[t][1]:
            changed_results[0] = 1
        else:
            changed_results[1] = 1
        label.append(changed_results)
    return data, label


### create saver
saver = tf.train.Saver()


### start the session ###
with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    ### GET DATA ###
    all_data = get_data(state)
    all_labels = get_results(state)
    data = all_data[0:int(len(all_data)*0.85)]
    labels = all_labels[0:int(len(all_labels)*0.85)]
    test_data = all_data[int(len(all_data)*0.85):len(all_data)]
    test_labels = all_labels[int(len(all_labels)*0.85):len(all_labels)]

    ### Run batches for training ###
    batches = 1000
    for counter in range(batches):
        data_batch, labels_batch, results_batch = get_batch(data, labels, 250)
        # create the training feed dict
        feed_dict_train = {x: data_batch, y_true: labels_batch}
        # run the network
        session.run(optimizer, feed_dict=feed_dict_train)
        output = session.run(output_layer, feed_dict=feed_dict_train)
        print(output[0])
        # Calculate the accuracy on the training-set.
        acc = get_win_loss(output, results_batch)
        saver.save(session, "./simple-ffnn.ckpt")
        msg = "Optimization Iteration: {0:>6}, Win_loss: {1:>6.4}"
        # Print it
        print("best win_loss: ", get_win_loss(labels_batch, results_batch))
        print(msg.format(counter, acc))

    ### Run validation ###
    print("Validation started")
    data_test, label_test = adapt_data(test_data, test_labels)
    feed_dict_test = {x: data_test, y_true: label_test}
    saver.save(session, "./simple-ffnn.ckpt")
    output_val = session.run(layer_fc1, feed_dict=feed_dict_test)
    print("Accuracy: ", get_win_loss(output_val, test_labels))