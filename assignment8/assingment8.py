import numpy as np
import random
import tensorflow as tf
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
# import data (already normalized) and combine datasets into one single set without the labels
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("", one_hot=True)
img_full = np.concatenate((mnist.test.images, mnist.train.images, mnist.validation.images), axis=0)
# generate the random input 
r_input = truncnorm.rvs(-1, 1, size=100)
z = tf.placeholder(tf.float32, [None, 100])
# show the first image
'''
image = np.reshape(img_full[0]*255, [28, 28])
plt.imshow(image)
plt.show()
'''

def getMiniBatch(size, arr):
    if (size > len(arr)):
        print("error not enough images left")
        return 0
    array = np.empty([size, arr.shape[1]], dtype=arr.dtype)
    for i in range(size):
        randomNo = random.randint(0, len(arr) - 1)
        print(randomNo)
        array[i] = arr[randomNo]
        arr = np.delete(arr, randomNo, axis=0)
    return array, arr

