import numpy as np
import matplotlib.pyplot as plt
from mpltoolkits.mplot3d import Axed3D
import tensorflow as tf


def activationfunction ( inputfloat ):

    solution = 1.7159*np.tanh((2/3)*inputfloat)
    return solution

sampleSize = 30

np.random.seed(1)
cats = np.random.normal(25, 5, sampleSize)
dogs = np.random.normal(45, 15, sampleSize)
x = tf.placeholder(tf.float32, 2)
W = tf.Variable(tf.zeros([2, 2]))







