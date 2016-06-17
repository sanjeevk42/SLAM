import tensorflow as tf

import numpy as np
import network as net


# model parameters

lstm_layers = 3
lstm_cells_in_layer = 100


def model(X):

    return

x = tf.placeholder("float", [None, 28, 28])
y = tf.placeholder("float", [None, 6])

py_x, state_size, init_state = model(X)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)