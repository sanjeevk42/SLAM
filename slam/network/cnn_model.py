"""
This constructs the graph for VGG16 CNN model.
"""

from kaffe.tensorflow.network import Network

import numpy as np
from slam.network.model_config import get_config_provider
from slam.utils.logging_utils import get_logger
from slam.utils.time_utils import time_it
import tensorflow as tf


class VGG16Model:
    
    def __init__(self, batch_size, rgbd_input_batch, output_dim, normalization_epsilon):
        self.network_input = rgbd_input_batch
        self.output_dim = output_dim
        self.logger = get_logger()
        self.batch_size = batch_size
        self.total_weights = 0
        self.initial_params = np.load('resources/VGG_16_4ch.npy').item()
        self.initial_params = {key.encode('utf-8'):self.initial_params[key] for key in self.initial_params}
        self.logger.info('Weight keys:{}'.format(self.initial_params.keys()))
        self.epsilon = normalization_epsilon
    """
     Builds the execution graph using VGG16-CNN architecture and 
     returns output of the network of shape [self.batch_size, 1, 1, self.output_dim]
     
    """
    def build_graph(self):
        
        with tf.variable_scope('vgg16'):
            
            filter_size = [3, 3]
            input_channels = self.network_input.get_shape()[3].value
            with tf.variable_scope('conv1'):
                conv1_1 = self.add_conv_layer('conv1_1', self.network_input, filter_size, input_channels, 64, should_init_wb=False)
                conv1_2 = self.add_conv_layer('conv1_2', conv1_1, filter_size, 64, 64)
                conv1_out = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
            
            with tf.variable_scope('conv2'):
                conv2_1 = self.add_conv_layer('conv2_1', conv1_out, filter_size, 64, 128)
                conv2_2 = self.add_conv_layer('conv2_2', conv2_1, filter_size, 128, 128)
                conv2_out = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
            
            with tf.variable_scope('conv3'):
                conv3_1 = self.add_conv_layer('conv3_1', conv2_out, filter_size, 128, 256)
                conv3_2 = self.add_conv_layer('conv3_2', conv3_1, filter_size, 256, 256)
                conv3_out = tf.nn.max_pool(conv3_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
                
            with tf.variable_scope('conv4'):
                conv4_1 = self.add_conv_layer('conv4_1', conv3_out, filter_size, 256, 512)
                conv4_2 = self.add_conv_layer('conv4_2', conv4_1, filter_size, 512, 512)
                conv4_3 = self.add_conv_layer('conv4_3', conv4_2, filter_size, 512, 512)
                conv4_out = tf.nn.max_pool(conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
                
            with tf.variable_scope('conv5'):
                conv5_1 = self.add_conv_layer('conv5_1', conv4_out, filter_size, 512, 512)
                conv5_2 = self.add_conv_layer('conv5_2', conv5_1, filter_size, 512, 512)
                conv5_3 = self.add_conv_layer('conv5_3', conv5_2, filter_size, 512, 512)
                conv5_out = tf.nn.max_pool(conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
                
            
            conv_out = tf.reshape(conv5_out, [self.batch_size, 7 * 7 * 512])
            
            fc1 = self.add_fc_layer('fc6-conv', conv_out, 7 * 7 * 512, 4096)
            fc2 = self.add_fc_layer('fc7-conv', fc1, 4096, 4096)
            
            with tf.variable_scope('out_layer'):
                out_weights = self.__get_variable('weights', [4096, self.output_dim], tf.float32, initializer=tf.truncated_normal_initializer(stddev=5e-2))
                out_bias = self.__get_variable('bias', [self.output_dim], tf.float32, initializer=tf.truncated_normal_initializer(stddev=5e-2))
                
                self.network_output = tf.matmul(fc2, out_weights) + out_bias
                
            tf.histogram_summary('cnn_output', self.network_output)
            
            self.logger.info('Total weights: {}'.format(self.total_weights))
            
        return self.network_output
    
    """
    Adds a convolution layer on top of layer_input with filter equal to filter_size. 
    Returns the output tensor by applying RLU activations.
    is_conv_layer - The calculation of the momentum and mean is different
    """
    def add_conv_layer(self, scope_name, layer_input, filter_size, input_channels,
                         output_channels, padding='SAME', should_init_wb=True):
        with tf.variable_scope(scope_name):
            weights_shape = filter_size + [input_channels, output_channels]
            initial_weights, initial_bias = self.__get_init_params(scope_name, should_init_wb)
            self.total_weights += weights_shape[0] * weights_shape[1] * weights_shape[2] * weights_shape[3]
            self.logger.info('Weight shape:{} for scope:{}'.format(weights_shape, tf.get_variable_scope().name))
            conv_weights = self.__get_variable('weights', weights_shape, tf.float32,
                                            initializer=initial_weights)
            
            tf.scalar_summary(scope_name + '/weight_sparsity', tf.nn.zero_fraction(conv_weights))
            tf.histogram_summary(scope_name + '/weights', conv_weights)
            
            conv = tf.nn.conv2d(layer_input, conv_weights,
                                    strides=[1 , 1 , 1, 1], padding=padding)

            # add batch normalization layer
            batch_mean, batch_var = tf.nn.moments(conv, axes=[0, 1, 2], keep_dims=False)
            scale = tf.Variable(tf.ones([output_channels]))
            beta = tf.Variable(tf.zeros([output_channels]))
            layer_output = tf.nn.batch_normalization(conv, batch_mean, batch_var, beta, scale, self.epsilon)
                
#             conv_biases = self.__get_variable('biases', [output_channels], tf.float32,
#                                             initializer=initial_bias)
#             layer_output = tf.nn.bias_add(conv, conv_biases)
             
            layer_output = tf.nn.relu(layer_output)  
            
            return layer_output
    
    def add_fc_layer(self, scope_name, layer_input, input_dim, out_dim):
        with tf.variable_scope(scope_name):
            init_weights, init_biases = self.__get_init_params(scope_name, should_init_wb=True)
            fc_weights = self.__get_variable('weights', [input_dim, out_dim] , tf.float32, initializer=init_weights)
            fc_biases = self.__get_variable('bias', [out_dim], tf.float32, initializer=init_biases)

            fc_out = tf.nn.relu(tf.nn.bias_add(tf.matmul(layer_input, fc_weights), fc_biases))
        return fc_out
            
    """
    Reads the initial values of weights and biases. 
    """
    def __get_init_params(self, scope_name, should_init_wb):
        if scope_name in self.initial_params and should_init_wb:
            initial_weights = tf.constant_initializer(self.initial_params[scope_name]['weights'])
            initial_bias = tf.constant_initializer(self.initial_params[scope_name]['biases'])
        else:
            self.logger.warn('No initial weights found for scope:{}. Initializing with random weights.'.format(scope_name))
            initial_weights = tf.truncated_normal_initializer(stddev=5e-2)
            initial_bias = tf.truncated_normal_initializer(stddev=5e-2)
        return initial_weights, initial_bias
    
    def add_loss(self, loss_weight, ground_truth):
        self.loss = tf.nn.l2_loss(tf.matmul(self.network_output - ground_truth, loss_weight), 'l2loss')
        self.loss = tf.Print(self.loss, [self.network_output, ground_truth, self.loss],
                                     'Value of cnn output layer, ground_truth and loss', summarize=6)
        tf.scalar_summary('cnn_loss', self.loss)
        return self.loss
    
    def add_optimizer(self):
        self.global_step = tf.Variable(0, trainable=False)
        
        learning_rate = tf.train.exponential_decay(0.01, self.global_step, 5,
                                   0.1, staircase=True)
        
        optimizer = self.get_optimizer(learning_rate)
        gradients = optimizer.compute_gradients(self.loss)
        
        for grad, var in gradients:
            if grad is not None:
                tf.histogram_summary(var.op.name + '/gradients', grad)
        # gradients = tf.Print([gradients], [gradients], 'The value of gradients:')
        self.apply_gradient_op = optimizer.apply_gradients(gradients, self.global_step)
        return self.apply_gradient_op
        
    def get_optimizer(self, learning_rate):
        config_provider = get_config_provider()
        opt = config_provider.optimizer()
        if opt == 'AdamOptimizer':
            return tf.train.AdamOptimizer(learning_rate)
        elif opt == 'GradientDescentOptimizer':
            return tf.train.GradientDescentOptimizer(learning_rate)
        elif opt == 'RMSPropOptimizer':
            return tf.train.RMSPropOptimizer(learning_rate)
    
    """
    Start training the model.
    """
    @time_it
    def start_training(self, max_steps):
        session = tf.Session()
        session.run(tf.initialize_all_variables())
        for step in xrange(max_steps):
            self.logger.info('Executing step:{}'.format(step))
            session.run([self.apply_gradient_op, self.loss])
    
    def __get_variable(self, name, shape, dtype, initializer):
        with tf.device('/cpu:0'):
            var = tf.get_variable(name, shape, dtype, initializer=initializer)
        return var


class VGG16(Network):
    
    def setup(self):
        (self.feed('input')
             .conv(3, 3, 64, 1, 1, name='conv1_1_4ch')
             .conv(3, 3, 64, 1, 1, name='conv1_2')
             .max_pool(2, 2, 2, 2, name='pool1')
             .conv(3, 3, 128, 1, 1, name='conv2_1')
             .conv(3, 3, 128, 1, 1, name='conv2_2')
             .max_pool(2, 2, 2, 2, name='pool2')
             .conv(3, 3, 256, 1, 1, name='conv3_1')
             .conv(3, 3, 256, 1, 1, name='conv3_2')
             .conv(3, 3, 256, 1, 1, name='conv3_3')
             .max_pool(2, 2, 2, 2, name='pool3')
             .conv(3, 3, 512, 1, 1, name='conv4_1')
             .conv(3, 3, 512, 1, 1, name='conv4_2')
             .conv(3, 3, 512, 1, 1, name='conv4_3')
             .max_pool(2, 2, 2, 2, name='pool4')
             .conv(3, 3, 512, 1, 1, name='conv5_1')
             .conv(3, 3, 512, 1, 1, name='conv5_2')
             .conv(3, 3, 512, 1, 1, name='conv5_3')
             .max_pool(2, 2, 2, 2, name='pool5')
             .conv(7, 7, 4096, 1, 1, padding='VALID', name='fc6-conv')
             .conv(1, 1, 4096, 1, 1, name='fc7-conv')
             .conv(1, 1, 1000, 1, 1, relu=False, name='fc8-conv')
             .softmax(name='prob'))
