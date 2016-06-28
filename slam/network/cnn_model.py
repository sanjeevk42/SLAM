"""
This constructs the graph for VGG16 CNN model.
"""

import numpy as np
from slam.network.model_input import get_queued_input_provider, \
    get_simple_input_provider
from slam.utils.logging_utils import get_logger
from slam.utils.time_utils import time_it
import tensorflow as tf
from slam.network.model_config import get_config_provider


class VGG16Model:
    
    def __init__(self, batch_size, rgbd_input_batch, output_dim):
        self.input_provider = get_queued_input_provider()
        self.rgbd_input_batch = rgbd_input_batch
        self.output_dim = output_dim
        self.logger = get_logger()
        self.batch_size = batch_size
        self.total_weights = 0
        self.initial_params = np.load('../resources/VGG_16_4ch.npy').item()
        self.initial_params = {key.encode('utf-8'):self.initial_params[key] for key in self.initial_params}
        self.logger.info('Weight keys:{}'.format(self.initial_params.keys()))
    """
     Builds the execution graph using VGG16-CNN architecture and 
     returns output of the network of shape [self.batch_size, 1, 1, self.output_dim]
     
    """
    def build_graph(self):
        with tf.variable_scope('vgg16'):
            
            filter_size = [3, 3]
            conv1 = self.__add_weight_layer('conv1', self.rgbd_input_batch, 2, filter_size, 4, 64, should_init_wb=False)
            conv2 = self.__add_weight_layer('conv2', conv1, 2, filter_size, 64, 128)
            conv3 = self.__add_weight_layer('conv3', conv2, 2, filter_size, 128, 256)
            conv4 = self.__add_weight_layer('conv4', conv3, 3, filter_size, 256, 512)
            conv5 = self.__add_weight_layer('conv5', conv4, 3, filter_size, 512, 512)
            fc1 = self.__add_conv_layer('fc6-conv', conv5, [7, 7], 512, 4096, padding='VALID')
            fc2 = self.__add_conv_layer('fc7-conv', fc1, [1, 1], 4096, 4096)
            fc3 = self.__add_conv_layer('fc8-conv', fc2, [1, 1], 4096, self.output_dim, should_init_wb=False)
            
            self.output_layer = tf.squeeze(fc3, squeeze_dims=[1 , 2])
            
            tf.histogram_summary('cnn_output', self.output_layer)
            self.logger.info('Total weights: {}'.format(self.total_weights))
            
            return self.output_layer
    
    """
    Adds a weight layer to the network in scope_name with layer_input tensor as layer input. 
    conv_layer - specifies no. of convolution layers to be stacked back to back. A max pool layer is added at the end. 
    filter_size - filter size to be used for each convolution layer.
    input_channels - number of channels(feature maps) in input tensor.
    output_channels - number of output channels from weight layer.
    should_init_wb - whether to initialize weights and biases or not.
    
    returns the tensor with created layer with specified config.
    
    """
    def __add_weight_layer(self, scope_name, layer_input, conv_layers, filter_size,
                           input_channels, output_channels, should_init_wb=True):
        next_layer_input = layer_input
        with tf.variable_scope(scope_name):
            for i in xrange(conv_layers):
                inner_scope_name = '{}_{}'.format(scope_name, i + 1)
                next_layer_input = self.__add_conv_layer(inner_scope_name, next_layer_input, filter_size,
                                            input_channels, output_channels, should_init_wb=should_init_wb)
                input_channels = output_channels
            return tf.nn.max_pool(next_layer_input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                   padding='VALID')
    
    """
    Adds a convolution layer on top of layer_input with filter equal to filter_size. 
    Returns the output tensor by applying RLU activations.
    """
    def __add_conv_layer(self, scope_name, layer_input, filter_size, input_channels,
                         output_channels, padding='SAME', should_init_wb=True):
        with tf.variable_scope(scope_name):
            weights_shape = filter_size + [input_channels, output_channels]
            initial_weights, initial_bias = self.__get_init_params(scope_name, should_init_wb)
            self.total_weights += weights_shape[0] * weights_shape[1] * weights_shape[2] * weights_shape[3]
            self.logger.info('Weight shape:{} for scope:{}'.format(weights_shape, tf.get_variable_scope().name))
            conv_weights = self.__get_variable('initial_params', weights_shape, tf.float32,
                                            initializer=initial_weights)
            conv_biases = self.__get_variable('biases', [output_channels], tf.float32,
                                            initializer=initial_bias)
            conv = tf.nn.conv2d(layer_input, conv_weights,
                                    strides=[1 , 1 , 1, 1], padding=padding)
            return tf.nn.relu(tf.nn.bias_add(conv, conv_biases))
    
    """
    Reads the initial values of weights and biases. 
    """
    def __get_init_params(self, scope_name, should_init_wb):
        if scope_name in self.initial_params and should_init_wb:
            initial_weights = tf.constant_initializer(self.initial_params[scope_name]['weights'])
            initial_bias = tf.constant_initializer(self.initial_params[scope_name]['biases'])
        else:
            self.logger.warn('No initial weights found for scope:{}. Initializing with random weights.'.format(scope_name))
            initial_weights = tf.random_normal_initializer()
            initial_bias = tf.random_normal_initializer()
        return initial_weights, initial_bias
    
    def add_loss(self, loss_weight, ground_truth):
        self.loss = tf.reduce_sum(tf.pow(tf.matmul(self.output_layer - ground_truth, loss_weight), 2)) / self.batch_size
        tf.scalar_summary('cnn_loss', self.loss)
        return self.loss
    
    def add_optimizer(self):
        self.global_step = tf.Variable(0, trainable=False)
        
        learning_rate = tf.train.exponential_decay(0.01, self.global_step, 5,
                                   0.1, staircase=True)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        gradients = optimizer.compute_gradients(self.loss)
        self.apply_gradient_op = optimizer.apply_gradients(gradients, self.global_step)
        return self.apply_gradient_op
        
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

if __name__ == '__main__':
    img_h = 224
    img_w = 224
    LOG_DIR = '/usr/prakt/s085/logs' 
    
    logger = get_logger()    
    config_provider = get_config_provider()
    epoch = config_provider.epoch()
    batch_size = config_provider.batch_size()
    sequence_length = config_provider.sequence_length()
    
    rgbd_input_batch = tf.placeholder(tf.float32, [batch_size, img_h, img_w, 4])
    groundtruth_batch = tf.placeholder(tf.float32, [batch_size, 6])
    
    vgg_model = VGG16Model(batch_size, rgbd_input_batch, 6)
    vgg_model.build_graph()
    
    loss_weight = tf.placeholder(tf.float32, [6, 6])
    loss = vgg_model.add_loss(loss_weight, groundtruth_batch)
    apply_gradient_op = vgg_model.add_optimizer()
    
    input_provider = get_simple_input_provider()
    session = tf.Session()
    session.run(tf.initialize_all_variables())
    
    merged_summary = tf.merge_all_summaries()
    summary_writer = tf.train.SummaryWriter(LOG_DIR, session.graph)
    
    for step in xrange(epoch):
        logger.info('Executing step:{}'.format(step))
        next_batch = input_provider.sequence_batch_itr(sequence_length, batch_size)
        for i, sequence_batch in enumerate(next_batch):
            logger.info('Step:{} Frame:{}'.format(step, i))
            logger.debug('Using rgb files:{}, depth files:{}, groundtruths:{} in current batch'.format(sequence_batch.rgb_filenames,
                                                                         sequence_batch.depth_filenames, sequence_batch.groundtruths))
            loss_weight_matrix = np.zeros([6, 6]) if i == 0 else np.identity(6)
            result = session.run([apply_gradient_op, loss, merged_summary], feed_dict={rgbd_input_batch:sequence_batch.rgbd_images,
                                        groundtruth_batch:sequence_batch.groundtruths, loss_weight:loss_weight_matrix})
            loss_value = result[1]
            logger.info('Loss :{}'.format(loss_value))
        summary_writer.add_summary(result[2], step)
        
    session.close()

