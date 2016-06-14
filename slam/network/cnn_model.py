import numpy as np
from slam.network.model_input import ModelInputProvider
from slam.utils.logging_utils import get_logger
from slam.utils.time_utils import time_it
import tensorflow as tf


class VGG16Model:
    
    def __init__(self, image_shape, batch_size, output_dim):
        self.input_provider = ModelInputProvider(batch_size)
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.output_dim = 1000
        self.logger = get_logger()
        self.initial_params = np.load('../../resources/VGG_16_4ch.npy').item()
        self.initial_params = {key.encode('utf-8'):self.initial_params[key] for key in self.initial_params}
        self.logger.info('Weight keys:{}'.format(self.initial_params.keys()))
    """
     Builds the execution graph using VGG16-CNN architecture.
    """
    def build_graph(self):
        with tf.variable_scope('main'):
            # # TODO: Need to get the input and output placeholder tensors from input provider...
            conv_input = tf.placeholder(tf.float32, shape=[self.batch_size] + self.image_shape)
            conv_output = tf.placeholder(tf.float32, shape=[self.batch_size, self.output_dim])
            images, outparams_batch = conv_input, conv_output
            self.ground_truth = outparams_batch
            filter_size = [3, 3]
            conv1 = self.__add_weight_layer('conv1', images, 2, filter_size, 4, 64, init_weights=False)
            conv2 = self.__add_weight_layer('conv2', conv1, 2, filter_size, 64, 128)
            conv3 = self.__add_weight_layer('conv3', conv2, 2, filter_size, 128, 256)
            conv4 = self.__add_weight_layer('conv4', conv3, 3, filter_size, 256, 512)
            conv5 = self.__add_weight_layer('conv5', conv4, 3, filter_size, 512, 512)
            fc1 = self.__add_conv_layer('fc6-conv', conv5, [10, 10], 512, 4096, padding='VALID', init_weights=False)
            fc2 = self.__add_conv_layer('fc7-conv', fc1, [1, 1], 4096, 4096)
            fc3 = self.__add_conv_layer('fc8-conv', fc2, [1, 1], 4096, 1000)
            # no need of softmax layer as we are not classifying ....
            self.output_layer = fc3;
            self.__add_loss()
            self.__add_optimizer()
            
    def __add_weight_layer(self, scope_name, layer_input, conv_layers, filter_size,
                           input_channels, output_channels, init_weights=True):
        next_layer_input = layer_input
        with tf.variable_scope(scope_name):
            for i in xrange(conv_layers):
                inner_scope_name = '{}_{}'.format(scope_name, i + 1)
                next_layer_input = self.__add_conv_layer(inner_scope_name, next_layer_input, filter_size,
                                            input_channels, output_channels, init_weights=init_weights)
                input_channels = output_channels
            return tf.nn.max_pool(next_layer_input, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1],
                                   padding='VALID')
    
    def __add_conv_layer(self, scope_name, layer_input, filter_size, input_channels,
                         output_channels, padding='SAME', init_weights=True):
        with tf.variable_scope(scope_name):
            weights_shape = filter_size + [input_channels, output_channels]
            initial_weights, initial_bias = self.__get_init_params(scope_name, init_weights)
            self.logger.info('Weight shape:{} for scope:{}'.format(weights_shape, tf.get_variable_scope().name))
            conv_weights = tf.get_variable('initial_params', weights_shape, tf.float32,
                                            initializer=initial_weights)
            conv_biases = tf.get_variable('biases', [output_channels], tf.float32,
                                            initializer=initial_bias)
            conv = tf.nn.conv2d(layer_input, conv_weights,
                                    strides=[1 , 1 , 1, 1], padding=padding)
            return tf.nn.relu(tf.nn.bias_add(conv, conv_biases))
    

    def __get_init_params(self, scope_name, init_weights):
        if scope_name in self.initial_params and init_weights:
            initial_weights = tf.constant_initializer(self.initial_params[scope_name]['weights'])
            initial_bias = tf.constant_initializer(self.initial_params[scope_name]['biases'])
        else:
            self.logger.warn('No initial weights found for scope:{}. Initializing with random weights.'.format(scope_name))
            initial_weights = tf.random_normal_initializer()
            initial_bias = tf.random_normal_initializer()
        return initial_weights, initial_bias
    
    def __add_loss(self):
        self.cost = tf.reduce_sum(tf.pow(self.output_layer - self.ground_truth, 2))
    
    def __add_optimizer(self):
        learning_rate = 0.1  # need to make adaptive ...
        self.global_step = tf.Variable(0, trainable=False)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        gradients = optimizer.compute_gradients(self.cost)
        self.apply_gradient_op = optimizer.apply_gradients(gradients, self.global_step)
        
    """
    Start training the model.
    """
    @time_it
    def train_model(self, max_steps):
        session = tf.Session()
        session.run(tf.initialize_all_variables())
        for step in xrange(max_steps):
            self.logger.info('Executing step:{}'.format(step))
            session.run([self.apply_gradient_op, self.cost])
    
    def evaluate_model(self):
        pass
    
    def predict(self):
        pass

if __name__ == '__main__':
    batch_size = 10
    img_h = 320
    img_w = 320
    vgg_model = VGG16Model([img_h, img_w, 4], batch_size, 10)
    vgg_model.build_graph()
    vgg_model.train_model(max_steps=400)
    vgg_model.evaluate_model()
    
    
