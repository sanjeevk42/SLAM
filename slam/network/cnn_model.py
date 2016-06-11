from slam.network.model_input import ModelInputProvider
from slam.utils.logging_utils import get_logger
from slam.utils.time_utils import time_it
import tensorflow as tf


class VGG16Model:
    
    def __init__(self, image_shape, batch_size, output_dim):
        self.input_provider = ModelInputProvider(batch_size)
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.output_dim = output_dim
        self.logger = get_logger()

    """
     Builds the execution graph using VGG16-CNN architecture.
    """
    def build_graph(self):
        with tf.variable_scope('main'):
            images, outparams_batch = self.input_provider.get_training_batch()
            self.ground_truth = outparams_batch
            #conv_input = tf.placeholder(tf.float32, shape=[self.batch_size] + self.image_shape)
            filter_size = [3, 3]
            conv1 = self.__add_weight_layer('conv1', images, 2, filter_size, 4, 64)
            conv2 = self.__add_weight_layer('conv2', conv1, 2, filter_size, 64, 128)
            conv3 = self.__add_weight_layer('conv3', conv2, 2, filter_size, 128, 256)
            conv4 = self.__add_weight_layer('conv4', conv3, 3, filter_size, 256, 512)
            conv5 = self.__add_weight_layer('conv5', conv4, 3, filter_size, 512, 512)
            reshape = tf.reshape(conv5, [self.batch_size, -1])
            input_dim = reshape.get_shape()[1].value
            fc1 = self.__add_fc_layer('fc1', reshape, input_dim, 4096)
            fc2 = self.__add_fc_layer('fc2', fc1, 4096, 4096)
            fc3 = self.__add_fc_layer('fc3', fc2, 4096, self.output_dim)
            # no need of softmax layer as we are not classifying ....
            self.output_layer = fc3;
            self.__add_loss()
            self.__add_optimizer()
            
    def __add_weight_layer(self, scope_name, layer_input, conv_layers, filter_size,
                           input_channels, output_channels):
        next_layer_input = layer_input
        with tf.variable_scope(scope_name):
            for i in xrange(conv_layers):
                scope_name = 'convlayer' + str(i + 1)
                next_layer_input = self.__add_conv_layer(scope_name, layer_input, filter_size,
                                                         input_channels, output_channels)
            return tf.nn.max_pool(next_layer_input, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1],
                                   padding='SAME')
    
    def __add_conv_layer(self, scope_name, layer_input, filter_size, input_channels, output_channels):
        with tf.variable_scope(scope_name):
            weights_shape = filter_size + [input_channels, output_channels]
            self.logger.info('Weight shape:{} for scope:{}'.format(weights_shape, tf.get_variable_scope().name))
            conv_weights = tf.get_variable('weights', weights_shape, tf.float32,
                                            initializer=tf.random_normal_initializer())
            conv_biases = tf.get_variable('biases', [output_channels], tf.float32,
                                            initializer=tf.random_normal_initializer())
            conv = tf.nn.conv2d(layer_input, conv_weights,
                                    strides=[1 , 1 , 1, 1], padding='SAME')
            return tf.nn.relu(tf.nn.bias_add(conv, conv_biases))
    
    def __add_fc_layer(self, scope_name, layer_input, input_dim, output_dim):
        with tf.variable_scope(scope_name):
            fc_weights = tf.get_variable('weights', [input_dim, output_dim], tf.float32,
                                            initializer=tf.random_normal_initializer())
            fc_biases = tf.get_variable('biases', [output_dim], tf.float32,
                                            initializer=tf.random_normal_initializer())
            return tf.nn.relu(tf.matmul(layer_input, fc_weights) + fc_biases)
    
    def __add_loss(self):
        self.cost = tf.reduce_sum(tf.pow(self.output_layer - self.ground_truth, 2))
    
    def __add_optimizer(self):
        learning_rate = 0.1  # need to make adaptive ...
        self.global_step = tf.Variable(0, trainable=False)
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate).apply_gradients(self.cost, self.global_step)
        
    """
    Start training the model.
    """
    @time_it
    def train_model(self, max_steps):
        session = tf.Session()
        session.run(tf.initialize_all_variables())
        for step in xrange(max_steps):
            session.run([self.optimizer, self.cost])
    
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