"""
This constructs the graph of the LSTM model
"""

from tensorflow.python.ops.rnn_cell import LSTMCell, MultiRNNCell

from slam.network.model_input import get_queued_input_provider
from slam.network.summary_helper import add_activation_summary, \
    add_loss_summaries
import tensorflow as tf


class LSTMmodel:

    """
    TODO: Implement tf.saver or initialization from file
    """
    def __init__(self, model_input, layer_size, layers, output_dim, ground_truth, batch_size):
        self.model_input = model_input
        self.batch_size = batch_size
        self.layer_size = layer_size
        self.layers = layers
        self.output_dim = output_dim
        self.forget_bias = 1.0
        self.ground_truth = ground_truth
        self.init_state = self.__init_state()

    def build_graph(self):
        with tf.variable_scope('lstm'):
            lstm_cell = LSTMCell(self.layer_size)
            rnn_cell = MultiRNNCell([lstm_cell] * self.layers)
            cell_output, self.init_state = rnn_cell(self.model_input, self.init_state)
            print("%i layers created" % self.layers)
            self.output_layer = self.__add_output_layer("fc_out", cell_output, self.layer_size, self.output_dim)
            
            add_activation_summary(self.output_layer)
            self.output_layer = tf.Print(self.output_layer, [self.output_layer, tf.convert_to_tensor(self.ground_truth)],
                                          'Value of output layer and ground truth:', summarize=6)
            return self.output_layer, rnn_cell.state_size, self.init_state

    def __init_state(self):
        return tf.zeros([self.batch_size, 2 * self.layer_size * self.layers], tf.float32)
        


    def __add_output_layer(self, scope_name, layer_input, input_dim, output_dim):
        with tf.variable_scope(scope_name):
            fc_weights = tf.get_variable('weights', [input_dim, output_dim], tf.float32,
                                         initializer=tf.random_normal_initializer())
            fc_biases = tf.get_variable('biases', [output_dim], tf.float32,
                                        initializer=tf.random_normal_initializer())
            return tf.matmul(layer_input, fc_weights) + fc_biases
        
    
    def add_loss(self, loss_weight):
        self.loss = tf.reduce_sum(tf.pow(tf.matmul(self.output_layer - self.ground_truth, loss_weight), 2))
        tf.scalar_summary('loss', self.loss)
        add_activation_summary(self.output_layer)
#         tf.add_to_collection('losses', self.loss)
        return self.loss
    
    def add_optimizer(self):
        self.global_step = tf.Variable(0, trainable=False)

        learning_rate = tf.train.exponential_decay(0.01, self.global_step, 50,
                                   0.1, staircase=True)
        
#         loss_averages_op = add_loss_summaries(self.loss)
#         with tf.control_dependencies([loss_averages_op]):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        gradients = optimizer.compute_gradients(self.loss)
        
        self.apply_gradient_op = optimizer.apply_gradients(gradients, self.global_step)
        
        for var in tf.trainable_variables():
            tf.histogram_summary(var.op.name, var)

        for grad, var in gradients:
            if grad is not None:
                tf.histogram_summary(var.op.name + '/gradients', grad)
          
        return self.apply_gradient_op


