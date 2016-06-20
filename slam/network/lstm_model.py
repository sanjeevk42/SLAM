import tensorflow as tf

"""
This constructs the graph of the LSTM model
"""
from tensorflow.python.ops.rnn_cell import LSTMCell, MultiRNNCell
class LSTMmodel:

    """
    TODO: Implement tf.saver or initialization from file
    """
    def __init__(self, model_input, layer_size, layers, output_dim, ground_truth):
        self.model_input = model_input
        self.layer_size = layer_size
        self.layers = layers
        self.output_dim = output_dim
        self.forget_bias = 1.0
        self.ground_truth = ground_truth
        self.init_state = self.__init_state()

    def build_graph(self):
        state = self.init_state
        with tf.variable_scope('main_lstm'):
            lstm_cell = LSTMCell(self.layer_size)
            rnn_cell = MultiRNNCell([lstm_cell] * self.layers)
            cell_output, state = rnn_cell(self.model_input, state)
            print("%i layers created" % self.layers)
            self.output_layer = self.__add_output_layer("fc_out", cell_output, self.layer_size, self.output_dim)
            return self.output_layer, rnn_cell.state_size, self.init_state


    def __init_state(self):
        return tf.zeros([1, 2 * self.layer_size * self.layers], tf.float32)


    def __add_output_layer(self, scope_name, layer_input, input_dim, output_dim):
        with tf.variable_scope(scope_name):
            fc_weights = tf.get_variable('weights', [input_dim, output_dim], tf.float32,
                                         initializer=tf.random_normal_initializer())
            fc_biases = tf.get_variable('biases', [output_dim], tf.float32,
                                        initializer=tf.random_normal_initializer())
            return tf.matmul(layer_input, fc_weights) + fc_biases
        
    
    def add_loss(self):
        self.cost = tf.reduce_sum(tf.pow(self.output_layer - self.ground_truth, 2))
        return self.cost
    
    def add_optimizer(self):
        learning_rate = 0.1  # need to make adaptive ...
        self.global_step = tf.Variable(0, trainable=False)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        gradients = optimizer.compute_gradients(self.cost)
        self.apply_gradient_op = optimizer.apply_gradients(gradients, self.global_step)
        return self.apply_gradient_op


