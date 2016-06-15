import tensorflow as tf

class LSTMmodel:

    """
    TODO: Implement tf.saver or initialization from file
    """
    def __init__(self, batch_size, input_shape):
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.layer_size = 100
        self.layers = 2
        self.output_dim = 10
        self.forget_bias = 1.0
        self.init_state = self.__init_state()

    def build_graph(self):
        with tf.variable_scope('main_lstm'):
            if self.layers == 1:
                lstm_layer = tf.nn.rnn_cell.BasicLSTMCell(self.layer_size, self.forget_bias)
                lstm_output, states = tf.nn.rnn(lstm_layer, self.input_shape, initial_state=self.init_state)
                print("%i layer created"%self.layers)
            else:
                layer = tf.nn.rnn_cell.BasicLSTMCell(self.layer_size)
                lstm_layer = tf.nn.rnn_cell.MultiRNNCell([layer] * self.layers)
                lstm_output, state = tf.nn.rnn(lstm_layer, self.input_shape, initial_state=self.init_state)
                print("%i layers created" % self.layers)
            output_layer = self.__add_output_layer("fc_out", lstm_output, self.layer_size, self.output_dim)
            return output_layer, lstm_layer.state_size, self.init_state


    def __init_state(self):
        return(tf.placeholder("float", [None, 2 * self.layer_size * self.layers]))


    def __add_fc_layer(self, scope_name, layer_input, input_dim, output_dim):
        with tf.variable_scope(scope_name):
            fc_weights = tf.get_variable('weights', [input_dim, output_dim], tf.float32,
                                         initializer=tf.random_normal_initializer())
            fc_biases = tf.get_variable('biases', [output_dim], tf.float32,
                                        initializer=tf.random_normal_initializer())
            return tf.nn.relu(tf.matmul(layer_input, fc_weights) + fc_biases)


    def __add_output_layer(self, scope_name, layer_input, input_dim, output_dim):
        with tf.variable_scope(scope_name):
            fc_weights = tf.get_variable('weights', [input_dim, output_dim], tf.float32,
                                         initializer=tf.random_normal_initializer())
            fc_biases = tf.get_variable('biases', [output_dim], tf.float32,
                                        initializer=tf.random_normal_initializer())
            return tf.matmul(layer_input[-1], fc_weights) + fc_biases

