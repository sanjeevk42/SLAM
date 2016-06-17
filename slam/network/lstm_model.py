import tensorflow as tf

"""
This constructs the graph of the LSTM model
"""
class LSTMmodel:

    """
    TODO: Implement tf.saver or initialization from file
    """
    def __init__(self, input, time_step_size, input_vector_size, layer_size, number_of_layers, output_length):
        self.input_shape = self.__prepare_input(input, time_step_size, input_vector_size)
        self.layer_size = layer_size
        self.layers = number_of_layers
        self.output_dim = output_length
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


    """
    Functions to create the correct input tensors
    """
    #def __prepare_input(self, scope_name, input_vector_size, time_step_size):
    def __prepare_input(self, input, time_step_size, input_vector_size):
        #with tf.variable_scope(scope_name):
            # X, input shape: (batch_size, input_vec_size, time_step_size)
            #x = tf.placeholder("float", [None, time_step_size, input_vector_size])

            # XT shape: (input_vec_size, batch_size, time_step_size)
            xt = tf.transpose(input, [1, 0, 2])  # permute time_step_size and batch_size
            # XR shape: (input vec_size, batch_size)
            xr = tf.reshape(xt, [-1, input_vector_size])  # each row has input for each lstm cell (lstm_size)
            # Each array shape: (batch_size, input_vec_size)
            x_split = tf.split(0, time_step_size, xr)  # split them to time_step_size (28 arrays)

            return x_split

"""
    def __prepare_image_input(scope_name, pixels_in_y, pixels_in_x, dim_as_time):
        '''

        :param scope_name: this parameter is self explaining
        :param pixels_in_y: number of pixels in vertical direction
        :param pixels_in_x: number of pixels in horizontal direction
        :param dim_as_time: 0: feeds a row at each time step, 1: feeds a column at each time step
        :return: a tensor of the correct input shape
        '''

        if(dim_as_time == 0):
            return self.____prepare_input(scope_name, pixels_in_x, pixels_in_y)

        if (dim_as_time == 1):
            return prepare_input(scope_name, pixels_in_y, pixels_in_x)
"""