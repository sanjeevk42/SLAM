import tensorflow as tf

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
#NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
WEIGHT_DECAY_REGULARIZATION = 5*10e-5
#LEARNING_RATE_DECAY_FACTOR = 0.1  #Learning rate decay factor.
#INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var

def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.
  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.
  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
  Returns:
    Variable Tensor
  """
  var = _variable_on_cpu(name, shape,
                         tf.truncated_normal_initializer(stddev=stddev))
  if wd is not None:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

def inference(images):
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().

  # ---------------------- CONV1 ---------------------- #
  with tf.variable_scope('conv1_1') as scope:
    kernel = _variable_with_weight_decay('weights', shape=[3, 3, 4, 64], stddev=1e-4, wd=WEIGHT_DECAY_REGULARIZATION)
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv1_1 = tf.nn.relu(bias, name=scope.name)
    #_activation_summary(conv1)

  with tf.variable_scope('conv1_2') as scope:
    kernel = _variable_with_weight_decay('weights', shape=[3, 3, 64, 64], stddev=1e-4, wd=WEIGHT_DECAY_REGULARIZATION)
    conv = tf.nn.conv2d(conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv1_2 = tf.nn.relu(bias, name=scope.name)

  pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

  # ---------------------- CONV2 ---------------------- #
  with tf.variable_scope('conv2_1') as scope:
    kernel = _variable_with_weight_decay('weights', shape=[3, 3, 32, 128], stddev=1e-4, wd=WEIGHT_DECAY_REGULARIZATION)
    conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv2_1 = tf.nn.relu(bias, name=scope.name)

  with tf.variable_scope('conv2_2') as scope:
    kernel = _variable_with_weight_decay('weights', shape=[3, 3, 128, 128], stddev=1e-4, wd=WEIGHT_DECAY_REGULARIZATION)
    conv = tf.nn.conv2d(conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv2_2 = tf.nn.relu(bias, name=scope.name)

  pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

  # ---------------------- CONV3 ---------------------- #
  with tf.variable_scope('conv3_1') as scope:
    kernel = _variable_with_weight_decay('weights', shape=[3, 3, 128, 256], stddev=1e-4, wd=WEIGHT_DECAY_REGULARIZATION)
    conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv3_1 = tf.nn.relu(bias, name=scope.name)

  with tf.variable_scope('conv3_2') as scope:
    kernel = _variable_with_weight_decay('weights', shape=[3, 3, 256, 256], stddev=1e-4, wd=WEIGHT_DECAY_REGULARIZATION)
    conv = tf.nn.conv2d(conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv3_2 = tf.nn.relu(bias, name=scope.name)

  with tf.variable_scope('conv3_3') as scope:
    kernel = _variable_with_weight_decay('weights', shape=[3, 3, 256, 256],
                                           stddev=1e-4, wd=WEIGHT_DECAY_REGULARIZATION)
    conv = tf.nn.conv2d(conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv3_3 = tf.nn.relu(bias, name=scope.name)

  pool3 = tf.nn.max_pool(conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

  # ---------------------- CONV4 ---------------------- #
  with tf.variable_scope('conv4_1') as scope:
    kernel = _variable_with_weight_decay('weights', shape=[3, 3, 128, 512], stddev=1e-4, wd=WEIGHT_DECAY_REGULARIZATION)
    conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv4_1 = tf.nn.relu(bias, name=scope.name)

  with tf.variable_scope('conv4_2') as scope:
    kernel = _variable_with_weight_decay('weights', shape=[3, 3, 512, 512], stddev=1e-4, wd=WEIGHT_DECAY_REGULARIZATION)
    conv = tf.nn.conv2d(conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv4_2 = tf.nn.relu(bias, name=scope.name)

  with tf.variable_scope('conv4_3') as scope:
    kernel = _variable_with_weight_decay('weights', shape=[3, 3, 512, 512], stddev=1e-4,
                                          wd=WEIGHT_DECAY_REGULARIZATION)
    conv = tf.nn.conv2d(conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv4_3 = tf.nn.relu(bias, name=scope.name)

  pool4 = tf.nn.max_pool(conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

  # ---------------------- CONV5 ---------------------- #
  with tf.variable_scope('conv5_1') as scope:
    kernel = _variable_with_weight_decay('weights', shape=[3, 3, 512, 512], stddev=1e-4, wd=WEIGHT_DECAY_REGULARIZATION)
    conv = tf.nn.conv2d(pool4, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv5_1 = tf.nn.relu(bias, name=scope.name)

  with tf.variable_scope('conv5_2') as scope:
    kernel = _variable_with_weight_decay('weights', shape=[3, 3, 512, 512], stddev=1e-4, wd=WEIGHT_DECAY_REGULARIZATION)
    conv = tf.nn.conv2d(conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv5_2 = tf.nn.relu(bias, name=scope.name)

  with tf.variable_scope('conv5_3') as scope:
    kernel = _variable_with_weight_decay('weights', shape=[3, 3, 512, 512], stddev=1e-4, wd=WEIGHT_DECAY_REGULARIZATION)
    conv = tf.nn.conv2d(conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv5_3 = tf.nn.relu(bias, name=scope.name)

  pool5 = tf.nn.max_pool(conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

  # ---------------------- FULLY CONNECTED NET ---------------------- #

# reshape the output of the CNN
reshape = tf.reshape(pool5, [1, -1])












'''
  # norm1
  norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')

  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights', shape=[5, 5, 64, 64],
                                         stddev=1e-4, wd=0.0)
    conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(bias, name=scope.name)
    #_activation_summary(conv2)

  # norm2
  norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
  # pool2
  pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool2')

  # local3
  with tf.variable_scope('local3') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
    dim = reshape.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    #_activation_summary(local3)

  # local4
  with tf.variable_scope('local4') as scope:
    weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
    local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
    #_activation_summary(local4)

  # softmax, i.e. softmax(WX + b)
  with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES],
                                          stddev=1/192.0, wd=0.0)
    biases = _variable_on_cpu('biases', [NUM_CLASSES],
                              tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
    #_activation_summary(softmax_linear)

  return softmax_linear
'''

def loss(logits, labels):
  """Add L2Loss to all the trainable variables.
  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]
  Returns:
    Loss tensor of type float.
  """
  # Calculate the average cross entropy loss across the batch.
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits, labels, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')
