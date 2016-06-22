from slam.network.cnn_model import VGG16Model
from slam.network.lstm_model import LSTMmodel
import tensorflow as tf
import numpy as np
from slam.utils.logging_utils import get_logger
from slam.network.model_config import get_config_provider
from slam.network.model_input import get_simple_input_provider


logger = get_logger()

def model():
    img_h = 224
    img_w = 224
    
    config_provider = get_config_provider()
    
    epoch = config_provider.epoch()
    batch_size = config_provider.batch_size()
    sequence_length = config_provider.sequence_length()
    lstm_layers = config_provider.lstm_layers()

    rgbd_input_batch = tf.placeholder(tf.float32, [batch_size, img_h, img_w, 4])
    groundtruth_batch = tf.placeholder(tf.float32, [batch_size, 6])
    
    vgg_model = VGG16Model(rgbd_input_batch, 4096)
    cnn_output = vgg_model.build_graph()
    lstm_model = LSTMmodel(cnn_output, layer_size=4096, layers=lstm_layers, output_dim=6,
                            ground_truth=groundtruth_batch, batch_size=batch_size)
    lstm_model.build_graph()
    
    loss_weight = tf.placeholder(tf.float32, [6, 6])
    
    loss = lstm_model.add_loss(loss_weight)
    apply_gradient_op = lstm_model.add_optimizer()
    
    session = tf.Session()
    session.run(tf.initialize_all_variables())
    tf.train.start_queue_runners(sess=session)
#     summary_writer = tf.train.SummaryWriter('/home/sanjeev/logs', session.graph)
#     
    input_provider = get_simple_input_provider()
    
    for step in xrange(epoch):
        logger.info('Executing step:{}'.format(step))
        
        input_batch = input_provider.get_next_batch(sequence_length, batch_size)
        for i, batch in input_batch:
            logger.info('Step:{} Frame:{}'.format(step, i))
            loss_weight_matrix = np.identity(6) if i == 0 else np.zeros([6, 6])
            loss_value = session.run([apply_gradient_op, loss], feed_dict={rgbd_input_batch:batch[0],
                                        groundtruth_batch:batch[1], loss_weight:loss_weight_matrix})
        
        logger.info('Step:{}, loss:{}'.format(step, loss_value))


if __name__ == '__main__':
    model()
