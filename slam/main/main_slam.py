import numpy as np
from slam.network.cnn_model import VGG16Model
from slam.network.lstm_model import LSTMmodel
from slam.network.model_config import get_config_provider
from slam.network.model_input import get_simple_input_provider
from slam.utils.logging_utils import get_logger
import tensorflow as tf


logger = get_logger()
LOG_DIR = '/usr/prakt/s085/logs' 
LEARNED_WEIGHTS_FILENAME = 'resources/learned_weights.ckpt'
img_h = 224
img_w = 224

def start_training():
    
    config_provider = get_config_provider()
    
    epoch = config_provider.epoch()
    batch_size = config_provider.batch_size()
    sequence_length = config_provider.sequence_length()
    lstm_layers = config_provider.lstm_layers()
    cnn_output_dim = config_provider.cnn_output_dim()
    normalization_epsilon = config_provider.normalization_epsilon()

    rgbd_input_batch = tf.placeholder(tf.float32, [batch_size, img_h, img_w, 4])
    groundtruth_batch = tf.placeholder(tf.float32, [batch_size, 6])
    lstm_init_state = tf.placeholder(tf.float32, [batch_size, 2 * cnn_output_dim * lstm_layers])
    
    network = build_complete_network(rgbd_input_batch, groundtruth_batch, lstm_init_state, batch_size, lstm_layers, cnn_output_dim, normalization_epsilon)
    
    lstm_model = network[2]
    
    loss_weight = tf.placeholder(tf.float32, [6, 6])
    loss = lstm_model.add_loss(loss_weight)
    apply_gradient_op = lstm_model.add_optimizer()
    
    session = tf.Session()
    session.run(tf.initialize_all_variables())
    tf.train.start_queue_runners(sess=session)
    
    merged_summary = tf.merge_all_summaries()
    summary_writer = tf.train.SummaryWriter(LOG_DIR, session.graph)
#     
    input_provider = get_simple_input_provider(config_provider.training_filenames)
    
    saver = tf.train.Saver()
    
    for step in xrange(epoch):
        logger.info('Executing epoc:{}'.format(step))
        
        input_batch = input_provider.sequence_batch_itr(sequence_length, batch_size)
        for i, sequence_batch in enumerate(input_batch):
            logger.debug('Using rgb files:{}, depth files:{}, groundtruths:{} in current batch'.format(sequence_batch.rgb_filenames,
                                                                         sequence_batch.depth_filenames, sequence_batch.groundtruths))
            loss_weight_matrix = np.zeros([6, 6]) if i == 0 else np.identity(6)
            result = session.run([apply_gradient_op, loss, merged_summary], feed_dict={rgbd_input_batch:sequence_batch.rgbd_images,
                                        groundtruth_batch:sequence_batch.groundtruths, loss_weight:loss_weight_matrix,
                                        lstm_init_state:np.zeros([batch_size, 2 * cnn_output_dim * lstm_layers], np.float32)})
            loss_value = result[1]
            logger.info('epoc:{}, sequence number:{}, loss:{}'.format(step, i, loss_value))
            
        summary_writer.add_summary(result[2], step)
        saver.save(session, LEARNED_WEIGHTS_FILENAME)
        
        logger.info('epoc:{}, loss:{}'.format(step, loss_value))



def build_complete_network(rgbd_input_batch, groundtruth_batch, lstm_init_state, batch_size, lstm_layers, cnn_output_dim, normalization_epsilon):
    cnn_model = VGG16Model(batch_size, rgbd_input_batch, cnn_output_dim, normalization_epsilon)
    cnn_output = cnn_model.build_graph()
    lstm_model = LSTMmodel(cnn_output, layer_size=cnn_output_dim, layers=lstm_layers, output_dim=6,
                            ground_truth=groundtruth_batch, batch_size=batch_size, init_state=lstm_init_state)
    lstm_output = lstm_model.build_graph()
    return cnn_model, cnn_output, lstm_model, lstm_output

def evaluate_model():

    config_provider = get_config_provider()    
    input_provider = get_simple_input_provider(config_provider.test_filenames)
    
    
    batch_size = 1
    sequence_length = 400
    lstm_layers = config_provider.lstm_layers()
    cnn_output_dim = config_provider.cnn_output_dim()
    
    rgbd_input_batch = tf.placeholder(tf.float32, [batch_size, img_h, img_w, 4])
    groundtruth_batch = tf.placeholder(tf.float32, [batch_size, 6])
    lstm_init_state = tf.placeholder(tf.float32, [batch_size, 2 * cnn_output_dim * lstm_layers])
    
    network = build_complete_network(rgbd_input_batch, groundtruth_batch, lstm_init_state, batch_size, lstm_layers, cnn_output_dim)
    
    lstm_output = network[3]

    session = tf.Session()
    session.run(tf.initialize_all_variables())
    
    saver = tf.train.Saver()
    
    saver.restore(session, LEARNED_WEIGHTS_FILENAME)
    
    total_rmse = 0
    
    for step in xrange(10):
        logger.info('Executing evaluation step:{} '.format(step))
        input_batch = input_provider.sequence_batch_itr(sequence_length, batch_size)
        for _, sequence_batch in enumerate(input_batch):
            result = session.run([lstm_output], feed_dict={rgbd_input_batch:sequence_batch.rgbd_images,
                                        groundtruth_batch:sequence_batch.groundtruths})
            rmse = (np.array(result) - sequence_batch.groundtruths) ** 2
            total_rmse += rmse
            logger.info('Input frame info: rgb file:{}, depth file:{}, groundtruth:{}, predicted params:{}, rmse:{} '.format(sequence_batch.rgb_filenames,
                                                                        sequence_batch.depth_filenames, sequence_batch.groundtruths, result, rmse))
            
    logger.info('Total rmse on test data:{}'.format(total_rmse))

if __name__ == '__main__':
    start_training()
#     evaluate_model()
    

