
import os

import tensorflow as tf
from slam.network.model_input import PoseNetInputProvider
from slam.utils.logging_utils import get_logger
from slam.network.google_net import GoogleNet
from slam.network.model_config import get_config_provider
from slam.network.model_input import get_simple_input_provider

"""
Google net with rgb dataset.
"""

if __name__ == '__main__':
    img_h = 224
    img_w = 224
    config_provider = get_config_provider()
    input_provider = get_simple_input_provider(config_provider.training_filenames)
    logger = get_logger()    

    base_dir = '/usr/prakt/s085/google-net/'
    LOG_DIR = os.path.join(base_dir, 'logs/')  
    LEARNED_WEIGHTS_FILENAME = os.path.join(base_dir, 'checkpoints/learned_weights.ckpt')
    
    epoch = 1000
    batch_size = 20
    sequence_length = 200
    
    rgb_input_batch = tf.placeholder(tf.float32, [batch_size, img_h, img_w, 4], name='rgbd_input')
    groundtruth_batch = tf.placeholder(tf.float32, [batch_size, 6], name='groundtruth')
    
    google_net = GoogleNet({'data':rgb_input_batch}, 6)
    loss = google_net.add_l2_loss(groundtruth_batch)
    apply_gradient_op = google_net.add_optimizer()
    
    session = tf.Session()
    session.run(tf.initialize_all_variables())
    
    merged_summary = tf.merge_all_summaries()
    summary_writer = tf.train.SummaryWriter(LOG_DIR, session.graph)
    saver = tf.train.Saver()
    
    for step in xrange(epoch):
        logger.info('Executing step:{}'.format(step))
        next_batch = input_provider.sequence_batch_itr(sequence_length, batch_size)
        for i, sequence_batch in enumerate(next_batch):
            logger.debug('epoc:{}, seq_no:{}, rgb files:{}, depth files:{}, groundtruths:{} in current batch'.format(step, i, sequence_batch.rgb_filenames,
                                                            sequence_batch.depth_filenames, sequence_batch.groundtruths))
            result = session.run([apply_gradient_op, loss, merged_summary], feed_dict={rgb_input_batch:sequence_batch.rgbd_images,
                                        groundtruth_batch:sequence_batch.groundtruths})
            loss_value = result[1]
            logger.info('epoc:{}, seq_no:{} loss :{}'.format(step, i, loss_value))
            summary_writer.add_summary(result[2], step * i + i)
        if step % 10 == 0:
            logger.info('Saving weights.')
            saver.save(session, LEARNED_WEIGHTS_FILENAME)
            
        logger.info('epoc:{}, loss:{}'.format(step, loss_value))
        
    session.close()


