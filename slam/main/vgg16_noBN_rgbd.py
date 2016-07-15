import os
import tensorflow as tf
import numpy as np

from slam.network.cnn_model_noBN import VGG16Model
from slam.network.model_config import get_config_provider
from slam.network.model_input import get_simple_input_provider
from slam.utils.logging_utils import get_logger

"""
 VGG16 with rgbd dataset.
"""
if __name__ == '__main__':
    img_h = 224
    img_w = 224

    logger = get_logger()
    config_provider = get_config_provider()

    base_dir = config_provider.base_log_dir()
    LOG_DIR = os.path.join(base_dir, 'logs/')
    LEARNED_WEIGHTS_FILENAME = os.path.join(base_dir, 'checkpoints/learned_weights.ckpt')

    epoch = config_provider.epoch()
    batch_size = config_provider.batch_size()
    sequence_length = config_provider.sequence_length()
    normalization_epsilon = config_provider.normalization_epsilon()

    rgbd_input_batch = tf.placeholder(tf.float32, [batch_size, img_h, img_w, 4], name='rgbd_input')
    groundtruth_batch = tf.placeholder(tf.float32, [batch_size, 6], name='groundtruth')

    vgg_model = VGG16Model(batch_size, rgbd_input_batch, 6, normalization_epsilon)
    vgg_model.build_graph()

    loss_weight = tf.placeholder(tf.float32, [6, 6])
    loss = vgg_model.add_loss(loss_weight, groundtruth_batch)
    apply_gradient_op = vgg_model.add_optimizer()

    input_provider = get_simple_input_provider(config_provider.training_filenames)
    session = tf.Session()
    session.run(tf.initialize_all_variables())

    merged_summary = tf.merge_all_summaries()
    summary_writer = tf.train.SummaryWriter(LOG_DIR, session.graph)
    saver = tf.train.Saver()
    for step in xrange(epoch):
        logger.info('Executing step:{}'.format(step))
        next_batch = input_provider.complete_seq_iter()
        for i, sequence_batch in enumerate(next_batch):
            logger.debug(
                'epoc:{}, seq_no:{}, rgb files:{}, depth files:{}, groundtruths:{} in current batch'.format(step, i,
                                                                                                            sequence_batch.rgb_filenames,
                                                                                                            sequence_batch.depth_filenames,
                                                                                                            sequence_batch.groundtruths))
            loss_weight_matrix = np.zeros([6, 6]) if i == 0 else np.identity(6)
            result = session.run([apply_gradient_op, loss, merged_summary],
                                 feed_dict={rgbd_input_batch: sequence_batch.rgbd_images,
                                            groundtruth_batch: sequence_batch.groundtruths,
                                            loss_weight: loss_weight_matrix})
            loss_value = result[1]
            logger.info('epoc:{}, seq_no:{} loss :{}'.format(step, i, loss_value))
            summary_writer.add_summary(result[2], step * i + i)
        if step % 10 == 0:
            logger.info('Saving weights.')
            saver.save(session, LEARNED_WEIGHTS_FILENAME)

        logger.info('epoc:{}, loss:{}'.format(step, loss_value))

    session.close()