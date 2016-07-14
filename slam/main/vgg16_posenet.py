import os

from tensorflow.python.training.adam import AdamOptimizer

import numpy as np
from slam.network.cnn_model import VGG16Model
from slam.network.model_input import PoseNetInputProvider
from slam.utils.logging_utils import get_logger
import tensorflow as tf

"""
VGG 16 with posenet dataset.
"""


def add_posenet_loss(output, groundtruth):
    x = groundtruth[:, :3]
    xp = output[:, :3]
    q = groundtruth[:, 3:]
    qp = output[:, 3:]
    # loss = tf.reduce_sum(xp - x) + 0.5 * tf.reduce_sum(qp - q_norm)
    loss = tf.add(tf.reduce_sum((xp - x) ** 2) , 1100 * tf.reduce_sum((qp - q) ** 2))
    loss = tf.Print(loss, [output, groundtruth, loss ], 'output, groundtruth, loss:', summarize=20)
    tf.scalar_summary('posenet_loss', loss)
    return loss

def add_optimizer(loss):
    global_step = tf.Variable(0, trainable=False)
    
    learning_rate = tf.train.exponential_decay(0.01, global_step, 5,
                               0.1, staircase=True)
    
    optimizer = AdamOptimizer(learning_rate)
    gradients = optimizer.compute_gradients(loss)
    
    for grad, var in gradients:
        if grad is not None:
            tf.histogram_summary(var.op.name + '/gradients', grad)
    # gradients = tf.Print([gradients], [gradients], 'The value of gradients:')
    apply_gradient_op = optimizer.apply_gradients(gradients, global_step)
    return apply_gradient_op

if __name__ == '__main__':
    img_h = 224
    img_w = 224
    input_provider = PoseNetInputProvider()
    logger = get_logger()    

    base_dir = '/home/sanjeev/posenet/'
    LOG_DIR = os.path.join(base_dir, 'logs/')  
    LEARNED_WEIGHTS_FILENAME = os.path.join(base_dir, 'checkpoints/learned_weights.ckpt')
    
    epoch = 1000
    batch_size = 1
    normalization_epsilon = 0.001
    
    rgbd_input_batch = tf.placeholder(tf.float32, [batch_size, img_h, img_w, 3], name='rgbd_input')
    groundtruth_batch = tf.placeholder(tf.float32, [batch_size, 7], name='groundtruth')
    
    vgg_model = VGG16Model(batch_size, rgbd_input_batch, 7, normalization_epsilon)
    network_output = vgg_model.build_graph()
    
    loss = add_posenet_loss(network_output, groundtruth_batch)
    
    apply_gradient_op = add_optimizer(loss)
    
    session = tf.Session()
    session.run(tf.initialize_all_variables())
    
    merged_summary = tf.merge_all_summaries()
    summary_writer = tf.train.SummaryWriter(LOG_DIR, session.graph)
    saver = tf.train.Saver()
    
    for step in xrange(epoch):
        logger.info('Executing step:{}'.format(step))
        next_batch = input_provider.sequence_batch_itr(batch_size)
        for i, sequence_batch in enumerate(next_batch):
            logger.debug('epoc:{}, seq_no:{}, rgb files:{}, groundtruths:{} in current batch'.format(step, i, sequence_batch.rgb_filenames,
                                                                         sequence_batch.groundtruths))
            loss_weight_matrix = np.zeros([6, 6]) if i == 0 else np.identity(6)
            result = session.run([apply_gradient_op, loss, merged_summary], feed_dict={rgbd_input_batch:sequence_batch.rgb_files,
                                        groundtruth_batch:sequence_batch.groundtruths})
            loss_value = result[1]
            logger.info('epoc:{}, seq_no:{} loss :{}'.format(step, i, loss_value))
            summary_writer.add_summary(result[2], step * i + i)
        if step % 10 == 0:
            logger.info('Saving weights.')
            saver.save(session, LEARNED_WEIGHTS_FILENAME)
            
        logger.info('epoc:{}, loss:{}'.format(step, loss_value))
        
    session.close()
