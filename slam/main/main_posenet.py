import os
from scipy import ndimage
from skimage import transform

import numpy as np
from slam.network.cnn_model import VGG16Model
from slam.utils.logging_utils import get_logger
import tensorflow as tf
from tensorflow.python.training.adam import AdamOptimizer

def read_rgb_image(filepath):
    rgb_img = ndimage.imread(filepath)
    width = height = 224
    img_width = rgb_img.shape[1]
    img_height = rgb_img.shape[0]

    # scale such that smaller dimension is 256
    if img_width < img_height:
        factor = 256.0/img_width
    else:
        factor = 256.0/img_height
    rgb_img = transform.rescale(rgb_img, factor)

    # crop randomly
    width_start = np.random.randint(0, rgb_img.shape[1]-width)
    height_start = np.random.randint(0, rgb_img.shape[0]-height)

    rgb_img = rgb_img[height_start:height_start+height, width_start:width_start+width]
    return rgb_img

class PoseNetInputProvider:
    
    BASE_DIR = '/home/sanjeev/Downloads/KingsCollege/'
    
    def __init__(self):
        self.sequence_info = np.loadtxt(os.path.join(self.BASE_DIR, 'dataset_train.txt'), dtype="str", unpack=False)
        
    class PoseNetBatch:pass
    
    class PoseNetIterator:
        
        def __init__(self, sequence_info, batch_size):
            self.sequence_info = sequence_info
            self.index = 0
            self.batch_size = batch_size
        
        def __iter__(self):
            return self
        
        def next(self):
            if self.index + self.batch_size < len(self.sequence_info):
                batch_info = self.sequence_info[self.index:self.index + self.batch_size]
                filenames = [os.path.join(PoseNetInputProvider.BASE_DIR, filename) for filename in batch_info[:, 0]]
                rgb_files = map(read_rgb_image, filenames)
                groundtruths = batch_info[:, 1:]
                batch = PoseNetInputProvider.PoseNetBatch()
                batch.rgb_files = rgb_files
                batch.groundtruths = groundtruths
                batch.rgb_filenames = filenames
                self.index += self.batch_size
                return batch
            else:
                raise StopIteration()
    
    def sequence_batch_itr(self, batch_size):
        return PoseNetInputProvider.PoseNetIterator(self.sequence_info, batch_size)
    
    
def add_posenet_loss(output, groundtruth):
    x = groundtruth[:, :3]
    xp = output[:, :3]
    q = groundtruth[:, 3:]
    qp = output[:, 3:]
    q_norm = q / tf.sqrt(q ** 2)
    loss = tf.reduce_sum(xp - x) + 0.5 * tf.reduce_sum(qp - q_norm)
    loss = tf.Print(loss, [q, q_norm], 'Value of q and qnorm:', summarize=20)
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
