import os
from os.path import isfile

import tensorflow as tf


class ModelInputProvider:
    
    DATA_DIR = ''
    
    def __init__(self, batch_size):
        self.batch_size = batch_size
    
    """
    Return the input data and ground truth tensors.
    """    
    def get_training_batch(self):
        filenames = [f for f in os.listdir(self.DATA_DIR) if isfile(os.path.join(self.DATA_DIR, f))]
        filename_queue = tf.train.string_input_producer(filenames, shuffle=False)
        image, outparams = self.read_rgbd_data(filename_queue)
        images, outparam_batch = tf.train.batch([image, outparams], batch_size=self.batch_size,
                        num_threads=20, capacity=4 * self.batch_size)
        return images, outparam_batch
    
    """
     Returns the rgbd tensor and output parameters tensor after reading from file name queue.
    """
    def read_rgbd_data(self, filename_queue):
        # TODO: use appropriate reader to read data
        pass
