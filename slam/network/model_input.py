import bisect
import os

import numpy as np
from slam.network.model_config import get_config_provider
import tensorflow as tf
from fileinput import filename
from slam.utils.logging_utils import get_logger


def _absolute_position(groundtruth):
    x = groundtruth[0:3]
    q = groundtruth[3:7]

    # compute euler angles
    phi = np.arctan2(2 * (q[0] * q[1] + q[2] * q[3]), 1 - 2 * (np.square(q[1]) + np.square(q[2])))
    theta = np.arcsin(2 * (q[0] * q[2] - q[3] * q[1]))
    psi = np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (np.square(q[2]) + np.square(q[3])))

    # compute rotation matrix
    rot_mat = np.zeros((3, 3))
    rot_mat[0, 0] = np.cos(theta) * np.cos(phi)
    rot_mat[1, 0] = np.cos(theta) * np.sin(phi)
    rot_mat[2, 0] = -np.sin(theta)

    rot_mat[0, 1] = np.sin(psi) * np.sin(theta) * np.cos(phi) - np.cos(psi) * np.sin(phi)
    rot_mat[1, 1] = np.sin(psi) * np.sin(theta) * np.sin(phi) + np.cos(psi) * np.cos(phi)
    rot_mat[2, 1] = np.sin(psi) * np.cos(theta)

    rot_mat[0, 2] = np.cos(psi) * np.sin(theta) * np.cos(phi) + np.sin(psi) * np.sin(phi)
    rot_mat[1, 2] = np.cos(psi) * np.sin(theta) * np.sin(phi) - np.sin(psi) * np.cos(phi)
    rot_mat[2, 2] = np.cos(psi) * np.cos(theta)

    # compute angular velocities
    w_abs = np.arccos((np.trace(rot_mat) - 1) / 2)
    w = np.zeros(3)
    w[0] = 1 / (2 * np.sin(w_abs)) * (rot_mat[2, 1] - rot_mat[1, 2]) * w_abs
    w[1] = 1 / (2 * np.sin(w_abs)) * (rot_mat[0, 2] - rot_mat[2, 0]) * w_abs
    w[2] = 1 / (2 * np.sin(w_abs)) * (rot_mat[1, 0] - rot_mat[0, 1]) * w_abs

    return np.concatenate((x, w), 0)


def _find_label(groundtruth, timestamp):
    return bisect.bisect_left(groundtruth, timestamp)


class ModelInputProvider:
    
    BASE_DATA_DIR = '/home/sanjeev/data/'  # '/usr/data/rgbd_datasets/tum_rgbd_benchmark/'
    
    def __init__(self):
        self.config_provider = get_config_provider()
        training_filenames = self.config_provider.get_training_filenames()
        self.training_filenames = [os.path.join(self.BASE_DATA_DIR, filename) for filename in training_filenames]
        self.logger = get_logger()
        self.batch_size = len(self.training_filenames)
    
    """
    Return the input data and ground truth tensors.

    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 4] size.
    outparam_batch: Labels. 1D tensor of [batch_size] size.
    """    
    def get_training_batch(self):
        self.logger.info('Going to create batch of images and ground truths')
        images_batch = []
        groundtruth_batch = []
        for filename in self.training_filenames:
            self.logger.info('Creating input queue for training sample at:{}'.format(filename))
            
            associations = np.loadtxt(os.path.join(filename, "associate.txt"), dtype="str", unpack=False)
            groundtruth = np.loadtxt(os.path.join(filename , "groundtruth.txt"), dtype="str", unpack=False)
            dataset_size = associations.shape[0]
            
            self.logger.info('The size of dataset:{} is {}'.format(filename, dataset_size))
            # compute absolute position
            abs_pos = np.zeros((dataset_size, 6))
            rel_pos = np.zeros((dataset_size, 6))
            for i in range(dataset_size):
                abs_pos[i] = _absolute_position(groundtruth[:, 1:][_find_label(groundtruth[:, 0], associations[i, 0])].astype(np.float32))
                if i > 0:
                    rel_pos[i] = abs_pos[i] - abs_pos[i - 1]
                else:
                    rel_pos[i] = np.zeros(6)

            rgb_filepaths = associations[:, 1]
            depth_filepaths = associations[:, 3]
            rgb_filepaths = [os.path.join(filename, filepath) for filepath in rgb_filepaths]
            depth_filepaths = [os.path.join(filename, filepath) for filepath in depth_filepaths]
            rgb_filepaths_tensor = tf.convert_to_tensor(rgb_filepaths)
            depth_filepaths_tensor = tf.convert_to_tensor(depth_filepaths)

            input_queue = tf.train.slice_input_producer([rgb_filepaths_tensor, depth_filepaths_tensor, rel_pos], shuffle=False)
        
            image, outparams = self.read_rgbd_data(input_queue)
            images_batch.append(image)
            groundtruth_batch.append(outparams)
        
#         images, outparam_batch = tf.train.batch([image, outparams], batch_size=self.batch_size,
#                         num_threads=20, capacity=4 * self.batch_size)
        return images_batch, groundtruth_batch
    
    """
     Returns the rgbd tensor and output parameters tensor after reading from file name queue.
    """
    def read_rgbd_data(self, input_queue):
        # original input size
        width_original = 480
        height_original = 640

        # input size
        width = 224
        height = 224

        value_rgb = tf.read_file(input_queue[0])
        value_depth = tf.read_file(input_queue[1])

        # Decoder
        png_rgb = tf.image.decode_png(value_rgb, channels=3)
        tf.image_summary('image', png_rgb)
        png_depth = tf.image.decode_png(value_depth, channels=1)

        # Reshape
        png_rgb = tf.reshape(png_rgb, [width_original, height_original, 3])
        png_depth = tf.reshape(png_depth, [width_original, height_original, 1])

        # Resize
        png_rgb = tf.image.resize_images(png_rgb, width, height)
        png_depth = tf.image.resize_images(png_depth, width, height)

        # Adjust brightness of depth image
        png_depth = tf.image.adjust_brightness(png_depth, 1)

        image = tf.concat(2, (png_rgb, png_depth))

        rel_pos = input_queue[2]
        rel_pos = tf.reshape(rel_pos, [1, 1, 6])
        
        return tf.cast(image, tf.float32), tf.cast(rel_pos, tf.float32)

    def get_batch_size(self):
        return self.batch_size


input_provider = ModelInputProvider()
def get_input_provider():
    return input_provider
