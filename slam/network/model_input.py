import os
from os.path import isfile

import tensorflow as tf
import numpy as np

import scipy.ndimage
import bisect


def _absolute_position(groundtruth):
    x = groundtruth[0:3]
    q = groundtruth[3:7]

    # compute euler angles
    phi = np.arctan2(2*(q[0]*q[1] + q[2]*q[3]), 1-2*(np.square(q[1]) + np.square(q[2])))
    theta = np.arcsin(2*(q[0]*q[2] - q[3]*q[1]))
    psi = np.arctan2(2*(q[0]*q[3] + q[1]*q[2]), 1-2*(np.square(q[2]) + np.square(q[3])))

    # compute rotation matrix
    rot_mat = np.zeros((3, 3))
    rot_mat[0, 0] = np.cos(theta)*np.cos(phi)
    rot_mat[1, 0] = np.cos(theta)*np.sin(phi)
    rot_mat[2, 0] = -np.sin(theta)

    rot_mat[0, 1] = np.sin(psi)*np.sin(theta)*np.cos(phi) - np.cos(psi)*np.sin(phi)
    rot_mat[1, 1] = np.sin(psi)*np.sin(theta)*np.sin(phi) + np.cos(psi)*np.cos(phi)
    rot_mat[2, 1] = np.sin(psi)*np.cos(theta)

    rot_mat[0, 2] = np.cos(psi)*np.sin(theta)*np.cos(phi) + np.sin(psi)*np.sin(phi)
    rot_mat[1, 2] = np.cos(psi)*np.sin(theta)*np.sin(phi) - np.sin(psi)*np.cos(phi)
    rot_mat[2, 2] = np.cos(psi)*np.cos(theta)

    # compute angular velocities
    w_abs = np.arccos((np.trace(rot_mat)-1)/2)
    w = np.zeros(3)
    w[0] = 1/(2*np.sin(w_abs)) * (rot_mat[2, 1]-rot_mat[1, 2]) * w_abs
    w[1] = 1/(2*np.sin(w_abs)) * (rot_mat[0, 2]-rot_mat[2, 0]) * w_abs
    w[2] = 1/(2*np.sin(w_abs)) * (rot_mat[1, 0]-rot_mat[0, 1]) * w_abs

    return np.concatenate((x, w), 0)


def _find_label(groundtruth, timestamp):
    return bisect.bisect_left(groundtruth, timestamp)


class ModelInputProvider:
    
    DATA_DIR = ''
    
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.dataset_size = 0
    
    """
    Return the input data and ground truth tensors.

    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 4] size.
    outparam_batch: Labels. 1D tensor of [batch_size] size.
    """    
    def get_training_batch(self):
        rgbd = np.loadtxt(self.DATA_DIR+"/associated_data.txt", dtype="str",  unpack=False)
        groundtruth = np.loadtxt(self.DATA_DIR+"/groundtruth.txt", dtype="str",  unpack=False)
        self.dataset_size = rgbd.shape[0]

        # compute absolute position
        abs_pos = np.zeros((self.dataset_size, 6))
        for i in range(self.dataset_size):
            abs_pos[i] = _absolute_position(groundtruth[:, 1:][_find_label(groundtruth[:, 0], rgbd[i, 0])].astype(np.float))

        filename_queue_rgb = tf.train.string_input_producer(self.DATA_DIR + "/" + tf.convert_to_tensor(rgbd[:, 1]), shuffle=False)
        filename_queue_depth = tf.train.string_input_producer(self.DATA_DIR + "/" + tf.convert_to_tensor(rgbd[:, 3]), shuffle=False)

        image, outparams = self.read_rgbd_data(filename_queue_rgb, filename_queue_depth, abs_pos)
        images, outparam_batch = tf.train.batch([image, outparams], batch_size=self.batch_size,
                        num_threads=20, capacity=4 * self.batch_size)
        return images, outparam_batch
    
    """
     Returns the rgbd tensor and output parameters tensor after reading from file name queue.
    """
    def read_rgbd_data(self, filename_queue_rgb, filename_queue_depth, abs_pos):
        reader = tf.WholeFileReader()

        key, value_rgb = reader.read(filename_queue_rgb)
        _, value_depth = reader.read(filename_queue_depth)

         # Decoder
        png_rgb = tf.image.decode_png(value_rgb)
        png_depth = tf.image.decode_png(value_depth)

        # Scale depth image to values between 0 and 255
        png_depth -= png_depth.min()
        png_depth = (png_depth * 255.0/png_depth.max())

        # Images are concatenated to 4 channels
        image = np.concatenate((png_rgb, png_depth), axis=2)

        # Scale down image for CNN input
        image = scipy.ndimage.zoom(image, (224.0/480, 224.0/640, 1))

        # Compute relative position
        if key > 0:
            rel_pos = abs_pos[key] - abs_pos[key-1]
        else:
            rel_pos = 0

        return image, rel_pos
