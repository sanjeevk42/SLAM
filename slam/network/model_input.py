import bisect
from fileinput import filename
import os
import random

import numpy as np
from slam.network.model_config import get_config_provider
from slam.utils.logging_utils import get_logger
import tensorflow as tf
from collections import OrderedDict


def _quat_to_transformation(groundtruth):
    t = groundtruth[0:3]
    q = groundtruth[3:7]

    # compute euler angles
    phi = np.arctan2(2 * (q[0] * q[1] + q[2] * q[3]), 1 - 2 * (np.square(q[1]) + np.square(q[2])))
    theta = np.arcsin(2 * (q[0] * q[2] - q[3] * q[1]))
    psi = np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (np.square(q[2]) + np.square(q[3])))

    # compute rotation matrix
    trans = np.zeros((4, 4))
    trans[0, 0] = np.cos(theta) * np.cos(phi)
    trans[1, 0] = np.cos(theta) * np.sin(phi)
    trans[2, 0] = -np.sin(theta)
    trans[3, 0] = 0

    trans[0, 1] = np.sin(psi) * np.sin(theta) * np.cos(phi) - np.cos(psi) * np.sin(phi)
    trans[1, 1] = np.sin(psi) * np.sin(theta) * np.sin(phi) + np.cos(psi) * np.cos(phi)
    trans[2, 1] = np.sin(psi) * np.cos(theta)
    trans[3, 1] = 0

    trans[0, 2] = np.cos(psi) * np.sin(theta) * np.cos(phi) + np.sin(psi) * np.sin(phi)
    trans[1, 2] = np.cos(psi) * np.sin(theta) * np.sin(phi) - np.sin(psi) * np.cos(phi)
    trans[2, 2] = np.cos(psi) * np.cos(theta)
    trans[3, 2] = 0

    trans[0, 3] = t[0]
    trans[1, 3] = t[1]
    trans[2, 3] = t[2]
    trans[3, 3] = 1

    return trans


def _trans_to_twist(trans):
    # compute angular velocities
    w_abs = np.arccos((np.trace(trans[0:3, 0:3]) - 1) / 2)
    w = np.zeros(3)
    w[0] = 1 / (2 * np.sin(w_abs)) * (trans[2, 1] - trans[1, 2]) * w_abs
    w[1] = 1 / (2 * np.sin(w_abs)) * (trans[0, 2] - trans[2, 0]) * w_abs
    w[2] = 1 / (2 * np.sin(w_abs)) * (trans[1, 0] - trans[0, 1]) * w_abs

    w_hat = np.zeros((3, 3))
    w_hat[0] = [0, -w[2], w[1]]
    w_hat[1] = [w[2], 0, -w[0]]
    w_hat[2] = [-w[1], w[0], 0]

    w_abs = np.linalg.norm(w)

    if w_abs == 0:
        return np.zeros(6)

    w = np.matrix(w)
    w_hat = np.matrix(w_hat)
    omega = (np.matrix((np.eye(3) - trans[0:3, 0:3]))*w_hat + w * np.transpose(w)) / (w_abs*w_abs)
    v = np.transpose(np.linalg.inv(omega) * np.matrix(trans[0:3, 3]))

    return np.concatenate((v, w), 1)


def _inverse_trans(trans):
    inv_trans = np.zeros((4, 4))
    inv_trans[0:3, 0:3] = np.transpose(trans[0:3, 0:3])
    inv_trans[0:3, 3] = -np.matrix(inv_trans[0:3, 0:3]) * np.matrix(trans[0:3, 3])
    inv_trans[3, 3] = 1

    return inv_trans


def _twist_to_trans(twist):
    v = np.matrix(twist[0, 0:3])
    w = np.matrix(twist[0, 3:6])
    w_abs = np.linalg.norm(w)
    w_hat = np.zeros((3, 3))
    w_hat[0] = [0, -w[0, 2], w[0, 1]]
    w_hat[1] = [w[0, 2], 0, -w[0, 0]]
    w_hat[2] = [-w[0, 1], w[0, 0], 0]
    w_hat = np.matrix(w_hat)

    rot = np.eye(3) + w_hat/w_abs * np.sin(w_abs) + w_hat * w_hat / (w_abs * w_abs) * (1 - np.cos(w_abs))

    trans = (np.matrix((np.eye(3) - rot)) * w_hat * np.transpose(v) + np.transpose(w) * w * np.transpose(v)) / (w_abs * w_abs)

    tr = np.zeros((4, 4))
    tr[0:3, 0:4] = np.concatenate((rot, trans), 1)
    tr[3, 3] = 1

    return tr


def _find_label(groundtruth, timestamp):
    return bisect.bisect_left(groundtruth, timestamp)


class QueuedInputProvider:
    
    BASE_DATA_DIR = '/home/sanjeev/data/'  # '/usr/data/rgbd_datasets/tum_rgbd_benchmark/'
    # focal length x fr1/fr2/fr3
    FX = [517.3, 520.9, 535.4]
    # focal length y
    FY = [516.5, 521.0, 539.2]
    # optical center x
    CX = [318.6, 325.1, 320.1]
    # optical center y
    CY = [255.3, 249.7, 247.6]
    # 16-bit PNG files
    FACTOR = 5000
    
    def __init__(self):
        self.config_provider = get_config_provider()
        training_filenames = self.config_provider.training_filenames()
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
        sequence_length = 300
        for filename in self.training_filenames:
            self.logger.info('Creating input queue for training sample at:{}'.format(filename))
            
            associations = np.loadtxt(os.path.join(filename, "associate.txt"), dtype="str", unpack=False)
            groundtruth = np.loadtxt(os.path.join(filename, "groundtruth.txt"), dtype="str", unpack=False)
            dataset_size = associations.shape[0]

            # select every nth image
            n = 1
            associations = associations[0::n]

            start_point = np.random.randint(0, dataset_size - sequence_length)
            
            self.logger.info('The size of dataset:{} is {}'.format(filename, dataset_size))
            # compute absolute position
            twist = np.zeros((sequence_length, 6))
            trans_old = np.zeros((4, 4))
            for ind in range(sequence_length):
                i = ind + start_point
                quat = groundtruth[:, 1:][_find_label(groundtruth[:, 0], associations[i, 0])].astype(np.float32)
                trans_new = _quat_to_transformation(quat)
                if i > 0:
                    twist[ind] = _trans_to_twist(_inverse_trans(trans_old) * trans_new)
                else:
                    twist[ind] = np.zeros(6)
                trans_old = trans_new

            rgb_filepaths = associations[start_point:start_point + sequence_length, 1]
            depth_filepaths = associations[start_point:start_point + sequence_length, 3]
            rgb_filepaths = [os.path.join(filename, filepath) for filepath in rgb_filepaths]
            depth_filepaths = [os.path.join(filename, filepath) for filepath in depth_filepaths]
            rgb_filepaths_tensor = tf.convert_to_tensor(rgb_filepaths)
            depth_filepaths_tensor = tf.convert_to_tensor(depth_filepaths)

            input_queue = tf.train.slice_input_producer([rgb_filepaths_tensor, depth_filepaths_tensor, twist], shuffle=False)
        
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

        # Normalize depth
        png_depth = tf.image.adjust_contrast(png_depth, 1) * 255.0/tf.reduce_max(png_depth)

        image = tf.concat(2, (png_rgb, png_depth))

        twist = tf.reshape(input_queue[2], [1, 1, 6])
        
        return tf.cast(image, tf.float32), tf.cast(twist, tf.float32)

    def get_batch_size(self):
        return self.batch_size

    def _point_cloud(self, depth_image, dataset_int):
        x = []
        y = []
        z = []
        for v in range(depth_image.height):
            for u in range(depth_image.width):
                z.append(depth_image[v, u] / self.FACTOR)
                x.append((u - self.CX[dataset_int - 1]) * z[-1] / self.FX[dataset_int - 1])
                y.append((v - self.CY[dataset_int - 1]) * z[-1] / self.FY[dataset_int - 1])
        return [x, y, z]


class SimpleInputProvider:

    class InputBatch:
    
        def __init__(self, input_provider, seqdir_vs_offset, sequence_length):
            self.counter = 0
            self.seqdir_vs_offset = seqdir_vs_offset
            self.input_provider = input_provider
            self.sequence_length = sequence_length
        
        def __iter__(self):
            return self
        
        def next(self):
            rgbd_batch = []
            groundtruth_batch = []
            if self.counter < self.sequence_length:
                for i, ele in enumerate(self.seqdir_vs_offset):
                    seqdir = ele[0]
                    offset = ele[1]
                    rgbd_file = self.input_provider.get_rgbd_file(seqdir, offset)
                    rgbd_batch.append(rgbd_file)
                    groundtruth = self.input_provider.get_ground_truth(seqdir, offset)
                    groundtruth_batch.append(groundtruth)
                    self.counter += 1
                    self.seqdir_vs_offset[i][1] = offset + 1
                return np.array(rgbd_batch), np.array(groundtruth_batch)
            else:
                raise StopIteration()
        
    BASE_DATA_DIR = '/home/sanjeev/data/'  # '/usr/data/rgbd_datasets/tum_rgbd_benchmark/'
    
    def __init__(self):
        self.config_provider = get_config_provider()
        training_filenames = self.config_provider.training_filenames()
        self.sequence_dirs = [os.path.join(self.BASE_DATA_DIR, filename) for filename in training_filenames]
        
        self.seq_dir_map = {}
        
        for seq_dir in self.sequence_dirs:
            associations = np.loadtxt(os.path.join(seq_dir, "associate.txt"), dtype="str", unpack=False)
            groundtruth = np.loadtxt(os.path.join(seq_dir , "groundtruth.txt"), dtype="str", unpack=False)
            if seq_dir not in self.seq_dir_map:
                self.seq_dir_map[seq_dir] = {}
            self.seq_dir_map[seq_dir]['associations'] = associations
            self.seq_dir_map[seq_dir]['groundtruth'] = groundtruth
            
            sequence_size = associations.shape[0]
            abs_pos = np.zeros((sequence_size, 6))
            rel_pos = np.zeros((sequence_size, 6))
            for i in range(sequence_size):
                abs_pos[i] = _absolute_position(groundtruth[:, 1:][_find_label(groundtruth[:, 0],
                                                         associations[i, 0])].astype(np.float32))
                if i > 0:
                    rel_pos[i] = abs_pos[i] - abs_pos[i - 1]
                else:
                    rel_pos[i] = np.zeros(6)
            self.seq_dir_map[seq_dir]['relpos'] = rel_pos
            
    def get_next_batch(self, sequence_length, batch_size):
        random.shuffle(self.sequence_dirs)
        training_sequences = self.sequence_dirs
        total_sequences = len(training_sequences)
        seqdir_vs_offset = []
        for i in xrange(batch_size):
            seq_dir = training_sequences[i % total_sequences]
            associations = self.seq_dir_map[seq_dir]['associations']
            total_frames = len(associations)
            offset = random.randint(0, total_frames - sequence_length)
            seqdir_vs_offset.append([seq_dir, offset])
            
        input_batch = self.InputBatch(self, seqdir_vs_offset, sequence_length)
        return input_batch
    
    def get_rgbd_file(self, dirname, offset):
        associations = self.seq_dir_map[dirname]['associations']
        rgb_file = os.path.join(dirname, associations[offset, 1])
        depth_file = os.path.join(dirname, associations[offset, 3])
       
        
        session = tf.InteractiveSession()
        
        rgb_file = tf.read_file(rgb_file)
        depth_file = tf.read_file(depth_file)
        
        png_rgb = tf.image.decode_png(rgb_file, channels=3)
        png_depth = tf.image.decode_png(depth_file, channels=1)
        
        width_original, height_original = 480, 640
        width = height = 224
        # Reshape
        png_rgb = tf.reshape(png_rgb, [width_original, height_original, 3])
        png_depth = tf.reshape(png_depth, [width_original, height_original, 1])

        # Resize
        png_rgb = tf.image.resize_images(png_rgb, width, height)
        png_depth = tf.image.resize_images(png_depth, width, height)

        # Adjust brightness of depth image
        png_depth = tf.image.adjust_brightness(png_depth, 1)

        image = tf.concat(2, (png_rgb, png_depth))

        
        image = tf.cast(image, tf.float32)
        rgbd_file = image.eval()
        
        session.close()
        
        return rgbd_file
    
    def get_ground_truth(self, dirname, offset):
        groundtruth = self.seq_dir_map[dirname]['relpos'][offset, :]
        groundtruth = np.reshape(groundtruth, [1, 1, 6])
        return groundtruth
            
queued_input_provider = QueuedInputProvider()
def get_queued_input_provider():
    return queued_input_provider

simple_input_provider = SimpleInputProvider()
def get_simple_input_provider():
    return simple_input_provider

if __name__ == '__main__':
    input_provider = SimpleInputProvider()
    
    input_batch = input_provider.get_next_batch(100, 20)
    for i, batch in enumerate(input_batch):
        print i, 'groundtruth: ', batch[1]

