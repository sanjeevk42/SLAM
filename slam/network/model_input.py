import bisect
from fileinput import filename
import os
import random
from scipy import ndimage, misc
from skimage import transform

import numpy as np
from slam.network.model_config import get_config_provider
from slam.utils.logging_utils import get_logger
from slam.utils.time_utils import time_it
import tensorflow as tf

def _quat_to_transformation(groundtruth):
    t = groundtruth[0:3]
    q = groundtruth[3:7]
    q_x = q[0]
    q_y = q[1]
    q_z = q[2]
    q_w = q[3]

    trans = np.zeros((4, 4))
    trans[0, 0] = q_w * q_w + q_x * q_x - q_y * q_y - q_z * q_z
    trans[1, 0] = 2 * (q_x * q_y + q_w * q_z)
    trans[2, 0] = 2 * (q_z * q_x - q_w * q_y)
    trans[3, 0] = 0

    trans[0, 1] = 2 * (q_x * q_y - q_w * q_z)
    trans[1, 1] = q_w * q_w - q_x * q_x + q_y * q_y - q_z * q_z
    trans[2, 1] = 2 * (q_y * q_z + q_w * q_x)
    trans[3, 1] = 0

    trans[0, 2] = 2 * (q_z * q_x + q_w * q_y)
    trans[1, 2] = 2 * (q_y * q_z - q_w * q_x)
    trans[2, 2] = q_w * q_w - q_x * q_x - q_y * q_y + q_z * q_z
    trans[3, 2] = 0

    trans[0, 3] = t[0]
    trans[1, 3] = t[1]
    trans[2, 3] = t[2]
    trans[3, 3] = 1

    return trans


def _trans_to_twist(trans):
    # compute angular velocities
    if (np.trace(trans[0:3, 0:3]) - 1) / 2 > 1:
        return np.zeros(6)

    argument = (np.trace(trans[0:3, 0:3]) - 1) / 2
    if np.abs(argument) > 1:
        print "Warning: numerical issue in groundtruth transformation"
        argument = 1 * argument/np.abs(argument)
    w_abs = np.arccos(argument)
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
    omega = (np.matrix((np.eye(3) - trans[0:3, 0:3])) * w_hat + w * np.transpose(w)) / (w_abs * w_abs)
    v = np.transpose(np.linalg.inv(omega) * np.transpose(np.matrix(trans[0:3, 3])))

    return np.concatenate((v, w), 1)


def _inverse_trans(trans):
    inv_trans = np.zeros((4, 4))
    inv_trans[0:3, 0:3] = np.transpose(trans[0:3, 0:3])
    inv_trans[0:3, 3] = np.squeeze(-np.matrix(inv_trans[0:3, 0:3]) * np.transpose(np.matrix(trans[0:3, 3])))
    inv_trans[3, 3] = 1
    
    return inv_trans


def _twist_to_trans(twist):
    v = np.matrix(twist[0:3])
    w = np.matrix(twist[3:6])
    w_abs = np.linalg.norm(w)
    w_hat = np.zeros((3, 3))
    w_hat[0] = [0, -w[0, 2], w[0, 1]]
    w_hat[1] = [w[0, 2], 0, -w[0, 0]]
    w_hat[2] = [-w[0, 1], w[0, 0], 0]
    w_hat = np.matrix(w_hat)

    rot = np.eye(3) + w_hat / w_abs * np.sin(w_abs) + w_hat * w_hat / (w_abs * w_abs) * (1 - np.cos(w_abs))

    trans = (np.matrix((np.eye(3) - rot)) * w_hat * np.transpose(v) + np.transpose(w) * w * np.transpose(v)) / (w_abs * w_abs)

    tr = np.zeros((4, 4))
    tr[0:3, 0:4] = np.concatenate((rot, trans), 1)
    tr[3, 3] = 1

    return tr


def _transform_pointcloud(pointcloud, trans):
        return np.dot(pointcloud, transform[0:3, 0:3]) + transform[0:3, 3]


def _overlap(backtransform):
    dim = [224.0, 224.0]
    c = 0.0
    for i in range(backtransform.shape[0]):
        if dim[0] >= backtransform[i, 0] >= 0 and dim[1] >= backtransform[i, 1] >= 0:
            c += 1
    return c / backtransform.shape[0]


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
    FACTOR = 1
    
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

            twist, rgb_filepaths, depth_filepaths = self._get_data(associations, groundtruth, sequence_length)

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
        png_depth = png_depth * 255.0 / tf.reduce_max(png_depth)

        image = tf.concat(2, (png_rgb, png_depth))

        twist = tf.reshape(input_queue[2], [1, 1, 6])
        
        return tf.cast(image, tf.float32), tf.cast(twist, tf.float32)

    def get_batch_size(self):
        return self.batch_size

    def _point_cloud(self, depth_image, dataset_int):
        pointcloud = np.zeros((depth_image.shape[0] * depth_image.shape[1], 3))
        for v in range(depth_image.shape[0]):
            for u in range(depth_image.shape[1]):
                pointcloud[u * v, 2] = depth_image[v, u] / self.FACTOR
                pointcloud[u * v, 0] = (u - self.CX[dataset_int - 1]) * pointcloud[u * v, 2] / self.FX[dataset_int - 1]
                pointcloud[u * v, 1] = (v - self.CY[dataset_int - 1]) * pointcloud[u * v, 2] / self.FY[dataset_int - 1]
        return pointcloud[np.all(pointcloud != 0.0, 1)]

    def _backtransform_pointcloud(self, pointcloud, dataset_int):
        uvd = np.zeros(pointcloud.shape)
        uvd[:, 2] = pointcloud[:, 2]
        uvd[:, 0] = pointcloud[:, 0] * self.FX[dataset_int - 1] / uvd[:, 2] + self.CX[dataset_int - 1]
        uvd[:, 1] = pointcloud[:, 1] * self.FY[dataset_int - 1] / uvd[:, 2] + self.CY[dataset_int - 1]
        return uvd

    """
    Return Twist and Filepaths
    dynamic drop -> images where the overlap is greater than the specified value are dropped (has to be between 0.0 and 1.0)
    static drop -> a fixed number of images is dropped
    """
    def _get_data(self, associations, groundtruth, sequence_length, dynamic_drop=0.0, static_drop=1):
        # select every nth image
        associations = associations[0::static_drop]
        # define filepaths
        rgb_filepaths = associations[:, 1]
        depth_filepaths = associations[:, 3]

        # set dataset length
        dataset_length = associations.shape[0]

        # initialize twist and old transformation
        twist = np.zeros((dataset_length, 6))
        trans_old = np.zeros((4, 4))

        # check if dynamic drop is chosen
        drop = 0.0 < dynamic_drop <= 1.0

        delete_list = []
        pointcloud_old = None
        for i in range(dataset_length):
            # get quaternion
            quat = groundtruth[:, 1:][_find_label(groundtruth[:, 0], associations[i, 0])].astype(np.float32)
            # compute transformation matrix from quaternion
            trans_new = _quat_to_transformation(quat)
            if drop:
                # compute pointcloud if dynamic drop is chosen
                pointcloud = self._point_cloud(misc.imread(os.path.join(filename, associations[i, 3])), 1)
            if i > 0:
                # compute relative transformation matrix
                relative_trans = np.dot(_inverse_trans(trans_old), trans_new)
                if drop:
                    # transform old pointcloud if drop is chosen
                    transformed_pointcloud = _transform_pointcloud(pointcloud_old, relative_trans)
                    # transform pointcloud back into (u,v,d)
                    back = self._backtransform_pointcloud(transformed_pointcloud, 1)
                    # compute overlap with current image
                    overlap = _overlap(back, [640, 480])
                    print overlap

                    if overlap < dynamic_drop:
                        # set twist, if overlap is smaller than maximal overlap
                        twist[i] = _trans_to_twist(relative_trans)
                    else:
                        delete_list.append(i)
                else:
                    twist[i] = _trans_to_twist(relative_trans)
            if i == 0 or (drop and overlap < dynamic_drop):
                # set old transformation to new transformation if image is not dropped
                trans_old = trans_new
            if drop:
                # set old pointcloud to new pointcloud
                pointcloud_old = pointcloud

        if drop:
            # delete dropped files
            twist = np.delete(twist, delete_list, 0)
            rgb_filepaths = np.delete(rgb_filepaths, delete_list, 0)
            depth_filepaths = np.delete(depth_filepaths, delete_list, 0)

        # set dataset length
        dataset_length = twist.shape[0]
        # take dataset length if it is smaller than sequence length
        sequence_length = np.min([sequence_length, dataset_length])

        # choose start point randomly
        start_point = 0
        if dataset_length > sequence_length:
            start_point = np.random.randint(0, dataset_length - sequence_length)

        # set output
        twist = twist[start_point:start_point + sequence_length]
        rgb_filepaths = rgb_filepaths[start_point:start_point + sequence_length]
        depth_filepaths = depth_filepaths[start_point:start_point + sequence_length]

        self.logger.info('The size of dataset:{} is {}'.format(filename, twist.shape[0]))

        return twist, rgb_filepaths, depth_filepaths


class SimpleInputProvider:

    class SequenceBatch:
        
        def __init__(self):
            self.rgb_filenames = []
            self.depth_filenames = []
            self.rgbd_images = []
            self.groundtruths = []
        
    class SequenceBatchIterator:
    
        def __init__(self, input_provider, seqdir_vs_offset, sequence_length):
            self.counter = 0
            self.seqdir_vs_offset = seqdir_vs_offset
            self.input_provider = input_provider
            self.sequence_length = sequence_length
            self.logger = get_logger()
        
        def __iter__(self):
            return self
        
        @time_it
        def next(self):
            self.logger.debug('Going to fetch next batch of frames. batch size:{}, frame no.:{} '.format(len(self.seqdir_vs_offset), self.counter))
            sequence_batch = SimpleInputProvider.SequenceBatch()
            if self.counter < self.sequence_length:
                for i, ele in enumerate(self.seqdir_vs_offset):
                    seqdir = ele[0]
                    offset = ele[1]
                    rgb_filename, depth_filename, rgbd_file = self.input_provider.get_rgbd_file(seqdir, offset)
                    groundtruth = self.input_provider.get_ground_truth(seqdir, offset)
                    
                    sequence_batch.rgbd_images.append(rgbd_file)
                    sequence_batch.groundtruths.append(groundtruth)
                    sequence_batch.rgb_filenames.append(rgb_filename)
                    sequence_batch.depth_filenames.append(depth_filename)
                    self.seqdir_vs_offset[i][1] = offset + 1
                self.counter += 1
                return sequence_batch
            else:
                raise StopIteration()
        
    BASE_DATA_DIR = '/usr/data/rgbd_datasets/tum_rgbd_benchmark/'  # '/home/sanjeev/data/'
    
    def __init__(self, filename_provider):
        training_filenames = filename_provider()
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
            twist = np.zeros((sequence_size, 6))
            trans_old = np.zeros((4, 4))
            for i in range(sequence_size):
                quat = groundtruth[:, 1:][_find_label(groundtruth[:, 0], associations[i, 0])].astype(np.float32)
                trans_new = _quat_to_transformation(quat)
                if i > 0:
                    twist[i] = _trans_to_twist(trans_new)
                else:
                    twist[i] = np.zeros(6)
                trans_old = trans_new
            
            self.seq_dir_map[seq_dir]['relpos'] = twist
            
    def sequence_batch_itr(self, sequence_length, batch_size):
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
            
        input_batch = self.SequenceBatchIterator(self, seqdir_vs_offset, sequence_length)
        return input_batch
    
    
    """
    Randomly selects one sequence out of all sequences and returns all frames of the sequence.
    """
    def complete_seq_iter(self):
        random.shuffle(self.sequence_dirs)
        training_sequences = self.sequence_dirs
        seq_dir = training_sequences[0]
        associations = self.seq_dir_map[seq_dir]['associations']
        total_frames = len(associations)
        seqdir_vs_offset = [[seq_dir, 0]]
        input_batch = self.SequenceBatchIterator(self, seqdir_vs_offset, total_frames)
        return input_batch
    
    def get_rgbd_file(self, dirname, offset):
        associations = self.seq_dir_map[dirname]['associations']
        
        if associations[offset, 1].startswith('depth'):
            rgb_filename = os.path.join(dirname, associations[offset, 3])
            depth_filename = os.path.join(dirname, associations[offset, 1])
        else:
            rgb_filename = os.path.join(dirname, associations[offset, 1])
            depth_filename = os.path.join(dirname, associations[offset, 3])
       
        rgb_img = ndimage.imread(rgb_filename)
        depth_img = ndimage.imread(depth_filename)
        width = height = 224

        # Reshape
        depth_img = np.reshape(depth_img, list(depth_img.shape) + [1])
        depth_img = 255 * depth_img / np.max(depth_img)

        rgbd_img = np.concatenate((rgb_img, depth_img), 2)

        # Resize
        rgbd_img = transform.resize(rgbd_img, [width, height], preserve_range=True)

        return rgb_filename, depth_filename, rgbd_img.astype(np.float32)
    
    def get_ground_truth(self, dirname, offset):
        groundtruth = self.seq_dir_map[dirname]['relpos'][offset, :]
        return groundtruth

class PoseNetInputProvider:
    
    BASE_DIR = '/usr/data/cvpr_shared/lingni/cambridge_pose_dataset/KingsCollege/'
    
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

queued_input_provider = QueuedInputProvider()
def get_queued_input_provider():
    return queued_input_provider

def get_simple_input_provider(filename_provider):
    return SimpleInputProvider(filename_provider)

if __name__ == '__main__':
    config_provider = get_config_provider()
    input_batch = get_simple_input_provider(config_provider.training_filenames).complete_seq_iter()
    for i, batch in enumerate(input_batch):
        print i, 'rgb files: ', batch.rgb_filenames, 'depth files:', batch.depth_filenames

