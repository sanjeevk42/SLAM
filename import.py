import tensorflow as tf
import numpy as np
import bisect
import scipy.ndimage


def _find_label(groundtruth, timestamp):
    return bisect.bisect_left(groundtruth, timestamp)

def _relative_position(groundtruth):
    q_0 = groundtruth[3]
    q_1 = groundtruth[4]
    q_2 = groundtruth[5]
    q_3 = groundtruth[6]
    roll = np.arctan2(2*(q_0*q_1 + q_2*q_3), 1-2*(np.square(q_1) + np.square(q_2)))
    pitch = np.arcsin(2*(q_0*q_2 - q_3*q_1))
    yaw = np.arctan2(2*(q_0*q_3 + q_1*q_2), 1-2*(np.square(q_2) + np.square(q_3)))

dataset = "rgbd_dataset_freiburg1_xyz"

rgbd = np.loadtxt(dataset+"/associated_data.txt", dtype="str",  unpack=False)
groundtruth = np.loadtxt(dataset+"/groundtruth.txt", dtype="str",  unpack=False)

labels = groundtruth[:, 1:]

# drop every nth element out of list
n = 3
x = np.delete(rgbd, np.arange(0, rgbd.size, 3))

# FIFO queue of filenames
filename_queue_rgb = tf.train.string_input_producer(dataset + "/" + tf.convert_to_tensor(rgbd[:, 1]))
filename_queue_depth = tf.train.string_input_producer(dataset + "/" + tf.convert_to_tensor(rgbd[:, 3]))

# Reader for the file format
reader = tf.WholeFileReader()
key_rgb, value_rgb = reader.read(filename_queue_rgb)
key_depth, value_depth = reader.read(filename_queue_depth)

# Decoder for record read by the reader
image_rgb = tf.image.decode_png(value_rgb)
image_depth = tf.image.decode_png(value_depth)

init_op = tf.initialize_all_variables()

images = np.zeros((480, 640, 4, rgbd[:, 1].size))

with tf.Session() as sess:
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    num_examples = rgbd[:, 1].size

    for index in range(num_examples):
        img_rgb = image_rgb.eval()
        img_depth = image_depth.eval()

        # scale depth image to values between 0 and 255
        img_depth -= img_depth.min()
        img_depth = (img_depth * 255.0/img_depth.max()) # maybe problem with float

        img = np.concatenate((img_rgb, img_depth), axis=2)

        # scale down image
        img = scipy.ndimage.zoom(img, (0.5, 0.5, 1))

        label = labels[_find_label(groundtruth[:, 0], rgbd[index, 0])]

        print(index)

    coord.request_stop()
    coord.join(threads)

