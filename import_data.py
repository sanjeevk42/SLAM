import tensorflow as tf
import os
import numpy as np


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


dataset = "rgbd_dataset_freiburg1_xyz"

rgb = np.loadtxt(dataset+"/rgb.txt", dtype="str",  unpack=False)
depth = np.loadtxt(dataset+"/depth.txt", dtype="str",  unpack=False)
groundtruth = np.loadtxt(dataset+"/groundtruth.txt", dtype="str",  unpack=False)

labels = groundtruth[:, 1:]

# FIFO queue of filenames
filename_queue_rgb = tf.train.string_input_producer(dataset + "/" + tf.convert_to_tensor(rgb[:, 1]))
filename_queue_depth = tf.train.string_input_producer(dataset + "/" + tf.convert_to_tensor(depth[:, 1]))

# Reader for the file format
reader = tf.WholeFileReader()
key_rgb, value_rgb = reader.read(filename_queue_rgb)
key_depth, value_depth = reader.read(filename_queue_depth)

# Decoder for record read by the reader
image_rgb = tf.image.decode_png(value_rgb)
image_depth = tf.image.decode_png(value_depth)

init_op = tf.initialize_all_variables()

images = np.zeros((480, 640, 4, rgb[:, 1].size))

with tf.Session() as sess:
    sess.run(init_op)

    # Start populating the filename queue.

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    num_examples = rgb[:, 1].size

    filename = os.path.join('records/', dataset + '.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(num_examples):
        img_rgb_raw = image_rgb.eval().tostring()
        img_depth_raw = image_depth.eval().tostring()

        label = labels[index]

        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(480),
            'width': _int64_feature(640),
            'depth_rgb': _int64_feature(3),
            'depth_depth': _int64_feature(1),

            'tx': _float_feature(label[0].astype(float)),
            'ty': _float_feature(label[1].astype(float)),
            'tz': _float_feature(label[2].astype(float)),
            'qx': _float_feature(label[3].astype(float)),
            'qy': _float_feature(label[4].astype(float)),
            'qz': _float_feature(label[5].astype(float)),
            'qw': _float_feature(label[6].astype(float)),

            'image_rgb_raw': _bytes_feature(img_rgb_raw),
            'image_depth_raw': _bytes_feature(img_depth_raw)}))
        print(index)
    writer.write(example.SerializeToString())

    coord.request_stop()
    coord.join(threads)

