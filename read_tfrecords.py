import tensorflow as tf
import numpy as np


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
      serialized_example,
      features={
          'image_rgb_raw': tf.FixedLenFeature([], tf.string),
          'image_depth_raw': tf.FixedLenFeature([], tf.string),

          'height': tf.FixedLenFeature([], tf.int64),
          'width': tf.FixedLenFeature([], tf.int64),
          'depth_rgb': tf.FixedLenFeature([], tf.int64),
          'depth_depth': tf.FixedLenFeature([], tf.int64),

          'tx': tf.FixedLenFeature([], tf.float32),
          'ty': tf.FixedLenFeature([], tf.float32),
          'tz': tf.FixedLenFeature([], tf.float32),
          'qx': tf.FixedLenFeature([], tf.float32),
          'qy': tf.FixedLenFeature([], tf.float32),
          'qz': tf.FixedLenFeature([], tf.float32),
          'qw': tf.FixedLenFeature([], tf.float32)
      })

    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    depth_rgb = tf.cast(features['depth_rgb'], tf.int32)
    depth_depth = tf.cast(features['depth_depth'], tf.int32)

    image_rgb = tf.decode_raw(features['image_rgb_raw'], tf.uint8)
    image_depth = tf.decode_raw(features['image_depth_raw'], tf.uint8)

    image_rgb = tf.reshape(image_rgb, tf.pack([height, width, depth_rgb]))
    image_depth = tf.reshape(image_depth, tf.pack([height, width, depth_depth]))

    labels = [features['tx'], features['ty'], features['tz'],
              features['qx'], features['qy'], features['qz'],
              features['qw']]

    return image_rgb, image_depth, labels

dataset = "rgbd_dataset_freiburg1_xyz"
filename_queue_records = tf.train.string_input_producer(['records/' + dataset + '.tfrecords'])
image_rgb, image_depth, labels = read_and_decode(filename_queue_records)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    tf.train.start_queue_runners(sess=sess)

    for index in range(700):
        print(index)
        rgb_img, depth_img, label = sess.run([image_rgb, image_depth, labels])
        print (label)

