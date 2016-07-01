import bisect
import os

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
from scipy import misc
from mpl_toolkits.mplot3d import Axes3D

import Image


def _find_label(groundtruth, timestamp):
    return bisect.bisect_left(groundtruth, timestamp)


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

    w = np.matrix(w)
    w_hat = np.matrix(w_hat)

    if w_abs == 0:
        return np.zeros(6)

    omega = (np.matrix((np.eye(3) - trans[0:3, 0:3])) * w_hat + w * np.transpose(w)) / (w_abs * w_abs)
    v = np.transpose(np.linalg.inv(omega) * np.matrix(trans[0:3, 3:4]))

    return np.concatenate((v, w), 1)


def _inverse_trans(trans):
    inv_trans = np.zeros((4, 4))
    inv_trans[0:3, 0:3] = np.transpose(trans[0:3, 0:3])
    inv_trans[0:3, 3:4] = - np.matrix(inv_trans[0:3, 0:3]) * np.matrix(trans[0:3, 3:4])
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

    if w_abs == 0:
        return np.eye(4)

    rot = np.eye(3) + w_hat/w_abs * np.sin(w_abs) + w_hat * w_hat / (w_abs * w_abs) * (1 - np.cos(w_abs))

    trans = (np.matrix((np.eye(3) - rot)) * w_hat * np.transpose(v) + np.transpose(w) * w * np.transpose(v)) / (w_abs * w_abs)

    tr = np.zeros((4, 4))
    tr[0:3, 0:4] = np.concatenate((rot, trans), 1)
    tr[3, 3] = 1

    return tr


def read_rgbd_data(input_queue):
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
    png_depth = tf.image.decode_png(value_depth, channels=1)

    # Reshape
    png_rgb = tf.reshape(png_rgb, [width_original, height_original, 3])
    png_depth = tf.reshape(png_depth, [width_original, height_original, 1])

    # Resize
    png_rgb = tf.image.resize_images(png_rgb, width, height)
    png_depth = tf.image.resize_images(png_depth, width, height)

    # Normalize depth
    png_depth = png_depth * 255.0/tf.reduce_max(png_depth)

    image = tf.concat(2, (png_rgb, png_depth))

    rel_pos = input_queue[2]
    rel_pos = tf.reshape(rel_pos, [1, 1, 6])

    return tf.cast(image, tf.float32), tf.cast(rel_pos, tf.float32)


def _point_cloud(depth_image, dataset_int):
    pointcloud = np.zeros((depth_image.shape[0] * depth_image.shape[1], 3))
    for v in range(depth_image.shape[0]):
        for u in range(depth_image.shape[1]):
            index = u + v*depth_image.shape[1]
            pointcloud[index, 2] = depth_image[v, u] / FACTOR
            pointcloud[index, 0] = (u - CX[dataset_int-1]) * pointcloud[index, 2] / FX[dataset_int-1]
            pointcloud[index, 1] = (v - CY[dataset_int-1]) * pointcloud[index, 2] / FY[dataset_int-1]
    return pointcloud[np.all(pointcloud != 0.0, 1)]


def _transform_pointcloud(pointcloud, trans):
    return np.dot(pointcloud, trans[0:3, 0:3]) + trans[0:3, 3]


def _backtransform_pointcloud(pointcloud, dataset_int):
    uvd = np.zeros(pointcloud.shape)
    uvd[:, 2] = pointcloud[:, 2]
    uvd[:, 0] = pointcloud[:, 0] * FX[dataset_int-1] / uvd[:, 2] + CX[dataset_int-1]
    uvd[:, 1] = pointcloud[:, 1] * FY[dataset_int-1] / uvd[:, 2] + CY[dataset_int-1]
    return uvd


def _overlap(backtransform, dim):
    c = 0.0
    for i in range(backtransform.shape[0]):
        if 0 <= backtransform[i, 0] <= dim[0] and 0 <= backtransform[i, 1] <= dim[1]:
            c += 1
    return c / backtransform.shape[0]

def _get_data(associations, groundtruth, sequence_length, dynamic_drop=0.0, static_drop=1):
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

    for i in range(dataset_length):
        # get quaternion
        quat = groundtruth[:, 1:][_find_label(groundtruth[:, 0], associations[i, 0])].astype(np.float32)
        # compute transformation matrix from quaternion
        trans_new = _quat_to_transformation(quat)
        if drop:
            # compute pointcloud if dynamic drop is chosen
            pointcloud = _point_cloud(misc.imread(os.path.join(filename, associations[i, 3])), 1)
        if i > 0:
            # compute relative transformation matrix
            relative_trans = np.dot(_inverse_trans(trans_old), trans_new)
            if drop:
                # transform old pointcloud if drop is chosen
                transformed_pointcloud = _transform_pointcloud(pointcloud_old, relative_trans)
                # transform pointcloud back into (u,v,d)
                back = _backtransform_pointcloud(transformed_pointcloud, 1)
                # compute overlap with current image
                overlap = _overlap(back, [640, 480])
                print overlap

                if overlap < dynamic_drop:
                    # set twist, if overlap is smaller than maximal overlap
                    twist[i] = _trans_to_twist(relative_trans)
                else:
                    twist[i] = None
                    rgb_filepaths[i] = None
                    depth_filepaths[i] = None
            else:
                twist[i] = _trans_to_twist(relative_trans)
        if i == 0 or (drop and overlap < dynamic_drop):
            # set old transformation to new transformation if image is not dropped
            trans_old = trans_new
        if drop:
            # set old pointcloud to new pointcloud
            pointcloud_old = pointcloud

    if drop:
        twist = [x for x in twist if x != None]
        print twist
        # rgb_filepaths = rgb_filepaths[np.all(rgb_filepaths != 0.0, 0)]
        # depth_filepaths = depth_filepaths[np.all(depth_filepaths != 0.0, 0)]

    dataset_length = twist.shape[0]
    sequence_length = np.min([sequence_length, dataset_length])
    start_point = 0
    if dataset_length > sequence_length:
        start_point = np.random.randint(0, dataset_length - sequence_length)

    twist = twist[start_point:start_point + sequence_length]
    rgb_filepaths = rgb_filepaths[start_point:start_point + sequence_length]
    depth_filepaths = depth_filepaths[start_point:start_point + sequence_length]

    return twist, rgb_filepaths, depth_filepaths


# folder, where the datasets are stored
path = "/home/aw/PycharmProjects/data/"

# Datasets as a List of strings
datasets = ["freiburg1_rpy", "freiburg1_xyz"]

FX = [517.3, 520.9, 535.4]
# focal length y
FY = [516.5, 521.0, 539.2]
# optical center x
CX = [318.6, 325.1, 320.1]
# optical center y
CY = [255.3, 249.7, 247.6]
# 16-bit PNG files
FACTOR = 1
# FACTOR = 5000

filename = path + datasets[0]

images_batch = []
groundtruth_batch = []

associations = np.loadtxt(os.path.join(filename, "associate.txt"), dtype="str", unpack=False)
groundtruth = np.loadtxt(os.path.join(filename, "groundtruth.txt"), dtype="str", unpack=False)

sequence_length = 200

twist, rgb_filepaths, depth_filepaths = _get_data(associations, groundtruth, sequence_length, dynamic_drop=0.9)

dataset_length = twist.shape[0]

rgb_filepaths = [os.path.join(filename, filepath) for filepath in rgb_filepaths]
depth_filepaths = [os.path.join(filename, filepath) for filepath in depth_filepaths]
rgb_filepaths_tensor = tf.convert_to_tensor(rgb_filepaths)
depth_filepaths_tensor = tf.convert_to_tensor(depth_filepaths)

input_queue = tf.train.slice_input_producer([rgb_filepaths_tensor, depth_filepaths_tensor, twist], shuffle=False)

image, outparams = read_rgbd_data(input_queue)

init_op = tf.initialize_all_variables()

np.set_printoptions(threshold=np.nan)

with tf.Session() as sess:
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for index in range(dataset_length):
        img, groundtruth = sess.run([image, outparams])
        print groundtruth

        # pointcloud = _point_cloud(img[:, :, 3], 1)
        # transform = _twist_to_trans(groundtruth[0, 0, :])
        # if index != 0:
        #     transformed_pointcloud = _transform_pointcloud(pointcloud_old, transform)
        #     back = _backtransform_pointcloud(transformed_pointcloud, 1)


        #     n = 20
        #     pc = pointcloud[0::n]
        #     pco = pointcloud_old[0::n]
        #     pct = transformed_pointcloud[0::n]
        #     bt = back[0::n]
        #     fig = plt.figure()
        #     ax = fig.add_subplot(111, projection='3d')
        #     ax.scatter(bt[:, 0], bt[:, 1], bt[:, 2], color="red")
        #     # ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], color="red")
        #     # ax.scatter(pct[:, 0], pct[:, 1], pct[:, 2], color="green")
        #     # ax.scatter(pco[:, 0], pco[:, 1], pco[:, 2], color="blue")
        #     plt.show()
        # pointcloud_old = pointcloud

        # im = Image.fromarray(img[:, :, 3])
        # im.convert('RGB').save("depth"+str(index)+".png")

        # plt.imshow(img[:, :, 3])



    coord.request_stop()
    coord.join(threads)

