from numpy import loadtxt
dataset = "rgbd_dataset_freiburg1_xyz"

rgb = loadtxt(dataset+"/rgb.txt", dtype="str",  unpack=False)
depth = loadtxt(dataset+"/depth.txt", dtype="str",  unpack=False)
groundtruth = loadtxt(dataset+"/groundtruth.txt", dtype="str",  unpack=False)

print groundtruth[:, 1:]
