
import numpy as np
import argparse
import re
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.misc
import matplotlib.pyplot as plt
import os


disp_pp = np.load("/home/geesara/diparity/0000000000_disp.npy")
disp_to_img = scipy.misc.imresize(disp_pp.squeeze(), [256, 512])
plt.imsave(os.path.join("./", "{}_disp.png".format("new_image")), disp_to_img, cmap='plasma')
