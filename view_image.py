
import numpy as np
import argparse
import re
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.misc
import matplotlib.pyplot as plt
import os


disp_pp = np.load("/home/geesara/diparity/disparities.npy")
for i in range(0,len(disp_pp)):
    disp_to_img = scipy.misc.imresize(disp_pp[i].squeeze(), [256, 512])
    plt.imsave(os.path.join("/home/geesara/diparity/images", "{}_{}disp.png".format(i, "new_image")), disp_to_img, cmap='plasma')
