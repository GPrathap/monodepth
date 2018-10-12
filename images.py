import pandas as pd
import numpy as np

import numpy as np
import argparse
import re
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.misc
import matplotlib.pyplot as plt
import os

for i in range(1,20):
    df = pd.read_csv("/home/geesara/inno/"+str(i)+"_disparity_map.csv")
    dataset = np.array(df.dropna(axis=1))
    disp_to_img = scipy.misc.imresize(dataset.squeeze(), [256, 512])
    plt.imsave(os.path.join("/home/geesara/diparity/images", "{}_{}--------.png".format(i, "new_image"))
                   , disp_to_img, cmap='plasma')

    print(dataset.shape)

# input_image = scipy.misc.imread('/dataset/kiit/2011_09_30/2011_09_30_drive_0020_sync/image_02/data/0000000000.png', mode="RGB")
# original_height, original_width, num_channels = input_image.shape
# input_image = scipy.misc.imresize(input_image, [256, 512])
# input_image = input_image.astype(np.float32) / 255
# input_images = np.stack((input_image, np.fliplr(input_image)))
#
# print(input_image.shape)