# Creates a graph.

# import tensorflow as tf
#
# a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
# b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
# c = tf.matmul(a, b)
# # Creates a session with log_device_placement set to True.
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# # Runs the op.
# print(sess.run(c))

import tensorflow as tf
image1 = tf.image.decode_png(tf.read_file('/dataset/images/data_tracking_image_2/training/image_02/0014/000066.png'))
print(image1.shape)
with tf.Session() as sess:
    img = sess.run(image1)
    print(img.shape, img[0][0])