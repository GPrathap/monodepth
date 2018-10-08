# Copyright UCL Business plc 2017. Patent Pending. All rights reserved. 
#
# The MonoDepth Software is licensed under the terms of the UCLB ACP-A licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
#
# For any other use of the software not covered by the UCLB ACP-A Licence, 
# please contact info@uclb.com

"""Fully convolutional model for monocular depth estimation
    by Clement Godard, Oisin Mac Aodha and Gabriel J. Brostow
    http://visual.cs.ucl.ac.uk/pubs/monoDepth/
"""

from __future__ import absolute_import, division, print_function

import math
from collections import namedtuple

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
histogram_summary = tf.summary.histogram
from Batch_Norm import batch_norm
from bilinear_sampler import *
from tensorflow.python.training import moving_averages

monodepth_parameters = namedtuple('parameters', 
                        'encoder, '
                        'height, width, '
                        'batch_size, '
                        'num_threads, '
                        'num_epochs, '
                        'do_stereo, '
                        'wrap_mode, '
                        'use_deconv, '
                        'alpha_image_loss, '
                        'disp_gradient_loss_weight, '
                        'lr_loss_weight, '
                        'sample_dir, '
                        'use_bn, '
                        'z_dim, '
                        'frozen_model_filename, '
                        'full_summary')

class MonodepthModel(object):
    """monodepth model"""
    def __init__(self, params, mode, left, right ,reuse_variables=None, model_index=0):
        self.params = params
        self.mode = mode
        self.left = left
        self.right = right
        self.model_collection = ['model_' + str(model_index)]
        self.width = self.params.width
        self.height = self.params.height
        self.reuse_variables = reuse_variables
        self.batch_size = self.params.batch_size
        self.sample_num = self.params.batch_size
        self.y_dim = None
        self.z_dim = params.z_dim
        self.gf_dim = 64
        self.df_dim = 64
        self.c_dim = 3
        self._extra_train_ops = []
        self.build_model()
        self.build_outputs()


        if self.mode == 'test' or self.mode == 'export':
            return

        self.build_losses()
        self.build_summaries()

    def gradient_x(self, img):
        gx = img[:,:,:-1,:] - img[:,:,1:,:]
        return gx

    def gradient_y(self, img):
        gy = img[:,:-1,:,:] - img[:,1:,:,:]
        return gy

    def upsample_nn(self, x, ratio):
        s = tf.shape(x)
        h = s[1]
        w = s[2]
        return tf.image.resize_nearest_neighbor(x, [h * ratio, w * ratio])

    def scale_pyramid(self, img, num_scales):
        scaled_imgs = [img]
        s = tf.shape(img)
        print(img[0])
        h = s[1]
        w = s[2]
        for i in range(num_scales - 1):
            ratio = 2 ** (i + 1)
            nh = h // ratio
            nw = w // ratio
            scaled_imgs.append(tf.image.resize_area(img, [nh, nw]))
        return scaled_imgs

    def generate_image_left(self, img, disp):
        return bilinear_sampler_1d_h(img, -disp)

    def generate_image_right(self, img, disp):
        return bilinear_sampler_1d_h(img, disp)

    def SSIM(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = slim.avg_pool2d(x, 3, 1, 'VALID')
        mu_y = slim.avg_pool2d(y, 3, 1, 'VALID')

        sigma_x  = slim.avg_pool2d(x ** 2, 3, 1, 'VALID') - mu_x ** 2
        sigma_y  = slim.avg_pool2d(y ** 2, 3, 1, 'VALID') - mu_y ** 2
        sigma_xy = slim.avg_pool2d(x * y , 3, 1, 'VALID') - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

        SSIM = SSIM_n / SSIM_d

        return tf.clip_by_value((1 - SSIM) / 2, 0, 1)

    def get_disparity_smoothness(self, disp, pyramid):
        disp_gradients_x = [self.gradient_x(d) for d in disp]
        disp_gradients_y = [self.gradient_y(d) for d in disp]

        image_gradients_x = [self.gradient_x(img) for img in pyramid]
        image_gradients_y = [self.gradient_y(img) for img in pyramid]

        weights_x = [tf.exp(-tf.reduce_mean(tf.abs(g), 3, keep_dims=True)) for g in image_gradients_x]
        weights_y = [tf.exp(-tf.reduce_mean(tf.abs(g), 3, keep_dims=True)) for g in image_gradients_y]

        smoothness_x = [disp_gradients_x[i] * weights_x[i] for i in range(4)]
        smoothness_y = [disp_gradients_y[i] * weights_y[i] for i in range(4)]
        return smoothness_x + smoothness_y

    def get_disp(self, x, batch_normalization_fn):
        disp = 0.3 * self.conv(x, 2, 3, 1, batch_normalization_fn, tf.nn.sigmoid)
        return disp

    def conv(self, x, num_out_layers, kernel_size, stride, batch_normalization_fn, activation_fn=tf.nn.elu):
        p = np.floor((kernel_size - 1) / 2).astype(np.int32)
        p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
        return slim.conv2d(p_x, num_out_layers, kernel_size, stride, 'VALID', normalizer_fn=batch_normalization_fn)

    def conv_block(self, x, num_out_layers, kernel_size, batch_normalization_fn):
        conv1 = self.conv(x,     num_out_layers, kernel_size, 1, batch_normalization_fn=batch_normalization_fn)
        conv2 = self.conv(conv1, num_out_layers, kernel_size, 2, batch_normalization_fn=batch_normalization_fn)
        return conv2

    def maxpool(self, x, kernel_size):
        p = np.floor((kernel_size - 1) / 2).astype(np.int32)
        p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
        return slim.max_pool2d(p_x, kernel_size)

    def resconv(self, x, num_layers, stride, batch_normalization_fn, apply_batch_norm=False, name="batch_norm"):
        do_proj = tf.shape(x)[3] != num_layers or stride == 2
        shortcut = []
        conv1 = self.conv(x,         num_layers, 1, 1, batch_normalization_fn=batch_normalization_fn)
        conv2 = self.conv(conv1,     num_layers, 3, stride, batch_normalization_fn=batch_normalization_fn)
        conv3 = self.conv(conv2, 4 * num_layers, 1, 1, batch_normalization_fn=batch_normalization_fn, activation_fn=None)
        if do_proj:
            shortcut = self.conv(x, 4 * num_layers, 1, stride, batch_normalization_fn=batch_normalization_fn, activation_fn=None)
        else:
            shortcut = x
        normalized_conv = conv3 + shortcut
        # if apply_batch_norm:
        #     normalized_conv = self._batch_norm(normalized_conv, name)
        return tf.nn.elu(normalized_conv)

    def resblock(self, x, num_layers, num_blocks, name, batch_normalization_fn):
        out = x
        for i in range(num_blocks - 1):
            out = self.resconv(out, num_layers, 1, batch_normalization_fn)
        out = self.resconv(out, num_layers, 2, batch_normalization_fn, apply_batch_norm=True, name=name)
        return out

    def get_feature_set(self):
        return tf.concat([tf.layers.flatten(self.disp1), tf.layers.flatten(self.disp2),
                          tf.layers.flatten(self.disp3), tf.layers.flatten(self.disp4)], axis=1)

    def upconv(self, x, num_out_layers, kernel_size, scale, batch_normalization_fn):
        upsample = self.upsample_nn(x, scale)
        conv = self.conv(upsample, num_out_layers, kernel_size, 1, batch_normalization_fn)
        return conv

    def deconv(self, x, num_out_layers, kernel_size, scale):
        p_x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])
        conv = slim.conv2d_transpose(p_x, num_out_layers, kernel_size, scale, 'SAME')
        return conv[:,3:-1,3:-1,:]

    # def _batch_norm(self, input_):
    #     """Batch normalization for a 4-D tensor"""
    #     assert len(input_.get_shape()) == 4
    #     filter_shape = input_.get_shape().as_list()
    #     mean, var = tf.nn.moments(input_, axes=[0, 1, 2])
    #     out_channels = filter_shape[3]
    #     offset = tf.Variable(tf.zeros([out_channels]))
    #     scale = tf.Variable(tf.ones([out_channels]))
    #     batch_norm = tf.nn.batch_normalization(input_, mean, var, offset, scale, 0.001)
    #     return batch_norm

    def _batch_norm(self, x, name):
        """Batch normalization."""
        with tf.variable_scope(name):
            params_shape = [x.get_shape()[-1]]

            beta = tf.get_variable(
                'beta', params_shape, tf.float32,
                initializer=tf.constant_initializer(0.0, tf.float32))
            gamma = tf.get_variable(
                'gamma', params_shape, tf.float32,
                initializer=tf.constant_initializer(1.0, tf.float32))

            if self.mode == 'train':
                mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')

                moving_mean = tf.get_variable(
                    'moving_mean', params_shape, tf.float32,
                    initializer=tf.constant_initializer(0.0, tf.float32),
                    trainable=False)
                moving_variance = tf.get_variable(
                    'moving_variance', params_shape, tf.float32,
                    initializer=tf.constant_initializer(1.0, tf.float32),
                    trainable=False)

                self._extra_train_ops.append(moving_averages.assign_moving_average(
                    moving_mean, mean, 0.9))
                self._extra_train_ops.append(moving_averages.assign_moving_average(
                    moving_variance, variance, 0.9))
            else:
                mean = tf.get_variable(
                    'moving_mean', params_shape, tf.float32,
                    initializer=tf.constant_initializer(0.0, tf.float32),
                    trainable=False)
                variance = tf.get_variable(
                    'moving_variance', params_shape, tf.float32,
                    initializer=tf.constant_initializer(1.0, tf.float32),
                    trainable=False)
                tf.summary.histogram(mean.op.name, mean)
                tf.summary.histogram(variance.op.name, variance)
            # epsilon used to be 1e-5. Maybe 0.001 solves NaN problem in deeper net.
            y = tf.nn.batch_normalization(
                x, mean, variance, beta, gamma, 0.001)
            y.set_shape(x.get_shape())
            return y

    def build_vgg(self):
        #set convenience functions
        conv = self.conv
        if self.params.use_deconv:
            upconv = self.deconv
        else:
            upconv = self.upconv

        with tf.variable_scope('encoder'):
            conv1 = self.conv_block(self.model_input,  32, 7, batch_normalization_fn = None) # H/2
            conv2 = self.conv_block(conv1,             64, 5, batch_normalization_fn = None) # H/4
            conv3 = self.conv_block(conv2,            128, 3, batch_normalization_fn = None) # H/8
            conv4 = self.conv_block(conv3,            256, 3, batch_normalization_fn = None) # H/16
            conv5 = self.conv_block(conv4,            512, 3, batch_normalization_fn = None) # H/32
            conv6 = self.conv_block(conv5,            512, 3, batch_normalization_fn = None) # H/64
            conv7 = self.conv_block(conv6,            512, 3, batch_normalization_fn = None) # H/128

        with tf.variable_scope('skips'):
            skip1 = conv1
            skip2 = conv2
            skip3 = conv3
            skip4 = conv4
            skip5 = conv5
            skip6 = conv6
        
        with tf.variable_scope('decoder'):
            upconv7 = upconv(conv7,  512, 3, 2) #H/64
            concat7 = tf.concat([upconv7, skip6], 3)
            iconv7  = conv(concat7,  512, 3, 1, batch_normalization_fn = None)

            upconv6 = upconv(iconv7, 512, 3, 2) #H/32
            concat6 = tf.concat([upconv6, skip5], 3)
            iconv6  = conv(concat6,  512, 3, 1, batch_normalization_fn = None)

            upconv5 = upconv(iconv6, 256, 3, 2) #H/16
            concat5 = tf.concat([upconv5, skip4], 3)
            iconv5  = conv(concat5,  256, 3, 1, batch_normalization_fn = None)

            upconv4 = upconv(iconv5, 128, 3, 2) #H/8
            concat4 = tf.concat([upconv4, skip3], 3)
            iconv4 = conv(concat4,  128, 3, 1, batch_normalization_fn = None)
            self.disp4 = self.get_disp(iconv4, batch_normalization_fn = None)
            udisp4  = self.upsample_nn(self.disp4, 2)

            upconv3 = upconv(iconv4,  64, 3, 2) #H/4
            concat3 = tf.concat([upconv3, skip2, udisp4], 3)
            iconv3  = conv(concat3,   64, 3, 1, batch_normalization_fn = None)
            self.disp3 = self.get_disp(iconv3, batch_normalization_fn = None)
            udisp3  = self.upsample_nn(self.disp3, 2)

            upconv2 = upconv(iconv3,  32, 3, 2) #H/2
            concat2 = tf.concat([upconv2, skip1, udisp3], 3)
            iconv2  = conv(concat2,   32, 3, 1, batch_normalization_fn = None)
            self.disp2 = self.get_disp(iconv2, batch_normalization_fn = None)
            udisp2 = self.upsample_nn(self.disp2, 2)

            upconv1 = upconv(iconv2,  16, 3, 2) #H
            concat1 = tf.concat([upconv1, udisp2], 3)
            iconv1 = conv(concat1,   16, 3, 1, batch_normalization_fn = None)
            self.disp1 = self.get_disp(iconv1, batch_normalization_fn = None)
            self.classification = tf.nn.sigmoid(iconv1)

    def build_resnet50(self):
        #set convenience functions
        conv = self.conv
        if self.params.use_deconv:
            upconv = self.deconv
        else:
            upconv = self.upconv

        with tf.variable_scope('encoder'):
            conv1 = conv(self.model_input, 64, 7, 2, batch_normalization_fn = None) # H/2  -   64D
            pool1 = self.maxpool(conv1,           3) # H/4  -   64D
            conv2 = self.resblock(pool1,      64, 3, "conv2batch", batch_normalization_fn = None ) # H/8  -  256D
            conv3 = self.resblock(conv2,     128, 4, "conv3batch", batch_normalization_fn = None ) # H/16 -  512D
            conv4 = self.resblock(conv3,     256, 6, "conv4batch", batch_normalization_fn = None ) # H/32 - 1024D
            conv5 = self.resblock(conv4,     512, 3, "conv5batch", batch_normalization_fn = None ) # H/64 - 2048D

        with tf.variable_scope('skips'):
            skip1 = conv1
            skip2 = pool1
            skip3 = conv2
            skip4 = conv3
            skip5 = conv4
        
        # DECODING
        with tf.variable_scope('decoder'):
            upconv6 = upconv(conv5, 512, 3, 2, batch_normalization_fn = None ) #H/32
            concat6 = tf.concat([upconv6, skip5], 3)
            iconv6 = conv(concat6, 512, 3, 1, batch_normalization_fn = None )

            upconv5 = upconv(iconv6, 256, 3, 2, batch_normalization_fn = None ) #H/16
            concat5 = tf.concat([upconv5, skip4], 3)
            iconv5 = conv(concat5, 256, 3, 1, batch_normalization_fn = None )

            upconv4 = upconv(iconv5,  128, 3, 2, batch_normalization_fn = None ) #H/8
            concat4 = tf.concat([upconv4, skip3], 3)
            iconv4  = conv(concat4,   128, 3, 1, batch_normalization_fn = None )
            self.disp4 = self.get_disp(iconv4, batch_normalization_fn = None )
            udisp4 = self.upsample_nn(self.disp4, 2)

            upconv3 = upconv(iconv4,   64, 3, 2, batch_normalization_fn = None ) #H/4
            concat3 = tf.concat([upconv3, skip2, udisp4], 3)
            iconv3  = conv(concat3,    64, 3, 1, batch_normalization_fn = None )
            self.disp3 = self.get_disp(iconv3, batch_normalization_fn = None )
            udisp3 = self.upsample_nn(self.disp3, 2)

            upconv2 = upconv(iconv3,   32, 3, 2, batch_normalization_fn = None ) #H/2
            concat2 = tf.concat([upconv2, skip1, udisp3], 3)
            iconv2  = conv(concat2,    32, 3, 1, batch_normalization_fn = None )
            self.disp2 = self.get_disp(iconv2, batch_normalization_fn = None )
            udisp2 = self.upsample_nn(self.disp2, 2)

            upconv1 = upconv(iconv2,  16, 3, 2, batch_normalization_fn = None ) #H
            concat1 = tf.concat([upconv1, udisp2], 3)
            iconv1 = conv(concat1,   16, 3, 1, batch_normalization_fn = None )
            self.disp1 = self.get_disp(iconv1, batch_normalization_fn = None)
            self.classification = tf.nn.sigmoid(iconv1)


    def build_model(self):
        if self.mode=="train":
            is_training = True
        else:
            is_training = False

        batch_norm_params = {'is_training': is_training}
        self.normalizer_fn = slim.batch_norm if self.params.use_bn else None
        normalizer_params = batch_norm_params if self.params.use_bn else None

        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], activation_fn=tf.nn.elu,
                             normalizer_fn=self.normalizer_fn, normalizer_params=normalizer_params):
            with tf.variable_scope('discriminator', reuse=self.reuse_variables):
                self.left_pyramid = self.scale_pyramid(self.left, 4)
                if self.mode == 'train':
                    self.right_pyramid = self.scale_pyramid(self.right, 4)

                if self.params.do_stereo:
                    self.model_input = tf.concat([self.left, self.right], 3)
                else:
                    self.model_input = self.left

                #build model
                if self.params.encoder == 'vgg':
                    self.build_vgg()
                elif self.params.encoder == 'resnet50':
                    self.build_resnet50()
                    # self.build_resnet101()
                else:
                    return None

    def build_outputs(self):
        # STORE DISPARITIES
        with tf.variable_scope('disparities'):
            self.disp_est  = [self.disp1, self.disp2, self.disp3, self.disp4]
            self.disp_left_est  = [tf.expand_dims(d[:,:,:,0], 3) for d in self.disp_est]
            self.disp_right_est = [tf.expand_dims(d[:,:,:,1], 3) for d in self.disp_est]

        if self.mode == 'test' or self.mode == "export":
            return

        # if self.is_generator_in_use:
        #     self.left_pyramid = self.scale_pyramid(self.model_input,4)
        # GENERATE IMAGES
        with tf.variable_scope('images'):
            self.left_est  = [self.generate_image_left(self.right_pyramid[i], self.disp_left_est[i])  for i in range(4)]
            self.right_est = [self.generate_image_right(self.left_pyramid[i], self.disp_right_est[i]) for i in range(4)]

        # LR CONSISTENCY
        with tf.variable_scope('left-right'):
            self.right_to_left_disp = [self.generate_image_left(self.disp_right_est[i], self.disp_left_est[i])  for i in range(4)]
            self.left_to_right_disp = [self.generate_image_right(self.disp_left_est[i], self.disp_right_est[i]) for i in range(4)]

        # DISPARITY SMOOTHNESS
        with tf.variable_scope('smoothness'):
            self.disp_left_smoothness = self.get_disparity_smoothness(self.disp_left_est,  self.left_pyramid)
            self.disp_right_smoothness = self.get_disparity_smoothness(self.disp_right_est, self.right_pyramid)

    def build_losses(self):
        with tf.variable_scope('losses', reuse=self.reuse_variables):
            # IMAGE RECONSTRUCTION
            # L1
            self.l1_left = [tf.abs( self.left_est[i] - self.left_pyramid[i]) for i in range(4)]
            self.l1_reconstruction_loss_left = [tf.reduce_mean(l) for l in self.l1_left]
            self.l1_right = [tf.abs(self.right_est[i] - self.right_pyramid[i]) for i in range(4)]
            self.l1_reconstruction_loss_right = [tf.reduce_mean(l) for l in self.l1_right]

            # SSIM
            self.ssim_left = [self.SSIM(self.left_est[i],  self.left_pyramid[i]) for i in range(4)]
            self.ssim_loss_left = [tf.reduce_mean(s) for s in self.ssim_left]
            self.ssim_right = [self.SSIM(self.right_est[i], self.right_pyramid[i]) for i in range(4)]
            self.ssim_loss_right = [tf.reduce_mean(s) for s in self.ssim_right]

            # WEIGTHED SUM
            self.image_loss_right = [self.params.alpha_image_loss * self.ssim_loss_right[i]
                                     + (1 - self.params.alpha_image_loss) * self.l1_reconstruction_loss_right[i] for i in range(4)]
            self.image_loss_left  = [self.params.alpha_image_loss * self.ssim_loss_left[i]
                                     + (1 - self.params.alpha_image_loss) * self.l1_reconstruction_loss_left[i]  for i in range(4)]
            self.image_loss = tf.add_n(self.image_loss_left + self.image_loss_right)

            # DISPARITY SMOOTHNESS
            self.disp_left_loss  = [tf.reduce_mean(tf.abs(self.disp_left_smoothness[i]))  / 2 ** i for i in range(4)]
            self.disp_right_loss = [tf.reduce_mean(tf.abs(self.disp_right_smoothness[i])) / 2 ** i for i in range(4)]
            self.disp_gradient_loss = tf.add_n(self.disp_left_loss + self.disp_right_loss)

            # LR CONSISTENCY
            self.lr_left_loss  = [tf.reduce_mean(tf.abs(self.right_to_left_disp[i] - self.disp_left_est[i]))  for i in range(4)]
            self.lr_right_loss = [tf.reduce_mean(tf.abs(self.left_to_right_disp[i] - self.disp_right_est[i])) for i in range(4)]
            self.lr_loss = tf.add_n(self.lr_left_loss + self.lr_right_loss)

            # TOTAL LOSS
            self.discriminator_total_loss = self.image_loss + self.params.disp_gradient_loss_weight * self.disp_gradient_loss + self.params.lr_loss_weight * self.lr_loss

    def build_summaries(self):
        # SUMMARIES
        with tf.device('/cpu:0'):
            for i in range(4):
                tf.summary.scalar('ssim_loss_' + str(i), self.ssim_loss_left[i] + self.ssim_loss_right[i])
                tf.summary.scalar('l1_loss_' + str(i), self.l1_reconstruction_loss_left[i] + self.l1_reconstruction_loss_right[i])
                tf.summary.scalar('image_loss_' + str(i), self.image_loss_left[i] + self.image_loss_right[i])
                tf.summary.scalar('disp_gradient_loss_' + str(i), self.disp_left_loss[i] + self.disp_right_loss[i])
                tf.summary.scalar('lr_loss_' + str(i), self.lr_left_loss[i] + self.lr_right_loss[i])
                tf.summary.image('disp_left_est_' + str(i), self.disp_left_est[i], max_outputs=4)
                tf.summary.image('disp_right_est_' + str(i), self.disp_right_est[i], max_outputs=4)

                if self.params.full_summary:
                    tf.summary.image('left_est_' + str(i), self.left_est[i], max_outputs=4)
                    tf.summary.image('right_est_' + str(i), self.right_est[i], max_outputs=4)
                    tf.summary.image('ssim_left_'  + str(i), self.ssim_left[i],  max_outputs=4)
                    tf.summary.image('ssim_right_' + str(i), self.ssim_right[i], max_outputs=4)
                    tf.summary.image('l1_left_'  + str(i), self.l1_left[i],  max_outputs=4)
                    tf.summary.image('l1_right_' + str(i), self.l1_right[i], max_outputs=4)

            if self.params.full_summary:
                tf.summary.image('left',  self.left,   max_outputs=4)
                tf.summary.image('right', self.right,  max_outputs=4)

