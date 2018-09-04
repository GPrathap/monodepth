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
                        'z_dim, '
                        'full_summary')

class MonodepthGenerateModel(object):
    """monodepth model"""
    def __init__(self, params, mode, z_vector, reuse_variables=None, model_index=0):
        self.params = params
        self.mode = mode
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
        self.z_vector = z_vector
        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn3 = batch_norm(name='g_bn3')
        self.build_model()
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

    def get_disp(self, x):
        disp = 0.3 * self.conv(x, 2, 3, 1, tf.nn.sigmoid)
        return disp

    def conv(self, x, num_out_layers, kernel_size, stride, activation_fn=tf.nn.elu):
        p = np.floor((kernel_size - 1) / 2).astype(np.int32)
        p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
        return slim.conv2d(p_x, num_out_layers, kernel_size, stride, 'VALID', activation_fn=activation_fn)

    def conv_block(self, x, num_out_layers, kernel_size):
        conv1 = self.conv(x,     num_out_layers, kernel_size, 1)
        conv2 = self.conv(conv1, num_out_layers, kernel_size, 2)
        return conv2

    def maxpool(self, x, kernel_size):
        p = np.floor((kernel_size - 1) / 2).astype(np.int32)
        p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
        return slim.max_pool2d(p_x, kernel_size)

    def resconv(self, x, num_layers, stride):
        do_proj = tf.shape(x)[3] != num_layers or stride == 2
        shortcut = []
        conv1 = self.conv(x,         num_layers, 1, 1)
        conv2 = self.conv(conv1,     num_layers, 3, stride)
        conv3 = self.conv(conv2, 4 * num_layers, 1, 1, None)
        if do_proj:
            shortcut = self.conv(x, 4 * num_layers, 1, stride, None)
        else:
            shortcut = x
        return tf.nn.elu(conv3 + shortcut)

    def resblock(self, x, num_layers, num_blocks):
        out = x
        for i in range(num_blocks - 1):
            out = self.resconv(out, num_layers, 1)
        out = self.resconv(out, num_layers, 2)
        return out

    def get_feature_set(self):
        # return tf.concat([tf.layers.flatten(self.disp1), tf.layers.flatten(self.disp2),
        #                   tf.layers.flatten(self.disp3), tf.layers.flatten(self.disp4)], axis=0)
        return tf.layers.flatten(self.disp2)

    def upconv(self, x, num_out_layers, kernel_size, scale):
        upsample = self.upsample_nn(x, scale)
        conv = self.conv(upsample, num_out_layers, kernel_size, 1)
        return conv

    def deconv(self, x, num_out_layers, kernel_size, scale):
        p_x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])
        conv = slim.conv2d_transpose(p_x, num_out_layers, kernel_size, scale, 'SAME')
        return conv[:,3:-1,3:-1,:]

    def conv_out_size_same(self, size, stride):
        return int(math.ceil(float(size) / float(stride)))

    # def deconv2d(self, input_, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="deconv2d", with_w=False):
    #     with tf.variable_scope(name):
    #         # filter : [height, width, output_channels, in_channels]
    #         w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
    #                             initializer=tf.random_normal_initializer(stddev=stddev))
    #         deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
    #                                         strides=[1, d_h, d_w, 1])
    #         biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
    #         deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
    #         if with_w:
    #             return deconv, w, biases
    #         else:
    #             return deconv

    def linear(self, input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
        shape = input_.get_shape().as_list()
        with tf.variable_scope(scope or "Linear"):
            try:
                matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                         tf.random_normal_initializer(stddev=stddev))
            except ValueError as err:
                msg = "NOTE: Usually, this is due to an issue with the image dimensions. " \
                      " Did you correctly set '--crop' or '--input_height' or '--output_height'?"
                err.args = err.args + (msg,)
                raise
            bias = tf.get_variable("bias", [output_size],
                                   initializer=tf.constant_initializer(bias_start))
            if with_w:
                return tf.matmul(input_, matrix) + bias, matrix, bias
            else:
                return tf.matmul(input_, matrix) + bias

    # def sampler(self, z, y=None):
    #     with tf.variable_scope("generator") as scope:
    #         scope.reuse_variables()
    #         s_h, s_w = self.height, self.width
    #         s_h2, s_w2 = self.conv_out_size_same(s_h, 2), self.conv_out_size_same(s_w, 2)
    #         s_h4, s_w4 = self.conv_out_size_same(s_h2, 2), self.conv_out_size_same(s_w2, 2)
    #         s_h8, s_w8 = self.conv_out_size_same(s_h4, 2), self.conv_out_size_same(s_w4, 2)
    #         s_h16, s_w16 = self.conv_out_size_same(s_h8, 2), self.conv_out_size_same(s_w8, 2)
    #
    #         # project `z` and reshape
    #         h0 = tf.reshape( self.linear(z, self.gf_dim * 8 * s_h16 * s_w16, 'g_h0_lin'),
    #             [-1, s_h16, s_w16, self.gf_dim * 8])
    #         h0 = tf.nn.relu(self.g_bn0(h0, train=False))
    #
    #         h1 = self.deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim * 4], name='g_h1')
    #         h1 = tf.nn.relu(self.g_bn1(h1, train=False))
    #
    #         h2 = self.deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2], name='g_h2')
    #         h2 = tf.nn.relu(self.g_bn2(h2, train=False))
    #
    #         h3 = self.deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim * 1], name='g_h3')
    #         h3 = tf.nn.relu(self.g_bn3(h3, train=False))
    #
    #         h4 = self.deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4')
    #
    #         return tf.nn.tanh(h4)
    #
    def generator(self, z, y=None):
        with tf.variable_scope("generator") as scope:
            s_h, s_w = self.height, self.width
            s_h2, s_w2 = self.conv_out_size_same(s_h, 2), self.conv_out_size_same(s_w, 2)
            s_h4, s_w4 = self.conv_out_size_same(s_h2, 2), self.conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = self.conv_out_size_same(s_h4, 2), self.conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = self.conv_out_size_same(s_h8, 2), self.conv_out_size_same(s_w8, 2)

            # project `z` and reshape
            self.z_, self.h0_w, self.h0_b = self.linear(
                z, self.gf_dim * 8 * s_h16 * s_w16, 'g_h0_lin', with_w=True)
            #
            h0 = tf.reshape(self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])
            # h0 = tf.nn.relu(self.h0)

            # h0 = tf.layers.dense(z, 4, activation=tf.identity)
            # h0 = tf.reshape(h0, shape=(-1, s_h16, s_w16, 3))

            # self.h1, self.h1_w, self.h1_b = self.deconv(
            #     h0, [self.batch_size, s_h8, s_w8, self.gf_dim * 4], name='g_h1', with_w=True)
            # h1 = tf.nn.relu(self.g_bn1(self.h1))
            #
            # h2, self.h2_w, self.h2_b = self.deconv(
            #     h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2], name='g_h2', with_w=True)
            # h2 = tf.nn.relu(self.g_bn2(h2))
            #
            # h3, self.h3_w, self.h3_b = self.deconv(
            #     h2, [self.batch_size, s_h2, s_w2, self.gf_dim * 1], name='g_h3', with_w=True)
            # h3 = tf.nn.relu(self.g_bn3(h3))
            #
            # h4, self.h4_w, self.h4_b = self.deconv(
            #     h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4', with_w=True)
            h1 = self.deconv(h0, s_h8, 3, 2)
            # h1 = tf.nn.relu(h1)

            h2 = self.deconv(h1, s_h4, 3, 2)
            # h2 = tf.nn.relu(h2)

            h3 = self.deconv(h2, s_h2, 3, 2)
            # h3 = tf.nn.relu(h3)

            h4 = self.deconv(h3, s_h, 3, 2)

            h5 = tf.layers.conv2d_transpose(h4, 3, [1, 1], strides=(1, 1), padding='SAME'
                                                , activation=tf.identity)

            return tf.nn.tanh(h5)

    def conv_cond_concat(self, x, y):
        x_shapes = x.get_shape()
        y_shapes = y.get_shape()
        return tf.concat([
            x, y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)

    def build_model(self):
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], activation_fn=tf.nn.elu):
            with tf.variable_scope('generator', reuse=self.reuse_variables):
                # self.left_pyramid = self.scale_pyramid(self.left, 4)
                # self.model_input = self.left
                self.generatored_model = self.generator(self.z_vector)
                self.left = self.generatored_model
                self.model_input = self.left

    def get_model(self):
        return self.generatored_model

    def build_summaries(self):
        # SUMMARIES
        with tf.device('/cpu:0'):
            print ("Still no summary has been added!")
        #     for i in range(4):
        #         tf.summary.scalar('ssim_loss_' + str(i), self.ssim_loss_left[i] + self.ssim_loss_right[i], collections=self.model_collection)
        #         tf.summary.scalar('l1_loss_' + str(i), self.l1_reconstruction_loss_left[i] + self.l1_reconstruction_loss_right[i], collections=self.model_collection)
        #         tf.summary.scalar('image_loss_' + str(i), self.image_loss_left[i] + self.image_loss_right[i], collections=self.model_collection)
        #         tf.summary.scalar('disp_gradient_loss_' + str(i), self.disp_left_loss[i] + self.disp_right_loss[i], collections=self.model_collection)
        #         tf.summary.scalar('lr_loss_' + str(i), self.lr_left_loss[i] + self.lr_right_loss[i], collections=self.model_collection)
        #         tf.summary.image('disp_left_est_' + str(i), self.disp_left_est[i], max_outputs=4, collections=self.model_collection)
        #         tf.summary.image('disp_right_est_' + str(i), self.disp_right_est[i], max_outputs=4, collections=self.model_collection)
        #
        #         if self.params.full_summary:
        #             tf.summary.image('left_est_' + str(i), self.left_est[i], max_outputs=4, collections=self.model_collection)
        #             tf.summary.image('right_est_' + str(i), self.right_est[i], max_outputs=4, collections=self.model_collection)
        #             tf.summary.image('ssim_left_'  + str(i), self.ssim_left[i],  max_outputs=4, collections=self.model_collection)
        #             tf.summary.image('ssim_right_' + str(i), self.ssim_right[i], max_outputs=4, collections=self.model_collection)
        #             tf.summary.image('l1_left_'  + str(i), self.l1_left[i],  max_outputs=4, collections=self.model_collection)
        #             tf.summary.image('l1_right_' + str(i), self.l1_right[i], max_outputs=4, collections=self.model_collection)
        #
        #     if self.params.full_summary:
        #         tf.summary.image('left',  self.left,   max_outputs=4, collections=self.model_collection)
        #         tf.summary.image('right', self.right,  max_outputs=4, collections=self.model_collection)

