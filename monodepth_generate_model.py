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
        self.model_index = model_index
        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn3 = batch_norm(name='g_bn3')
        self.build_model()
        self.build_summaries()

    def deconv(self, x, num_out_layers, kernel_size, scale):
        p_x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])
        conv = slim.conv2d_transpose(p_x, num_out_layers, kernel_size, scale, 'SAME')
        return conv[:,3:-1,3:-1,:]

    def conv_out_size_same(self, size, stride):
        return int(math.ceil(float(size) / float(stride)))

    def deconv2d(self, input_, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="deconv2d", with_w=False):
        with tf.variable_scope(name):
            # filter : [height, width, output_channels, in_channels]
            w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                                initializer=tf.random_normal_initializer(stddev=stddev))
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                            strides=[1, d_h, d_w, 1])
            biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
            deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
            if with_w:
                return deconv, w, biases
            else:
                return deconv

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

    def sampler(self, z, y=None):

            s_h, s_w = self.height, self.width
            s_h2, s_w2 = self.conv_out_size_same(s_h, 2), self.conv_out_size_same(s_w, 2)
            s_h4, s_w4 = self.conv_out_size_same(s_h2, 2), self.conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = self.conv_out_size_same(s_h4, 2), self.conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = self.conv_out_size_same(s_h8, 2), self.conv_out_size_same(s_w8, 2)
            s_h32, s_w32 = self.conv_out_size_same(s_h16, 2), self.conv_out_size_same(s_w16, 2)
            s_h64, s_w64 = self.conv_out_size_same(s_h32, 2), self.conv_out_size_same(s_w32, 2)

            # project `z` and reshape
            h0 = tf.reshape( self.linear(z, self.gf_dim * 32 * s_h64 * s_w64, 'g_h0_lin'),
                [-1, s_h64, s_w64, self.gf_dim * 32])
            h0 = tf.nn.elu(self.g_bn0(h0, train=False))

            h1 = self.deconv2d(h0, [self.batch_size, s_h32, s_w32, self.gf_dim * 16], name='g_h1')
            h1 = tf.nn.elu(self.g_bn1(h1, train=False))

            h1 = self.deconv2d(h1, [self.batch_size, s_h16, s_w16, self.gf_dim *8], name='g_h2')
            h1 = tf.nn.elu(self.g_bn1(h1, train=False))

            h1 = self.deconv2d(h1, [self.batch_size, s_h8, s_w8, self.gf_dim * 4], name='g_h3')
            h1 = tf.nn.elu(self.g_bn1(h1, train=False))

            h2 = self.deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2], name='g_h4')
            h2 = tf.nn.elu(self.g_bn2(h2, train=False))

            h3 = self.deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim * 1], name='g_h5')
            h3 = tf.nn.elu(self.g_bn3(h3, train=False))

            h4 = self.deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h6')

            return tf.nn.tanh(h4)

    def generator(self, z, y=None):
        # with tf.variable_scope("generator") as scope:
        s_h, s_w = self.height, self.width
        s_h2, s_w2 = self.conv_out_size_same(s_h, 2), self.conv_out_size_same(s_w, 2)
        s_h4, s_w4 = self.conv_out_size_same(s_h2, 2), self.conv_out_size_same(s_w2, 2)
        s_h8, s_w8 = self.conv_out_size_same(s_h4, 2), self.conv_out_size_same(s_w4, 2)
        s_h16, s_w16 = self.conv_out_size_same(s_h8, 2), self.conv_out_size_same(s_w8, 2)
        s_h32, s_w32 = self.conv_out_size_same(s_h16, 2), self.conv_out_size_same(s_w16, 2)
        s_h64, s_w64 = self.conv_out_size_same(s_h32, 2), self.conv_out_size_same(s_w32, 2)

        self.z_, self.h0_w, self.h0_b = self.linear(
            z, self.gf_dim * 32 * s_h64 * s_w64, 'g_h0_lin', with_w=True)

        self.h0 = tf.reshape(
            self.z_, [-1, s_h64, s_w64, self.gf_dim * 32])
        h0 = tf.nn.elu(self.g_bn0(self.h0))


        self.h11, self.h11_w, self.h11_b = self.deconv2d(
            h0, [self.batch_size, s_h32, s_w32, self.gf_dim * 16], name='g_h1', with_w=True)
        h11 = tf.nn.elu(self.g_bn1(self.h11))

        self.h12, self.h12_w, self.h12_b = self.deconv2d(
            h11, [self.batch_size, s_h16, s_w16, self.gf_dim * 8], name='g_h2', with_w=True)
        h12 = tf.nn.elu(self.g_bn1(self.h12))

        self.h13, self.h13_w, self.h13_b = self.deconv2d(
            h12, [self.batch_size, s_h8, s_w8, self.gf_dim * 4], name='g_h3', with_w=True)
        h13 = tf.nn.elu(self.g_bn1(self.h13))

        h2, self.h2_w, self.h2_b = self.deconv2d(
            h13, [self.batch_size, s_h4, s_w4, self.gf_dim * 2], name='g_h4', with_w=True)
        h2 = tf.nn.elu(self.g_bn2(h2))

        h3, self.h3_w, self.h3_b = self.deconv2d(
            h2, [self.batch_size, s_h2, s_w2, self.gf_dim * 1], name='g_h5', with_w=True)
        h3 = tf.nn.elu(self.g_bn3(h3))

        h4, self.h4_w, self.h4_b = self.deconv2d(
            h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h6', with_w=True)
        return tf.nn.tanh(h4)

    def build_model(self):
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], activation_fn=tf.nn.elu):
            with tf.variable_scope('generator', reuse=self.reuse_variables):
                self.generatored_model = self.generator(self.z_vector)
                self.samplter_network = self.sampler(self.z_vector)
                self.left = self.generatored_model
                self.model_input = self.left

    def get_model(self):
        return self.generatored_model

    def build_summaries(self):
        print("")
        # with tf.device('/cpu:0'):
        #         tf.summary.image('fake_image_' + str(self.model_index), self.model_input
        #                          , max_outputs=50, collections=self.model_collection)
