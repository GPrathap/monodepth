# Copyright UCL Business plc 2017. Patent Pending. All rights reserved.
#
# The MonoDepth Software is licensed under the terms of the UCLB ACP-A licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
#
# For any other use of the software not covered by the UCLB ACP-A Licence,
# please contact info@uclb.com

# from __future__ import absolute_import, division, print_function

# only keep warnings and errors
import os

from monodepth_generate_model import MonodepthGenerateModel
from utils import save_images, image_manifold_size
from utilsgan import params_with_name

os.environ['TF_CPP_MIN_LOG_LEVEL']='1'

import numpy as np
import argparse
import re
import time

import tensorflow.contrib.slim as slim

from monodepth_model import *
from monodepth_dataloader import *
from average_gradients import *

image_summary = tf.summary.image


from tensorflow.contrib.framework.python.ops import variables as contrib_variables_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops.distributions import distribution as ds
from tensorflow.python.ops.losses import losses
from tensorflow.python.ops.losses import util
from tensorflow.python.summary import summary

parser = argparse.ArgumentParser(description='Monodepth TensorFlow implementation.')
parser.add_argument('--mode',                      type=str,   help='train or test', default='train')
parser.add_argument('--model_name',                type=str,   help='model name', default='monodepth')
parser.add_argument('--encoder',                   type=str,   help='type of encoder, vgg or resnet50', default='vgg')
parser.add_argument('--dataset',                   type=str,   help='dataset to train on, kitti, or cityscapes', default='kitti')
parser.add_argument('--z_dim',                     type=int,   help='default set to 100', default=100)
parser.add_argument('--data_path',                 type=str,   help='path to the data', default="/home/a.gabdullin/geesara/2011_kia/")
parser.add_argument('--filenames_file',            type=str,   help='path to the filenames text file', default="./utils/filenames/kitti_train_files.txt")
parser.add_argument('--input_height',              type=int,   help='input height', default=256)
parser.add_argument('--input_width',               type=int,   help='input width', default=512)
parser.add_argument('--batch_size',                type=int,   help='batch size', default=8)
parser.add_argument('--num_epochs',                type=int,   help='number of epochs', default=50)
parser.add_argument('--learning_rate',             type=float, help='initial learning rate', default=1e-4)
parser.add_argument('--lr_loss_weight',            type=float, help='left-right consistency weight', default=1.0)
parser.add_argument('--alpha_image_loss',          type=float, help='weight between SSIM and L1 in the image loss', default=0.85)
parser.add_argument('--disp_gradient_loss_weight', type=float, help='disparity smoothness weigth', default=0.1)
parser.add_argument('--do_stereo',                             help='if set, will train the stereo model', action='store_true')
parser.add_argument('--wrap_mode',                 type=str,   help='bilinear sampler wrap mode, edge or border', default='border')
parser.add_argument('--use_deconv',                            help='if set, will use transposed convolutions', action='store_true')
parser.add_argument('--num_gpus',                  type=int,   help='number of GPUs to use for training', default=1)
parser.add_argument('--num_threads',               type=int,   help='number of threads to use for data loading', default=2)
parser.add_argument('--output_directory',          type=str,   help='output directory for test disparities, if empty outputs to checkpoint folder', default='')
parser.add_argument('--log_directory',             type=str,   help='directory to save checkpoints and summaries', default='o')
parser.add_argument('--checkpoint_path',           type=str,   help='path to a specific checkpoint to load', default='')
parser.add_argument('--retrain',                               help='if used with checkpoint_path, will restart training from step zero', action='store_true')
parser.add_argument('--full_summary',                          help='if set, will keep more data for each summary. Warning: the file can become very large', action='store_true')
parser.add_argument('--sample_dir',                type=str,   help='sample directory', default='./dataset/images')

args = parser.parse_args()

def post_process_disparity(disp):
    _, h, w = disp.shape
    l_disp = disp[0,:,:]
    r_disp = np.fliplr(disp[1,:,:])
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

def count_text_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return len(lines)

def data_preprocessing_for_wasserstein_loss(real_data, generated_data):
    # if real_data.shape.ndims is None:
    #     raise ValueError('`real_data` can\'t have unknown rank.')
    # if generated_data.shape.ndims is None:
    #     raise ValueError('`generated_data` can\'t have unknown rank.')

    differences = tf.subtract(generated_data, real_data)
    batch_size = differences.shape[0].value or array_ops.shape(differences)[0]
    alpha_shape = [batch_size] + [1] * (differences.shape.ndims - 1)
    alpha = random_ops.random_uniform(shape=alpha_shape)
    interpolates = real_data + (alpha * differences)
    return interpolates

def em_loss(y_coefficients, y_pred):
    return tf.reduce_mean(tf.multiply(y_coefficients, y_pred))

# Wasserstein losses from `Wasserstein GAN` (https://arxiv.org/abs/1701.07875).
def wasserstein_generator_loss(
    discriminator_gen_outputs,
    weights=1.0,
    scope=None,
    loss_collection=ops.GraphKeys.LOSSES,
    reduction=losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
    add_summaries=False):
  with ops.name_scope(scope, 'generator_wasserstein_loss', (
      discriminator_gen_outputs, weights)) as scope:
    discriminator_gen_outputs = math_ops.to_float(discriminator_gen_outputs)

    loss = - discriminator_gen_outputs
    loss = losses.compute_weighted_loss(
        loss, weights, scope, loss_collection, reduction)

  summary.scalar('generator_wass_loss', loss)

  return loss


def wasserstein_discriminator_loss(
    discriminator_real_outputs,
    discriminator_gen_outputs,
    real_weights=1.0,
    generated_weights=1.0,
    scope=None,
    loss_collection=ops.GraphKeys.LOSSES,
    reduction=losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
    add_summaries=False):
  with ops.name_scope(scope, 'discriminator_wasserstein_loss', (
      discriminator_real_outputs, discriminator_gen_outputs, real_weights,
      generated_weights)) as scope:
    discriminator_real_outputs = math_ops.to_float(discriminator_real_outputs)
    discriminator_gen_outputs = math_ops.to_float(discriminator_gen_outputs)
    discriminator_real_outputs.shape.assert_is_compatible_with(
        discriminator_gen_outputs.shape)

    loss_on_generated = losses.compute_weighted_loss(
        discriminator_gen_outputs, generated_weights, scope,
        loss_collection=None, reduction=reduction)
    loss_on_real = losses.compute_weighted_loss(
        discriminator_real_outputs, real_weights, scope, loss_collection=None,
        reduction=reduction)
    loss = loss_on_generated - loss_on_real
    util.add_loss(loss, loss_collection)


  # summary.scalar('discriminator_gen_wass_loss', loss_on_generated)
  # summary.scalar('discriminator_real_wass_loss', loss_on_real)
  # summary.scalar('discriminator_wass_loss', loss)

  return loss

def train(params):
    """Training loop."""
    # with tf.Graph().as_default(), tf.device('/cpu:0'):
    with tf.Graph().as_default(), tf.device('/device:GPU:0'):
        global_step = tf.Variable(0, trainable=False)
        # OPTIMIZER
        num_training_samples = count_text_lines(args.filenames_file)
        steps_per_epoch = np.ceil(num_training_samples / params.batch_size).astype(np.int32)
        num_total_steps = params.num_epochs * steps_per_epoch
        start_learning_rate = args.learning_rate
        boundaries = [np.int32((3/5) * num_total_steps), np.int32((4/5) * num_total_steps)]
        values = [args.learning_rate, args.learning_rate / 2, args.learning_rate / 4]
        learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
        print("Total number of samples: {}".format(num_training_samples))
        print("Total number of steps: {}".format(num_total_steps))

        z = tf.placeholder(tf.float32, [params.batch_size, params.z_dim], name='z_noise')
        dataloader = MonodepthDataloader(args.data_path, args.filenames_file, params, args.dataset, args.mode)
        left = dataloader.left_image_batch
        right = dataloader.right_image_batch

        fake_generated_left_image = []
        reuse_variables = tf.AUTO_REUSE
        # split for each gpu
        model_generator = MonodepthGenerateModel(params, args.mode, z, reuse_variables, 0)
        left_splits = tf.split(left,  args.num_gpus, 0)[0]
        left_splits_fake = tf.split(model_generator.get_model(), args.num_gpus, 0)[0]
        right_splits = tf.split(right, args.num_gpus, 0)[0]

        with tf.variable_scope('discriminator', reuse=reuse_variables):
                differences = tf.subtract(left_splits_fake, left_splits)
                alpha_shape = [params.batch_size] + [1] * (differences.shape.ndims - 1)
                alpha = tf.random_uniform(shape=alpha_shape, minval=0., maxval=1.)
                interpolates = left_splits + (alpha * differences)
                left_splits_wasserstein_model = MonodepthModel(params, args.mode, interpolates,                                                               right_splits, reuse_variables, left_splits_fake, 1)
                gradients = tf.gradients(left_splits_wasserstein_model.logistic,
                                     [interpolates], stop_gradients=interpolates, colocate_gradients_with_ops=True)[0]
                slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
                gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
                _gradient_penalty = 10 * gradient_penalty

        with tf.variable_scope('discriminator', reuse=reuse_variables):
            model_real = MonodepthModel(params, args.mode, left_splits,
                                                  right_splits, reuse_variables,None, 0)
        loss_discriminator_real = model_real.discriminator_total_loss
        real_feature_set = model_real.get_feature_set()

        with tf.variable_scope('discriminator', reuse=reuse_variables):
            model_fake = MonodepthModel(params, args.mode, left_splits_fake, right_splits,
                                    reuse_variables,None, 10)

        fake_feature_set = model_fake.get_feature_set()
        loss_discriminator = loss_discriminator_real
        generator_loss = tf.nn.l2_loss((real_feature_set - fake_feature_set))
        # total_loss_generator = tf.reduce_mean(generator_loss) + wasserstein_generator_loss(model_fake.logistic_linear)
        # total_loss_discriminator = wasserstein_discriminator_loss(model_real.logistic_linear, model_fake.logistic_linear)+ \
        #                             loss_discriminator + _gradient_penalty
        total_loss_generator = tf.reduce_mean(generator_loss)-tf.reduce_mean(model_fake.logistic)
        total_loss_discriminator = tf.reduce_mean(model_fake.logistic) - tf.reduce_mean(model_real.logistic) \
                                   + loss_discriminator + _gradient_penalty

        # with tf.device('/device:GPU:0'):
        opt_discriminator_step = tf.train.AdamOptimizer(learning_rate)
        opt_generator_step = tf.train.AdamOptimizer(learning_rate)

        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator/*")
            d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator/*")
            d_optim = opt_discriminator_step.minimize(total_loss_discriminator, var_list=d_vars)
            g_optim = opt_generator_step.minimize(total_loss_generator, var_list=g_vars)

        tf.summary.scalar('learning_rate', learning_rate, ['discriminator_0'])
        tf.summary.scalar('total_loss', total_loss_discriminator, ['discriminator_0'])
        summary_op = tf.summary.merge_all('discriminator_0')

        # SESSION
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # SAVER
        summary_writer = tf.summary.FileWriter(args.log_directory + '/' + args.model_name, sess.graph)
        train_saver = tf.train.Saver()

        # COUNT PARAMS
        total_num_parameters = 0
        for variable in tf.trainable_variables():
            total_num_parameters += np.array(variable.get_shape().as_list()).prod()
        print("Number of trainable parameters: {}".format(total_num_parameters))

        # INIT
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coordinator = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

        # LOAD CHECKPOINT IF SET
        if args.checkpoint_path != '':
            train_saver.restore(sess, args.checkpoint_path.split(".")[0])

            if args.retrain:
                sess.run(global_step.assign(0))

        # GO!
        start_step = global_step.eval(session=sess)
        start_time = time.time()
        sample_dataset = np.random.uniform(low=-1, high=1, size=(params.batch_size, params.z_dim)).astype(np.float32)

        for step in range(start_step, num_total_steps):
            before_op_time = time.time()

            batch_z = np.random.uniform(low=-1, high=1, size=(params.batch_size, params.z_dim)).astype(np.float32)
            _, loss_value_discriminator = sess.run([d_optim, total_loss_discriminator], feed_dict={z: batch_z})
            # _, loss_value_discriminator, images_original = sess.run([d_optim, total_loss_discriminator, dataloader.left_image_batch], feed_dict={z: batch_z})
            # print("size-------> {}".format(images_original))
            for _ in range(2):
                _, loss_value_generator, generated_images = sess.run([g_optim, total_loss_generator, model_generator.samplter_network],
                                               feed_dict={z: batch_z})
            duration = time.time() - before_op_time
            if step and step % 100 == 0:
                _, loss_value_generator, generated_images = sess.run(
                    [g_optim, total_loss_generator, model_generator.samplter_network],
                    feed_dict={z: sample_dataset})
                save_images(generated_images, image_manifold_size(generated_images.shape[0]),
                             '{}/train_{:02d}_{:04d}.png'.format(params.sample_dir, step, 1))
                examples_per_sec = params.batch_size / duration
                time_sofar = (time.time() - start_time) / 3600
                training_time_left = (num_total_steps / step - 1.0) * time_sofar
                print_string = 'batch {:>6} | examples/s: {:4.2f} | loss_discriminator: {:.5f} | time elapsed: {:.2f}h ' \
                               '| time left: {:.2f}h'
                print(print_string.format(step, examples_per_sec, loss_value_discriminator, time_sofar,
                                          training_time_left))
                print_string = 'batch {:>6} | examples/s: {:4.2f} | loss_generator: {:.5f} | time elapsed: {:.2f}h | ' \
                               'time left: {:.2f}h'
                print(print_string.format(step, examples_per_sec, loss_value_generator, time_sofar,
                                          training_time_left))
                summary_str = sess.run(summary_op, feed_dict={z: batch_z})
                summary_writer.add_summary(summary_str, global_step=step)
            if step and step % 10000 == 0:
                train_saver.save(sess, args.log_directory + '/' + args.model_name + '/model', global_step=step)

        train_saver.save(sess, args.log_directory + '/' + args.model_name + '/model', global_step=num_total_steps)

def model_validate(params):
        """Test function."""
        dataloader = MonodepthDataloader(args.data_path, args.filenames_file, params, args.dataset, args.mode)
        left  = dataloader.left_image_batch
        right = dataloader.right_image_batch

        model = MonodepthModel(params, args.mode, left, right)

        # SESSION
        config = tf.ConfigProto(allow_soft_placement=True)
        sess = tf.Session(config=config)

        # SAVER
        train_saver = tf.train.Saver()

        # INIT
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coordinator = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

        # RESTORE
        if args.checkpoint_path == '':
            restore_path = tf.train.latest_checkpoint(args.log_directory + '/' + args.model_name)
        else:
            restore_path = args.checkpoint_path.split(".")[0]
        train_saver.restore(sess, restore_path)

        num_test_samples = count_text_lines(args.filenames_file)

        print('now testing {} files'.format(num_test_samples))
        disparities = np.zeros((num_test_samples, params.height, params.width), dtype=np.float32)
        disparities_pp = np.zeros((num_test_samples, params.height, params.width), dtype=np.float32)
        for step in range(num_test_samples):
            disp = sess.run(model.disp_left_est[0])
            disparities[step] = disp[0].squeeze()
            disparities_pp[step] = post_process_disparity(disp.squeeze())

        print('done.')

        print('writing disparities.')
        if args.output_directory == '':
            output_directory = os.path.dirname(args.checkpoint_path)
        else:
            output_directory = args.output_directory
        np.save(output_directory + '/disparities.npy',    disparities)
        np.save(output_directory + '/disparities_pp.npy', disparities_pp)

        print('done.')

def main():
    params = monodepth_parameters(
        encoder=args.encoder,
        height=args.input_height,
        width=args.input_width,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
        num_epochs=args.num_epochs,
        do_stereo=args.do_stereo,
        wrap_mode=args.wrap_mode,
        use_deconv=args.use_deconv,
        alpha_image_loss=args.alpha_image_loss,
        disp_gradient_loss_weight=args.disp_gradient_loss_weight,
        lr_loss_weight=args.lr_loss_weight,
        full_summary=args.full_summary,
        sample_dir=args.sample_dir,
        z_dim=100)

    if args.mode == 'train':
        train(params)
    elif args.mode == 'test':
        model_validate(params)

# if __name__ == '__main__':
#     tf.app.run()

main()
