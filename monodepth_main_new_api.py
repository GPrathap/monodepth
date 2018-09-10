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

from monodepth_generate_model_gan import MonodepthGenerateModel
from utils import save_images, image_manifold_size

os.environ['TF_CPP_MIN_LOG_LEVEL']='1'

import numpy as np
import argparse
import re
import time

import tensorflow.contrib.slim as slim

from monodepth_model_gan import *
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

tfgan = tf.contrib.gan


parser = argparse.ArgumentParser(description='Monodepth TensorFlow implementation.')
parser.add_argument('--mode',                      type=str,   help='train or test', default='train')
parser.add_argument('--model_name',                type=str,   help='model name', default='monodepth')
parser.add_argument('--encoder',                   type=str,   help='type of encoder, vgg or resnet50', default='resnet50')
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

        model_real = MonodepthModel(params, args.mode, left_splits,
                                                  right_splits, reuse_variables,None, 0)
        loss_discriminator_real = model_real.discriminator_total_loss
        real_feature_set = model_real.get_feature_set()


        model_fake = MonodepthModel(params, args.mode, left_splits_fake, right_splits,
                                    reuse_variables,None, 10)
        fake_feature_set = model_fake.get_feature_set()

        loss_discriminator = loss_discriminator_real
        generator_loss = tf.nn.l2_loss((real_feature_set - fake_feature_set))

        total_loss_generator = tf.reduce_mean(generator_loss) + wasserstein_generator_loss(model_fake.logistic)
        total_loss_discriminator = wasserstein_discriminator_loss(model_real.logistic, model_fake.logistic)+ \
                                    loss_discriminator


        gan_model = tfgan.gan_model(
            generator_fn=model_generator.generator,
            discriminator_fn=model_real.get_discriminator,
            real_data=left_splits,
            generator_inputs=z)



        improved_wgan_loss = tfgan.gan_loss(
            gan_model,
            # We make the loss explicit for demonstration, even though the default is
            # Wasserstein loss.
            generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
            discriminator_loss_fn=tfgan.losses.wasserstein_discriminator_loss,
            gradient_penalty_weight=1.0)


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
