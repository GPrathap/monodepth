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

from tensorflow.python.saved_model import signature_def_utils, signature_constants, tag_constants

from monodepth_generate_model import MonodepthGenerateModel
from utils import save_images, image_manifold_size

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

parser = argparse.ArgumentParser(description='Monodepth TensorFlow implementation.')
parser.add_argument('--mode',                      type=str,   help='train or test', default='train')
parser.add_argument('--model_name',                type=str,   help='model name', default='monodepth')
parser.add_argument('--encoder',                   type=str,   help='type of encoder, vgg or resnet50', default='resnet50')
parser.add_argument('--dataset',                   type=str,   help='datasetuhuu  quruqr to train on, kitti, or cityscapes', default='kitti')
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
parser.add_argument('--use_bn',                type=bool,   help='is using batch normalization', default=True)

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

        left_splits = tf.split(left,  args.num_gpus, 0)
        left_splits_fake = tf.split(model_generator.get_model(), args.num_gpus, 0)
        right_splits = tf.split(right, args.num_gpus, 0)

        model_real = MonodepthModel(params, args.mode, left_splits[0],
                                                  right_splits[0], reuse_variables, 0)
        loss_discriminator_real = model_real.discriminator_total_loss
        real_feature_set = model_real.get_feature_set()

        model_fake = MonodepthModel(params, args.mode, left_splits_fake[0], right_splits[0],
                                    reuse_variables, 10)

        fake_feature_set = model_fake.get_feature_set()
        #loss_discriminator_fake = model_fake.discriminator_total_loss

        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=model_real.classification, labels=tf.ones_like(model_real.classification)))  # real == 1
        # discriminator: images from generator (fake) are labelled as 0
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=model_fake.classification, labels=tf.zeros_like(model_fake.classification)))  # fake == 0
        loss_discriminator = loss_discriminator_real

        g_loss1 = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=model_fake.classification, labels=tf.ones_like(model_fake.classification)))

        generator_loss = tf.nn.l2_loss((real_feature_set - fake_feature_set))
        total_loss_discriminator = tf.reduce_mean(loss_discriminator) + d_loss_real + d_loss_fake
        total_loss_generator = tf.reduce_mean(generator_loss) + g_loss1

        opt_discriminator_step = tf.train.AdamOptimizer(learning_rate)
        opt_generator_step = tf.train.AdamOptimizer(learning_rate)

        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator/*")
            d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator/*")
            d_optim = opt_discriminator_step.minimize(total_loss_discriminator, var_list=d_vars)
            g_optim = opt_generator_step.minimize(total_loss_generator, var_list=g_vars)

        tf.summary.scalar('learning_rate', learning_rate)
        tf.summary.scalar('total_loss', total_loss_discriminator)
        summary_op = tf.summary.merge_all()

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
                             './{}/{:02d}_train_{:04d}.png'.format(params.sample_dir, step, 1))
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

        # left = tf.split(left, args.num_gpus, 0)[0]
        # right = tf.split(right, args.num_gpus, 0)[0]

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
        # if args.checkpoint_path == '':
        #     restore_path = tf.train.latest_checkpoint(args.log_directory + '/' + args.model_name)
        #     print("Model name {} ".format(restore_path))
        # else:
        #     restore_path = args.checkpoint_path.split(".")[0]
        restore_path = "/home/a.gabdullin/geesara/monodepth/o/monodepth/model-50000"
        train_saver.restore(sess, restore_path)
        # train_saver.restore(sess, restore_path)

        num_test_samples = count_text_lines(args.filenames_file)

        print('Now testing {} files'.format(num_test_samples))
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
            print("ffffff")
        output_directory = args.log_directory + '/' + args.model_name
        np.save(output_directory + '/disparities.npy',    disparities)
        np.save(output_directory + '/disparities_pp.npy', disparities_pp)

        print('done.')


def export_model(params):
    """Test function."""
    dataloader = MonodepthDataloader(args.data_path, args.filenames_file, params, args.dataset, args.mode)
    left = dataloader.left_image_batch
    right = dataloader.right_image_batch

    # left = tf.split(left, args.num_gpus, 0)[0]
    # right = tf.split(right, args.num_gpus, 0)[0]

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
    # if args.checkpoint_path == '':
    #     restore_path = tf.train.latest_checkpoint(args.log_directory + '/' + args.model_name)
    #     print("Model name {} ".format(restore_path))
    # else:
    #     restore_path = args.checkpoint_path.split(".")[0]
    restore_path = "/home/a.gabdullin/geesara/monodepth/o/monodepth/model-50000"
    train_saver.restore(sess, restore_path)
    # train_saver.restore(sess, restore_path)

    signature = signature_def_utils.build_signature_def(
        inputs=model.left,
        outputs=model.disp_left_est,
        method_name=signature_constants.PREDICT_METHOD_NAME)

    signature_map = {signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                         signature}

    model_builder = tf.saved_model.builder.SavedModelBuilder("/home/a.gabdullin/geesara/monodepth/o/monodepth")
    model_builder.add_meta_graph_and_variables(sess,
                                               tags=[tag_constants.SERVING],
                                               signature_def_map=signature_map,
                                               clear_devices=True)
    model_builder.save()

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
        use_bn=True,
        z_dim=100)

    if args.mode == 'train':
        train(params)
    elif args.mode == 'test':
        model_validate(params)
    elif args.mode == 'export':
        export_model(params)

# if __name__ == '__main__':
#     tf.app.run()

main()
