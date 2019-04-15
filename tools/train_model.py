#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-7-3 下午4:31
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/attentive-gan-derainnet
# @File    : train_model.py.py
# @IDE: PyCharm
"""
模型训练脚本
"""
import os
import os.path as ops
import argparse
import time

import tensorflow as tf
import numpy as np
import glog as log

from data_provider import data_feed_pipline
from config import global_config
from attentive_gan_model import derain_drop_net

CFG = global_config.cfg
VGG_MEAN = [103.939, 116.779, 123.68]


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, help='The dataset dir')
    parser.add_argument('--weights_path', type=str,
                        help='The pretrained weights path', default=None)

    return parser.parse_args()


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)

    return average_grads


def compute_net_gradients(images, labels, net, optimizer=None, is_net_first_initialized=False):
    """
    Calculate gradients for single GPU
    :param images: images for training
    :param labels: labels corresponding to images
    :param net: classification model
    :param optimizer: network optimizer
    :param is_net_first_initialized: if the network is initialized
    :return:
    """
    net_loss = net.compute_loss(input_tensor=images,
                                labels=labels,
                                name='nsfw_cls_model',
                                reuse=is_net_first_initialized)
    if optimizer is not None:
        grads = optimizer.compute_gradients(net_loss)
    else:
        grads = None

    return net_loss, grads


def train_model(dataset_dir, weights_path=None):
    """

    :param dataset_dir:
    :param gpu_id:
    :param weights_path:
    :return:
    """
    # 构建数据集
    with tf.device('/cpu:0'):

        train_dataset = data_feed_pipline.DerainDataFeeder(dataset_dir=dataset_dir, flags='train')
        val_dataset = data_feed_pipline.DerainDataFeeder(dataset_dir=dataset_dir, flags='val')

        train_input_tensor, train_label_tensor, train_mask_tensor = train_dataset.inputs(CFG.TRAIN.BATCH_SIZE, 1)
        val_input_tensor, val_label_tensor, val_mask_tensor = val_dataset.inputs(CFG.TRAIN.BATCH_SIZE, 1)

    with tf.device('/gpu:1'):

        # define network
        derain_net = derain_drop_net.DeRainNet(phase=tf.constant('train', dtype=tf.string))

        # calculate train loss and validation loss
        train_gan_loss, train_discriminative_loss, train_net_output = derain_net.compute_loss(
            input_tensor=train_input_tensor,
            gt_label_tensor=train_label_tensor,
            mask_label_tensor=train_mask_tensor,
            name='derain_net',
            reuse=False
        )

        val_gan_loss, val_discriminative_loss, val_net_output = derain_net.compute_loss(
            input_tensor=val_input_tensor,
            gt_label_tensor=val_label_tensor,
            mask_label_tensor=val_mask_tensor,
            name='derain_net',
            reuse=True
        )

        # calculate train ssim, psnr and validation ssim, psnr
        train_label_tensor_scale = tf.image.convert_image_dtype(
            image=(train_label_tensor + 1.0) / 2.0,
            dtype=tf.uint8
        )
        train_net_output_tensor_scale = tf.image.convert_image_dtype(
            image=(train_net_output + 1.0) / 2.0,
            dtype=tf.uint8
        )
        val_label_tensor_scale = tf.image.convert_image_dtype(
            image=(val_label_tensor + 1.0) / 2.0,
            dtype=tf.uint8
        )
        val_net_output_tensor_scale = tf.image.convert_image_dtype(
            image=(val_net_output + 1.0) / 2.0,
            dtype=tf.uint8
        )

        train_label_tensor_scale = tf.image.rgb_to_grayscale(
            images=tf.reverse(train_label_tensor_scale, axis=[-1])
        )
        train_net_output_tensor_scale = tf.image.rgb_to_grayscale(
            images=tf.reverse(train_net_output_tensor_scale, axis=[-1])
        )
        val_label_tensor_scale = tf.image.rgb_to_grayscale(
            images=tf.reverse(val_label_tensor_scale, axis=[-1])
        )
        val_net_output_tensor_scale = tf.image.rgb_to_grayscale(
            images=tf.reverse(val_net_output_tensor_scale, axis=[-1])
        )

        train_ssim = tf.reduce_mean(tf.image.ssim(
            train_label_tensor_scale, train_net_output_tensor_scale, max_val=255),
            name='avg_train_ssim'
        )
        train_psnr = tf.reduce_mean(tf.image.psnr(
            train_label_tensor_scale, train_net_output_tensor_scale, max_val=255),
            name='avg_train_psnr'
        )
        val_ssim = tf.reduce_mean(tf.image.ssim(
            val_label_tensor_scale, val_net_output_tensor_scale, max_val=255),
            name='avg_val_ssim'
        )
        val_psnr = tf.reduce_mean(tf.image.psnr(
            val_label_tensor_scale, val_net_output_tensor_scale, max_val=255),
            name='avg_val_psnr'
        )

        # collect trainable vars to update
        train_vars = tf.trainable_variables()

        d_vars = [tmp for tmp in train_vars if 'discriminative_loss' in tmp.name]
        g_vars = [tmp for tmp in train_vars if 'attentive_' in tmp.name and 'vgg_feats' not in tmp.name]
        vgg_vars = [tmp for tmp in train_vars if "vgg_feats" in tmp.name]

        # set optimizer
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(CFG.TRAIN.LEARNING_RATE, global_step,
                                                   100000, 0.1, staircase=True)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            d_optim = tf.train.AdamOptimizer(learning_rate).minimize(
                train_discriminative_loss, var_list=d_vars)
            g_optim = tf.train.MomentumOptimizer(
                learning_rate=learning_rate,
                momentum=tf.constant(0.9, tf.float32)).minimize(train_gan_loss, var_list=g_vars)

        # Set tf saver
        saver = tf.train.Saver()
        model_save_dir = 'model/derain_gan'
        if not ops.exists(model_save_dir):
            os.makedirs(model_save_dir)
        train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        model_name = 'derain_gan_{:s}.ckpt'.format(str(train_start_time))
        model_save_path = ops.join(model_save_dir, model_name)

        # Set tf summary
        tboard_save_path = 'tboard/derain_gan'
        if not ops.exists(tboard_save_path):
            os.makedirs(tboard_save_path)

        train_g_loss_scalar = tf.summary.scalar(name='train_gan_loss', tensor=train_gan_loss)
        train_d_loss_scalar = tf.summary.scalar(name='train_discriminative_loss', tensor=train_discriminative_loss)
        train_ssim_scalar = tf.summary.scalar(name='train_image_ssim', tensor=train_ssim)
        train_psnr_scalar = tf.summary.scalar(name='train_image_psnr', tensor=train_psnr)
        val_g_loss_scalar = tf.summary.scalar(name='val_gan_loss', tensor=val_gan_loss)
        val_d_loss_scalar = tf.summary.scalar(name='val_discriminative_loss', tensor=val_discriminative_loss)
        val_ssim_scalar = tf.summary.scalar(name='val_image_ssim', tensor=val_ssim)
        val_psnr_scalar = tf.summary.scalar(name='val_image_psnr', tensor=val_psnr)

        lr_scalar = tf.summary.scalar(name='learning_rate', tensor=learning_rate)

        train_summary_op = tf.summary.merge(
            [train_g_loss_scalar, train_d_loss_scalar, train_ssim_scalar, train_psnr_scalar, lr_scalar]
        )
        val_summary_op = tf.summary.merge(
            [val_g_loss_scalar, val_d_loss_scalar, val_ssim_scalar, val_psnr_scalar]
        )

        # Set sess configuration
        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TRAIN.GPU_MEMORY_FRACTION
        sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
        sess_config.gpu_options.allocator_type = 'BFC'

        sess = tf.Session(config=sess_config)

        summary_writer = tf.summary.FileWriter(tboard_save_path)
        summary_writer.add_graph(sess.graph)

        # Set the training parameters
        train_epochs = CFG.TRAIN.EPOCHS

        log.info('Global configuration is as follows:')
        log.info(CFG)

        with sess.as_default():

            tf.train.write_graph(graph_or_graph_def=sess.graph, logdir='',
                                 name='{:s}/derain_gan.pb'.format(model_save_dir))

            if weights_path is None:
                log.info('Training from scratch')
                init = tf.global_variables_initializer()
                sess.run(init)
            else:
                log.info('Restore model from last model checkpoint {:s}'.format(weights_path))
                saver.restore(sess=sess, save_path=weights_path)

            # 加载预训练参数
            pretrained_weights = np.load(
                './data/vgg16.npy',
                encoding='latin1').item()

            for vv in vgg_vars:
                weights_key = vv.name.split('/')[-3]
                try:
                    weights = pretrained_weights[weights_key][0]
                    _op = tf.assign(vv, weights)
                    sess.run(_op)
                except Exception as e:
                    continue

            # train loop
            for epoch in range(train_epochs):
                # training part
                t_start = time.time()

                # update network and calculate loss and evaluate statics
                d_op, g_op, train_d_loss, train_g_loss, train_avg_ssim, \
                train_avg_psnr, train_summary, val_summary = sess.run(
                    [d_optim, g_optim, train_discriminative_loss, train_gan_loss, train_ssim,
                     train_psnr, train_summary_op, val_summary_op]
                )

                summary_writer.add_summary(train_summary, global_step=epoch)
                summary_writer.add_summary(val_summary, global_step=epoch)

                cost_time = time.time() - t_start

                log.info('Epoch_Train: {:d} D_loss: {:.5f} G_loss: '
                         '{:.5f} SSIM: {:.5f} PSNR: {:.5f} Cost_time: {:.5f}s'.format(
                    epoch, train_d_loss, train_g_loss, train_avg_ssim, train_avg_psnr, cost_time)
                )

                # Evaluate model
                if epoch % 500 == 0:
                    val_d_loss, val_g_loss, val_avg_ssim, val_avg_psnr = sess.run(
                        [val_discriminative_loss, val_gan_loss, val_ssim, val_psnr]
                    )
                    log.info('Epoch_Val: {:d} D_loss: {:.5f} G_loss: '
                             '{:.5f} SSIM: {:.5f} PSNR: {:.5f} Cost_time: {:.5f}s'.format(
                        epoch, val_d_loss, val_g_loss, val_avg_ssim, val_avg_psnr, cost_time)
                    )

                # Save Model
                if epoch % 5000 == 0:
                    saver.save(sess=sess, save_path=model_save_path, global_step=epoch)

        sess.close()

    return


def train_multi_gpu(dataset_dir, weights_path=None):
    """

    :param dataset_dir:
    :param weights_path:
    :return:
    """
    raise NotImplementedError


if __name__ == '__main__':
    # init args
    args = init_args()

    # train model
    train_model(args.dataset_dir, weights_path=args.weights_path)
