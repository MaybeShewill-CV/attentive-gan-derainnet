#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-7-3 下午4:31
# @Author  : Luo Yao
# @Site    : http://icode.baidu.com/repos/baidu/personal-code/Luoyao
# @File    : train_model.py.py
# @IDE: PyCharm
"""
模型训练脚本
"""
import os
import os.path as ops
import argparse
import time

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import glog as log
import cv2

from data_provider import data_provider
from config import global_config
from attentive_gan_model import derain_drop_net
from attentive_gan_model import tf_ssim

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


def train_model(dataset_dir, weights_path=None):
    """

    :param dataset_dir:
    :param gpu_id:
    :param weights_path:
    :return:
    """

    # 构建数据集
    with tf.device('/gpu:0'):
        train_dataset = data_provider.DataSet(ops.join(dataset_dir, 'train.txt'))

        # 声明tensor
        input_tensor = tf.placeholder(dtype=tf.float32,
                                      shape=[CFG.TRAIN.BATCH_SIZE, CFG.TRAIN.IMG_HEIGHT, CFG.TRAIN.IMG_WIDTH, 3],
                                      name='input_tensor')
        label_tensor = tf.placeholder(dtype=tf.float32,
                                      shape=[CFG.TRAIN.BATCH_SIZE, CFG.TRAIN.IMG_HEIGHT, CFG.TRAIN.IMG_WIDTH, 3],
                                      name='label_tensor')
        mask_tensor = tf.placeholder(dtype=tf.float32,
                                     shape=[CFG.TRAIN.BATCH_SIZE, CFG.TRAIN.IMG_HEIGHT, CFG.TRAIN.IMG_WIDTH, 1],
                                     name='mask_tensor')
        lr_tensor = tf.placeholder(dtype=tf.float32,
                                   shape=[],
                                   name='learning_rate')
        phase_tensor = tf.placeholder(dtype=tf.string, shape=[], name='phase')

        # 声明ssim计算类
        ssim_computer = tf_ssim.SsimComputer()

        # 声明网络
        derain_net = derain_drop_net.DeRainNet(phase=phase_tensor)

        gan_loss, discriminative_loss, net_output = derain_net.compute_loss(
            input_tensor=input_tensor,
            gt_label_tensor=label_tensor,
            mask_label_tensor=mask_tensor,
            name='derain_net_loss')

        train_vars = tf.trainable_variables()

        # ssim = tf.image.ssim(tf.image.rgb_to_grayscale(label_tensor),
        #                      tf.image.rgb_to_grayscale(net_output),
        #                      max_val=1.0)
        ssim = ssim_computer.compute_ssim(tf.image.rgb_to_grayscale(net_output),
                                          tf.image.rgb_to_grayscale(label_tensor))

        d_vars = [tmp for tmp in train_vars if 'discriminative_loss' in tmp.name]
        g_vars = [tmp for tmp in train_vars if 'attentive_' in tmp.name and 'vgg_feats' not in tmp.name]
        vgg_vars = [tmp for tmp in train_vars if "vgg_feats" in tmp.name]

        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(lr_tensor, global_step,
                                                   100000, 0.1, staircase=True)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            d_optim = tf.train.AdamOptimizer(learning_rate).minimize(
                discriminative_loss, var_list=d_vars)
            g_optim = tf.train.MomentumOptimizer(
                learning_rate=learning_rate,
                momentum=tf.constant(0.9, tf.float32)).minimize(gan_loss, var_list=g_vars)

        # Set tf saver
        saver = tf.train.Saver()
        model_save_dir = 'model/derain_gan_tensorflow10'
        if not ops.exists(model_save_dir):
            os.makedirs(model_save_dir)
        train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        model_name = 'derain_gan_{:s}.ckpt'.format(str(train_start_time))
        model_save_path = ops.join(model_save_dir, model_name)

        # Set tf summary
        tboard_save_path = 'tboard/derain_gan_tensorflow10'
        if not ops.exists(tboard_save_path):
            os.makedirs(tboard_save_path)
        g_loss_scalar = tf.summary.scalar(name='gan_loss', tensor=gan_loss)
        d_loss_scalar = tf.summary.scalar(name='discriminative_loss', tensor=discriminative_loss)
        ssim_scalar = tf.summary.scalar(name='image_ssim', tensor=ssim)
        lr_scalar = tf.summary.scalar(name='learning_rate', tensor=lr_tensor)
        d_summary_op = tf.summary.merge([d_loss_scalar, lr_scalar])
        g_summary_op = tf.summary.merge([g_loss_scalar, ssim_scalar])

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

                gt_imgs, label_imgs, mask_imgs = train_dataset.next_batch(CFG.TRAIN.BATCH_SIZE)

                mask_imgs = [np.expand_dims(tmp, axis=-1) for tmp in mask_imgs]

                # Update discriminative Network
                _, d_loss, d_summary = sess.run(
                    [d_optim, discriminative_loss, d_summary_op],
                    feed_dict={input_tensor: gt_imgs,
                               label_tensor: label_imgs,
                               mask_tensor: mask_imgs,
                               lr_tensor: CFG.TRAIN.LEARNING_RATE,
                               phase_tensor: 'train'})

                # Update attentive gan Network
                _, g_loss, g_summary, ssim_val = sess.run(
                    [g_optim, gan_loss, g_summary_op, ssim],
                    feed_dict={input_tensor: gt_imgs,
                               label_tensor: label_imgs,
                               mask_tensor: mask_imgs,
                               lr_tensor: CFG.TRAIN.LEARNING_RATE,
                               phase_tensor: 'train'})

                summary_writer.add_summary(d_summary, global_step=epoch)
                summary_writer.add_summary(g_summary, global_step=epoch)

                cost_time = time.time() - t_start

                log.info('Epoch: {:d} D_loss: {:.5f} G_loss: '
                         '{:.5f} Ssim: {:.5f} Cost_time: {:.5f}s'.format(epoch, d_loss, g_loss,
                                                                         ssim_val, cost_time))
                if epoch % 5000 == 0:
                    saver.save(sess=sess, save_path=model_save_path, global_step=epoch)
        sess.close()

    return


if __name__ == '__main__':
    # init args
    args = init_args()

    # train model
    train_model(args.dataset_dir, weights_path=args.weights_path)
