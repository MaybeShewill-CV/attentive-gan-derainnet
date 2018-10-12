#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-7-19 上午10:28
# @Author  : Luo Yao
# @Site    : http://icode.baidu.com/repos/baidu/personal-code/Luoyao
# @File    : test_model.py
# @IDE: PyCharm
"""
test model
"""
import os
import os.path as ops
import argparse

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr

from attentive_gan_model import derain_drop_net
from config import global_config

CFG = global_config.cfg


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, help='The input image path')
    parser.add_argument('--weights_path', type=str, help='The model weights path')

    return parser.parse_args()


def minmax_scale(input_arr):
    """

    :param input_arr:
    :return:
    """
    min_val = np.min(input_arr)
    max_val = np.max(input_arr)

    output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

    return output_arr


def test_model(image_path, weights_path):
    """

    :param image_path:
    :param weights_path:
    :return:
    """
    assert ops.exists(image_path)

    with tf.device('/gpu:0'):
        input_tensor = tf.placeholder(dtype=tf.float32,
                                      shape=[CFG.TEST.BATCH_SIZE, CFG.TEST.IMG_HEIGHT, CFG.TEST.IMG_WIDTH, 3],
                                      name='input_tensor'
                                      )

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (CFG.TEST.IMG_WIDTH, CFG.TEST.IMG_HEIGHT))
    image_vis = image
    image = np.divide(image, 127.5) - 1

    phase = tf.constant('train', tf.string)

    with tf.device('/gpu:0'):
        net = derain_drop_net.DeRainNet(phase=phase)
        output, attention_maps = net.build(input_tensor=input_tensor, name='derain_net_loss')

    # Set sess configuration
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TRAIN.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    saver = tf.train.Saver()

    with tf.device('/gpu:0'):
        with sess.as_default():
            saver.restore(sess=sess, save_path=weights_path)

            output_image, atte_maps = sess.run(
                [output, attention_maps],
                feed_dict={input_tensor: np.expand_dims(image, 0)})

            output_image = output_image[0]
            for i in range(output_image.shape[2]):
                output_image[:, :, i] = minmax_scale(output_image[:, :, i])

            output_image = np.array(output_image, np.uint8)

            # Image metrics计算
            image_ssim = ssim(
                image_vis,
                output_image,
                data_range=output_image.max() - output_image.min())
            image_psnr = psnr(image_vis, output_image)

            print('Image ssim: {:.5f}'.format(image_ssim))
            print('Image psnr: {:.5f}'.format(image_psnr))

            # 保存并可视化结果
            cv2.imwrite('src_img.png', image_vis)
            cv2.imwrite('derain_ret.png', output_image)

            plt.figure('src_image')
            plt.imshow(image_vis[:, :, (2, 1, 0)])
            plt.figure('derain_ret')
            plt.imshow(output_image[:, :, (2, 1, 0)])
            plt.figure('atte_map_1')
            plt.imshow(atte_maps[0][0, :, :, 0], cmap='jet')
            plt.savefig('atte_map_1.png')
            plt.figure('atte_map_2')
            plt.imshow(atte_maps[1][0, :, :, 0], cmap='jet')
            plt.savefig('atte_map_2.png')
            plt.figure('atte_map_3')
            plt.imshow(atte_maps[2][0, :, :, 0], cmap='jet')
            plt.savefig('atte_map_3.png')
            plt.figure('atte_map_4')
            plt.imshow(atte_maps[3][0, :, :, 0], cmap='jet')
            plt.savefig('atte_map_4.png')
            plt.show()

    return


if __name__ == '__main__':
    # init args
    args = init_args()

    # test model
    test_model(args.image_path, args.weights_path)
