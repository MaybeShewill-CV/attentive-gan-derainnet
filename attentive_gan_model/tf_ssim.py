#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-6-4 下午3:00
# @Author  : Luo Yao
# @Site    : http://icode.baidu.com/repos/baidu/personal-code/Luoyao
# @File    : tf_ssim.py
# @IDE: PyCharm Community Edition
"""
Implement ssim compute using tensorflow
"""
import tensorflow as tf
import numpy as np


class SsimComputer(object):
    """

    """
    def __init__(self):
        """

        """
        pass

    @staticmethod
    def _tf_fspecial_gauss(size, sigma):
        """
        Function to mimic the 'fspecial' gaussian function
        :param size:
        :param sigma:
        :return:
        """
        x_data, y_data = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]

        x_data = np.expand_dims(x_data, axis=-1)
        x_data = np.expand_dims(x_data, axis=-1)

        y_data = np.expand_dims(y_data, axis=-1)
        y_data = np.expand_dims(y_data, axis=-1)

        x = tf.constant(x_data, dtype=tf.float32)
        y = tf.constant(y_data, dtype=tf.float32)

        g = tf.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
        return g / tf.reduce_sum(g)

    def compute_ssim(self, img1, img2, cs_map=False, mean_metric=True, size=11, sigma=1.5):
        """

        :param img1: image need to be gray scale
        :param img2:
        :param cs_map:
        :param mean_metric:
        :param size:
        :param sigma:
        :return:
        """
        assert img1.get_shape().as_list()[-1] == 1, 'Image must be gray scale'
        assert img2.get_shape().as_list()[-1] == 1, 'Image must be gray scale'

        window = self._tf_fspecial_gauss(size, sigma)  # window shape [size, size]
        K1 = 0.01  # origin parameter in paper
        K2 = 0.03  # origin parameter in paper
        L = 1  # depth of image (255 in case the image has a differnt scale)
        C1 = (K1 * L) ** 2
        C2 = (K2 * L) ** 2
        mu1 = tf.nn.conv2d(img1, window, strides=[1, 1, 1, 1], padding='VALID')
        mu2 = tf.nn.conv2d(img2, window, strides=[1, 1, 1, 1], padding='VALID')
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = tf.nn.conv2d(img1 * img1, window, strides=[1, 1, 1, 1], padding='VALID') - mu1_sq
        sigma2_sq = tf.nn.conv2d(img2 * img2, window, strides=[1, 1, 1, 1], padding='VALID') - mu2_sq
        sigma12 = tf.nn.conv2d(img1 * img2, window, strides=[1, 1, 1, 1], padding='VALID') - mu1_mu2
        if cs_map:
            value = (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                  (sigma1_sq + sigma2_sq + C2)),
                     (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
        else:
            value = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                 (sigma1_sq + sigma2_sq + C2))

        if mean_metric:
            value = tf.reduce_mean(value)
        return value


if __name__ == '__main__':
    """
    测试tf ssim和scikit image ssim区别
    """
    import cv2
    from skimage.measure import compare_ssim as ssim
    from skimage.measure import compare_psnr as psnr

    image_file_path = '/media/baidu/Data/Gan_Derain_Dataset/train/data/0_rain.png'
    label_file_path = '/media/baidu/Data/Gan_Derain_Dataset/train/gt_label/0_clean.png'

    # tensorflow获取图像数据
    image_string = tf.read_file(image_file_path)
    label_string = tf.read_file(label_file_path)

    image_tensor = tf.image.decode_png(image_string)
    label_tensor = tf.image.decode_png(label_string)

    image_tensor = tf.image.convert_image_dtype(image_tensor, tf.float32)
    label_tensor = tf.image.convert_image_dtype(label_tensor, tf.float32)

    # numpy获取图像数据
    image = cv2.imread(image_file_path, cv2.IMREAD_UNCHANGED)
    label = cv2.imread(label_file_path, cv2.IMREAD_UNCHANGED)

    # skimage ssim, psnr接口计算ssim psnr值
    image_ssim_py = ssim(
        np.array(image, np.float32),
        np.array(label, np.float32),
        data_range=label.max() - label.min(),
        multichannel=True)

    image_psnr_py = psnr(
        image,
        label,
        data_range=label.max() - label.min())

    image_tensor_v1 = tf.placeholder(dtype=tf.float32, shape=[1, 480, 720, 3])
    label_tensor_v1 = tf.placeholder(dtype=tf.float32, shape=[1, 480, 720, 3])

    # tensorflow接口获取ssim, psnr值
    image_ssim_tf = tf.image.ssim(image_tensor_v1, label_tensor_v1, max_val=1.0)
    image_psnr_tf = tf.image.psnr(image_tensor_v1, label_tensor_v1, max_val=1.0)

    with tf.Session() as sess:

        ssim_val = sess.run(image_ssim_tf,
                            feed_dict={image_tensor_v1: [np.array(image / 255, np.float32)],
                                       label_tensor_v1: [np.array(label / 255, np.float32)]})
        psnr_val = sess.run(image_psnr_tf,
                            feed_dict={image_tensor_v1: [np.array(image / 255, np.float32)],
                                       label_tensor_v1: [np.array(label / 255, np.float32)]})

        print('Skimage ssim {:.5f}'.format(image_ssim_py))
        print('Tf ssim {:.5f}'.format(ssim_val[0]))

        print('Skimage psnr {:.5f}'.format(image_psnr_py))
        print('Tf psnr {:.5f}'.format(psnr_val[0]))

# TODO LUOYAO(luoyao@baidu.com) tensorflow对比skimage, ssim计算结果有差异, psnr计算结果完全相同
