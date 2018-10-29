#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-6-26 下午8:52
# @Author  : Luo Yao
# @Site    : http://icode.baidu.com/repos/baidu/personal-code/Luoyao
# @File    : vgg16.py
# @IDE: PyCharm
"""
实现pretrained vgg用于特征提取计算attentive gan损失
"""
from collections import OrderedDict

import tensorflow as tf

from attentive_gan_model import cnn_basenet


class VGG16Encoder(cnn_basenet.CNNBaseModel):
    """
    实现了一个基于vgg16的特征编码类
    """
    def __init__(self, phase):
        """

        :param phase:
        """
        super(VGG16Encoder, self).__init__()
        self._train_phase = tf.constant('train', dtype=tf.string)
        self._test_phase = tf.constant('test', dtype=tf.string)
        self._phase = phase
        self._is_training = self._init_phase()
        print('VGG16 Network init complete')

    def _init_phase(self):
        """

        :return:
        """
        return tf.equal(self._phase, self._train_phase)

    def _conv_stage(self, input_tensor, k_size, out_dims, name,
                    stride=1, pad='SAME', reuse=False):
        """
        将卷积和激活封装在一起
        :param input_tensor:
        :param k_size:
        :param out_dims:
        :param name:
        :param stride:
        :param pad:
        :param reuse:
        :return:
        """
        with tf.variable_scope(name, reuse=reuse):
            conv = self.conv2d(inputdata=input_tensor, out_channel=out_dims,
                               kernel_size=k_size, stride=stride,
                               use_bias=False, padding=pad, name='conv')

            # bn = self.layerbn(inputdata=conv, is_training=self._is_training, name='bn')

            relu = self.relu(inputdata=conv, name='relu')

            return relu

    def _fc_stage(self, input_tensor, out_dims, name, use_bias=False, reuse=False):
        """

        :param input_tensor:
        :param out_dims:
        :param name:
        :param use_bias:
        :return:
        """
        with tf.variable_scope(name, reuse=reuse):
            fc = self.fullyconnect(inputdata=input_tensor, out_dim=out_dims, use_bias=use_bias,
                                   name='fc')

            # bn = self.layerbn(inputdata=fc, is_training=self._is_training, name='bn')

            relu = self.relu(inputdata=fc, name='relu')

        return relu

    def extract_feats(self, input_tensor, name, reuse=False):
        """
        根据vgg16框架对输入的tensor进行编码
        :param input_tensor:
        :param name:
        :param reuse:
        :return: 输出vgg16编码特征
        """
        with tf.variable_scope(name, reuse=reuse):
            # conv stage 1_1
            conv_1_1 = self._conv_stage(input_tensor=input_tensor, k_size=3,
                                        out_dims=64, name='conv1_1')

            # conv stage 1_2
            conv_1_2 = self._conv_stage(input_tensor=conv_1_1, k_size=3,
                                        out_dims=64, name='conv1_2')

            # pool stage 1
            pool1 = self.maxpooling(inputdata=conv_1_2, kernel_size=2,
                                    stride=2, name='pool1')

            # conv stage 2_1
            conv_2_1 = self._conv_stage(input_tensor=pool1, k_size=3,
                                        out_dims=128, name='conv2_1')

            # conv stage 2_2
            conv_2_2 = self._conv_stage(input_tensor=conv_2_1, k_size=3,
                                        out_dims=128, name='conv2_2')

            # pool stage 2
            pool2 = self.maxpooling(inputdata=conv_2_2, kernel_size=2,
                                    stride=2, name='pool2')

            # conv stage 3_1
            conv_3_1 = self._conv_stage(input_tensor=pool2, k_size=3,
                                        out_dims=256, name='conv3_1')

            # conv_stage 3_2
            conv_3_2 = self._conv_stage(input_tensor=conv_3_1, k_size=3,
                                        out_dims=256, name='conv3_2')

            # conv stage 3_3
            conv_3_3 = self._conv_stage(input_tensor=conv_3_2, k_size=3,
                                        out_dims=256, name='conv3_3')

            ret = (conv_1_1, conv_1_2, conv_2_1, conv_2_2,
                   conv_3_1, conv_3_2, conv_3_3)

            # pool stage 3
            # pool3 = self.maxpooling(inputdata=conv_3_3, kernel_size=2,
            #                         stride=2, name='pool3')

            # # conv stage 4_1
            # conv_4_1 = self._conv_stage(input_tensor=pool3, k_size=3,
            #                             out_dims=512, name='conv4_1')
            #
            # # conv stage 4_2
            # conv_4_2 = self._conv_stage(input_tensor=conv_4_1, k_size=3,
            #                             out_dims=512, name='conv4_2')
            #
            # # conv stage 4_3
            # conv_4_3 = self._conv_stage(input_tensor=conv_4_2, k_size=3,
            #                             out_dims=512, name='conv4_3')
            #
            # # pool stage 4
            # pool4 = self.maxpooling(inputdata=conv_4_3, kernel_size=2,
            #                         stride=2, name='pool4')
            #
            # # conv stage 5_1
            # conv_5_1 = self._conv_stage(input_tensor=pool4, k_size=3,
            #                             out_dims=512, name='conv5_1')
            #
            # # conv stage 5_2
            # conv_5_2 = self._conv_stage(input_tensor=conv_5_1, k_size=3,
            #                             out_dims=512, name='conv5_2')
            #
            # # conv stage 5_3
            # conv_5_3 = self._conv_stage(input_tensor=conv_5_2, k_size=3,
            #                             out_dims=512, name='conv5_3')
            #
            # # pool stage 5
            # pool5 = self.maxpooling(inputdata=conv_5_3, kernel_size=2,
            #                         stride=2, name='pool5')

            # fc stage 1
            # fc6 = self._fc_stage(input_tensor=pool5, out_dims=4096, name='fc6',
            #                      use_bias=False)

            # fc stage 2
            # fc7 = self._fc_stage(input_tensor=fc6, out_dims=4096, name='fc7',
            #                      use_bias=False)

        return ret


if __name__ == '__main__':
    a = tf.placeholder(dtype=tf.float32, shape=[1, 256, 256, 3], name='input')
    encoder = VGG16Encoder(phase=tf.constant('train', dtype=tf.string))
    ret = encoder.extract_feats(a, name='encode')
