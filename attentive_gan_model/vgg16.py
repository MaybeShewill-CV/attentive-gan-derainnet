#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-6-26 下午8:52
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/attentive-gan-derainnet
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

    def _conv_stage(self, input_tensor, k_size, out_dims, name, group_size=32,
                    stride=1, pad='SAME', reuse=False):
        """
        将卷积和激活封装在一起
        :param input_tensor:
        :param k_size:
        :param out_dims:
        :param name:
        :param group_size:
        :param stride:
        :param pad:
        :param reuse:
        :return:
        """
        with tf.variable_scope(name, reuse=reuse):
            conv = self.conv2d(inputdata=input_tensor, out_channel=out_dims,
                               kernel_size=k_size, stride=stride,
                               use_bias=False, padding=pad, name='conv')
            if group_size:
                gn = self.layergn(inputdata=conv, group_size=group_size, name='gn')
                relu = self.relu(inputdata=gn, name='relu')
            else:
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
                                        group_size=0, out_dims=64, name='conv1_1')

            # conv stage 1_2
            conv_1_2 = self._conv_stage(input_tensor=conv_1_1, k_size=3,
                                        group_size=0, out_dims=64, name='conv1_2')

            # pool stage 1
            pool1 = self.maxpooling(inputdata=conv_1_2, kernel_size=2,
                                    stride=2, name='pool1')

            # conv stage 2_1
            conv_2_1 = self._conv_stage(input_tensor=pool1, k_size=3,
                                        group_size=0, out_dims=128, name='conv2_1')

            # conv stage 2_2
            conv_2_2 = self._conv_stage(input_tensor=conv_2_1, k_size=3,
                                        group_size=0, out_dims=128, name='conv2_2')

            # pool stage 2
            pool2 = self.maxpooling(inputdata=conv_2_2, kernel_size=2,
                                    stride=2, name='pool2')

            # conv stage 3_1
            conv_3_1 = self._conv_stage(input_tensor=pool2, k_size=3,
                                        group_size=0, out_dims=256, name='conv3_1')

            # conv_stage 3_2
            conv_3_2 = self._conv_stage(input_tensor=conv_3_1, k_size=3,
                                        group_size=0, out_dims=256, name='conv3_2')

            # conv stage 3_3
            conv_3_3 = self._conv_stage(input_tensor=conv_3_2, k_size=3,
                                        group_size=0, out_dims=256, name='conv3_3')

            ret = (conv_1_1, conv_1_2, conv_2_1, conv_2_2,
                   conv_3_1, conv_3_2, conv_3_3)

        return ret


if __name__ == '__main__':
    a = tf.placeholder(dtype=tf.float32, shape=[1, 256, 256, 3], name='input')
    encoder = VGG16Encoder(phase=tf.constant('train', dtype=tf.string))
    ret = encoder.extract_feats(a, name='encode')
