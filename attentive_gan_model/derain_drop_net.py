#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-7-17 下午7:54
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/attentive-gan-derainnet
# @File    : derain_drop_net.py
# @IDE: PyCharm
"""
Attentive Gan Derain drop Network
"""
import tensorflow as tf

from attentive_gan_model import attentive_gan_net
from attentive_gan_model import discriminative_net


class DeRainNet(object):
    """

    """
    def __init__(self, phase):
        """

        """
        self._phase = phase
        self._attentive_gan = attentive_gan_net.GenerativeNet(self._phase)
        self._discriminator = discriminative_net.DiscriminativeNet(self._phase)

    def compute_loss(self, input_tensor, gt_label_tensor, mask_label_tensor, name, reuse=False):
        """
        计算gan损失和discriminative损失
        :param input_tensor: 含有雨滴的图像
        :param gt_label_tensor: 不含有雨滴的图像
        :param mask_label_tensor: mask标签用来计算attentive rnn
        :param name:
        :param reuse:
        :return:
        """
        with tf.variable_scope(name, reuse=reuse):

            # 计算attentive rnn loss
            attentive_rnn_loss, attentive_rnn_output = self._attentive_gan.compute_attentive_rnn_loss(
                input_tensor=input_tensor,
                label_tensor=mask_label_tensor,
                name='attentive_rnn_loss',
                reuse=reuse
            )

            auto_encoder_input = tf.concat((attentive_rnn_output, input_tensor),
                                           axis=-1, name='autoencoder_input')

            auto_encoder_loss, auto_encoder_output = self._attentive_gan.compute_autoencoder_loss(
                input_tensor=auto_encoder_input,
                label_tensor=gt_label_tensor,
                name='attentive_autoencoder_loss',
                reuse=reuse
            )

            gan_loss = tf.add(attentive_rnn_loss, auto_encoder_loss, name='gan_loss')

            discriminative_inference, discriminative_loss = self._discriminator.compute_loss(
                input_tensor=auto_encoder_output,
                label_tensor=gt_label_tensor,
                attention_map=attentive_rnn_output,
                name='discriminative_loss',
                reuse=reuse
            )

            l_gan = tf.reduce_mean(tf.log(tf.subtract(tf.constant(1.0), discriminative_inference)) * 0.01)
            l_gan = tf.identity(l_gan, name='l_gan_loss')

            gan_loss = tf.add(gan_loss, l_gan, name='total_gan_loss')

        return gan_loss, discriminative_loss, auto_encoder_output

    def inference(self, input_tensor, name, reuse=False):
        """
        生成最后的结果图
        :param input_tensor:
        :param name:
        :param reuse:
        :return:
        """
        with tf.variable_scope(name, reuse=reuse):

            attentive_rnn_out = self._attentive_gan.build_attentive_rnn(
                input_tensor=input_tensor,
                name='attentive_rnn_loss/attentive_inference'
            )

            attentive_autoencoder_input = tf.concat((attentive_rnn_out['final_attention_map'],
                                                     input_tensor), axis=-1)

            output = self._attentive_gan.build_autoencoder(
                input_tensor=attentive_autoencoder_input,
                name='attentive_autoencoder_loss/autoencoder_inference'
            )

        return output['skip_3'], attentive_rnn_out['attention_map_list']


if __name__ == '__main__':
    """
    test
    """
    input_tensor = tf.placeholder(dtype=tf.float32, shape=[5, 256, 256, 3])
    label_tensor = tf.placeholder(dtype=tf.float32, shape=[5, 256, 256, 3])
    mask_tensor = tf.placeholder(dtype=tf.float32, shape=[5, 256, 256, 1])

    net = DeRainNet(tf.constant('train', tf.string))

    g_loss, d_loss, _ = net.compute_loss(input_tensor, label_tensor, mask_tensor, 'loss')
    g_loss2, d_loss2, _ = net.compute_loss(input_tensor, label_tensor, mask_tensor, 'loss', reuse=True)

    for vv in tf.trainable_variables():
        print(vv.name)
