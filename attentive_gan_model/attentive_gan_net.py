#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-6-26 上午11:45
# @Author  : Luo Yao
# @Site    : http://icode.baidu.com/repos/baidu/personal-code/Luoyao
# @File    : attentive_gan_net.py
# @IDE: PyCharm
"""
实现Attentive GAN Network中的Attentive-Recurrent Network
"""
import tensorflow as tf

from attentive_gan_model import cnn_basenet
from attentive_gan_model import vgg16


class GenerativeNet(cnn_basenet.CNNBaseModel):
    """
    实现Attentive GAN Network中的生成网络 Fig(2)中的generator部分
    """
    def __init__(self, phase):
        """

        :return:
        """
        super(GenerativeNet, self).__init__()
        self._vgg_extractor = vgg16.VGG16Encoder(phase='test')
        self._train_phase = tf.constant('train', dtype=tf.string)
        self._test_phase = tf.constant('test', dtype=tf.string)
        self._phase = phase
        self._is_training = self._init_phase()

    def _init_phase(self):
        """

        :return:
        """
        return tf.equal(self._phase, self._train_phase)

    def build(self, input_tensor):
        """

        :param input_tensor:
        :return:
        """
        pass

    def _residual_block(self, input_tensor, name):
        """
        attentive recurrent net中的residual block
        :param input_tensor:
        :param name:
        :return:
        """
        output = None
        with tf.variable_scope(name):
            for i in range(5):
                if i == 0:
                    conv_1 = self.conv2d(inputdata=input_tensor,
                                         out_channel=32,
                                         kernel_size=3,
                                         padding='SAME',
                                         stride=1,
                                         use_bias=False,
                                         name='block_{:d}_conv_1'.format(i))
                    gn_1 = self.layergn(inputdata=conv_1,
                                        group_size=8,
                                        name='block_{:d}_gn_1'.format(i + 1))
                    relu_1 = self.lrelu(inputdata=gn_1, name='block_{:d}_relu_1'.format(i + 1))
                    output = relu_1
                    input_tensor = output
                else:
                    conv_1 = self.conv2d(inputdata=input_tensor,
                                         out_channel=32,
                                         kernel_size=1,
                                         padding='SAME',
                                         stride=1,
                                         use_bias=False,
                                         name='block_{:d}_conv_1'.format(i))
                    gn_1 = self.layergn(inputdata=conv_1,
                                        group_size=8,
                                        name='block_{:d}_gn_1'.format(i))
                    relu_1 = self.lrelu(inputdata=gn_1, name='block_{:d}_conv_1'.format(i + 1))
                    conv_2 = self.conv2d(inputdata=relu_1,
                                         out_channel=32,
                                         kernel_size=1,
                                         padding='SAME',
                                         stride=1,
                                         use_bias=False,
                                         name='block_{:d}_conv_2'.format(i))
                    gn_2 = self.layergn(inputdata=conv_2,
                                        group_size=8,
                                        name='block_{:d}_gn_2'.format(i))
                    relu_2 = self.lrelu(inputdata=gn_2, name='block_{:d}_conv_2'.format(i + 1))

                    output = self.lrelu(inputdata=tf.add(relu_2, input_tensor),
                                        name='block_{:d}_add'.format(i))
                    input_tensor = output

        return output

    def _conv_lstm(self, input_tensor, input_cell_state, name):
        """
        attentive recurrent net中的convolution lstm 见公式(3)
        :param input_tensor:
        :param input_cell_state:
        :param name:
        :return:
        """
        with tf.variable_scope(name):
            conv_i = self.conv2d(inputdata=input_tensor, out_channel=32, kernel_size=3, padding='SAME',
                                 stride=1, use_bias=False, name='conv_i')
            sigmoid_i = self.sigmoid(inputdata=conv_i, name='sigmoid_i')

            conv_f = self.conv2d(inputdata=input_tensor, out_channel=32, kernel_size=3, padding='SAME',
                                 stride=1, use_bias=False, name='conv_f')
            sigmoid_f = self.sigmoid(inputdata=conv_f, name='sigmoid_f')

            cell_state = sigmoid_f * input_cell_state + \
                         sigmoid_i * tf.nn.tanh(self.conv2d(inputdata=input_tensor,
                                                            out_channel=32,
                                                            kernel_size=3,
                                                            padding='SAME',
                                                            stride=1,
                                                            use_bias=False,
                                                            name='conv_c'))
            conv_o = self.conv2d(inputdata=input_tensor, out_channel=32, kernel_size=3, padding='SAME',
                                 stride=1, use_bias=False, name='conv_o')
            sigmoid_o = self.sigmoid(inputdata=conv_o, name='sigmoid_o')

            lstm_feats = sigmoid_o * tf.nn.tanh(cell_state)

            attention_map = self.conv2d(inputdata=lstm_feats, out_channel=1, kernel_size=3, padding='SAME',
                                        stride=1, use_bias=False, name='attention_map')
            attention_map = self.sigmoid(inputdata=attention_map)

            ret = {
                'attention_map': attention_map,
                'cell_state': cell_state,
                'lstm_feats': lstm_feats
            }

            return ret

    def build_attentive_rnn(self, input_tensor, name):
        """
        Generator的attentive recurrent部分, 主要是为了找到attention部分
        :param input_tensor:
        :param name:
        :return:
        """
        [batch_size, tensor_h, tensor_w, _] = input_tensor.get_shape().as_list()
        with tf.variable_scope(name):
            init_attention_map = tf.constant(0.5, dtype=tf.float32,
                                             shape=[batch_size, tensor_h, tensor_w, 1])
            init_cell_state = tf.constant(0.0, dtype=tf.float32,
                                          shape=[batch_size, tensor_h, tensor_w, 32])
            init_lstm_feats = tf.constant(0.0, dtype=tf.float32,
                                          shape=[batch_size, tensor_h, tensor_w, 32])

            attention_map_list = []

            for i in range(4):
                attention_input = tf.concat((input_tensor, init_attention_map), axis=-1)
                conv_feats = self._residual_block(input_tensor=attention_input,
                                                  name='residual_block_{:d}'.format(i + 1))
                lstm_ret = self._conv_lstm(input_tensor=conv_feats,
                                           input_cell_state=init_cell_state,
                                           name='conv_lstm_block_{:d}'.format(i + 1))
                init_attention_map = lstm_ret['attention_map']
                init_cell_state = lstm_ret['cell_state']
                init_lstm_feats = lstm_ret['lstm_feats']

                attention_map_list.append(lstm_ret['attention_map'])

        ret = {
            'final_attention_map': init_attention_map,
            'final_lstm_feats': init_lstm_feats,
            'attention_map_list': attention_map_list
        }

        return ret

    def compute_attentive_rnn_loss(self, input_tensor, label_tensor, name):
        """
        计算attentive rnn损失
        :param input_tensor:
        :param label_tensor:
        :param name:
        :return:
        """
        with tf.variable_scope(name):
            inference_ret = self.build_attentive_rnn(input_tensor=input_tensor,
                                                     name='attentive_inference')
            loss = tf.constant(0.0, tf.float32)
            n = len(inference_ret['attention_map_list'])
            for index, attention_map in enumerate(inference_ret['attention_map_list']):
                mse_loss = tf.pow(0.8, n - index + 1) * \
                           tf.losses.mean_squared_error(labels=label_tensor,
                                                        predictions=attention_map)
                loss = tf.add(loss, mse_loss)

        return loss, inference_ret['final_attention_map']

    def build_autoencoder(self, input_tensor, name):
        """
        Generator的autoencoder部分, 负责获取图像上下文信息
        :param input_tensor:
        :return:
        """
        with tf.variable_scope(name):
            conv_1 = self.conv2d(inputdata=input_tensor, out_channel=64, kernel_size=5,
                                 padding='SAME',
                                 stride=1, use_bias=False, name='conv_1')
            gn_1 = self.layergn(inputdata=conv_1, group_size=16, name='gn_1')
            relu_1 = self.lrelu(inputdata=gn_1, name='relu_1')

            conv_2 = self.conv2d(inputdata=relu_1, out_channel=128, kernel_size=3,
                                 padding='SAME',
                                 stride=2, use_bias=False, name='conv_2')
            gn_2 = self.layergn(inputdata=conv_2, name='gn_2')
            relu_2 = self.lrelu(inputdata=gn_2, name='relu_2')

            conv_3 = self.conv2d(inputdata=relu_2, out_channel=128, kernel_size=3,
                                 padding='SAME',
                                 stride=1, use_bias=False, name='conv_3')
            gn_3 = self.layergn(inputdata=conv_3, name='gn_3')
            relu_3 = self.lrelu(inputdata=gn_3, name='relu_3')

            conv_4 = self.conv2d(inputdata=relu_3, out_channel=128, kernel_size=3,
                                 padding='SAME',
                                 stride=2, use_bias=False, name='conv_4')
            gn_4 = self.layergn(inputdata=conv_4, name='gn_4')
            relu_4 = self.lrelu(inputdata=gn_4, name='relu_4')

            conv_5 = self.conv2d(inputdata=relu_4, out_channel=256, kernel_size=3,
                                 padding='SAME',
                                 stride=1, use_bias=False, name='conv_5')
            gn_5 = self.layergn(inputdata=conv_5, name='gn_5')
            relu_5 = self.lrelu(inputdata=gn_5, name='relu_5')

            conv_6 = self.conv2d(inputdata=relu_5, out_channel=256, kernel_size=3,
                                 padding='SAME',
                                 stride=1, use_bias=False, name='conv_6')
            gn_6 = self.layergn(inputdata=conv_6, name='gn_6')
            relu_6 = self.lrelu(inputdata=gn_6, name='relu_6')

            dia_conv1 = self.dilation_conv(input_tensor=relu_6, k_size=3, out_dims=256, rate=2,
                                           padding='SAME', use_bias=False, name='dia_conv_1')
            gn_7 = self.layergn(inputdata=dia_conv1, name='gn_7')
            relu_7 = self.lrelu(gn_7, name='relu_7')

            dia_conv2 = self.dilation_conv(input_tensor=relu_7, k_size=3, out_dims=256, rate=4,
                                           padding='SAME', use_bias=False, name='dia_conv_2')
            gn_8 = self.layergn(inputdata=dia_conv2, name='gn_8')
            relu_8 = self.lrelu(gn_8, name='relu_8')

            dia_conv3 = self.dilation_conv(input_tensor=relu_8, k_size=3, out_dims=256, rate=8,
                                           padding='SAME', use_bias=False, name='dia_conv_3')
            gn_9 = self.layergn(inputdata=dia_conv3, name='gn_9')
            relu_9 = self.lrelu(gn_9, name='relu_9')

            dia_conv4 = self.dilation_conv(input_tensor=relu_9, k_size=3, out_dims=256, rate=16,
                                           padding='SAME', use_bias=False, name='dia_conv_4')
            gn_10 = self.layergn(inputdata=dia_conv4, name='gn_10')
            relu_10 = self.lrelu(gn_10, name='relu_10')

            conv_7 = self.conv2d(inputdata=relu_10, out_channel=256, kernel_size=3,
                                 padding='SAME', stride=1, use_bias=False,
                                 name='conv_7')
            gn_11 = self.layergn(inputdata=conv_7, name='gn_11')
            relu_11 = self.lrelu(inputdata=gn_11, name='relu_11')

            conv_8 = self.conv2d(inputdata=relu_11, out_channel=256, kernel_size=3,
                                 padding='SAME', stride=1, use_bias=False,
                                 name='conv_8')
            gn_12 = self.layergn(inputdata=conv_8, name='gn_12')
            relu_12 = self.lrelu(inputdata=gn_12, name='relu_12')

            deconv_1 = self.deconv2d(inputdata=relu_12, out_channel=128, kernel_size=4,
                                     stride=2, padding='SAME', use_bias=False, name='deconv_1')
            gn_13 = self.layergn(inputdata=deconv_1, name='gn_13')
            avg_pool_1 = self.avgpooling(inputdata=gn_13, kernel_size=2, stride=1, padding='SAME',
                                         name='avg_pool_1')
            relu_13 = self.lrelu(inputdata=avg_pool_1, name='relu_13')

            conv_9 = self.conv2d(inputdata=tf.add(relu_13, relu_3), out_channel=128, kernel_size=3,
                                 padding='SAME', stride=1, use_bias=False,
                                 name='conv_9')
            gn_14 = self.layergn(inputdata=conv_9, name='gn_14')
            relu_14 = self.lrelu(inputdata=gn_14, name='relu_14')

            deconv_2 = self.deconv2d(inputdata=relu_14, out_channel=64, kernel_size=4,
                                     stride=2, padding='SAME', use_bias=False, name='deconv_2')
            gn_15 = self.layergn(inputdata=deconv_2, group_size=16, name='gn_15')
            avg_pool_2 = self.avgpooling(inputdata=gn_15, kernel_size=2, stride=1, padding='SAME',
                                         name='avg_pool_2')
            relu_15 = self.lrelu(inputdata=avg_pool_2, name='relu_15')

            conv_10 = self.conv2d(inputdata=tf.add(relu_15, relu_1), out_channel=32, kernel_size=3,
                                  padding='SAME', stride=1, use_bias=False,
                                  name='conv_10')
            gn_16 = self.layergn(inputdata=conv_10, group_size=8, name='gn_16')
            relu_16 = self.lrelu(inputdata=gn_16, name='relu_16')

            skip_output_1 = self.conv2d(inputdata=relu_12, out_channel=3, kernel_size=3,
                                        padding='SAME', stride=1, use_bias=False,
                                        name='skip_ouput_1')

            skip_output_2 = self.conv2d(inputdata=relu_14, out_channel=3, kernel_size=3,
                                        padding='SAME', stride=1, use_bias=False,
                                        name='skip_output_2')

            skip_output_3 = self.conv2d(inputdata=relu_16, out_channel=3, kernel_size=3,
                                        padding='SAME', stride=1, use_bias=False,
                                        name='skip_output_3')

            # 传统GAN输出层都使用tanh函数激活
            skip_output_3 = tf.nn.tanh(skip_output_3, name='skip_output_3_tanh')

            ret = {
                'skip_1': skip_output_1,
                'skip_2': skip_output_2,
                'skip_3': skip_output_3
            }

        return ret

    def compute_autoencoder_loss(self, input_tensor, label_tensor, name):
        """
        计算自编码器损失函数
        :param input_tensor:
        :param label_tensor:
        :param name:
        :return:
        """
        [_, ori_height, ori_width, _] = label_tensor.get_shape().as_list()
        label_tensor_ori = label_tensor
        label_tensor_resize_2 = tf.image.resize_bilinear(images=label_tensor,
                                                         size=(int(ori_height / 2), int(ori_width / 2)))
        label_tensor_resize_4 = tf.image.resize_bilinear(images=label_tensor,
                                                         size=(int(ori_height / 4), int(ori_width / 4)))
        label_list = [label_tensor_resize_4, label_tensor_resize_2, label_tensor_ori]
        lambda_i = [0.6, 0.8, 1.0]
        # 计算lm_loss(见公式(5))
        lm_loss = tf.constant(0.0, tf.float32)
        with tf.variable_scope(name):
            inference_ret = self.build_autoencoder(input_tensor=input_tensor, name='autoencoder_inference')
            output_list = [inference_ret['skip_1'], inference_ret['skip_2'], inference_ret['skip_3']]
            for index, output in enumerate(output_list):
                mse_loss = tf.losses.mean_squared_error(output, label_list[index]) * lambda_i[index]
                lm_loss = tf.add(lm_loss, mse_loss)

            # 计算lp_loss(见公式(6))
            src_vgg_feats = self._vgg_extractor.extract_feats(input_tensor=label_tensor,
                                                              name='vgg_feats',
                                                              reuse=False)
            pred_vgg_feats = self._vgg_extractor.extract_feats(input_tensor=output_list[-1],
                                                               name='vgg_feats',
                                                               reuse=True)

            lp_losses = []
            for index, feats in enumerate(src_vgg_feats):
                lp_losses.append(tf.losses.mean_squared_error(src_vgg_feats[index], pred_vgg_feats[index]))
            lp_loss = tf.reduce_mean(lp_losses)

            loss = tf.add(lm_loss, lp_loss)

        return loss, inference_ret['skip_3']


if __name__ == '__main__':
    input_image = tf.placeholder(dtype=tf.float32, shape=[1, 256, 256, 3])
    auto_label_image = tf.placeholder(dtype=tf.float32, shape=[1, 256, 256, 3])
    rnn_label_image = tf.placeholder(dtype=tf.float32, shape=[1, 256, 256, 1])
    net = GenerativeNet(phase=tf.constant('train', tf.string))
    rnn_loss = net.compute_attentive_rnn_loss(input_image, rnn_label_image, name='rnn_loss')
    autoencoder_loss = net.compute_autoencoder_loss(input_image, auto_label_image, name='autoencoder_loss')

    for vv in tf.trainable_variables():
        print(vv.name)
