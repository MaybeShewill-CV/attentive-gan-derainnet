#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2020/1/16 下午4:42
# @Author  : LuoYao
# @Site    : ICode
# @File    : freeze_attentive_gan_derain_net.py
# @IDE: PyCharm
"""
Freeze model tools
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import graph_util

from attentive_gan_model import derain_drop_net

MODEL_WEIGHTS_FILE_PATH = './test.ckpt'

# construct compute graph
input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 240, 360, 3], name='input_tensor')
net = derain_drop_net.DeRainNet(phase=tf.constant('test', tf.string))
output, attention_maps = net.inference(input_tensor=input_tensor, name='derain_net')
output = tf.squeeze(output, axis=0, name='final_output')
# attention_maps = tf.squeeze(attention_maps, axis=0, name='final_attention_maps')

# create a session
saver = tf.train.Saver()

sess_config = tf.ConfigProto(allow_soft_placement=True)
sess_config.gpu_options.per_process_gpu_memory_fraction = 0.85
sess_config.gpu_options.allow_growth = False
sess_config.gpu_options.allocator_type = 'BFC'

sess = tf.Session(config=sess_config)

# import best model
saver.restore(sess, MODEL_WEIGHTS_FILE_PATH)  # variables

# get graph definition
gd = sess.graph.as_graph_def()

# fix batch norm nodes
for node in gd.node:
    # print(node.name)
    if node.op == 'RefSwitch':
        node.op = 'Switch'
        for index in range(len(node.input)):
            if 'moving_' in node.input[index]:
                node.input[index] = node.input[index] + '/read'
    elif node.op == 'AssignSub':
        node.op = 'Sub'
        if 'use_locking' in node.attr:
            del node.attr['use_locking']

# generate protobuf
converted_graph_def = graph_util.convert_variables_to_constants(
    sess, gd, ["input_tensor", "final_output"])
tf.train.write_graph(converted_graph_def, './', 'attentive_gan_derain.pb', as_text=False)
