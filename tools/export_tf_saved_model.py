#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 19-3-8 上午10:46
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/attentive-gan-derainnet
# @File    : export_tf_saved_model.py
# @IDE: PyCharm
"""
Export tensorflow saved model
"""
import os.path as ops
import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import saved_model as sm

from config import global_config
from attentive_gan_model import derain_drop_net


CFG = global_config.cfg

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
_CHANNEL_MEANS = [_B_MEAN, _G_MEAN, _R_MEAN]


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--export_dir', type=str, help='The model export dir')
    parser.add_argument('--ckpt_path', type=str, help='The pretrained ckpt model weights file path')

    return parser.parse_args()


def build_saved_model(ckpt_path, export_dir):
    """
    Convert source ckpt weights file into tensorflow saved model
    :param ckpt_path:
    :param export_dir:
    :return:
    """

    if ops.exists(export_dir):
        raise ValueError('Export dir must be a dir path that does not exist')

    assert ops.exists(ops.split(ckpt_path)[0])

    # build inference tensorflow graph
    image_tensor = tf.placeholder(dtype=tf.float32,
                                  shape=[1, CFG.TRAIN.CROP_IMG_HEIGHT, CFG.TRAIN.CROP_IMG_WIDTH, 3],
                                  name='input_tensor')
    # set nsfw net
    phase = tf.constant('test', dtype=tf.string)
    derain_net = derain_drop_net.DeRainNet(phase=phase)

    # compute inference logits
    output, attention_maps = derain_net.inference(input_tensor=image_tensor, name='derain_net')

    # scale image
    output = tf.squeeze(output, 0)
    b, g, r = tf.split(output, num_or_size_splits=3, axis=-1)
    scaled_channel = []
    for channel in [b, g, r]:
        tmp = (channel - tf.reduce_min(channel)) * 255.0 / (tf.reduce_max(channel) - tf.reduce_min(channel))
        scaled_channel.append(tmp)
    output = tf.concat(values=scaled_channel, axis=-1)
    output = tf.cast(output, tf.uint8, name='derain_image_result')

    # set tensorflow saver
    saver = tf.train.Saver()

    # Set sess configuration
    sess_config = tf.ConfigProto(device_count={"GPU": 0})
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TEST.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    with sess.as_default():

        saver.restore(sess=sess, save_path=ckpt_path)

        # set model save builder
        saved_builder = sm.builder.SavedModelBuilder(export_dir)

        # add tensor need to be saved
        saved_input_tensor = sm.utils.build_tensor_info(image_tensor)
        saved_prediction_tensor = sm.utils.build_tensor_info(output)

        # build SignatureDef protobuf
        signatur_def = sm.signature_def_utils.build_signature_def(
            inputs={'input_tensor': saved_input_tensor},
            outputs={'prediction': saved_prediction_tensor},
            method_name='derain_predict'
        )

        # add graph into MetaGraphDef protobuf
        saved_builder.add_meta_graph_and_variables(
            sess,
            tags=[sm.tag_constants.SERVING],
            signature_def_map={sm.signature_constants.REGRESS_INPUTS: signatur_def}
        )

        # save model
        saved_builder.save()

    return


def test_load_saved_model(saved_model_dir):
    """

    :param saved_model_dir:
    :return:
    """
    image = cv2.imread('data/test_data/test_1.png', cv2.IMREAD_COLOR)
    image = cv2.resize(src=image,
                       dsize=(CFG.TRAIN.CROP_IMG_WIDTH, CFG.TRAIN.CROP_IMG_HEIGHT),
                       interpolation=cv2.INTER_LINEAR)
    image_vis = image
    image = np.divide(np.array(image, np.float32), 127.5) - 1.0
    image = np.expand_dims(image, 0)

    # Set sess configuration
    sess_config = tf.ConfigProto(device_count={"GPU": 0})
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TEST.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    with sess.as_default():

        meta_graphdef = sm.loader.load(
            sess,
            tags=[sm.tag_constants.SERVING],
            export_dir=saved_model_dir)

        signature_def_d = meta_graphdef.signature_def
        signature_def_d = signature_def_d[sm.signature_constants.REGRESS_INPUTS]

        image_input_tensor = signature_def_d.inputs['input_tensor']
        prediction_tensor = signature_def_d.outputs['prediction']

        input_tensor = sm.utils.get_tensor_from_tensor_info(image_input_tensor, sess.graph)
        predictions = sm.utils.get_tensor_from_tensor_info(prediction_tensor, sess.graph)

        prediction_val = sess.run(predictions, feed_dict={input_tensor: image})

        plt.figure('source image')
        plt.imshow(image_vis[:, :, (2, 1, 0)])
        plt.figure('derain image')
        plt.imshow(prediction_val[:, :, (2, 1, 0)])

        plt.show()


if __name__ == '__main__':
    """
    build saved model
    """
    # init args
    args = init_args()

    # build saved model
    build_saved_model(args.ckpt_path, args.export_dir)

    # test build saved model
    test_load_saved_model(args.export_dir)
