#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 19-3-1 上午10:41
# @Author  : Luo Yao
# @Site    : http://icode.baidu.com/repos/baidu/personal-code/Luoyao
# @File    : tf_io_pipline_tools.py
# @IDE: PyCharm
"""
Some tensorflow records io tools
"""
import os
import os.path as ops

import cv2
import numpy as np
import tensorflow as tf
import glog as log

from context import global_config

CFG = global_config.cfg

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
_CHANNEL_MEANS = [_B_MEAN, _G_MEAN, _R_MEAN]


def int64_feature(value):
    """

    :return:
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def bytes_feature(value):
    """

    :param value:
    :return:
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def morph_process(image):
    """
    图像形态学变化
    :param image:
    :return:
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    close_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    open_image = cv2.morphologyEx(close_image, cv2.MORPH_OPEN, kernel)

    return open_image


def write_example_tfrecords(rain_images_paths, clean_images_paths, tfrecords_path):
    """
    write tfrecords
    :param rain_images_paths:
    :param clean_images_paths:
    :param tfrecords_path:
    :return:
    """
    _tfrecords_dir = ops.split(tfrecords_path)[0]
    os.makedirs(_tfrecords_dir, exist_ok=True)

    log.info('Writing {:s}....'.format(tfrecords_path))

    with tf.python_io.TFRecordWriter(tfrecords_path) as _writer:
        for _index, _rain_image_path in enumerate(rain_images_paths):

            with open(_rain_image_path, 'rb') as f:
                check_chars = f.read()[-2:]
            if check_chars != b'\xff\xd9':
                log.error('Image file {:s} is not complete'.format(_rain_image_path))
                continue
            else:
                # prepare rain image
                _rain_image = cv2.imread(_rain_image_path, cv2.IMREAD_COLOR)
                _rain_image = cv2.resize(_rain_image,
                                         dsize=(CFG.TRAIN.IMG_WIDTH, CFG.TRAIN.IMG_HEIGHT),
                                         interpolation=cv2.INTER_LINEAR)
                _rain_image_raw = _rain_image.tostring()

                # prepare clean image
                _clean_image = cv2.imread(clean_images_paths[_index], cv2.IMREAD_COLOR)
                _clean_image = cv2.resize(_clean_image,
                                          dsize=(CFG.TRAIN.IMG_WIDTH, CFG.TRAIN.IMG_HEIGHT),
                                          interpolation=cv2.INTER_LINEAR)
                _clean_image_raw = _clean_image.tostring()

                # prepare mask image
                _diff_image = np.abs(np.array(_rain_image, np.float32) - np.array(_clean_image, np.float32))
                _diff_image = _diff_image.sum(axis=2)

                _mask_image = np.zeros(_diff_image.shape, np.float32)

                _mask_image[np.where(_diff_image >= 50)] = 1.
                _mask_image = morph_process(_mask_image)
                _mask_image = _mask_image.tostring()

                _example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'rain_image_raw': bytes_feature(_rain_image_raw),
                            'clean_image_raw': bytes_feature(_clean_image_raw),
                            'mask_image_raw': bytes_feature(_mask_image)
                        }))
                _writer.write(_example.SerializeToString())

    log.info('Writing {:s} complete'.format(tfrecords_path))

    return


def decode(serialized_example):
    """
    Parses an image and label from the given `serialized_example`
    :param serialized_example:
    :return:
    """
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'depth': tf.FixedLenFeature([], tf.int64)
        })

    # decode image
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image_shape = tf.stack([CFG.TRAIN.IMG_HEIGHT, CFG.TRAIN.IMG_WIDTH, 3])
    image = tf.reshape(image, image_shape)

    # Convert label from a scalar int64 tensor to an int32 scalar.
    label = tf.cast(features['label'], tf.int32)

    return image, label


def augment_for_train(image, label):
    """

    :param image:
    :param label:
    :return:
    """
    # first apply random crop
    image = tf.image.random_crop(value=image,
                                 size=[CFG.TRAIN.CROP_IMG_HEIGHT, CFG.TRAIN.CROP_IMG_WIDTH, 3],
                                 seed=tf.set_random_seed(1234),
                                 name='crop_image')
    # apply random flip
    image = tf.image.random_flip_left_right(image=image, seed=tf.set_random_seed(1234))

    return image, label


def augment_for_validation(image, label):
    """

    :param image:
    :param label:
    :return:
    """
    assert CFG.TRAIN.IMG_HEIGHT == CFG.TRAIN.IMG_WIDTH
    assert CFG.TRAIN.CROP_IMG_HEIGHT == CFG.TRAIN.CROP_IMG_WIDTH

    # apply central crop
    central_fraction = CFG.TRAIN.CROP_IMG_HEIGHT / CFG.TRAIN.IMG_HEIGHT
    image = tf.image.central_crop(image=image, central_fraction=central_fraction)

    return image, label


def normalize(image, label):
    """
    Normalize the image data by substracting the imagenet mean value
    :param image:
    :param label:
    :return:
    """

    if image.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0]')

    image_fp = tf.cast(image, dtype=tf.float32)
    means = tf.expand_dims(tf.expand_dims(_CHANNEL_MEANS, 0), 0)

    return image_fp - means, label