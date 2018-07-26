#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-7-18 下午5:27
# @Author  : Luo Yao
# @Site    : http://icode.baidu.com/repos/baidu/personal-code/Luoyao
# @File    : test_tmp.py
# @IDE: PyCharm
"""

"""
import os
import os.path as ops

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
try:
    from cv2 import cv2
except ImportError:
    pass

config = tf.ConfigProto(device_count={'GPU': 0})
serialized = config.SerializeToString()

print(list(map(hex, serialized)))

a = np.ones([15, 1], np.float32)
a[7] = 0
a[9] = 0
a[1] = 0

a_t = tf.convert_to_tensor(a)
b = tf.where(tf.not_equal(a_t, 0), a_t, a_t + 0.5)

with tf.Session() as sess:
    print(sess.run(b))


RAIN_IMAGE_PATH = '/media/baidu/Data/Gan_Derain_Dataset/train/data/1_rain.png'
CLEAN_IMAGE_PATH = '/media/baidu/Data/Gan_Derain_Dataset/train/gt_label/1_clean.png'

rain_image = cv2.imread(RAIN_IMAGE_PATH, cv2.IMREAD_COLOR)
clean_image = cv2.imread(CLEAN_IMAGE_PATH, cv2.IMREAD_COLOR)

diff_image = np.abs(np.array(cv2.cvtColor(rain_image, cv2.COLOR_BGR2GRAY), np.float32) -
                    np.array(cv2.cvtColor(clean_image, cv2.COLOR_BGR2GRAY), np.float32))

tmp_image = np.zeros(diff_image.shape)

tmp_image[np.where(diff_image > 30)] = 255

plt.figure('rain_image')
plt.imshow(rain_image, cmap='gray')
plt.figure('clean_image')
plt.imshow(clean_image, cmap='gray')
plt.figure('diff_image')
plt.imshow(tmp_image, cmap='gray')
plt.show()
