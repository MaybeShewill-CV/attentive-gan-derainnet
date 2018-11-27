#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-7-3 下午4:28
# @Author  : Luo Yao
# @Site    : http://icode.baidu.com/repos/baidu/personal-code/Luoyao
# @File    : data_provider.py
# @IDE: PyCharm
"""
训练数据生成器
"""
import os.path as ops

import numpy as np
import cv2

from config import global_config

CFG = global_config.cfg


class DataSet(object):
    """
    实现数据集类
    """

    def __init__(self, dataset_info_file):
        """

        :param dataset_info_file:
        """
        self._gt_img_list, self._gt_label_list = self._init_dataset(dataset_info_file)
        self._random_dataset()
        self._next_batch_loop_count = 0

    def _init_dataset(self, dataset_info_file):
        """

        :param dataset_info_file:
        :return:
        """
        gt_img_list = []
        gt_label_list = []

        assert ops.exists(dataset_info_file), '{:s}　不存在'.format(dataset_info_file)

        with open(dataset_info_file, 'r') as file:
            for _info in file:
                info_tmp = _info.strip(' ').split()

                gt_img_list.append(info_tmp[1])
                gt_label_list.append(info_tmp[0])

        return gt_img_list, gt_label_list

    def _random_dataset(self):
        """

        :return:
        """
        assert len(self._gt_img_list) == len(self._gt_label_list)

        random_idx = np.random.permutation(len(self._gt_img_list))
        new_gt_img_list = []
        new_gt_label_list = []

        for index in random_idx:
            new_gt_img_list.append(self._gt_img_list[index])
            new_gt_label_list.append(self._gt_label_list[index])

        self._gt_img_list = new_gt_img_list
        self._gt_label_list = new_gt_label_list

    @staticmethod
    def _generate_training_pathches(gt_img, label_img, mask_img, patch_nums=1, patch_size=128):
        """
        在标签图像和原始图像上随机扣取图像对
        :param gt_img:
        :param label_img:
        :param mask_img:
        :param patch_nums:
        :param patch_size:
        :return:
        """
        np.random.seed(1234)

        gt_img_patch = []
        label_img_patch = []
        mask_img_patch = []

        if gt_img.shape != label_img.shape:
            label_img = cv2.resize(label_img, dsize=(gt_img.shape[1], gt_img.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)

        img_h = gt_img.shape[0]
        img_w = gt_img.shape[1]

        for i in range(patch_nums):
            seed_x = np.random.randint(int(patch_size / 2 + 1), img_w - int(patch_size / 2) - 1)
            seed_y = np.random.randint(int(patch_size / 2 + 1), img_h - int(patch_size / 2) - 1)

            gt_img_patch.append(gt_img[
                                seed_y - int(patch_size / 2):seed_y + int(patch_size / 2),
                                seed_x - int(patch_size / 2):seed_x + int(patch_size / 2),
                                :])

            mask_img_patch.append(mask_img[
                                  seed_y - int(patch_size / 2):seed_y + int(patch_size / 2),
                                  seed_x - int(patch_size / 2):seed_x + int(patch_size / 2)])

            tmp = label_img[seed_y - int(patch_size / 2):seed_y + int(patch_size / 2),
                            seed_x - int(patch_size / 2):seed_x + int(patch_size / 2),
                            :]
            label_img_patch.append(tmp)

        return gt_img_patch, label_img_patch, mask_img_patch

    @staticmethod
    def _morph_process(image):
        """
        图像形态学变化
        :param image:
        :return:
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
        close_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        open_image = cv2.morphologyEx(close_image, cv2.MORPH_OPEN, kernel)

        return open_image

    def next_batch(self, batch_size):
        """

        :param batch_size:
        :return:
        """
        assert len(self._gt_label_list) == len(self._gt_img_list)

        idx_start = batch_size * self._next_batch_loop_count
        idx_end = batch_size * self._next_batch_loop_count + batch_size

        if idx_end > len(self._gt_label_list):
            self._random_dataset()
            self._next_batch_loop_count = 0
            return self.next_batch(batch_size)
        else:
            gt_img_list = self._gt_img_list[idx_start:idx_end]
            gt_label_list = self._gt_label_list[idx_start:idx_end]

            gt_imgs = []
            gt_labels = []
            mask_labels = []
            gt_img_paths = []

            for index, gt_img_path in enumerate(gt_img_list):
                gt_image = cv2.imread(gt_img_path, cv2.IMREAD_COLOR)
                label_image = cv2.imread(gt_label_list[index], cv2.IMREAD_COLOR)

                gt_image = cv2.resize(gt_image, (CFG.TRAIN.IMG_WIDTH, CFG.TRAIN.IMG_HEIGHT),
                                      interpolation=cv2.INTER_CUBIC)
                label_image = cv2.resize(label_image, (CFG.TRAIN.IMG_WIDTH, CFG.TRAIN.IMG_HEIGHT),
                                         interpolation=cv2.INTER_CUBIC)

                diff_image = np.abs(np.array(gt_image, np.float32) - np.array(label_image, np.float32))
                diff_image = diff_image.sum(axis=2)

                mask_image = np.zeros(diff_image.shape, np.float32)

                mask_image[np.where(diff_image >= 50)] = 1.
                mask_image = self._morph_process(mask_image)

                gt_image = np.divide(gt_image.astype(np.float32), 127.5) - 1
                label_image = np.divide(label_image.astype(np.float32), 127.5) - 1

                gt_imgs.append(gt_image)
                gt_labels.append(label_image)
                mask_labels.append(mask_image)
                gt_img_paths.append(gt_img_path)

            self._next_batch_loop_count += 1
            return gt_imgs, gt_labels, mask_labels, gt_img_paths


if __name__ == '__main__':
    val = DataSet('/media/baidu/Data/Gan_Derain_Dataset/train/train.txt')
    a1, a2, a3, a4 = val.next_batch(848)
    import matplotlib.pyplot as plt
    for index, gt_image in enumerate(a1):
        plt.figure('test_{:d}_src'.format(index))
        plt.imshow(gt_image[:, :, (2, 1, 0)])
        plt.figure('test_{:d}_label'.format(index))
        plt.imshow(a2[index][:, :, (2, 1, 0)])
        plt.figure('test_{:d}_mask'.format(index))
        plt.imshow(a3[index], cmap='gray')
        plt.show()
