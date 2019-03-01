#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 19-3-1 下午1:47
# @Author  : Luo Yao
# @Site    : http://icode.baidu.com/repos/baidu/personal-code/Luoyao
# @File    : context.py
# @IDE: PyCharm
"""
environment module
"""
import os.path as ops
import sys
sys.path.insert(0, ops.abspath(ops.join(ops.dirname(__file__), '..')))

import config.global_config as global_config
from data_provider import tf_io_pipline_tools as tf_io_pipline_tools
from data_provider import data_feed_pipline as data_feed_pipline