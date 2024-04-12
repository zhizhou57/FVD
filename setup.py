# -*- coding: utf-8 -*-
# @Time    : 2024/4/12 11:32
# @Author  : yesliu
# @File    : setup.py.py
from setuptools import setup, find_packages

setup(
    name='fvdcal',
    version='v1.0',
    description='A module for FVD calculations',
    packages=find_packages(),
    author='yesheng liu',
    author_email='yes_liu@whu.edu.cn',
    url='https://github.com/zhizhou57/FVD',
    requires=['decord','torch'],
    license='MIT'
)