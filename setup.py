#!/usr/bin/env python
# -*- coding: utf-8 -*-
desc = """\
Python Image Displacement Identification
=============
This module supports the identification of disploacement from image date

For the showcase see: https://github.com/ladisk/pyIDI/blob/master/pyIDI%20Showcase.ipynb
"""

from setuptools import setup
setup(name='pyIDI',
      version='0.10',
      author='Klemen Zaletelj, Domen Gorjup, Janko Slavič',
      author_email='janko.slavic@fs.uni-lj.si, ladisk@gmail.com',
      description='Python Image Displacement Identification.',
      url='https://github.com/ladisk/pyIDI',
      py_modules=['pyIDI'],
      long_description=desc,
      install_requires=['numpy', 'tqdm', 'matplotlib']
      )