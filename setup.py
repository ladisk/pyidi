#!/usr/bin/env python
# -*- coding: utf-8 -*-
desc = """\
Python Image Displacement Identification
========================================
This module supports the identification of disploacement from image date

For the showcase see: `github.com/ladisk/pyIDI <https://github.com/ladisk/pyIDI/blob/master/Showcase.ipynb>`_
"""

from setuptools import setup

setup(name='pyidi',
      version='0.19',
      author='Klemen Zaletelj, Domen Gorjup, Janko Slavič',
      author_email='janko.slavic@fs.uni-lj.si, ladisk@gmail.com',
      description='Python Image Displacement Identification.',
      url='https://github.com/ladisk/pyidi',
      packages=['pyidi', 'pyidi.methods'],
      long_description=desc,
      install_requires=['numpy>=1.15.4', 'scipy>=1.1.0', 'tqdm', 'matplotlib>=3.0.0', 'pyMRAW>=0.22'],
      keywords='computer vision dic gradient-based image identification',
      )
