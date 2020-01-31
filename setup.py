#!/usr/bin/env python
# -*- coding: utf-8 -*-
desc = """\
Python Image Displacement Identification
========================================
This module supports the identification of disploacement from image date

For the showcase see: `github.com/ladisk/pyIDI <https://github.com/ladisk/pyIDI/blob/master/Showcase.ipynb>`_
"""

import os
import re
from setuptools import setup

regexp = re.compile(r'.*__version__ = [\'\"](.*?)[\'\"]', re.S)

base_path = os.path.dirname(__file__)

init_file = os.path.join(base_path, 'pyidi', '__init__.py')
with open(init_file, 'r') as f:
    module_content = f.read()

    match = regexp.match(module_content)
    if match:
        version = match.group(1)
    else:
        raise RuntimeError(
            'Cannot find __version__ in {}'.format(init_file))

with open('README.md', 'r') as f:
    readme = f.read()


def parse_requirements(filename):
    ''' Load requirements from a pip requirements file '''
    with open(filename, 'r') as fd:
        lines = []
        for line in fd:
            line.strip()
            if line and not line.startswith("#"):
                lines.append(line)
    return lines

requirements = parse_requirements('requirements.txt')

print(version, requirements, readme)

setup(name='pyidi',
      version=version,
      author='Klemen Zaletelj, Domen Gorjup, Janko Slavič',
      author_email='janko.slavic@fs.uni-lj.si, ladisk@gmail.com',
      description='Python Image Displacement Identification.',
      url='https://github.com/ladisk/pyidi',
      packages=['pyidi', 'pyidi.methods'],
      long_description=readme,
      install_requires=requirements,
      keywords='computer vision dic gradient-based image identification',
      )
