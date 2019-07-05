#!/usr/bin/env python

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

# for your packages to be recognized by python
d = generate_distutils_setup(
 packages=['support_surface_saver_ros'],
 package_dir={'support_surface_saver_ros': 'ros/src/support_surface_saver_ros'}
)

setup(**d)
