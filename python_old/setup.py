#!/usr/bin/env python

"""
setup.py file for SWIG example
"""

from distutils.core import setup, Extension
import numpy
import os

abarth_module = Extension('_abarth',
                           sources=['abarth_wrap.cxx', 'abarth.cpp'],
                           )

setup (name = 'abarth',
       version = '0.1',
       author      = "SWIG Docs",
       description = """Simple swig example from docs""",
       include_dirs = [numpy.get_include(),'.'],
       ext_modules = [abarth_module],
       py_modules = ["abarth"],
       )

