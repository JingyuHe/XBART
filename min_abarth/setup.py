#!/usr/bin/env python

"""
setup.py file for SWIG example
"""

from distutils.core import setup, Extension
import numpy
import os

abarth_module = Extension('_abarth',
                           sources=['abarth_wrap.cxx', 'abarth.cpp',
                                    'utility.cpp','fit_std_main_loop.cpp',
                                      "sample_int_crank.cpp",  "treefuns.cpp",
                                        "common.cpp" ,   "forest.cpp",    "tree.cpp"

                                    ],
                           extra_compile_args=['-std=c++11',"-larmadillo", "-llapack", "-lblas"]
                           )

setup (name = 'abarth',
       version = '0.1',
       author      = "SWIG Docs",
       description = """Simple swig example from docs""",
       include_dirs = [numpy.get_include(),'.'],
       ext_modules = [abarth_module],
       #libraries=['random'],
       py_modules = ["abarth"],
       )

