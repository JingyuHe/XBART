#!/usr/bin/env python

"""
setup.py file for SWIG example
"""
import setuptools
from distutils.core import setup, Extension
import numpy
import os


# os.environ["CC"] = "g++-4.7" 
# os.environ["CXX"] = "g++-4.7"
abarth_module = Extension('_abarth',
                           sources=['abarth_wrap.cxx', 'abarth.cpp',
                                    'src/utility.cpp','src/fit_std_main_loop.cpp',
                                      "src/sample_int_crank.cpp",  "src/treefuns.cpp",
                                        "src/common.cpp" ,   "src/forest.cpp",    "src/tree.cpp"

                                    ],
                          # sources=['abarth_wrap.cxx', 'abarth.cpp',
                          #           'utility.cpp','fit_std_main_loop.cpp',
                          #             "sample_int_crank.cpp",  "treefuns.cpp",
                          #               "common.cpp" ,   "forest.cpp",    "tree.cpp"

                          #          ],

                            language= "c++",
                            #libraries =["/Library/Frameworks/Python.framework/Versions/3.6/lib"],
                            #include_dirs = ['/Library/Frameworks/Python.framework/Versions/3.6/include/python3.6m'], # temp...
                            include_dirs = [numpy.get_include(),'.', "src"],
                           extra_compile_args=["-mmacosx-version-min=10.9",'-std=gnu++11']#,"-larmadillo", "-llapack", "-lblas"]
                           )

setup (name = 'abarth',
       version = '0.01',
       author      = "Saar Yalov",
       description = """Abarth project""",
       include_dirs = [numpy.get_include(),'.',"src"],
       ext_modules = [abarth_module],
       sources = ["abarth.py"],
      install_requires=['numpy'],
       py_modules = ["abarth"],
       include_package_data=True
       )

