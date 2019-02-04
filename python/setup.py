#!/usr/bin/env python

"""
setup.py file for SWIG example
"""
import setuptools
from setuptools import setup, Extension
import numpy

from sys import platform
compile_args = ["-std=gnu++11", "-v","-DNDEBUG"]
if platform == "darwin":
  compile_args.append("-mmacosx-version-min=10.9") # To ensure gnu+11 and all std libs

abarth_module = Extension('_abarth',
                           sources=['abarth_wrap.cxx', 'abarth.cpp',
                                    "src/utility.cpp",'src/fit_std_main_loop.cpp',
                                      "src/sample_int_crank.cpp",  "src/treefuns.cpp",
                                        "src/common.cpp" ,   "src/forest.cpp",    "src/tree.cpp",

                                    ],


                            language= "c++",
                            #libraries =["/Library/Frameworks/Python.framework/Versions/3.6/lib"],
                            #include_dirs = ['/Library/Frameworks/Python.framework/Versions/3.6/include/python3.6m'], # temp...
                            include_dirs = [numpy.get_include(),'.', "src"],
                           extra_compile_args=compile_args#,"-larmadillo", "-llapack", "-lblas"]
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

