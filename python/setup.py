from setuptools import setup, Extension,find_packages
import numpy
import os

from sys import platform
if platform == "win32":
  compile_args = []
else:
  compile_args = ["-std=gnu++11", "-fpic",  "-g"]
if platform == "darwin":
  compile_args.append("-mmacosx-version-min=10.9") # To ensure gnu+11 and all std libs
  os.environ["MACOSX_DEPLOYMENT_TARGET"] = "10.9"

XBART_cpp_module = Extension('_xbart_cpp_',
                           sources=['xbart/xbart_wrap.cxx', 'xbart/xbart.cpp',
                                    "src/utility.cpp",'src/fit_std_main_loop.cpp',
                                      "src/sample_int_crank.cpp",
                                        "src/common.cpp" ,   "src/forest.cpp",    
                                        "src/tree.cpp","src/thread_pool.cpp",
                                        "src/cdf.cpp","src/json_io.cpp"
                                    ],
                            language= "c++",
                            include_dirs = [numpy.get_include(),'.', "src","xbart"],
                            extra_compile_args=compile_args
                           )


def readme():
    with open('README.md') as f:
        return f.read()

setup (name = 'xbart',
       version = '0.1.6',
       author      = "Jingyu He, Saar Yalov, P. Richard Hahn, Lee Reeves",
       description = """XBART project""",
       long_descripition = readme(),
       include_dirs = [numpy.get_include(),'.',"src","xbart"],
       ext_modules = [XBART_cpp_module],
       install_requires=['numpy'],
       license="Apache-2.0",
       py_modules = ["xbart"],
       python_requires='>3.6',
       packages=find_packages()
       )

