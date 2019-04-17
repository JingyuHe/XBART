from setuptools import setup, Extension,find_packages
import numpy

from sys import platform
if platform == "win32":
  compile_args = []
else:
  compile_args = ["-std=gnu++11", "-fpic",  "-g"]
if platform == "darwin":
  compile_args.append("-mmacosx-version-min=10.9") # To ensure gnu+11 and all std libs

XBART_cpp_module = Extension('_xbart_cpp_',
                           sources=['xbart/xbart_wrap.cxx', 'xbart/xbart.cpp',
                                    "src/utility.cpp",'src/fit_std_main_loop.cpp',
                                      "src/sample_int_crank.cpp",
                                        "src/common.cpp" ,   "src/forest.cpp",    
                                        "src/tree.cpp","src/thread_pool.cpp"

                                    ],
                            language= "c++",
                            include_dirs = [numpy.get_include(),'.', "src","xbart"],
                           extra_compile_args=compile_args#,"-larmadillo", "-llapack", "-lblas"]
                           )

setup (name = 'xbart',
       version = '0.1.2',
       author      = "Jingyu He, Saar Yalov, P. Richard Hahn",
       description = """XBART project""",
       include_dirs = [numpy.get_include(),'.',"src","xbart"],
       ext_modules = [XBART_cpp_module],
       install_requires=['numpy'],
       license="Apache-2.0",
       py_modules = ["xbart"],
       packages=find_packages()
       )

