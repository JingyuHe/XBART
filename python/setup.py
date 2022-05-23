from setuptools import setup, Extension, find_packages
from distutils.sysconfig import get_config_var
from distutils.version import LooseVersion
import numpy
import os
import sys
import platform 
if sys.platform == "win32":
    compile_args = []
    link_args = []
else:
    compile_args = ["-std=c++17", "-fpic",  "-g"]
    link_args = ["-larmadillo"]
if sys.platform == "darwin":
    if 'MACOSX_DEPLOYMENT_TARGET' not in os.environ:
        current_system = LooseVersion(platform.mac_ver()[0])
        python_target = LooseVersion(
            get_config_var('MACOSX_DEPLOYMENT_TARGET'))
        if python_target < '10.9' and current_system >= '10.9':
            os.environ['MACOSX_DEPLOYMENT_TARGET'] = '10.9'


XBART_cpp_module = Extension('_xbart_cpp_',
                             sources=['xbart/xbart_wrap.cxx', 'xbart/xbart.cpp',
                                      "src/utility.cpp", 'src/mcmc_loop.cpp',
                                      "src/sample_int_crank.cpp",
                                      "src/common.cpp",  
                                      "src/tree.cpp", "src/thread_pool.cpp",
                                      "src/cdf.cpp", "src/json_io.cpp","src/model.cpp"
                                      ],
                             language="c++",
                             include_dirs=[
                                 numpy.get_include(), '.', "src", "xbart"],
                             extra_compile_args=compile_args,
                             extra_link_args = link_args
                             )


def readme():
    with open('README.md') as f:
        return f.read()


setup(name='xbart',
      version='0.1.7',
      author="Jingyu He, Saar Yalov, P. Richard Hahn",
      description="""XBART project""",
      long_description=readme(),
      include_dirs=[numpy.get_include(), '.', "src", "xbart"],
      ext_modules=[XBART_cpp_module],
      install_requires=['numpy'],
      license="Apache-2.0",
      py_modules=["xbart"],
      python_requires='>3.5',
      packages=find_packages()
      )
