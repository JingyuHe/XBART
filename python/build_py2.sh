#! /bin/bash
cp -r ../src .
swig -c++ -python abarth.i
#swig -c++  -includeall  -python abarth.i
/usr/bin/python setup.py build_ext --inplace
rm -rf src
/usr/bin/python tests/test.py

