#! /bin/bash
cp -r ../src src
swig -c++ -python -py3 abarth.i
#swig -c++  -includeall  -python abarth.i
python3 setup.py build_ext --inplace
python3 tests/test.py
