#! /bin/bash
cp -r ../src .
swig -c++ -python -py3 abarth.i
python setup.py build_ext --inplace
rm -rf src
python tests/test.py
