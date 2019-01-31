#! /bin/bash
swig -c++ -python abarth.i
python setup.py build_ext --inplace
python test.py
