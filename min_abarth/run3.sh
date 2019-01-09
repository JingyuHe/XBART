#! /bin/bash
swig -c++ -python -py3 abarth.i
python3 setup.py build_ext --inplace
python3 test.py
