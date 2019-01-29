swig -c++ -python abarth.i
python setup.py sdist
pip install dist/abarth-0.01.tar.gz --user
python tests/test.py
