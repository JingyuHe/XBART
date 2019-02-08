cp -r ../src .
swig -c++ -python -py3  abarth.i
python3 setup.py sdist --formats=gztar,zip bdist_wheel 
pip3 install dist/abarth-0.1.tar.gz --user
python3 tests/test.py