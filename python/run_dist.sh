cp -r ../src src
swig -c++ -python abarth.i
python setup.py sdist --formats=gztar,zip bdist_wheel 
pip install dist/abarth-0.01.tar.gz --user 
python tests/test.py
