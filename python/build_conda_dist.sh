cp -r ../src .
swig -c++ -python -py3  abarth.i
python setup.py sdist --formats=gztar,zip bdist_wheel 
rm -rf src
pip install dist/abarth-0.1.tar.gz --user
python tests/test.py