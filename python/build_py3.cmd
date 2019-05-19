rem For Windows
mkdir src
copy ..\src src
swig -c++ -python -py3 XBART.i
python setup.py sdist bdist_wheel
python tests/test.py
