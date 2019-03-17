#! /bin/bash
echo Building python
cd ../python/
./remove.sh
./dist_remove.sh
cp -r ../src .
ver=$(python -V 2>&1 | sed 's/.* \([0-9]\).\([0-9]\).*/\1\2/')
if [ "$ver" -lt "27" ]; then
    echo "Running script with python $ver" 
    swig -c++ -python XBART.i
else
	echo "Running script with python $ver" 
	swig -c++ -python -py3  XBART.i
fi
python setup.py build_ext --inplace
python ../tests/test.py