#! /bin/bash
echo Building python
cd ../python/
bash build_py.sh -s -d
python tests/test.py

