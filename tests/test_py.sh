#! /bin/bash
echo Building python
cd ../python/
bash build_py.sh -d
python tests/test.py

