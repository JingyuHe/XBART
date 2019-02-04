#! /bin/bash
[ -e _abarth.so ] && rm _abarth.so 
[ -e build ] && rm -rf build

[ -e src/abarth.py ] && rm src/abarth.py
[ -e src/abarth.pyc ] && rm src/abarth.pyc
[ -e src/abarth_wrap.cxx  ] && rm src/abarth_wrap.cxx 

[ -e python/abarth.py ] && rm python/abarth.py
[ -e python/abarth.pyc ] && rm python/abarth.pyc
[ -e python/abarth_wrap.cxx  ] && rm python/abarth_wrap.cxx 

[ -e abarth.py ] && rm abarth.py
[ -e abarth.pyc ] && rm abarth.pyc
[ -e abarth_wrap.cxx  ] && rm abarth_wrap.cxx 

[ -e _abarth.cpython-36m-x86_64-linux-gnu.so ]&& rm _abarth.cpython-36m-x86_64-linux-gnu.so
[ -e _abarth.cpython-36m-darwin.so ] && rm _abarth.cpython-36m-darwin.so
[ -e __pycache__/abarth.cpython-36.pyc ] && rm __pycache__/abarth.cpython-36.pyc