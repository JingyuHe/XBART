#! /bin/bash
rm -rf __pycache__/
rm -rf XBART.egg-info
rm -rf dist

[ -e _XBART.so ] && rm _XBART.so 
[ -e build ] && rm -rf build
[ -e XBART.egg-info ] && rm -rf XBART.egg-info

[ -e src/XBART.py ] && rm src/XBART.py
[ -e src/XBART.pyc ] && rm src/XBART.pyc
[ -e src/XBART_wrap.cxx  ] && rm src/XBART_wrap.cxx 

[ -e XBART.py ] && rm XBART.py
[ -e XBART.pyc ] && rm XBART.pyc
[ -e XBART_wrap.cxx  ] && rm XBART_wrap.cxx 

[ -e _XBART.cpython-36m-x86_64-linux-gnu.so ]&& rm _XBART.cpython-36m-x86_64-linux-gnu.so
[ -e _XBART.cpython-36m-darwin.so ] && rm _XBART.cpython-36m-darwin.so
[ -e __pycache__/XBART.cpython-36.pyc ] && rm __pycache__/XBART.cpython-36.pyc