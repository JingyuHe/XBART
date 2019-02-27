#! /bin/bash
PYTHON_BIN=python
DIST_FLAG=false

function usage(){
    cat << ENDUSAGE
    usage:
    $0 [-p|--python  PYTHON_BIN] [-d|--dist] [-h|--help]
    
    options:
    -p|--python : path to python bin 
    -d| --dist  : include if building dist
    -h|--help   : see documantation

ENDUSAGE
}

while [[ "$#" > 0 ]]
do
    case $1 in
      -p|--python) PYTHON_BIN=$2;shift;;
      -h|--help) usage; exit; ;;
      -d|--dist) DIST_FLAG=true;shift;;
      -*|--*) printf "\n\n   ERROR: Unsupported option $1\n\n"; usage; exit; ;;
    esac
    shift
done

echo Building python
if [[ ! $(command -v "$PYTHON_BIN") ]]  
then 
  echo "error $PYTHON_BIN not found. Make sure PYTHON_BIN is properly defined"
  exit 1
fi

./remove.sh
./dist_remove.sh
$PYTHON_BIN -m pip uninstall abarth 
cp -r ../src .

ver=$($PYTHON_BIN -V 2>&1 | sed 's/.* \([0-9]\).\([0-9]\).*/\1\2/')
if [ "$ver" -le "27" ]; then
    echo "Running script with python $ver" 
    swig -c++ -python abarth.i
else
  echo "Running script with python $ver" 
  swig -c++ -python -py3  abarth.i
fi

if $DIST_FLAG;then
  $PYTHON_BIN setup.py sdist --formats=gztar,zip bdist_wheel 
  $PYTHON_BIN -m pip install dist/*.tar.gz --user 
else
  $PYTHON_BIN setup.py build_ext --inplace
fi
rm -rf src
$PYTHON_BIN tests/test.py