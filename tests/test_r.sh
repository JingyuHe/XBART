#! /bin/bash
echo Building R
cd ../../
R CMD REMOVE abarth
R CMD INSTALL abarth
cd abarth/tests/
echo Testing R
Rscript test.R