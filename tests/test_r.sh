#! /bin/bash
echo Building R
cd ../../
R CMD REMOVE XBART
R CMD INSTALL XBART
cd XBART/tests/
echo Testing R
Rscript test.R
