#! /bin/bash
R CMD REMOVE abarth
rm abarth_2.5.tar.gz
R CMD build abarth
R CMD INSTALL abarth
Rscript cat_test.R