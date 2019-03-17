#! /bin/bash
R CMD REMOVE XBART
rm XBART_2.5.tar.gz
R CMD build XBART
R CMD INSTALL XBART
Rscript cat_test.R