#ifndef GUARD_utility_h
#define GUARD_utility_h

#include "common.h"
#include "tree.h"
#include "treefuns.h"

// copy NumericMatrix to STD matrix
xinfo copy_xinfo(Rcpp::NumericMatrix& X);

// copy IntegerMatrix to STD matrix
xinfo_sizet copy_xinfo_sizet(Rcpp::IntegerMatrix& X);

// // initialize STD matrix
xinfo ini_xinfo(size_t p, size_t N);

// // initialize STD integer matrix
xinfo_sizet ini_xinfo_sizet(size_t p, size_t N);

#endif