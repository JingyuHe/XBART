#ifndef GUARD_utility_h
#define GUARD_utility_h

#include "common.h"
#include "tree.h"
#include "treefuns.h"

// copy NumericMatrix to STD matrix
xinfo create_xinfo(Rcpp::NumericMatrix& X);

// copy IntegerMatrix to STD matrix
xinfo_sizet create_xinfo_sizet(Rcpp::IntegerMatrix& X);


#endif