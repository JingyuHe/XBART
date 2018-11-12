#ifndef GUARD_sample_int_crank_h
#define GUARD_sample_int_crank_h


#include "utility.h"

using namespace Rcpp;

#include <queue>


void check_args2(int n, int size, const std::vector<double> &prob);


std::vector<size_t> sample_int_crank2(int n, int size, std::vector<double> prob);


std::vector<double> sample_int_ccrank2(int n, int size, std::vector<double> prob);


std::vector<size_t> sample_int_expj2(int n, int size, std::vector<double> prob);


std::vector<size_t> sample_int_expjs2(int n, int size, std::vector<double> prob);



void check_args(int n, int size, const NumericVector &prob);


IntegerVector sample_int_crank(int n, int size, NumericVector prob);


SEXP sample_int_ccrank(int n, int size, NumericVector prob);


IntegerVector sample_int_expj(int n, int size, NumericVector prob);

IntegerVector sample_int_expjs(int n, int size, NumericVector prob);



#endif
