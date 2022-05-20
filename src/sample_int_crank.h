#ifndef GUARD_sample_int_crank_h
#define GUARD_sample_int_crank_h

#include "utility.h"

// ADDED
#include "rn.h"

// using namespace Rcpp;
//#ifndef SWIG
#include <queue>
//#endif

void check_args(size_t n, size_t size, const std::vector<double> &prob);

std::vector<size_t> sample_int_ccrank(size_t n, size_t size, std::vector<double> prob, std::mt19937 &gen);

#endif
