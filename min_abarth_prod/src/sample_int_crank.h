#ifndef GUARD_sample_int_crank_h
#define GUARD_sample_int_crank_h


#include "utility.h"


//ADDED
#include "rn.h"

// using namespace Rcpp;
//#ifndef SWIG
#include <queue>
//#endif


void check_args(int n, int size, const std::vector<double> &prob);


//std::vector<size_t> sample_int_crank(int n, int size, std::vector<double> prob);


std::vector<double> sample_int_ccrank(int n, int size, std::vector<double> prob);


// std::vector<size_t> sample_int_expj(int n, int size, std::vector<double> prob);


// std::vector<size_t> sample_int_expjs(int n, int size, std::vector<double> prob);



#endif
