#ifndef GUARD_utility_h
#define GUARD_utility_h

#include "common.h"


// copy NumericMatrix to STD matrix
xinfo copy_xinfo(Rcpp::NumericMatrix& X);

// copy IntegerMatrix to STD matrix
xinfo_sizet copy_xinfo_sizet(Rcpp::IntegerMatrix& X);

// // initialize STD matrix
xinfo ini_xinfo(size_t N, size_t p);

// // initialize STD integer matrix
xinfo_sizet ini_xinfo_sizet(size_t N, size_t p);

std::vector<double> row_sum(xinfo& X);

std::vector<double> col_sum(xinfo& X);

double sum_squared(std::vector<double> v);

double sum_vec(std::vector<double>& v);

void seq_gen(size_t start, size_t end, size_t length_out, arma::uvec& vec);

void cumsum_chunk(arma::vec& y, arma::uvec& ind, arma::vec& y_cumsum_chunk);

void calculate_y_cumsum(arma::vec& y, arma::uvec& ind, arma::vec& y_cumsum, arma::vec& y_cumsum_inv);


#endif