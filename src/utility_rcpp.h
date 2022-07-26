#ifndef GUARD_utility_rcpp_
#define GUARD_utility_rcpp_

// #include<RcppArmadillo.h>
#include "Rcpp.h"
#include <armadillo>
#include "X_struct.h"

using namespace arma;

// utility functions that rely on Rcpp packages

void rcpp_to_std2(arma::mat &y, arma::mat &X, arma::mat &Xtest, std::vector<double> &y_std, double &y_mean, Rcpp::NumericMatrix &X_std, Rcpp::NumericMatrix &Xtest_std, matrix<size_t> &Xorder_std);

void rcpp_to_std2(arma::mat &y, arma::mat &X, std::vector<double> &y_std, double &y_mean, Rcpp::NumericMatrix &X_std, matrix<size_t> &Xorder_std);

void rcpp_to_std2(arma::mat &X, Rcpp::NumericMatrix &X_std, matrix<size_t> &Xorder_std);

void rcpp_to_std2(arma::mat &y, arma::mat &Z, arma::mat &X, arma::mat &Ztest, arma::mat &Xtest, std::vector<double> &y_std, double &y_mean, matrix<double> &Z_std, Rcpp::NumericMatrix &X_std, matrix<double> &Ztest_std, Rcpp::NumericMatrix &Xtest_std, matrix<size_t> &Xorder_std);

void rcpp_to_std2(arma::mat &y, arma::mat &Z, arma::mat &X_ps, arma::mat &X_trt, std::vector<double> &y_std, double &y_mean, matrix<double> &Z_std, Rcpp::NumericMatrix &X_std_ps, Rcpp::NumericMatrix &X_std_trt, matrix<size_t> &Xorder_std_ps, matrix<size_t> &Xorder_std_trt);

void Matrix_to_NumericMatrix(matrix<double> &a, Rcpp::NumericMatrix &b);

#endif
