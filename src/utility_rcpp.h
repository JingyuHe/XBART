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

void rcpp_to_std2(arma::mat &y, arma::mat &Z, arma::mat &X_con, arma::mat &X_mod, std::vector<double> &y_std, double &y_mean, matrix<double> &Z_std, Rcpp::NumericMatrix &X_std_con, Rcpp::NumericMatrix &X_std_mod, matrix<size_t> &Xorder_std_con, matrix<size_t> &Xorder_std_mod);

void Matrix_to_NumericMatrix(matrix<double> &a, Rcpp::NumericMatrix &b);

#endif
