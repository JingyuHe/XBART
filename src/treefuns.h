/*
 *  BART: Bayesian Additive Regression Trees
 *  Copyright (C) 2017 Robert McCulloch and Rodney Sparapani
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, a copy is available at
 *  https://www.R-project.org/Licenses/GPL-2
 */

#ifndef GUARD_treefuns_h
#define GUARD_treefuns_h

#include "tree.h"

// [[Rcpp::plugins(cpp11)]]


//--------------------------------------------------
//write cutpoint information to screen
void prxi(xinfo& xi);
//--------------------------------------------------
//evaluate tree tr on grid xi, write to os
void grm(tree& tr, xinfo& xi, std::ostream& os);
//--------------------------------------------------
//fit tree at matrix of x, matrix is stacked columns x[i,j] is *(x+p*i+j)
void fit(tree& t, xinfo& xi, size_t p, size_t n, double *x,  double* fv);
void fit_std(tree& t, size_t p, size_t n, double *x, std::vector<double>& fv);
void fit_noise_std(tree& t, size_t p, size_t n, double *x, std::vector<double>& fv);
//--------------------------------------------------
//does a (bottom) node have variables you can split on?
bool cansplit(tree::tree_p n, xinfo& xi);
//--------------------------------------------------
//find variables n can split on, put their indices in goodvars
void getgoodvars(tree::tree_p n, xinfo& xi,  std::vector<size_t>& goodvars);
//--------------------------------------------------
arma::vec fit_new(tree& tree, arma::mat& Xnew);
arma::vec fit_new_theta_noise(tree& tree, arma::mat& Xnew);


void fit_new_void(tree& tree, arma::mat& Xnew, arma::mat& pred, size_t& ind);





//--------------------------------------------------
//find best split variable and value, CART
void split_error(const arma::umat& Xorder, arma::vec& y, arma::uvec& best_split, arma::vec& least_error);
void split_xorder_std(xinfo_sizet& Xorder_left, xinfo_sizet& Xorder_right, xinfo_sizet& Xorder, double *  X, size_t split_var, size_t split_point, size_t N, size_t p);
void BART_likelihood(const arma::umat& Xorder, arma::vec& y, arma::vec& loglike, double tau, double sigma, size_t depth, size_t Nmin, double alpha, double beta);
void BART_likelihood_adaptive(const arma::umat& Xorder, arma::vec& y, arma::vec& loglike, double tau, double sigma, size_t depth, size_t Nmin, double alpha, double beta);
void BART_likelihood_std(size_t N, size_t p, xinfo_sizet& Xorder, double* y, std::vector<double>& loglike, double& tau, double& sigma, size_t& depth, double& alpha, double& beta);
void cumulative_sum_std(std::vector<double>& y_cumsum, std::vector<double>& y_cumsum_inv, double& y_sum, double* y, xinfo_sizet& Xorder, size_t& i, size_t& N);
//--------------------------------------------------
//split Xorder matrix for two subnodes 
void split_xorder(arma::umat& Xorder_left, arma::umat& Xorder_right, arma::umat& Xorder, arma::mat& X, size_t split_var, size_t split_point);
//--------------------------------------------------
//BART likelihood function
arma::vec BART_likelihood_function(arma::vec& n1, arma::vec& n2, arma::vec& s1, arma::vec& s2, double& tau, double& sigma, double& alpha, double& penalty);
//--------------------------------------------------
//generate a vector of integers
arma::uvec range(size_t start, size_t end);



#endif
