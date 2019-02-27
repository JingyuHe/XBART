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

using namespace std;
// [[Rcpp::plugins(cpp11)]]

//--------------------------------------------------
//write cutpoint information to screen
void prxi(xinfo &xi);
//--------------------------------------------------
//evaluate tree tr on grid xi, write to os
void grm(tree &tr, xinfo &xi, std::ostream &os);
//--------------------------------------------------
//fit tree at matrix of x, matrix is stacked columns x[i,j] is *(x+p*i+j)
void fit(tree &t, xinfo &xi, size_t p, size_t n, double *x, double *fv);
void fit_std(tree &t, size_t p, size_t n, double *x, std::vector<double> &fv);
void fit_noise_std(tree &t, size_t p, size_t n, double *x, std::vector<double> &fv);
//--------------------------------------------------
//does a (bottom) node have variables you can split on?
bool cansplit(tree::tree_p n, xinfo &xi);
//--------------------------------------------------
//find variables n can split on, put their indices in goodvars
void getgoodvars(tree::tree_p n, xinfo &xi, std::vector<size_t> &goodvars);
//--------------------------------------------------

// Removed
// arma::vec fit_new(tree& tree, arma::mat& Xnew);
// arma::vec fit_new_theta_noise(tree& tree, arma::mat& Xnew);
void fit_new_theta_noise_std(tree &tree, const double *X, size_t p, size_t N, std::vector<double> &output);

// REmoved
//arma::vec fit_new_theta_noise_test(tree& tree, arma::mat& Xnew, const double * X_std, size_t p, size_t N, std::vector<double>& output);

//void fit_new_void(tree& tree, arma::mat& Xnew, arma::mat& pred, size_t& ind);
double sum_residual_squared(tree &tree, const double *X, const std::vector<double> &y, size_t p);

void fit_new_std(tree &tree, const double *X_std, size_t N, size_t p, std::vector<double> &output);
// void update_sufficient_stat(tree& tree, arma::mat& y, arma::mat& X, tree::npv& bv, tree::npv& bv2, double& tau, double& sigma, double& alpha, double& beta);

#endif
