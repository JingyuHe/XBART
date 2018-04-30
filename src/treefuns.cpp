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

#include "treefuns.h"

//--------------------------------------------------
//write cutpoint information to screen
void prxi(xinfo& xi)
{
   cout << "xinfo: \n";
   for(size_t v=0;v!=xi.size();v++) {
      cout << "v: " << v << std::endl;
      for(size_t j=0;j!=xi[v].size();j++) cout << "j,xi[v][j]: " << j << ", " << xi[v][j] << std::endl;
   }
   cout << "\n\n";
}
//--------------------------------------------------
//evalute tree tr on grid given by xi and write to os
void grm(tree& tr, xinfo& xi, std::ostream& os)
{
   size_t p = xi.size();
   if(p!=2) {
      cout << "error in grm, p !=2\n";
      return;
   }
   size_t n1 = xi[0].size();
   size_t n2 = xi[1].size();
   tree::tree_p bp; //pointer to bottom node
   double *x = new double[2];
   for(size_t i=0;i!=n1;i++) {
      for(size_t j=0;j!=n2;j++) {
         x[0] = xi[0][i];
         x[1] = xi[1][j];
         bp = tr.bn(x,xi);
         os << x[0] << " " << x[1] << " " << bp->gettheta() << " " << bp->nid() << std::endl;
      }
   }
   delete[] x;
}
//--------------------------------------------------
//fit tree at matrix of x, matrix is stacked columns x[i,j] is *(x+p*i+j)
void fit(tree& t, xinfo& xi, size_t p, size_t n, double *x,  double* fv)
{
   tree::tree_p bn;
   for(size_t i=0;i<n;i++) {
      bn = t.bn(x+i*p,xi);
      fv[i] = bn->gettheta();
   }
}


void fit_std(tree& t, size_t p, size_t n, double *x, std::vector<double>& fv){
    tree::tree_p bn;
    for(size_t i=0;i<n;i++){
        bn = t.bn_std(x+i*p);
        fv[i] = bn->gettheta();
    }
}

void fit_noise_std(tree& t, size_t p, size_t n, double *x, std::vector<double>& fv){
    tree::tree_p bn;
    for(size_t i = 0; i < n; i ++ ){
        bn = t.bn_std(x + i * p);
        fv[i] = bn->gettheta_noise();
    }
}



//--------------------------------------------------
//does this bottom node n have any variables it can split on.
bool cansplit(tree::tree_p n, xinfo& xi)
{
   size_t L,U;
   bool v_found = false; //have you found a variable you can split on
   size_t v=0;
   while(!v_found && (v < xi.size())) { //invar: splitvar not found, vars left
      L=0; U = xi[v].size()-1;
      n->rg(v,&L,&U);
      if(U>=L) v_found=true;
      v++;
   }
   return v_found;
}
//--------------------------------------------------
//find variables n can split on, put their indices in goodvars
void getgoodvars(tree::tree_p n, xinfo& xi,  std::vector<size_t>& goodvars)
{
   goodvars.clear();
   size_t L,U;
   for(size_t v=0;v!=xi.size();v++) {//try each variable
      L=0; U = xi[v].size()-1;
      n->rg(v,&L,&U);
      if(U>=L) goodvars.push_back(v);
   }
}


arma::vec fit_new(tree& tree, arma::mat& Xnew){
    // size_t p = Xnew.n_cols;
    size_t N = Xnew.n_rows;
    arma::vec result(N);
    tree::tree_p bn;
    arma::mat temp;
    for(size_t i = 0; i < N; i ++){
        // cout << i << endl;
        temp = Xnew.row(i);
        bn = tree.search_bottom(temp);
        result(i) = bn -> gettheta();
    }
    return result;
}



arma::vec fit_new_theta_noise(tree& tree, arma::mat& Xnew){
    // size_t p = Xnew.n_cols;
    size_t N = Xnew.n_rows;
    arma::vec result(N);
    tree::tree_p bn;
    arma::mat temp;
    for(size_t i = 0; i < N; i ++){
        // cout << i << endl;
        temp = Xnew.row(i);
        bn = tree.search_bottom(temp);
        result(i) = bn -> gettheta_noise();
    }
    return result;
}

void fit_new_void(tree& tree, arma::mat& Xnew, arma::mat& pred, size_t& ind){
    // size_t p = Xnew.n_cols;
    size_t N = Xnew.n_rows;
    arma::vec result(N);
    tree::tree_p bn;
    arma::mat temp;
    for(size_t i = 0; i < N; i ++){
        // cout << i << endl;
        temp = Xnew.row(i);
        bn = tree.search_bottom(temp);
        pred(i, ind) = bn -> gettheta();
    }
    return;
}


void split_xorder(arma::umat& Xorder_left, arma::umat& Xorder_right, arma::umat& Xorder, arma::mat& X, size_t split_var, size_t split_point){
    // preserve order of other variables
    size_t N = Xorder.n_rows;
    size_t left_ix = 0;
    size_t right_ix = 0;
    for(size_t i = 0; i < Xorder.n_cols; i ++){
            left_ix = 0;
            right_ix = 0;
            for(size_t j = 0; j < N; j ++){
                // loop over all observations
                if(X(Xorder(j, i), split_var) <= X(Xorder(split_point, split_var), split_var)){
                    Xorder_left(left_ix, i) = Xorder(j, i);
                    left_ix = left_ix + 1;
                }else{
                    Xorder_right(right_ix, i) = Xorder(j, i);
                    right_ix = right_ix + 1;
                // }

            }
        }
    }
    return;
}


void split_xorder_std(xinfo_sizet& Xorder_left, xinfo_sizet& Xorder_right, xinfo_sizet& Xorder, double *  X, size_t split_var, size_t split_point, size_t N, size_t p){
    // N is number of rows for Xorder
    size_t left_ix = 0;
    size_t right_ix = 0;
    for(size_t i = 0; i < p; i ++ ){
        left_ix = 0;
        right_ix = 0;
        for(size_t j = 0; j < N; j ++){
            // Xorder(j, i), jth row and ith column
            // look at X(Xorder(j, i), split_var)
            // X[split_var][Xorder[i][j]]
            // X[split_var][Xorder[split_var][split_point]]
            if( *(X + p * split_var + Xorder[i][j])<= *(X + p * split_var + Xorder[split_var][split_point])){
                // copy a row
                for(size_t k = 0; k < p; k ++){
                    Xorder_left[i][left_ix] = Xorder[i][j];
                    left_ix = left_ix + 1;
                }
            }else{
                for(size_t k = 0; k < p; k ++){
                    Xorder_right[i][right_ix] = Xorder[i][j];
                    right_ix = right_ix + 1;
                }
            }
        }
    }
    return;
}





arma::vec BART_likelihood_function(arma::vec& n1, arma::vec& n2, arma::vec& s1, arma::vec& s2, double& tau, double& sigma, double& alpha, double& penalty){
    // log - likelihood of BART model
    // n1 is number of observations in group 1
    // s1 is sum of group 1
    arma::vec result;
    double sigma2 = pow(sigma, 2);
    arma::vec n1tau = n1 * tau;
    arma::vec n2tau = n2 * tau;
    result = - 0.5 * log(n1tau + sigma2) - 0.5 * log(n2tau + sigma2) + 0.5 * tau * pow(s1, 2) / (sigma2 * (n1tau + sigma2)) + 0.5 * tau * pow(s2, 2)/(sigma2 * (n2tau + sigma2));
    // result(result.n_elem - 1) = result(result.n_elem - 1) - penalty;
    // double temp = result.min();
    // result(0) = temp;
    // result(result.n_elem - 1) = temp;

    // the last entry is probability of no split
    // alpha is the prior probability of split, multiply it
    // result = result + log(alpha);
    result(result.n_elem - 1) = result(result.n_elem - 1) + log(1.0 - alpha) - log(alpha);
    return result;
}




void split_error(const arma::umat& Xorder, arma::vec& y, arma::uvec& best_split, arma::vec& least_error){
    // regular CART algorithm, compute sum of squared loss error

    size_t N = Xorder.n_rows;
    size_t p = Xorder.n_cols;
    // arma::mat errormat = arma::zeros(N, p);
    // loop over all variables and observations and compute error

    double y_error = arma::as_scalar(arma::sum(pow(y(Xorder.col(0)) - arma::mean(y(Xorder.col(0))), 2)));

    double ee;
    double temp_error = y_error;
    arma::vec y_cumsum(y.n_elem);
    arma::vec y2_cumsum(y.n_elem);

    y_cumsum = arma::cumsum(y(Xorder.col(0)));
    y2_cumsum = arma::cumsum(pow(y(Xorder.col(0)), 2));

    double y_sum = y_cumsum(y_cumsum.n_elem - 1);
    double y2_sum = y2_cumsum(y2_cumsum.n_elem - 1);

    arma::vec y2 = pow(y, 2);
    for(size_t i = 0; i < p; i++){ // loop over variables 
        temp_error = 100.0;
        y_cumsum = arma::cumsum(y(Xorder.col(i)));
        y2_cumsum = arma::cumsum(pow(y(Xorder.col(i)), 2));
        for(size_t j = 1; j < N - 1; j++){ // loop over cutpoints

            ee = y2_cumsum(j) - pow(y_cumsum(j), 2) / (double) (j+ 1) + y2_sum - y2_cumsum(j) - pow((y_sum - y_cumsum(j)), 2) / (double) (N - j - 1) ;

            if(ee < temp_error || temp_error == 100.0){
                best_split(i) = j; // Xorder(j,i) coordinate;
                temp_error = ee;
                least_error(i) = ee;
            }
        }
    }
    return;
}





void BART_likelihood(const arma::umat& Xorder, arma::vec& y, arma::vec& loglike, double tau, double sigma, size_t depth, size_t Nmin, double alpha, double beta){
    // compute BART posterior (loglikelihood + logprior penalty)
    // randomized

    // faster than split_error_3
    // use stacked vector loglike instead of a matrix, stacked by column
    // length of loglike is p * (N - 1) + 1
    // N - 1 has to be greater than 2 * Nmin

    size_t N = Xorder.n_rows;
    size_t p = Xorder.n_cols;
    
    arma::vec y_cumsum;

    double y_sum;

    arma::vec y_cumsum_inv;

    arma::vec n1tau = tau * arma::linspace(1, N - 1, N - 1);
    arma::vec n2tau = tau * arma::linspace(N-1, 1, N - 1);
    arma::vec temp_likelihood;
    arma::uvec temp_ind;

    double sigma2 = pow(sigma, 2);
    
    // double penalty = log(alpha) - beta * log(1.0 + depth);

    for(size_t i = 0; i < p; i++){ // loop over variables 
        y_cumsum = arma::cumsum(y(Xorder.col(i)));
        y_sum = y_cumsum(y_cumsum.n_elem - 1);
        y_cumsum_inv = y_sum - y_cumsum;  // redundant copy!

        // loglike.col(i) = BART_likelihood(ind1, ind2, y_cumsum, y_cumsum_inv, tau, sigma, alpha, penalty);
        loglike(arma::span(i * (N - 1), i * (N - 1) + N - 2)) = - 0.5 * log(n1tau + sigma2) - 0.5 * log(n2tau + sigma2) + 0.5 * tau * pow(y_cumsum(arma::span(0, N - 2)), 2) / (sigma2 * (n1tau + sigma2)) + 0.5 * tau * pow(y_cumsum_inv(arma::span(0, N - 2)), 2)/(sigma2 * (n2tau + sigma2));
        // temp_likelihood(arma::span(1, N-2)) = temp_likelihood(arma::span(1, N - 2)) + penalty;
        // temp_ind = arma::sort_index(temp_likelihood, "descend"); // decreasing order, pick the largest value
        // best_split(i) = arma::index_max(temp_error); // maximize likelihood
        // best_split.col(i) = temp_ind;
        // loglike.col(i) = temp_likelihood(best_split.col(i));
        
    }
    loglike(loglike.n_elem - 1) = - 0.5 * log(N * tau + sigma2) - 0.5 * log(sigma2) + 0.5 * tau * pow(y_sum, 2) / (sigma2 * (N * tau + sigma2)) - beta * log(1.0 + depth) + beta * log(depth) + log(1.0 - alpha) - log(alpha);
    // add penalty term
    // loglike.row(N - 1) = loglike.row(N - 1) - beta * log(1.0 + depth) + beta * log(depth) + log(1.0 - alpha) - log(alpha);
    
    loglike = loglike - max(loglike);
    loglike = exp(loglike);
    loglike = loglike / arma::as_scalar(arma::sum(loglike));

    // if((N - 1) > 2 * Nmin){
    //     for(size_t i = 0; i < p; i ++ ){
    //         // delete some candidates, otherwise size of the new node can be smaller than Nmin
    //         loglike(arma::span(i * (N - 1), i * (N - 1) + Nmin)).fill(0);
    //         loglike(arma::span(i * (N - 1) + N - 2 - Nmin, i * (N - 1) + N - 2)).fill(0);
    //     }
    // }
    return;
}





void BART_likelihood_adaptive(const arma::umat& Xorder, arma::vec& y, arma::vec& loglike, double tau, double sigma, size_t depth, size_t Nmin, double alpha, double beta){
    // compute BART posterior (loglikelihood + logprior penalty)
    // randomized

    // faster than split_error_3
    // use stacked vector loglike instead of a matrix, stacked by column
    // length of loglike is p * (N - 1) + 1
    // N - 1 has to be greater than 2 * Nmin

    size_t N = Xorder.n_rows;
    size_t p = Xorder.n_cols;
    
    arma::vec y_cumsum;

    double y_sum;

    arma::vec y_cumsum_inv;

    arma::vec n1tau = tau * arma::linspace(1, N - 1, N - 1);
    arma::vec n2tau = tau * arma::linspace(N-1, 1, N - 1);
    arma::vec temp_likelihood;
    arma::uvec temp_ind;

    double sigma2 = pow(sigma, 2);
    
    // double penalty = log(alpha) - beta * log(1.0 + depth);

    for(size_t i = 0; i < p; i++){ // loop over variables 
        y_cumsum = arma::cumsum(y(Xorder.col(i)));
        y_sum = y_cumsum(y_cumsum.n_elem - 1);
        y_cumsum_inv = y_sum - y_cumsum;  // redundant copy!

        // loglike.col(i) = BART_likelihood(ind1, ind2, y_cumsum, y_cumsum_inv, tau, sigma, alpha, penalty);
        loglike(arma::span(i * (N - 1), i * (N - 1) + N - 2)) = - 0.5 * log(n1tau + sigma2) - 0.5 * log(n2tau + sigma2) + 0.5 * tau * pow(y_cumsum(arma::span(0, N - 2)), 2) / (sigma2 * (n1tau + sigma2)) + 0.5 * tau * pow(y_cumsum_inv(arma::span(0, N - 2)), 2)/(sigma2 * (n2tau + sigma2));
        // temp_likelihood(arma::span(1, N-2)) = temp_likelihood(arma::span(1, N - 2)) + penalty;
        // temp_ind = arma::sort_index(temp_likelihood, "descend"); // decreasing order, pick the largest value
        // best_split(i) = arma::index_max(temp_error); // maximize likelihood
        // best_split.col(i) = temp_ind;
        // loglike.col(i) = temp_likelihood(best_split.col(i));
        
    }
    loglike(loglike.n_elem - 1) = - 0.5 * log(N * tau + sigma2) - 0.5 * log(sigma2) + 0.5 * tau * pow(y_sum, 2) / (sigma2 * (N * tau + sigma2)) - beta * log(1.0 + depth) + beta * log(depth) + log(1.0 - alpha) - log(alpha);
    // add penalty term
    // loglike.row(N - 1) = loglike.row(N - 1) - beta * log(1.0 + depth) + beta * log(depth) + log(1.0 - alpha) - log(alpha);
    
    loglike = loglike - max(loglike);
    loglike = exp(loglike);
    loglike = loglike / arma::as_scalar(arma::sum(loglike));

    if((N - 1) > 2 * Nmin){
        for(size_t i = 0; i < p; i ++ ){
            // delete some candidates, otherwise size of the new node can be smaller than Nmin
            loglike(arma::span(i * (N - 1), i * (N - 1) + Nmin)).fill(0);
            loglike(arma::span(i * (N - 1) + N - 2 - Nmin, i * (N - 1) + N - 2)).fill(0);
        }
    }
    return;
}



void BART_likelihood_std(size_t N, size_t p, xinfo_sizet& Xorder, double* y, std::vector<double>& loglike, double& tau, double& sigma, size_t& depth, double& alpha, double& beta){
    std::vector<double> y_cumsum(N);
    double y_sum;
    std::vector<double> y_cumsum_inv(N);
    std::vector<double> n1tau;
    std::vector<double> n2tau;
    std::vector<size_t> temp_ind;
    double sigma2 = pow(sigma, 2);
    for(size_t i = 0; i < p; i ++){
        // calculate cumulative sum, reorder y as the i-th column of Xorder matrix (i-th variable)
        cumulative_sum_std(y_cumsum, y_cumsum_inv, y_sum, y, Xorder, i, N);
        y_sum = y_cumsum[N - 1]; // the last one
        // y_cumsum_inv = 
    }
    return;
}


void cumulative_sum_std(std::vector<double>& y_cumsum, std::vector<double>& y_cumsum_inv, double& y_sum, double* y, xinfo_sizet& Xorder, size_t& i, size_t& N){
    // y_cumsum is the output cumulative sum
    // y is the original data
    // Xorder is sorted index matrix
    // i means take the i-th column of Xorder
    // N is length of y and y_cumsum
    if(N > 1){
        y_cumsum[0] = y[Xorder[i][0]];
        for(size_t j = 1; j < N; j++){
            y_cumsum[j] = y_cumsum[j - 1] + y[Xorder[i][j]];
        }
    }else{
        y_cumsum[0] = y[Xorder[i][0]];
    }
    y_sum = y_cumsum[N - 1];

    for(size_t j = 1; j < N; j ++){
        y_cumsum_inv[j] = y_sum - y_cumsum[j];
    }
    return;
}





arma::uvec range(size_t start, size_t end){
    // generate integers from start to end
    size_t N = end - start;
    arma::uvec output(N);
    for(size_t i = 0; i < N; i ++){
        output(i) = start + i;
    }
    return output;
}

