#include "tree.h"
#include "treefuns.h"
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]
#include <RcppArmadillo.h>
// double region_error(arma::vec& y);
// double error_function(int split_point, arma::vec& y, arma::uvec& Xorder_vec);
// void search_split_point(arma::uvec& Xorder_vec, arma::vec& y, int& split_ind, double& error_split);
// [[Rcpp::export]]
Rcpp::List singletree(arma::vec y, arma::mat X, int depth, int max_depth = 100, int Nmin = 5, double tau = 10, double sigma = 1, double alpha = 0.95, double beta = 2){
    // sort each column of X, create a matrix for order of elements in each column
    arma::umat Xorder(X.n_rows, X.n_cols);
    for(int i = 0; i < X.n_cols; i++){
        Xorder.col(i) = arma::sort_index(X.col(i));
    }

    int p = X.n_cols; // number of X variables


    tree root;
    double y_mean = arma::as_scalar(arma::mean(y));
    root.grow_tree(y, y_mean, Xorder, X, depth, max_depth, Nmin, tau, sigma, alpha, beta);
    std::stringstream treess;  //string stream to write trees to  

    treess << root ;


    tree_p tree_l;
    tree_p tree_r;

    tree_l = root.getl();
    tree_r = root.getr();

    tree::cnpv nds;
    root.getnodes(nds);
    treess.precision(10);

    

    return Rcpp::List::create(Rcpp::Named("trees") = Rcpp::CharacterVector(treess.str()));
}

