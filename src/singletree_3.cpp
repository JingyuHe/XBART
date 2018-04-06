#include "tree.h"
#include "treefuns.h"
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]
#include <RcppArmadillo.h>
// double region_error(arma::vec& y);
// double error_function(size_t split_point, arma::vec& y, arma::uvec& Xorder_vec);
// void search_split_point(arma::uvec& Xorder_vec, arma::vec& y, size_t& split_ind, double& error_split);
// [[Rcpp::export]]
Rcpp::List singletree_3(arma::vec y, arma::mat X, size_t depth, size_t max_depth = 100, size_t Nmin = 5, double tau = 10, double sigma = 1, double alpha = 0.95, double beta = 2){
    // sort each column of X, create a matrix for order of elements in each column
    arma::umat Xorder(X.n_rows, X.n_cols);
    for(size_t i = 0; i < X.n_cols; i++){
        Xorder.col(i) = arma::sort_index(X.col(i));
    }


    tree root;
    double y_mean = arma::as_scalar(arma::mean(y));

    root.grow_tree_2(y, y_mean, Xorder, X, depth, max_depth, Nmin, tau, sigma, alpha, beta);
    std::stringstream treess;  //string stream to write trees to  

    treess << root ;


    tree::tree_p tree_l;
    tree::tree_p tree_r;

    tree_l = root.getl();
    tree_r = root.getr();

    tree::cnpv nds;
    root.getnodes(nds);
    treess.precision(10);

    

    return Rcpp::List::create(Rcpp::Named("trees") = Rcpp::CharacterVector(treess.str()));
}

