#include "tree.h"
#include "treefuns.h"
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]
#include <RcppArmadillo.h>
// double region_error(arma::vec& y);
// double error_function(int split_point, arma::vec& y, arma::uvec& Xorder_vec);
// void search_split_point(arma::uvec& Xorder_vec, arma::vec& y, int& split_ind, double& error_split);
// [[Rcpp::export]]
Rcpp::List predict_tree(Rcpp::List trees, arma::mat Xnew){
    Rcpp::List output;
    // reconstruct the tree structure
    Rcpp::CharacterVector itrees(Rcpp::wrap(trees["trees"])); 

    std::string itv(itrees[0]);

    std::stringstream ttss(itv);

    tree tree_model;
    ttss >> tree_model;

    arma::vec pred = fit_new(tree_model, Xnew);

    output["predict"] = pred;
    return output;
}

