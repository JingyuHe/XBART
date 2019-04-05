#include <ctime>
#include <RcppArmadillo.h>
#include "tree.h"
#include "forest.h"
#include <chrono>
#include "fit_std_main_loop.h"
#include <utility.h>

// [[Rcpp::export]]
Rcpp::List xbart_predict(arma::mat X,size_t L ,double y_mean,Rcpp::XPtr<std::vector<std::vector<tree>>> tree_pnt){

	// Size of data
	size_t N = X.n_rows;
    size_t p = X.n_cols;

    // Init X_std matrix
    Rcpp::NumericMatrix X_std(N, p);
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < p; j++)
        {
            X_std(i, j) = X(i, j);
        }
    }
    double *Xpointer = &X_std[0];

    // Trees
   	std::vector<std::vector<tree>> *trees =  tree_pnt;

    // Result Container
    xinfo yhats_test_xinfo;
    size_t N_sweeps = (*trees).size();
    size_t M = (*trees)[0].size();
    ini_xinfo(yhats_test_xinfo, N, N_sweeps);

    // Predict
	predict_std(Xpointer, N, p, M, L, N_sweeps, 
		yhats_test_xinfo, *trees, y_mean);

	// Convert back to Rcpp
	Rcpp::NumericMatrix yhats(N, N_sweeps);
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < N_sweeps; j++)
        {
            yhats(i, j) = yhats_test_xinfo[j][i];
        }
    }

	return Rcpp::List::create(Rcpp::Named("yhats") = yhats);
}