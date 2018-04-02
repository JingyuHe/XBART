#include <ctime>
#include <RcppArmadillo.h>
#include "tree.h"
#include "treefuns.h"
#include "forest.h"

// [[Rcpp::plugins(cpp11)]]

// [[Rcpp::export]]
Rcpp::List train_forest_2(arma::mat y, arma::mat X, arma::mat Xtest, int M, int L, int N_sweeps, arma::vec max_depth, int Nmin, double alpha, double beta, double tau, bool draw_sigma){

    int N = X.n_rows;

    arma::umat Xorder(X.n_rows, X.n_cols);
    for(int i = 0; i < X.n_cols; i++){
        Xorder.col(i) = arma::sort_index(X.col(i));
    }


    arma::mat yhats = arma::zeros<arma::mat>(X.n_rows, N_sweeps);

    arma::mat yhats_test = arma::zeros<arma::mat>(Xtest.n_rows, N_sweeps);

    arma::mat predictions(X.n_rows, M);

    arma::mat predictions_test(Xtest.n_rows, M);

    arma::vec yhat = arma::sum(predictions, 1);

    arma::vec yhat_test = arma::zeros<arma::vec>(Xtest.n_rows);

    arma::vec residual;

    double sigma;

    double var_y = arma::as_scalar(arma::var(y));

    // double tau;

    forest::forest trees(M);

    arma::vec reshat;

    arma::vec reshat_test;

    tree::tree_p current_tree;


    for(int mc = 0; mc < L; mc ++ ){

        predictions.fill(arma::as_scalar(arma::mean(y)) / (double) M);

        predictions_test.fill(arma::as_scalar(arma::mean(y)) / (double) M);


        yhat = arma::sum(predictions, 1);

        yhat_test = arma::sum(predictions_test, 1);

        residual = y - yhat;

        for(int sweeps = 0; sweeps < N_sweeps; sweeps ++){

            for(int tree = 0; tree < M; tree ++){

                // current_tree = &trees.t[tree];

                residual = residual + predictions.col(tree);

                

                if(draw_sigma == true){
                    sigma = 1.0 / sqrt(arma::as_scalar(arma::randg(1, arma::distr_param(N / 2.0, 2.0 / as_scalar(sum(pow(residual, 2)))))));

                    // sigma = 1.0 / Rcpp::rgamma(1, N / 2.0, 2.0 / as_scalar(sum(pow(residual, 2))))[0];
                }else{
                    sigma = sqrt(arma::as_scalar(arma::mean(pow(residual, 2))));
                }

                yhat = yhat - predictions.col(tree);

                yhat_test = yhat_test - predictions_test.col(tree);


                trees.t[tree].grow_tree_2(residual, arma::as_scalar(mean(residual)), Xorder, X, 0, max_depth(sweeps), Nmin, tau, sigma, alpha, beta);

                reshat = fit_new(trees.t[tree], X);

                predictions.col(tree) = reshat;

                // fit_new_void(trees.t[tree], X, predictions, tree);

                reshat_test = fit_new(trees.t[tree], Xtest);

                predictions_test.col(tree) = reshat_test;

                // fit_new_void(trees.t[tree], Xtest, predictions_test, tree);

                residual = residual - predictions.col(tree);

                yhat = yhat + predictions.col(tree);

                yhat_test = yhat_test + predictions_test.col(tree);

            }
        yhats.col(sweeps) = yhat;
        yhats_test.col(sweeps) = yhat_test;
        }

    }

    return Rcpp::List::create(Rcpp::Named("yhats") = yhats, Rcpp::Named("yhats_test") = yhats_test);
}

