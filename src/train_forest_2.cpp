#include <ctime>
#include <RcppArmadillo.h>
#include "tree.h"
#include "treefuns.h"
#include "forest.h"

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]]
Rcpp::List train_forest_2(arma::mat y, arma::mat X, arma::mat Xtest, size_t M, size_t L, size_t N_sweeps, arma::vec max_depth, size_t Nmin, double alpha, double beta, double tau, bool draw_sigma, double kap = 16, double s = 4, bool verbose = false, bool m_update_sigma = false){

    size_t N = X.n_rows;

    arma::umat Xorder(X.n_rows, X.n_cols);
    for(size_t i = 0; i < X.n_cols; i++){
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

    // double tau;

    forest trees(M);

    arma::vec reshat;

    arma::vec reshat_test;


    for(size_t mc = 0; mc < L; mc ++ ){

        predictions.fill(arma::as_scalar(arma::mean(y)) / (double) M);

        predictions_test.fill(arma::as_scalar(arma::mean(y)) / (double) M);


        yhat = arma::sum(predictions, 1);

        yhat_test = arma::sum(predictions_test, 1);

        residual = y - yhat;

        for(size_t sweeps = 0; sweeps < N_sweeps; sweeps ++){

            if(verbose == true){
            cout << "--------------------------------" << endl;
            cout << "number of sweeps " << sweeps << endl;
            cout << "--------------------------------" << endl;
            }

            for(size_t tree_ind = 0; tree_ind < M; tree_ind ++){


                if(m_update_sigma == true){
                                if(draw_sigma == true){
                     sigma = 1.0 / sqrt(arma::as_scalar(arma::randg(1, arma::distr_param( (N + kap) / 2.0, 2.0 / as_scalar(sum(pow(residual, 2)) + s)))));

                    // sigma = 1.0 / Rcpp::rgamma(1, N / 2.0, 2.0 / as_scalar(sum(pow(residual, 2))))[0];
                }else{
                    // sigma = sqrt(arma::as_scalar(arma::mean(pow(residual, 2))));
                    sigma = 0.1;
                }
                }


                // add prediction of current tree back to residual
                // then it's m - 1 trees residual
                residual = residual + predictions.col(tree_ind);


                if(m_update_sigma == false){

                if(draw_sigma == true){
                    sigma = 1.0 / sqrt(arma::as_scalar(arma::randg(1, arma::distr_param( (N + kap) / 2.0, 2.0 / as_scalar(sum(pow(residual, 2)) + s)))));

                    // sigma = 1.0 / Rcpp::rgamma(1, N / 2.0, 2.0 / as_scalar(sum(pow(residual, 2))))[0];
                }else{
                    // sigma = sqrt(arma::as_scalar(arma::mean(pow(residual, 2))));
                    sigma = 0.1;
                }
                }

                yhat = yhat - predictions.col(tree_ind);

                yhat_test = yhat_test - predictions_test.col(tree_ind);


                trees.t[tree_ind].grow_tree_2(residual, arma::as_scalar(mean(residual)), Xorder, X, 0, max_depth(sweeps), Nmin, tau, sigma, alpha, beta);


                if(verbose == true){
                cout << "tree " << tree_ind << " size is " << trees.t[tree_ind].treesize() << endl;
                }
                
                reshat = fit_new(trees.t[tree_ind], X);

                predictions.col(tree_ind) = reshat;

                // fit_new_void(trees.t[tree], X, predictions, tree);

                reshat_test = fit_new(trees.t[tree_ind], Xtest);

                predictions_test.col(tree_ind) = reshat_test;

                // fit_new_void(trees.t[tree], Xtest, predictions_test, tree);

                residual = residual - predictions.col(tree_ind);



                yhat = yhat + predictions.col(tree_ind);

                yhat_test = yhat_test + predictions_test.col(tree_ind);

            }
        yhats.col(sweeps) = yhat;
        yhats_test.col(sweeps) = yhat_test;
        }

    }

    return Rcpp::List::create(Rcpp::Named("yhats") = yhats, Rcpp::Named("yhats_test") = yhats_test);
}

