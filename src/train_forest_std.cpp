#include <ctime>
#include <RcppArmadillo.h>
#include "tree.h"
#include "treefuns.h"
#include "forest.h"
// #include "utility.h"

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]]
Rcpp::List train_forest_std(Rcpp::NumericMatrix y_rcpp, Rcpp::NumericMatrix X_rcpp, Rcpp::NumericMatrix Xtest_rcpp, Rcpp::IntegerMatrix Xorder_rcpp, size_t M, size_t L, size_t N_sweeps, Rcpp::NumericMatrix max_depth_rcpp, size_t Nmin, double alpha, double beta, double tau, bool draw_sigma, double kap = 16, double s = 4, bool verbose = false, bool m_update_sigma = false, bool draw_mu = false){

    // matrices are stacked by column
    size_t p = X_rcpp.ncol();
    size_t N = X_rcpp.nrow();
    size_t N_test = Xtest_rcpp.nrow();

    // random number generator
    std::default_random_engine generator;
    std::gamma_distribution<double> gamma_d;

    // define matrices
    xinfo_sizet Xorder = copy_xinfo_sizet(Xorder_rcpp);

    // X and Xtest are pointers to the matrix
    // matrix are p * n, stack by row
    double *X = &X_rcpp[0];
    double *Xtest = &Xtest_rcpp[0];

    // max_depth is a matrix, number of trees by number of sweeps
    // double *max_depth = &max_depth_rcpp[0];
    size_t max_depth;

    // xinfo yhats;
    xinfo yhats = ini_xinfo(N, p); // vector of vectors, stack by column
    xinfo yhats_test = ini_xinfo(N_test, p);
    xinfo predictions = ini_xinfo(N, M);
    xinfo predictions_test = ini_xinfo(N_test, M);
    std::vector<double> yhat(N);
    std::vector<double> yhat_test(N_test);

    std::vector<double> residual(N);
    std::vector<double> residual_theta_noise(N);
    xinfo sigma_draw = ini_xinfo(M, N_sweeps);

    double sigma;

    forest trees(M);

    std::vector<double> reshat(N);
    std::vector<double> reshat_test(N_test);


    for(size_t mc = 0; mc < L; mc ++ ){

        for(size_t sweeps = 0; sweeps < N_sweeps; sweeps ++ ){

            if(verbose == true){
                cout << "--------------------------------" << endl;
                cout << "number of sweeps " << sweeps << endl;
                cout << "--------------------------------" << endl;
            }

            for(size_t tree_ind = 0; tree_ind < M; tree_ind ++ ){
        
                if(m_update_sigma == true){
                    // sampling sigma
                    gamma_d = std::gamma_distribution<double>((N + kap) / 2.0, 2.0 / sum_squared(residual) + s);
                    sigma = 1.0 / gamma_d(generator);
                }
                sigma_draw[sweeps][tree_ind] = sigma;

                // add prediction of current tree back to residual
                // then it's m - 1 trees residual
                for(size_t i = 0; i < N; i ++ ){
                    residual[i] = residual[i] + predictions[tree_ind][i];
                    // prediction of m - 1 trees
                    yhat[i] = yhat[i] - predictions[tree_ind][i];
                }

                // do the samething for residual_theta_noise, residual of m - 1 trees
                for(size_t i = 0; i < N; i ++ ){
                    residual_theta_noise[i] = residual_theta_noise[i] + predictions[tree_ind][i];
                }

                for(size_t i = 0; i < N_test; i ++ ){
                    yhat_test[i] = yhat_test[i] - predictions_test[tree_ind][i];
                }

                double mean_y = 0;
                for(size_t i = 0; i < N; i ++ ){
                    mean_y = mean_y + residual[i];
                }
                mean_y = mean_y / (double) N;
                
                // update the current tree
                trees.t[tree_ind].grow_tree_std(&residual[0], mean_y, Xorder, X, N, p, 0, max_depth_rcpp(tree_ind, sweeps), Nmin, tau, sigma, alpha, beta, &residual[0], draw_sigma, draw_mu);

                if(verbose == true){
                    cout << "tree" << tree_ind << " size is " << trees.t[tree_ind].treesize() << endl;
                }

                // update prediction of current tree
                fit_std(trees.t[tree_ind], p, N, X, predictions[tree_ind]);

                // update prediction of current tree, on test set
                fit_std(trees.t[tree_ind], p, N_test, Xtest, predictions_test[tree_ind]);

                // update sigma based on residual of m - 1 trees, rather than residual of m trees
                if(m_update_sigma == false){
                    gamma_d = std::gamma_distribution<double>((N + kap) / 2.0, 2.0 / sum_squared(residual) + s);
                    sigma = 1.0 / gamma_d(generator);
                }

                // update residuals, now it's residual of m trees
                for(size_t i = 0; i < N; i ++ ){
                    residual[i] = residual[i] - predictions[tree_ind][i];
                    yhat[i] = yhat[i] + predictions[tree_ind][i];
                }

                // update yhat_test
                for(size_t i = 0; i < N_test; i ++ ){
                    yhat_test[i] = yhat_test[i] + predictions[tree_ind][i];
                }

                
            }
        
        yhats[sweeps] = yhat;
        yhats_test[sweeps] = yhat_test;

        }
    }


    return Rcpp::List::create(Rcpp::Named("aaa") = 1.0);
}

