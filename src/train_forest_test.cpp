#include <ctime>
#include <RcppArmadillo.h>
#include "tree.h"
#include "treefuns.h"
#include "forest.h"


// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]]
Rcpp::List train_forest_test(arma::mat y, arma::mat X, arma::mat Xtest, size_t M, size_t L, size_t N_sweeps, arma::mat max_depth, size_t Nmin, size_t Ncutpoints, double alpha, double beta, double tau, bool draw_sigma, double kap = 16, double s = 4, bool verbose = false, bool m_update_sigma = false, bool draw_mu = false, bool parallel = true){

    size_t N = X.n_rows;
    size_t p = X.n_cols;
    size_t N_test = Xtest.n_rows;

    arma::umat Xorder(X.n_rows, X.n_cols);
    for(size_t i = 0; i < X.n_cols; i++){
        Xorder.col(i) = arma::sort_index(X.col(i));
    }


    ///////////////////////////////////////////////////////////////////
    // create copies for STD version
    std::vector<double> y_std(N);
    for(size_t i = 0; i < N; i ++ ){
        y_std[i] = y(i,0);
    }
    
    Rcpp::NumericMatrix X_std(N, p);
    for(size_t i = 0; i < N; i ++ ){
        for(size_t j = 0; j < p; j ++ ){
            X_std(i, j) = X(i, j);
        }
    }
    Rcpp::NumericMatrix Xtest_std(N_test, p);
    for(size_t i = 0; i < N_test; i ++ ){
        for(size_t j = 0; j < p; j ++ ){
            Xtest_std(i, j) = Xtest(i, j);
        }
    }
    double * ypointer = &y_std[0];
    double * Xpointer = &X_std[0];
    double * Xtestpointer = &Xtest_std[0];
    
    xinfo_sizet Xorder_std;
    ini_xinfo_sizet(Xorder_std, N, p);
    for(size_t i = 0; i < N; i ++ ){
        for(size_t j = 0; j < p; j ++){
            Xorder_std[j][i] = Xorder(i, j);
        }
    }

    xinfo yhats_std;
    ini_xinfo(yhats_std, N, N_sweeps);
    xinfo yhats_test_std;
    ini_xinfo(yhats_test_std, Xtest.n_rows, N_sweeps);

    // save predictions of each tree
    std::vector< std::vector<double> > predictions_std;
    ini_xinfo(predictions_std, N, M);

    xinfo predictions_test_std;
    ini_xinfo(predictions_test_std, Xtest.n_rows, M);

    std::vector<double> yhat_std(N);
    row_sum(predictions_std, yhat_std);
    std::vector<double> yhat_test_std(Xtest.n_rows);
    row_sum(predictions_test_std, yhat_test_std);

 
    // current residual
    std::vector<double> residual_std(N);

    xinfo sigma_draw_std;
    ini_xinfo(sigma_draw_std, M, N_sweeps);


    forest trees_std(M);

    std::vector<double> reshat_std;
    std::vector<double> reshat_test_std;

    std::vector<double> prob_std(2, 0.5);

    std::random_device rd_std;
    std::mt19937 gen_std(rd_std());
    std::discrete_distribution<> d_std(prob_std.begin(), prob_std.end());   


    ///////////////////////////////////////////////////////////////////







    arma::mat yhats = arma::zeros<arma::mat>(X.n_rows, N_sweeps);
    arma::mat yhats_test = arma::zeros<arma::mat>(Xtest.n_rows, N_sweeps);
    // save predictions of each tree
    arma::mat predictions(X.n_rows, M);
    // save predictions (based on theta_noise) of each tree
    // arma::mat predictions_theta_noise(X.n_rows, M);
    arma::mat predictions_test(Xtest.n_rows, M);
    arma::vec yhat = arma::sum(predictions, 1);
    arma::vec yhat_test = arma::zeros<arma::vec>(Xtest.n_rows);
    // current residual
    arma::mat residual;
    // current residual (based on theta_noise)
    // arma::vec residual_theta_noise;
    arma::mat sigma_draw(M, N_sweeps);
    double sigma;
    // double tau;
    forest trees(M);
    arma::vec reshat;
    arma::vec reshat_test;
    std::vector<double> prob(2, 0.5);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> d(prob.begin(), prob.end());
    // // sample one index of split point
    size_t prune;  


    // size_t count = 0;
    for(size_t mc = 0; mc < L; mc ++ ){

        // initialize
        predictions.fill(arma::as_scalar(arma::mean(y)) / (double) M);

        // predictions_theta_noise.fill(arma::as_scalar(arma::mean(y)) / (double) M);

        predictions_test.fill(arma::as_scalar(arma::mean(y)) / (double) M);

        yhat = arma::sum(predictions, 1);

        yhat_test = arma::sum(predictions_test, 1);

        residual = y - yhat;



        // residual_theta_noise = y - yhat;

        for(size_t sweeps = 0; sweeps < N_sweeps; sweeps ++){

            // if(verbose == true){
            // cout << "--------------------------------" << endl;
            // cout << "number of sweeps " << sweeps << endl;
            // cout << "--------------------------------" << endl;
            // }

            for(size_t tree_ind = 0; tree_ind < M; tree_ind ++){


                // if update sigma based on residual of all m trees
                if(m_update_sigma == true){
                    sigma = 1.0 / sqrt(arma::as_scalar(arma::randg(1, arma::distr_param( (N + kap) / 2.0, 2.0 / as_scalar(sum(pow(residual, 2)) + s)))));
                    sigma_draw(tree_ind, sweeps) = sigma;
                }

                // save sigma
                sigma_draw(tree_ind, sweeps) = sigma;

                // add prediction of current tree back to residual
                // then it's m - 1 trees residual
                residual = residual + predictions.col(tree_ind);

                for(size_t kk = 0; kk < N; kk ++ ){
                    residual_std[kk] = residual(kk);
                }

                // do the samething for residual_theta_noise, residual of m - 1 trees
                // residual_theta_noise = residual_theta_noise + predictions_theta_noise.col(tree_ind);

                // prediction of m - 1 trees
                yhat = yhat - predictions.col(tree_ind);

                // prediction of m - 1 trees on testing set
                yhat_test = yhat_test - predictions_test.col(tree_ind);

                trees.t[tree_ind].grow_tree_adaptive_test(arma::as_scalar(mean(residual)), 0, max_depth(tree_ind, sweeps), Nmin, Ncutpoints, tau, sigma, alpha, beta, draw_sigma, draw_mu, parallel, residual_std, Xorder_std, Xpointer);

                // trees.t[tree_ind].grow_tree_adaptive(residual, arma::as_scalar(mean(residual)), Xorder, X, 0, max_depth(tree_ind, sweeps), Nmin, Ncutpoints, tau, sigma, alpha, beta, draw_sigma, draw_mu, parallel);

                
                // cout << "tree size " << trees.t[tree_ind].treesize() << endl;

                if(verbose == true){
                    cout << "tree " << tree_ind << " size is " << trees.t[tree_ind].treesize() << endl;
                }
                
                // update prediction of current tree
                predictions.col(tree_ind) = fit_new(trees.t[tree_ind], X);

                // update prediction of current tree, test set
                predictions_test.col(tree_ind) = fit_new(trees.t[tree_ind], Xtest);

                // update sigma based on residual of m - 1 trees, residual_theta_noise
                if(m_update_sigma == false){
                    sigma = 1.0 / sqrt(arma::as_scalar(arma::randg(1, arma::distr_param( (N + kap) / 2.0, 2.0 / as_scalar(sum(pow(residual, 2)) + s)))));
                    sigma_draw(tree_ind, sweeps) = sigma;
                }

                // update residual, now it's residual of m trees
                residual = residual - predictions.col(tree_ind);

                // residual_theta_noise = residual_theta_noise - predictions_theta_noise.col(tree_ind);

                yhat = yhat + predictions.col(tree_ind);

                yhat_test = yhat_test + predictions_test.col(tree_ind);

            }
        yhats.col(sweeps) = yhat;
        yhats_test.col(sweeps) = yhat_test;

        }

    }

    return Rcpp::List::create(Rcpp::Named("yhats") = yhats, Rcpp::Named("yhats_test") = yhats_test, Rcpp::Named("sigma") = sigma_draw);
}

