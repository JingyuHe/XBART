#include <ctime>
#include <RcppArmadillo.h>
#include "tree.h"
#include "treefuns.h"
#include "forest.h"


// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]]
Rcpp::List train_forest_root(arma::mat y, arma::mat X, arma::mat Xtest, size_t M, size_t L, size_t N_sweeps, arma::mat max_depth, size_t Nmin, size_t Ncutpoints, double alpha, double beta, double tau, bool draw_sigma, double kap = 16, double s = 4, bool verbose = false, bool m_update_sigma = false, bool draw_mu = false, bool parallel = true){

    size_t N = X.n_rows;

    arma::umat Xorder(X.n_rows, X.n_cols);
    for(size_t i = 0; i < X.n_cols; i++){
        Xorder.col(i) = arma::sort_index(X.col(i));
    }


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

                // do the samething for residual_theta_noise, residual of m - 1 trees
                // residual_theta_noise = residual_theta_noise + predictions_theta_noise.col(tree_ind);

                // prediction of m - 1 trees
                yhat = yhat - predictions.col(tree_ind);

                // prediction of m - 1 trees on testing set
                yhat_test = yhat_test - predictions_test.col(tree_ind);

                // grow a tree

                // if(sweeps < 30){


                // if(sweeps < 1){
                    trees.t[tree_ind].grow_tree_adaptive(residual, arma::as_scalar(mean(residual)), Xorder, X, 0, max_depth(tree_ind, sweeps), Nmin, Ncutpoints, tau, sigma, alpha, beta, residual, draw_sigma, draw_mu, parallel);
                    cout << "tree size " << trees.t[tree_ind].treesize() << endl;
                // }else{ 
                //     //    trees.t[tree_ind].sample_theta(residual, X, tau, sigma, draw_mu);

                // // }
                
                //     // trees.t[tree_ind].prune_regrow(residual, arma::as_scalar(mean(residual)), X, 0, max_depth(tree_ind, sweeps), Nmin, Ncutpoints, tau, sigma, alpha, beta, residual, draw_sigma, draw_mu, parallel);
                // //     // cout << "tree size, prune and regrow " << trees.t[tree_ind].treesize() << endl;
                // //     cout << "+++++++++++++++++++++++++++" << endl;


                //     prune = d(gen);
                //     if(prune == 0){
                //         // cout << " grow " << endl;
                //         trees.t[tree_ind].one_step_grow(residual, arma::as_scalar(mean(residual)), X, 0, max_depth(tree_ind, sweeps), Nmin, Ncutpoints, tau, sigma, alpha, beta, residual, draw_sigma, draw_mu, parallel);
                //         // cout << "grow, before tree size " << trees.t[tree_ind].treesize() << endl;

                //     }else{
                //         if(trees.t[tree_ind].treesize() > 5){
                //         // cout << " prune " << endl;
                //         trees.t[tree_ind].one_step_prune(residual, arma::as_scalar(mean(residual)), X, 0, max_depth(tree_ind, sweeps), Nmin, Ncutpoints, tau, sigma, alpha, beta, residual, draw_sigma, draw_mu, parallel);
                //         // cout << "prune, before tree size " << trees.t[tree_ind].treesize() << endl;
                //         }
                //     }

                // }   
                // trees.t[tree_ind].grow_tree_adaptive(residual, arma::as_scalar(mean(residual)), Xorder, X, 0, max_depth(tree_ind, sweeps), Nmin, Ncutpoints, tau, sigma, alpha, beta, residual, draw_sigma, draw_mu, parallel);

                // trees.t[tree_ind].one_step_prune(residual, arma::as_scalar(mean(residual)), X, 0, max_depth(tree_ind, sweeps), Nmin, Ncutpoints, tau, sigma, alpha, beta, residual, draw_sigma, draw_mu, parallel); 
                // trees.t[tree_ind].one_step_grow(residual, arma::as_scalar(mean(residual)), X, 0, max_depth(tree_ind, sweeps), Nmin, Ncutpoints, tau, sigma, alpha, beta, residual, draw_sigma, draw_mu, parallel);
          


                
                // cout << "after tree size " << trees.t[tree_ind].treesize() << endl;
                // cout << "+++++++++++++++++++++++++++" << endl;


                if(verbose == true){
                    cout << "tree " << tree_ind << " size is " << trees.t[tree_ind].treesize() << endl;
                }
                
                // update prediction of current tree
                predictions.col(tree_ind) = fit_new(trees.t[tree_ind], X);


                // cout << "prediction sum of squares " << sum(pow(predictions.col(tree_ind), 2)) << endl;

                // update prediction (theta_noise) of current tree
                // predictions_theta_noise.col(tree_ind) = fit_new_theta_noise(trees.t[tree_ind], X);

                // update prediction of current tree, test set
                predictions_test.col(tree_ind) = fit_new(trees.t[tree_ind], Xtest);

                // fit_new_void(trees.t[tree], Xtest, predictions_test, tree);

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
        // cout << "+++++++++++++++++++++++++++++++++++" << endl;
        // cout << "end of one sweep" << endl;
        // cout << "+++++++++++++++++++++++++++++++++++" << endl;

        }

    }

    return Rcpp::List::create(Rcpp::Named("yhats") = yhats, Rcpp::Named("yhats_test") = yhats_test, Rcpp::Named("sigma") = sigma_draw);
}

