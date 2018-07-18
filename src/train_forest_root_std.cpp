#include <ctime>
#include <RcppArmadillo.h>
#include "tree.h"
#include "treefuns.h"
#include "forest.h"
#include <Rcpp.h>
// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]]
Rcpp::List train_forest_root_std(Rcpp::NumericVector y, Rcpp::NumericMatrix X, Rcpp::NumericMatrix Xtest, size_t M, size_t L, size_t N_sweeps, Rcpp::IntegerMatrix max_depth, size_t Nmin, size_t Ncutpoints, double alpha, double beta, double tau, bool draw_sigma, double kap = 16, double s = 4, bool verbose = false, bool m_update_sigma = false, bool draw_mu = false, bool parallel = true){

    size_t N = X.nrow();
    size_t p = X.ncol();
    size_t N_test = Xtest.nrow();


    // random number generator
    std::default_random_engine generator;
    std::gamma_distribution<double> gamma_d;


    xinfo X_std;
    ini_xinfo(X_std, N, p);
    for(size_t i = 0; i < p; i++){
        for(size_t j = 0; j < N; j ++ ){
            X_std[i][j] = X(j,i);
        }
    }




    xinfo_sizet Xorder;
    ini_xinfo_sizet(Xorder, N, p);

    // Rcpp::NumericMatrix::Column temp;
    // generate Xorder matrix
    for(size_t i = 0; i < p; i ++ ){
        Xorder[i] = sort_indexes(X_std[i]);
    }
    // Xorder is correct


    // caclulate mean of y
    double y_mean = 0.0;
    for(size_t i = 0; i < N; i ++){
        y_mean += y[i];
    }
    y_mean = y_mean / (double) N;
    
    

    // for(size_t i = 0; i < p; i++){
    //     // Xorder.col(i) = arma::sort_index(X.col(i));
    //     Xorder[i] = sort_indexes(X( _,i));
    // }

    xinfo yhats;
    ini_xinfo(yhats, N, N_sweeps);
    xinfo yhats_test;
    ini_xinfo(yhats_test, Xtest.nrow(), N_sweeps);

    // save predictions of each tree
    std::vector< std::vector<double> > predictions;
    ini_xinfo(predictions, N, M);

    xinfo predictions_test;
    ini_xinfo(predictions_test, Xtest.nrow(), M);

    std::vector<double> yhat(N);
    row_sum(predictions, yhat);
    std::vector<double> yhat_test(Xtest.nrow());
    row_sum(predictions_test, yhat_test);

    // current residual
    std::vector<double> residual(N);

    xinfo sigma_draw;
    ini_xinfo(sigma_draw, M, N_sweeps);

    double sigma = 1.0; 

    forest trees(M);

    std::vector<double> reshat;
    std::vector<double> reshat_test;

    std::vector<double> prob(2, 0.5);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> d(prob.begin(), prob.end());

    size_t prune;

    double* y_pointer = &y[0];
    double* X_pointer = &X[0];
    double* Xtest_pointer = &Xtest[0];


    for(size_t mc = 0; mc < L; mc ++){
    
        for(size_t i = 0; i < M; i ++ ){
            // initialize predictions with mean of y
            std::fill(predictions[i].begin(), predictions[i].end(), y_mean);
            std::fill(predictions_test[i].begin(), predictions_test[i].end(), y_mean);
        }
        // predictions.fill(y_mean);

        row_sum(predictions, yhat);
        row_sum(predictions_test, yhat_test);
        
        // residual = y - yhat;

        for(size_t sweeps = 0; sweeps < N_sweeps; sweeps ++ ){
            // loop over sweeps
            if(verbose == true){
                cout << "--------------------------------" << endl;
                cout << "number of sweeps " << sweeps << endl;
                cout << "--------------------------------" << endl;
            }

            for(size_t tree_ind = 0; tree_ind < M; tree_ind ++ ){
                // loop over trees in each sweep
                if(m_update_sigma == true){
                    // if update sigma based on residual of all M trees
                    gamma_d = std::gamma_distribution<double>((N + kap) / 2.0, 2.0 / sum_squared(residual) + s);
                    sigma = 1.0 / gamma_d(generator);
                    sigma_draw[sweeps][tree_ind] = sigma;
                }

                // save sigma
                // add prediction of current tree back to residual
                // then it's M - 1 trees residual

                residual = residual + predictions[tree_ind];

                // prediction of M - 1 trees
                yhat = yhat - predictions[tree_ind];

                // prediction of M - 1 trees on testing set
                yhat_test = yhat_test - predictions_test[tree_ind];

                // cout << p << "  " << N << "   " << Nmin << "   "<< Ncutpoints << "   " << tau << "   " << sigma << "   " << alpha<< "   " << beta << endl;


                cout << "------------------------OK1" << endl;
                trees.t[tree_ind].grow_tree_adaptive_std(residual, y_mean, Xorder, X_pointer, p, N, N, 0, (size_t) max_depth(tree_ind, sweeps), Nmin, Ncutpoints, tau, sigma, alpha, beta, draw_sigma, draw_mu, parallel);
                cout << "------------------------OK2" << endl;




            //     // grow a tree
            //     if(sweeps < 1){
            //         trees.t[tree_ind].grow_tree_adaptive_std();
            //     }else{
            //         prune = d(gen);

            //         if(prune == 0){
            //             trees.t[tree_ind].one_step_grow_std;

            //         }else{
            //             trees.t[tree_ind].one_step_prune;
            //         }


            //         trees.t[tree_ind].sample_theta_std();

            //     }

                if(verbose == true){
                    cout << "tree " << tree_ind << " size is " << trees.t[tree_ind].treesize() << endl;
                }

            //     // update prediction of current tree
            //     predictions[tree_ind] = fit_new_theta_noise_std;

                fit_new_theta_noise_std(trees.t[tree_ind], X_pointer, p, N, predictions[tree_ind]);

            //     predictions_test[tree_ind] = fit_new_theta_noise_std;
                fit_new_theta_noise_std(trees.t[tree_ind], Xtest_pointer, p, N_test, predictions_test[tree_ind]);

            //     // update sigma based on residual of M - 1 trees, residual_theta_noise

                if(m_update_sigma == false){
                    sigma_draw[sweeps][tree_ind] = sigma;
                }


            //     // update residual, now it's residual of M trees
                residual = residual - predictions[tree_ind];

                yhat = yhat + predictions[tree_ind];

                yhat_test = yhat_test + predictions_test[tree_ind];
            }


        
            // save to output object
            yhats[sweeps] = yhat;
            yhats_test[sweeps] = yhat_test;

        }


        // NumericVector output = wrap(x); convert std vector to NumericVector object in Rcpp

    }



    return Rcpp::List::create(Rcpp::Named("yhats") = Rcpp::wrap(yhats), Rcpp::Named("yhats_test") = Rcpp::wrap(yhats_test), Rcpp::Named("sigma") = Rcpp::wrap(sigma_draw));
}

