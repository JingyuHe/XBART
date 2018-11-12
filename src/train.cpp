#include <ctime>
#include <RcppArmadillo.h>
#include "tree.h"
#include "treefuns.h"
#include "forest.h"
#include <chrono>

using namespace std;

using namespace chrono;

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]]


// Helpers
struct AbarthParams{
            size_t M;
            size_t L;size_t N_sweeps; size_t Nmin; size_t Ncutpoints;
            size_t burnin; size_t mtry;size_t max_depth_num;
            double alpha;double beta;double tau;double kap;double s;
            bool draw_sigma;bool verbose; bool m_update_sigma;
            bool draw_mu;bool parallel;
};



void rcpp_to_std(arma::mat y, arma::mat X, arma::mat Xtest,
    std::vector<double>& y_std,double& y_mean,
    Rcpp::NumericMatrix& X_std,Rcpp::NumericMatrix& Xtest_std,
    xinfo_sizet& Xorder_std){
    // The goal of this function is to convert RCPP object to std objects

    // TODO: Refactor to remove N,p,N_test
    // TODO: Refactor code so for loops are self contained functions
    // TODO: Why RCPP and not std? 
    // TODO: inefficient Need Replacement?

    size_t N = X.n_rows;
    size_t p = X.n_cols;
    size_t N_test = Xtest.n_rows;

    // Create y_std
    for (size_t i = 0; i < N; i++)
    {
        y_std[i] = y(i, 0);
        y_mean = y_mean + y_std[i];
    }
    y_mean = y_mean / (double)N;

    // X_std
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < p; j++)
        {
            X_std(i, j) = X(i, j);
        }
    }


    //X_std_test
    for (size_t i = 0; i < N_test; i++)
    {
        for (size_t j = 0; j < p; j++)
        {
            Xtest_std(i, j) = Xtest(i, j);
        }
    }

    // Create Xorder
    // Order
    arma::umat Xorder(X.n_rows, X.n_cols);
    for (size_t i = 0; i < X.n_cols; i++)
    {
        Xorder.col(i) = arma::sort_index(X.col(i));
    }
    // Create
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < p; j++)
        {
            Xorder_std[j][i] = Xorder(i, j);
        }
    }

    return;
}



Rcpp::List abarth_train(arma::mat y, arma::mat X, arma::mat Xtest, 
    size_t M, size_t L, size_t N_sweeps, arma::mat max_depth, 
    size_t Nmin, size_t Ncutpoints, double alpha, double beta, 
    double tau, size_t burnin = 1, size_t mtry = 0, 
    bool draw_sigma = false, double kap = 16, double s = 4, 
    bool verbose = false, bool m_update_sigma = false, 
    bool draw_mu = false, bool parallel = true)
{

    
    auto start = system_clock::now();
    // RCPP -> STD
    // Container for std types to "return"
    size_t N = X.n_rows;
    size_t p = X.n_cols;
    size_t N_test = Xtest.n_rows;
    // y containers
    std::vector<double> y_std(N);
    double y_mean = 0.0;
    // x containers
    Rcpp::NumericMatrix X_std(N, p);
    Rcpp::NumericMatrix Xtest_std(N_test, p);
    // xorder containers
    xinfo_sizet Xorder_std;
    ini_xinfo_sizet(Xorder_std, N, p);
    // convert rcpp to std
    rcpp_to_std(y,X,Xtest,y_std,y_mean,X_std,Xtest_std,Xorder_std);


    // Assertions
    assert(mtry <= p);
    assert(burnin <= N_sweeps);

    if (mtry == 0)
    {
        mtry = p;
    }

    if (mtry != p)
    {
        cout << "Sample " << mtry << " out of " << p << " variables when grow each tree." << endl;
    }

    // Set Random Generator
    std::default_random_engine(generator);

    // Pointers for data
    double *ypointer = &y_std[0];
    double *Xpointer = &X_std[0];
    double *Xtestpointer = &Xtest_std[0];

    // save predictions of each tree
    std::vector<std::vector<double>> predictions_std;
    ini_xinfo(predictions_std, N, M);

    xinfo predictions_test_std;
    ini_xinfo(predictions_test_std, N_test, M);

    std::vector<double> yhat_std(N);
    row_sum(predictions_std, yhat_std);
    std::vector<double> yhat_test_std(N_test);
    row_sum(predictions_test_std, yhat_test_std);

    xinfo sigma_draw_std;
    ini_xinfo(sigma_draw_std, M, N_sweeps);


    ///////////////////////////////////////////////////////////////////


    // void fit(mc,M,predictions_std,predictions_test_std,y_mean)


    // Cpp native objects to return
    xinfo yhats_xinfo;
    ini_xinfo(yhats_xinfo, N, N_sweeps);

    xinfo yhats_test_xinfo;
    ini_xinfo(yhats_test_xinfo, N, N_sweeps);

    xinfo sigma_draw_xinfo;
    ini_xinfo(sigma_draw_xinfo, M, N_sweeps);


    // Other R Objects - CHANGE TO STD
    // Rcpp::NumericMatrix split_count_all_tree(p, M); // initialize at 0

    xinfo split_count_all_tree;  // initialize at 0
    ini_xinfo(split_count_all_tree, p, M);

    // split_count_all_tree = split_count_all_tree + 1; // initialize at 1
    std::vector<double> split_count_current_tree(p, 1.0);
    std::vector<double> mtry_weight_current_tree(p, 1.0);

    // Rcpp::NumericVector split_count_current_tree(p, 1);
    // Rcpp::NumericVector mtry_weight_current_tree(p, 1);

    double sigma;
    // double tau;
    forest trees(M);
    std::vector<double> prob(2, 0.5);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> d(prob.begin(), prob.end());
    // // sample one index of split point
    size_t prune;

    // current residual
    std::vector<double> residual_std(N);

    // std::vector<double> split_var_count(p);
    // std::fill(split_var_count.begin(), split_var_count.end(), 1);
    // Rcpp::NumericVector split_var_count(p, 1);


    

    // double *split_var_count_pointer = &split_var_count[0];


    // in the burnin samples, use all variables
    std::vector<size_t> subset_vars(p);
    std::iota(subset_vars.begin() + 1, subset_vars.end(), 1);
    
    // cout << subset_vars << endl;

    Rcpp::IntegerVector var_index_candidate(p);
    for (size_t i = 0; i < p; i++)
    {
        var_index_candidate[i] = i;
    }

    double run_time = 0.0;

    // save tree objects to strings
    // std::stringstream treess;
    // treess.precision(10);
    // treess << L << " " << M << " " << p << endl;

    // L, number of samples
    // M, number of trees

    bool use_all = true;

    for (size_t mc = 0; mc < L; mc++)
    {

        // initialize predcitions and predictions_test
        for (size_t ii = 0; ii < M; ii++)
        {
            std::fill(predictions_std[ii].begin(), predictions_std[ii].end(), y_mean / (double)M);
            std::fill(predictions_test_std[ii].begin(), predictions_test_std[ii].end(), y_mean / (double)M);
        }

        row_sum(predictions_std, yhat_std);
        row_sum(predictions_test_std, yhat_test_std);

        residual_std = y_std - yhat_std;

        for (size_t sweeps = 0; sweeps < N_sweeps; sweeps++)
        {

            if (verbose == true)
            {
                cout << "--------------------------------" << endl;
                cout << "number of sweeps " << sweeps << endl;
                cout << "--------------------------------" << endl;
            }

            for (size_t tree_ind = 0; tree_ind < M; tree_ind++)
            {

                // if update sigma based on residual of all m trees
                if (m_update_sigma == true)
                {

                    std::gamma_distribution<double> gamma_samp((N + kap) / 2.0, 2.0 / (sum_squared(residual_std) + s));

                    sigma = 1.0 / sqrt(gamma_samp(generator));

                    sigma_draw_xinfo[sweeps][tree_ind] = sigma;
                }

                // save sigma
                sigma_draw_xinfo[sweeps][tree_ind] = sigma;

                // add prediction of current tree back to residual
                // then it's m - 1 trees residual

                residual_std = residual_std + predictions_std[tree_ind];

                // do the samething for residual_theta_noise, residual of m - 1 trees

                yhat_std = yhat_std - predictions_std[tree_ind];

                yhat_test_std = yhat_test_std - predictions_test_std[tree_ind];

                if (use_all && (sweeps > burnin) && (mtry != p))
                {
                    // subset_vars = Rcpp::as<std::vector<size_t>>(sample(var_index_candidate, mtry, false, split_var_count));
                    use_all = false;
                }

                // cout << "variables used " << subset_vars << endl;
                // cout << "------------------" << endl;

                // clear counts of splits for one tree
                std::fill(split_count_current_tree.begin(), split_count_current_tree.end(), 0.0);

                for(int i=0;i<p;i++){
                    mtry_weight_current_tree[i] = mtry_weight_current_tree[i] - split_count_all_tree[tree_ind][i];
                }
                // mtry_weight_current_tree = mtry_weight_current_tree - split_count_all_tree(Rcpp::_, tree_ind);

                // cout << "before " << mtry_weight_current_tree << endl;

                trees.t[tree_ind].grow_tree_adaptive_abarth_train(sum_vec(residual_std) / (double)N, 0, max_depth(tree_ind, sweeps), Nmin, Ncutpoints, tau, sigma, alpha, beta, draw_sigma, draw_mu, parallel, residual_std, Xorder_std, Xpointer, mtry, run_time, var_index_candidate, use_all, mtry_weight_current_tree, split_count_current_tree);

                mtry_weight_current_tree = mtry_weight_current_tree + split_count_current_tree;

                // cout << "after " << mtry_weight_current_tree << endl; 


                for(int i = 0;i<p;i++){
                    split_count_all_tree[tree_ind][i] = split_count_current_tree[i];
                }
                // split_count_all_tree(Rcpp::_, tree_ind) = split_count_current_tree; 


                if (verbose == true)
                {
                    cout << "tree " << tree_ind << " size is " << trees.t[tree_ind].treesize() << endl;
                }

                // update prediction of current tree
                fit_new_std(trees.t[tree_ind], Xpointer, N, p, predictions_std[tree_ind]);

                // update prediction of current tree, test set
                fit_new_std(trees.t[tree_ind], Xtestpointer, N_test, p, predictions_test_std[tree_ind]);

                // update sigma based on residual of m - 1 trees, residual_theta_noise
                if (m_update_sigma == false)
                {

                    std::gamma_distribution<double> gamma_samp((N + kap) / 2.0, 2.0 / (sum_squared(residual_std) + s));

                    sigma = 1.0 / sqrt(gamma_samp(generator));

                    sigma_draw_xinfo[sweeps][tree_ind] = sigma;
                }

                // update residual, now it's residual of m trees

                residual_std = residual_std - predictions_std[tree_ind];

                yhat_std = yhat_std + predictions_std[tree_ind];
                yhat_test_std = yhat_test_std + predictions_test_std[tree_ind];

                // treess << trees.t[tree_ind];
            }

            // save predictions to output matrix
            yhats_xinfo[sweeps] = yhat_std;
            yhats_test_xinfo[sweeps] = yhat_test_std;

        }
    }

    // Convert STD -> RCPP outputs
    // R Objects to Return
    Rcpp::NumericMatrix yhats(N, N_sweeps);
    Rcpp::NumericMatrix yhats_test(N_test, N_sweeps);
    Rcpp::NumericMatrix sigma_draw(M, N_sweeps);     // save predictions of each tree


    // TODO: Make these functions
    for(size_t i = 0;i<N_test;i++){
        for(size_t j =0;j<N_sweeps;j++){
            yhats(i,j) = yhats_xinfo[j][i];
        }
    }
    for(size_t i = 0;i<N_test;i++){
        for(size_t j =0;j<N_sweeps;j++){
            yhats_test(i,j) = yhats_test_xinfo[j][i];
        }
    }
    for(size_t i = 0;i<N_test;i++){
        for(size_t j =0;j<M;j++){
            sigma_draw(i,j) = sigma_draw_xinfo[j][i];
        }
    }

    auto end = system_clock::now();

    auto duration = duration_cast<microseconds>(end - start);

    cout << "Total running time " << double(duration.count()) * microseconds::period::num / microseconds::period::den << endl;

    cout << "Running time of split Xorder " << run_time << endl;

    cout << "Count of splits for each variable " << mtry_weight_current_tree << endl;

    // return Rcpp::List::create(Rcpp::Named("yhats") = yhats, Rcpp::Named("yhats_test") = yhats_test, Rcpp::Named("sigma") = sigma_draw, Rcpp::Named("trees") = Rcpp::CharacterVector(treess.str()));
    return Rcpp::List::create(Rcpp::Named("yhats") = yhats, Rcpp::Named("yhats_test") = yhats_test, Rcpp::Named("sigma") = sigma_draw);
}
