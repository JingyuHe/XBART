#include <ctime>
// #include <RcppArmadillo.h>
#include "Rcpp.h"
#include <armadillo>
#include "tree.h"
#include <chrono>
#include "mcmc_loop.h"
#include "X_struct.h"
#include "utility_rcpp.h"
#include "json_io.h"

using namespace std;
using namespace chrono;

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]]
Rcpp::List XBART_heterosk_cpp(arma::mat y,
                              arma::mat X,
                              arma::mat Xtest,
                              size_t num_sweeps,
                              size_t burnin,
                              size_t p_categorical,
                              size_t mtry,
                              double no_split_penalty_m,
                              size_t num_trees_m,
                              size_t max_depth_m,
                              size_t n_min_m,
                              size_t num_cutpoints_m,
                              double tau_m,
                              double no_split_penalty_v,
                              size_t num_trees_v,
                              size_t max_depth_v,
                              size_t n_min_v,
                              size_t num_cutpoints_v,
                              double a_v, double b_v, // shape and rate
                              double ini_var, // optional initialization for variance
                              double kap = 16, double s = 4,
                              double tau_kap = 3, double tau_s = 0.5,
                              double alpha = 0.95, double beta = 1.25, //BART tree params
                              bool verbose = false,
                              bool sampling_tau = true,
                              bool parallel = true,
                              bool set_random_seed = false,
                              size_t random_seed = 0,
                              bool sample_weights = true,
                              double nthread = 0
                              )
{
    //COUT << "In source." << endl;
    //double var = ini_var;
    if (parallel)
    {
        thread_pool.start(nthread);
    }

    size_t N = X.n_rows;

    // number of total variables
    size_t p = X.n_cols;

    size_t N_test = Xtest.n_rows;

    // number of continuous variables
    size_t p_continuous = p - p_categorical;

    // suppose first p_continuous variables are continuous, then categorical

    assert(mtry <= p);

    assert(burnin <= num_sweeps);

    if (mtry == 0)
    {
        mtry = p;
    }

    if (mtry != p)
    {
        COUT << "Sample " << mtry << " out of " << p << " variables when grow each tree." << endl;
    }

    arma::umat Xorder(X.n_rows, X.n_cols);
    matrix<size_t> Xorder_std;
    ini_matrix(Xorder_std, N, p);

    std::vector<double> y_std(N);
    double y_mean = 0.0;

    Rcpp::NumericMatrix X_std(N, p);
    Rcpp::NumericMatrix Xtest_std(N_test, p);

    rcpp_to_std2(y, X, Xtest, y_std, y_mean, X_std, Xtest_std, Xorder_std);

    ///////////////////////////////////////////////////////////////////

    // double *ypointer = &y_std[0];
    double *Xpointer = &X_std[0];
    double *Xtestpointer = &Xtest_std[0];

    matrix<double> res_m;
    ini_matrix(res_m, N, num_sweeps);
    matrix<double> res_v;
    ini_matrix(res_v, N, num_sweeps);

    matrix<double> yhats_test_xinfo;
    ini_matrix(yhats_test_xinfo, N_test, num_sweeps);

    matrix<double> sigma2_test_xinfo;
    ini_matrix(sigma2_test_xinfo, N_test, num_sweeps);

    matrix<double> mhats_test_xinfo;
    ini_matrix(mhats_test_xinfo, N_test, num_sweeps);

    matrix<double> vhats_test_xinfo;
    ini_matrix(vhats_test_xinfo, N_test, num_sweeps);

    // TODO: check and remove -- this is likely an unnecessary storage
    matrix<double> sigma_draw_xinfo;
    ini_matrix(sigma_draw_xinfo, num_trees_m, num_sweeps);
/*
    // Create trees for the mean model
    vector<vector<tree>> *trees_mean = new vector<vector<tree>>(num_sweeps);
    for (size_t i = 0; i < num_sweeps; i++)
    {
        (*trees_mean)[i] = vector<tree>(num_trees_m);
    }

    // Create trees for the variance model
    vector<vector<tree>> *trees_var = new vector<vector<tree>>(num_sweeps);
    for (size_t i = 0; i < num_sweeps; i++)
    {
        (*trees_var)[i] = vector<tree>(num_trees_v);
    }
*/
    vector<vector<tree>> trees_mean(num_sweeps);
    vector<vector<tree>> trees_var(num_sweeps);

    for (size_t i = 0; i < num_sweeps; i++)
    {
        trees_mean[i].resize(num_trees_m);
        trees_var[i].resize(num_trees_v);
    }

    // COUT << "Objects init." << endl;
    // define the mean model
    hskNormalModel *model_m = new hskNormalModel(kap, s, tau_m, alpha, beta, sampling_tau, tau_kap, tau_s);

//    NormalModel *model_m = new NormalModel(kap, s, tau_m, alpha, beta, sampling_tau, tau_kap, tau_s);
    // cout << "after define model " << model->tau << " " << model->tau_mean << endl;
    model_m->setNoSplitPenalty(no_split_penalty_m);

    // define the variance model
    // TODO:update the first two inputs
    logNormalModel *model_v = new logNormalModel(a_v, b_v, kap, s, tau_m, alpha, beta);
    // cout << "after define model " << model->tau << " " << model->tau_mean << endl;
    model_v->setNoSplitPenalty(no_split_penalty_v);

    // initialize X_struct
    std::vector<double> initial_theta_m(1, y_mean / (double)num_trees_m);
    X_struct x_struct_m(Xpointer, &y_std, N, Xorder_std, p_categorical, p_continuous, &initial_theta_m, num_trees_m);
    //COUT << "var: " << ini_var << endl;
    std::vector<double> initial_theta_v(1, exp(log(1.0/ ini_var) / (double)num_trees_v));
    //std::vector<double> initial_theta_v(1, 1);
    X_struct x_struct_v(Xpointer, &y_std, N, Xorder_std, p_categorical, p_continuous, &initial_theta_v, num_trees_v);

    // State settings for the mean model
 //   NormalState state_m(Xpointer, Xorder_std, N, p, num_trees_m, p_categorical, p_continuous, set_random_seed, random_seed, n_min_m, num_cutpoints_m, mtry, Xpointer, num_sweeps, sample_weights, &y_std, 1.0, max_depth_m, y_mean, burnin, model_m->dim_residual, nthread, parallel);
    std::vector<double> sigma_vec(N, ini_var); // initialize vector of heterogeneous sigmas

    HeteroskedasticState state(Xpointer, Xorder_std,
                               N, p,
                               num_trees_m, num_trees_v,
                               p_categorical, p_continuous,
                               set_random_seed, random_seed,
                               n_min_m, n_min_v,
                               num_cutpoints_m, num_cutpoints_v,
                               mtry, Xpointer,
                               num_sweeps, sample_weights,
                               &y_std, 1.0,
                               max_depth_m, max_depth_v,
                               y_mean, burnin,
                               model_v->dim_residual, nthread,
                               parallel, sigma_vec);

/*
    // Create trees for the mean model
    vector<vector<tree>> *trees1_m = new vector<vector<tree>>(num_sweeps);
    for (size_t i = 0; i < num_sweeps; i++)
    {
        (*trees1_m)[i] = vector<tree>(num_trees_m);
    }

    // Create trees for the variance model
    vector<vector<tree>> *trees1_v = new vector<vector<tree>>(num_sweeps);
    for (size_t i = 0; i < num_sweeps; i++)
    {
        (*trees1_v)[i] = vector<tree>(num_trees_v);
    }
*/
    mcmc_loop_heteroskedastic(Xorder_std, verbose, state, model_m, trees_mean, x_struct_m, model_v, trees_var, x_struct_v);

/*
    HeteroskedasticState state_m(Xpointer, Xorder_std,
                               N, p,
                               num_trees_m, num_trees_m,
                               p_categorical, p_continuous,
                               set_random_seed, random_seed,
                               n_min_m, n_min_m,
                               num_cutpoints_m, num_cutpoints_m,
                               mtry, Xpointer,
                               num_sweeps, sample_weights,
                               &y_std, 1.0,
                               max_depth_m, max_depth_m,
                               y_mean, burnin,
                               model_v->dim_residual, nthread,
                               parallel, sigma_vec);

    HeteroskedasticState state_v(Xpointer, Xorder_std,
                               N, p,
                               num_trees_v, num_trees_v,
                               p_categorical, p_continuous,
                               set_random_seed, random_seed,
                               n_min_v, n_min_v,
                               num_cutpoints_v, num_cutpoints_v,
                               mtry, Xpointer,
                               num_sweeps, sample_weights,
                               &y_std, 1.0,
                               max_depth_v, max_depth_v,
                               y_mean, burnin,
                               model_v->dim_residual, nthread,
                               parallel, sigma_vec);

    //COUT << "Running the model." << endl;
    ////////////////////////////////////////////////////////////////
    // mcmc loop
    mcmc_loop_hsk(Xorder_std, verbose, sigma_draw_xinfo, *trees_mean, state_m, model_m, x_struct_m, *trees_var, state_v, model_v, x_struct_v,
                  res_m, res_v);

*/
    // stop parallel computing
    thread_pool.stop();

    //COUT << "Predict." << endl;
    // TODO: check how predict function will be different
    model_m->predict_std(Xtestpointer, N_test, p, num_trees_m, num_sweeps, mhats_test_xinfo, trees_mean);
    model_v->predict_std(Xtestpointer, N_test, p, num_trees_v, num_sweeps, vhats_test_xinfo, trees_var);

    //state_m.reset();
    //state_v.reset();
    //x_struct_m.reset();
    //x_struct_v.reset();

    //COUT << "Re-format objects for output." << endl;

    for(size_t i = 0; i < N_test; i++) {
        for(size_t j = 0; j < num_sweeps; j++) {
            sigma2_test_xinfo[j][i] = 1.0 / vhats_test_xinfo[j][i];
            yhats_test_xinfo[j][i] = mhats_test_xinfo[j][i];
        }
    }
    // R Objects to Return
    Rcpp::NumericMatrix yhats_test(N_test, num_sweeps);
    Rcpp::NumericMatrix sigma2_test(N_test, num_sweeps);
    Rcpp::NumericVector split_count_sum_mean(p, 0);
    Rcpp::NumericVector split_count_sum_var(p, 0);

    // copy from std vector to Rcpp Numeric Matrix objects
    Matrix_to_NumericMatrix(yhats_test_xinfo, yhats_test);
    Matrix_to_NumericMatrix(sigma2_test_xinfo, sigma2_test);
    for (size_t i = 0; i < p; i++)
    {
        split_count_sum_mean(i) = (int)(*state.split_count_all_m)[i];
        split_count_sum_var(i) = (int)(*state.split_count_all_v)[i];
    }

    // print out tree structure, for usage of BART warm-start
    Rcpp::StringVector output_tree_mean(num_sweeps);
    Rcpp::StringVector output_tree_var(num_sweeps);

    tree_to_string(trees_mean, output_tree_mean, num_sweeps, num_trees_m, p);
    tree_to_string(trees_var, output_tree_var, num_sweeps, num_trees_v, p);

    Rcpp::StringVector tree_json_mean(1);
    Rcpp::StringVector tree_json_var(1);
    json j = get_forest_json(trees_mean, y_mean);
    json j2 = get_forest_json(trees_var, y_mean);
    tree_json_mean[0] = j.dump(4);
    tree_json_var[0] = j2.dump(4);


    return Rcpp::List::create(
        Rcpp::Named("yhats_test") = yhats_test,
        Rcpp::Named("sigma2hats_test") = sigma2_test,
        Rcpp::Named("importance_mean") = split_count_sum_mean,
        Rcpp::Named("importance_variance") = split_count_sum_var,
        Rcpp::Named("model_list") = Rcpp::List::create(Rcpp::Named("y_mean") = y_mean, Rcpp::Named("p") = p),
        Rcpp::Named("tree_json_mean") = tree_json_mean,
        Rcpp::Named("tree_json_variance") = tree_json_var,
        Rcpp::Named("tree_string_mean") = output_tree_mean,
        Rcpp::Named("tree_string_variance") = output_tree_var
        );
}