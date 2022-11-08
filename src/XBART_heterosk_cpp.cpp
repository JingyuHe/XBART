#include <ctime>
// #include <RcppArmadillo.h>
#include "Rcpp.h"
#include <armadillo>
#include "tree.h"
#include <chrono>
#include "mcmc_loop.h"
#include "X_struct.h"
#include "utility_rcpp.h"

using namespace std;
using namespace chrono;

// TODO: determine the appropriate set of input variables
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

    // Create trees for the mean model
    vector<vector<tree>> *trees2_m = new vector<vector<tree>>(num_sweeps);
    for (size_t i = 0; i < num_sweeps; i++)
    {
        (*trees2_m)[i] = vector<tree>(num_trees_m);
    }

    // Create trees for the variance model
    vector<vector<tree>> *trees2_v = new vector<vector<tree>>(num_sweeps);
    for (size_t i = 0; i < num_sweeps; i++)
    {
        (*trees2_v)[i] = vector<tree>(num_trees_v);
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

    // State settings for the mean model
 //   NormalState state_m(Xpointer, Xorder_std, N, p, num_trees_m, p_categorical, p_continuous, set_random_seed, random_seed, n_min_m, num_cutpoints_m, mtry, Xpointer, num_sweeps, sample_weights, &y_std, 1.0, max_depth_m, y_mean, burnin, model_m->dim_residual, nthread, parallel);
    std::vector<double> sigma_vec(N, ini_var); // initialize vector of heterogeneous sigmas
    hskState state_m(Xpointer, Xorder_std, N, p, num_trees_m,
                                                p_categorical, p_continuous, set_random_seed,
                                                random_seed, n_min_m, num_cutpoints_m,
                                                mtry, Xpointer, num_sweeps, sample_weights,
                                                &y_std, 1.0, max_depth_m, y_mean, burnin, model_m->dim_residual,
                                                nthread, parallel, sigma_vec); //last input is nthread, need update*/
    // State settings for the variance model
    NormalState state_v(Xpointer, Xorder_std, N, p, num_trees_v,
                                                   p_categorical, p_continuous, set_random_seed,
                                                   random_seed, n_min_v, num_cutpoints_v,
                                                   mtry, Xpointer, num_sweeps, sample_weights,
                                                   &y_std, 1.0, max_depth_v, y_mean, burnin, model_v->dim_residual,
                                                   nthread, parallel); //last input is nthread, need update

    // initialize X_struct
    std::vector<double> initial_theta_m(1, y_mean / (double)num_trees_m);
    X_struct x_struct_m(Xpointer, &y_std, N, Xorder_std, p_categorical, p_continuous, &initial_theta_m, num_trees_m);
    //COUT << "var: " << ini_var << endl;
    std::vector<double> initial_theta_v(1, exp(log(1.0/ ini_var) / (double)num_trees_v));
    //std::vector<double> initial_theta_v(1, 1);
    X_struct x_struct_v(Xpointer, &y_std, N, Xorder_std, p_categorical, p_continuous, &initial_theta_v, num_trees_v);

    //COUT << "Running the model." << endl;
    ////////////////////////////////////////////////////////////////
    // mcmc loop
    mcmc_loop_hsk(Xorder_std, verbose, sigma_draw_xinfo, *trees2_m, state_m, model_m, x_struct_m, *trees2_v, state_v, model_v, x_struct_v,
                  res_m, res_v);

    //mcmc_loop_hsk_test(Xorder_std, verbose, sigma_draw_xinfo, *trees2_m, state_m, model_m, x_struct_m);

    // stop parallel computing
    thread_pool.stop();

    //COUT << "Predict." << endl;
    // TODO: check how predict function will be different
    model_m->predict_std(Xtestpointer, N_test, p, num_trees_m, num_sweeps, mhats_test_xinfo, *trees2_m);
    model_v->predict_std(Xtestpointer, N_test, p, num_trees_v, num_sweeps, vhats_test_xinfo, *trees2_v);

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
    // Rcpp::NumericMatrix yhats(N, num_sweeps);
    Rcpp::NumericMatrix yhats_test(N_test, num_sweeps);
    Rcpp::NumericMatrix sigma2_test(N_test, num_sweeps);

    Rcpp::NumericMatrix res_mm(N, num_sweeps);
    Rcpp::NumericMatrix res_vm(N, num_sweeps);

    // copy from std vector to Rcpp Numeric Matrix objects
    Matrix_to_NumericMatrix(yhats_test_xinfo, yhats_test);
    Matrix_to_NumericMatrix(sigma2_test_xinfo, sigma2_test);
    Matrix_to_NumericMatrix(res_m, res_mm);
    Matrix_to_NumericMatrix(res_v, res_vm);

    // TODO: check outputs
    return Rcpp::List::create(
        Rcpp::Named("yhats_test") = yhats_test,
        Rcpp::Named("sigma2hats_test") = sigma2_test,
        Rcpp::Named("res_mm") = res_mm,
        Rcpp::Named("res_vm") = res_vm
        );
}