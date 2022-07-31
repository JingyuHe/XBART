#include <ctime>
// #include <RcppArmadillo.h>
#include "tree.h"
#include <chrono>
#include "mcmc_loop.h"
#include "X_struct.h"
#include "utility_rcpp.h"
#include "json_io.h"

using namespace std;
using namespace chrono;
using namespace arma;

void tree_to_string(vector<vector<tree>> &trees, Rcpp::StringVector &output_tree, size_t num_sweeps, size_t num_trees, size_t p)
{
    std::stringstream treess;
    for (size_t i = 0; i < num_sweeps; i++)
    {
        treess.precision(10);

        treess.str(std::string());
        treess << num_trees << " " << p << endl;

        for (size_t t = 0; t < num_trees; t++)
        {
            treess << (trees)[i][t];
        }

        output_tree(i) = treess.str();
    }
    return;
}

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]]
Rcpp::List XBCF_continuous_cpp(arma::mat y, arma::mat Z, arma::mat X_ps, arma::mat X_trt, size_t num_trees_ps, size_t num_trees_trt, size_t num_sweeps, size_t max_depth, size_t n_min, size_t num_cutpoints, double alpha_ps, double beta_ps, double alpha_trt, double beta_trt, double tau_ps, double tau_trt, double no_split_penality, size_t burnin = 1, size_t mtry_ps = 0, size_t mtry_trt = 0, size_t p_categorical_ps = 0, size_t p_categorical_trt = 0, double kap = 16, double s = 4, double tau_ps_kap = 3, double tau_ps_s = 0.5, double tau_trt_kap = 3, double tau_trt_s = 0.5, bool verbose = false, bool sampling_tau = true, bool parallel = true, bool set_random_seed = false, size_t random_seed = 0, bool sample_weights = true, double nthread = 0)
{
    if (parallel)
    {
        thread_pool.start(nthread);
        COUT << "Running in parallel with " << nthread << " threads." << endl;
    }
    else
    {
        COUT << "Running with single thread." << endl;
    }

    size_t N = X_ps.n_rows;

    // number of total variables
    size_t p_ps = X_ps.n_cols;
    size_t p_trt = X_trt.n_cols;

    COUT << "size of X_ps " << X_ps.n_rows << " " << X_ps.n_cols << endl;
    COUT << "size of X_trt " << X_trt.n_rows << " " << X_trt.n_cols << endl;

    // number of basis functions (1 in the case of the OG bcf)
    size_t p_z = Z.n_cols;

    // number of continuous variables
    size_t p_continuous_ps = p_ps - p_categorical_ps;
    size_t p_continuous_trt = p_trt - p_categorical_trt;

    // suppose first p_continuous variables are continuous, then categorical
    assert(mtry_ps <= p_ps);

    assert(mtry_trt <= p_trt);

    assert(burnin <= num_sweeps);

    if (mtry_ps == 0)
    {
        mtry_ps = p_ps;
    }

    if (mtry_trt == 0)
    {
        mtry_trt = p_trt;
    }

    if (mtry_ps != p_ps)
    {
        COUT << "Sample " << mtry_ps << " out of " << p_ps << " variables when grow each prognostic tree." << endl;
    }

    if (mtry_trt != p_trt)
    {
        COUT << "Sample " << mtry_trt << " out of " << p_trt << " variables when grow each treatment tree." << endl;
    }

    arma::umat Xorder_ps(X_ps.n_rows, X_ps.n_cols);
    matrix<size_t> Xorder_std_ps;
    ini_matrix(Xorder_std_ps, N, p_ps);

    arma::umat Xorder_trt(X_trt.n_rows, X_trt.n_cols);
    matrix<size_t> Xorder_std_trt;
    ini_matrix(Xorder_std_trt, N, p_trt);

    std::vector<double> y_std(N);
    double y_mean = 0.0;

    for (size_t i = 0; i < N; i++)
    {
        y_mean += y[i] / Z[i];
    }
    y_mean = y_mean / N;
    cout << "y mean is " << y_mean << endl;

    Rcpp::NumericMatrix X_std_ps(N, p_ps);
    Rcpp::NumericMatrix X_std_trt(N, p_trt);

    matrix<double> Z_std;
    ini_matrix(Z_std, N, p_z);

    rcpp_to_std2(y, Z, X_ps, X_trt, y_std, y_mean, Z_std, X_std_ps, X_std_trt, Xorder_std_ps, Xorder_std_trt);

    ///////////////////////////////////////////////////////////////////

    double *Xpointer_ps = &X_std_ps[0];
    double *Xpointer_trt = &X_std_trt[0];

    matrix<double> sigma_draw_xinfo;
    ini_matrix(sigma_draw_xinfo, num_trees_ps + num_trees_trt, num_sweeps);

    // create trees
    vector<vector<tree>> trees_ps(num_sweeps);
    vector<vector<tree>> trees_trt(num_sweeps);

    for (size_t i = 0; i < num_sweeps; i++)
    {
        trees_ps[i].resize(num_trees_ps);
        trees_trt[i].resize(num_trees_trt);
    }

    // define model
    XBCFContinuousModel *model = new XBCFContinuousModel(kap, s, tau_ps, tau_trt, alpha_ps, beta_ps, alpha_trt, beta_trt, sampling_tau, tau_ps_kap, tau_ps_s, tau_trt_kap, tau_trt_s);
    model->setNoSplitPenality(no_split_penality);

    // State settings
    std::unique_ptr<State> state(new NormalLinearState(&Z_std, Xpointer_ps, Xpointer_trt, Xorder_std_ps, Xorder_std_trt, N, p_ps, p_trt, num_trees_ps, num_trees_trt, p_categorical_ps, p_categorical_trt, p_continuous_ps, p_continuous_trt, set_random_seed, random_seed, n_min, num_cutpoints, mtry_ps, mtry_trt, Xpointer_ps, num_sweeps, sample_weights, &y_std, 1.0, max_depth, y_mean, burnin, model->dim_residual, nthread, parallel)); // last input is nthread, need update

    // initialize X_struct
    std::vector<double> initial_theta_ps(1, 0);
    std::unique_ptr<X_struct> x_struct_ps(new X_struct(Xpointer_ps, &y_std, N, Xorder_std_ps, p_categorical_ps, p_continuous_ps, &initial_theta_ps, num_trees_ps));

    std::vector<double> initial_theta_trt(1, y_mean / (double)num_trees_trt);
    std::unique_ptr<X_struct> x_struct_trt(new X_struct(Xpointer_trt, &y_std, N, Xorder_std_trt, p_categorical_trt, p_continuous_trt, &initial_theta_trt, num_trees_trt));

    ////////////////////////////////////////////////////////////////
    mcmc_loop_linear(Xorder_std_ps, Xorder_std_trt, verbose, sigma_draw_xinfo, trees_ps, trees_trt, no_split_penality, state, model, x_struct_ps, x_struct_trt);

    // R Objects to Return
    Rcpp::NumericMatrix sigma_draw(num_trees_ps + num_trees_trt, num_sweeps); // save predictions of each tree
    Rcpp::NumericVector split_count_sum_ps(p_ps, 0);                          // split counts
    Rcpp::NumericVector split_count_sum_trt(p_trt, 0);
    // Rcpp::XPtr<std::vector<std::vector<tree>>> tree_pnt_ps(trees_ps, true);
    // Rcpp::XPtr<std::vector<std::vector<tree>>> tree_pnt_trt(trees_trt, true);
    //
    // copy from std vector to Rcpp Numeric Matrix objects
    Matrix_to_NumericMatrix(sigma_draw_xinfo, sigma_draw);

    for (size_t i = 0; i < p_ps; i++)
    {
        split_count_sum_ps(i) = (int)state->split_count_all_ps[i];
    }

    for (size_t i = 0; i < p_trt; i++)
    {
        split_count_sum_trt(i) = (int)state->split_count_all_trt[i];
    }

    // clean memory
    // delete model;
    state.reset();
    x_struct_ps.reset();
    x_struct_trt.reset();

    // print out tree structure, for usage of BART warm-start

    Rcpp::StringVector output_tree_ps(num_sweeps);
    Rcpp::StringVector output_tree_trt(num_sweeps);

    tree_to_string(trees_trt, output_tree_trt, num_sweeps, num_trees_trt, p_trt);
    tree_to_string(trees_ps, output_tree_ps, num_sweeps, num_trees_ps, p_ps);

    Rcpp::StringVector tree_json_trt(1);
    Rcpp::StringVector tree_json_ps(1);
    json j = get_forest_json(trees_trt, y_mean);
    json j2 = get_forest_json(trees_ps, y_mean);
    tree_json_trt[0] = j.dump(4);
    tree_json_ps[0] = j2.dump(4);

    thread_pool.stop();

    return Rcpp::List::create(
        Rcpp::Named("sigma") = sigma_draw,
        Rcpp::Named("importance_prognostic") = split_count_sum_ps,
        Rcpp::Named("importance_treatment") = split_count_sum_trt,
        Rcpp::Named("model_list") = Rcpp::List::create(Rcpp::Named("y_mean") = y_mean, Rcpp::Named("p_ps") = p_ps, Rcpp::Named("p_trt") = p_trt),
        Rcpp::Named("tree_json_trt") = tree_json_trt,
        Rcpp::Named("tree_json_ps") = tree_json_ps,
        Rcpp::Named("tree_string_trt") = output_tree_trt,
        Rcpp::Named("tree_string_ps") = output_tree_ps);
}
