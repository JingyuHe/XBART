#include <ctime>
// #include <RcppArmadillo.h>
#include "tree.h"
#include <chrono>
#include "mcmc_loop.h"
#include "X_struct.h"
#include "utility_rcpp.h"

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
Rcpp::List XBCF_continuous_cpp(arma::mat y, arma::mat Z, arma::mat X, arma::mat X_ps, arma::mat X_trt, arma::mat Xtest, arma::mat Xtest_ps, arma::mat Xtest_trt, arma::mat Ztest, size_t num_trees, size_t num_trees_ps, size_t num_trees_trt, size_t num_sweeps, size_t max_depth, size_t n_min, size_t num_cutpoints, double alpha, double beta, double tau, double no_split_penality, size_t burnin = 1, size_t mtry = 0, size_t mtry_ps = 0, size_t mtry_trt = 0, size_t p_categorical = 0, size_t p_categorical_ps = 0, size_t p_categorical_trt = 0, double kap = 16, double s = 4, double tau_kap = 3, double tau_s = 0.5, bool verbose = false, bool sampling_tau = true, bool parallel = true, bool set_random_seed = false, size_t random_seed = 0, bool sample_weights = true, double nthread = 0)
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

    size_t N = X.n_rows;

    // number of total variables
    size_t p = X.n_cols;
    size_t p_ps = X.n_cols;
    size_t p_trt = X_trt.n_cols;

    COUT << "size of x " << X.n_rows << " " << X.n_cols << endl;

    size_t N_test = Xtest.n_rows;

    // number of basis functions (1 in the case of the OG bcf)
    size_t p_z = Z.n_cols;

    // number of continuous variables
    size_t p_continuous = p - p_categorical;

    size_t p_continuous_ps = p_ps - p_categorical_ps;
    size_t p_continuous_trt = p_trt - p_categorical_trt;

    // suppose first p_continuous variables are continuous, then categorical

    COUT << "mtry " << mtry << " " << p << endl;

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

    Rcpp::NumericMatrix X_std(N, p);
    Rcpp::NumericMatrix X_std_ps(N, p_ps);
    Rcpp::NumericMatrix X_std_trt(N, p_trt);

    Rcpp::NumericMatrix Xtest_std(N_test, p);
    Rcpp::NumericMatrix Xtest_std_ps(N_test, p_ps);
    Rcpp::NumericMatrix Xtest_std_trt(N_test, p_trt);

    matrix<double> Z_std;
    ini_matrix(Z_std, N, p_z);
    matrix<double> Ztest_std;
    ini_matrix(Ztest_std, N_test, p_z);

    rcpp_to_std2(y, Z, X, X_ps, X_trt, Ztest, Xtest, Xtest_ps, Xtest_trt, y_std, y_mean, Z_std, X_std, X_std_ps, X_std_trt, Ztest_std, Xtest_std, Xtest_std_ps, Xtest_std_trt, Xorder_std, Xorder_std_ps, Xorder_std_trt);

    ///////////////////////////////////////////////////////////////////

    double *Xpointer = &X_std[0];
    double *Xpointer_ps = &X_std_ps[0];
    double *Xpointer_trt = &X_std_trt[0];
    double *Xtestpointer = &Xtest_std[0];
    double *Xtestpointer_ps = &Xtest_std_ps[0];
    double *Xtestpointer_trt = &Xtest_std_trt[0];

    // matrix<double> yhats_xinfo;
    // ini_matrix(yhats_xinfo, N, num_sweeps);

    matrix<double> yhats_test_xinfo;
    ini_matrix(yhats_test_xinfo, N_test, num_sweeps);

    matrix<double> prognostic_xinfo;
    ini_matrix(prognostic_xinfo, N_test, num_sweeps);

    matrix<double> treatment_xinfo;
    ini_matrix(treatment_xinfo, N_test, num_sweeps);

    matrix<double> sigma_draw_xinfo;
    ini_matrix(sigma_draw_xinfo, num_trees, num_sweeps);

    // // Create trees
    vector<vector<tree>> *trees_ps = new vector<vector<tree>>(num_sweeps);
    for (size_t i = 0; i < num_sweeps; i++)
    {
        (*trees_ps)[i] = vector<tree>(num_trees);
    }

    vector<vector<tree>> *trees_trt = new vector<vector<tree>>(num_sweeps);
    for (size_t i = 0; i < num_sweeps; i++)
    {
        (*trees_trt)[i] = vector<tree>(num_trees);
    }

    // define model
    XBCFContinuousModel *model = new XBCFContinuousModel(kap, s, tau, alpha, beta, sampling_tau, tau_kap, tau_s);
    model->setNoSplitPenality(no_split_penality);

    // State settings
    std::unique_ptr<State> state(new NormalLinearState(&Z_std, Xpointer, Xpointer_ps, Xpointer_trt, Xorder_std, Xorder_std_ps, Xorder_std_trt, N, p, p_ps, p_trt, num_trees, num_trees_ps, num_trees_trt, p_categorical, p_categorical_ps, p_categorical_trt, p_continuous, p_continuous_ps, p_continuous_trt, set_random_seed, random_seed, n_min, num_cutpoints, mtry, mtry_ps, mtry_trt, Xpointer, num_sweeps, sample_weights, &y_std, 1.0, max_depth, y_mean, burnin, model->dim_residual, nthread, parallel)); // last input is nthread, need update

    // initialize X_struct
    std::vector<double> initial_theta_ps(1, 0);
    std::unique_ptr<X_struct> x_struct_ps(new X_struct(Xpointer_ps, &y_std, N, Xorder_std_ps, p_categorical_ps, p_continuous_ps, &initial_theta_ps, num_trees_ps));

    std::vector<double> initial_theta_trt(1, y_mean / (double)num_trees);
    std::unique_ptr<X_struct> x_struct_trt(new X_struct(Xpointer_trt, &y_std, N, Xorder_std_trt, p_categorical_trt, p_continuous_trt, &initial_theta_trt, num_trees_trt));

    ////////////////////////////////////////////////////////////////
    mcmc_loop_linear(Xorder_std_ps, Xorder_std_trt, verbose, sigma_draw_xinfo, *trees_ps, *trees_trt, no_split_penality, state, model, x_struct_ps, x_struct_trt);

    model->predict_std(Ztest_std, Xtestpointer, N_test, p, num_trees, num_sweeps, yhats_test_xinfo, prognostic_xinfo, treatment_xinfo, *trees_ps, *trees_trt);

    // R Objects to Return
    Rcpp::NumericMatrix yhats_test(N_test, num_sweeps);
    Rcpp::NumericMatrix prognostic(N_test, num_sweeps);
    Rcpp::NumericMatrix treatment(N_test, num_sweeps);

    Rcpp::NumericMatrix sigma_draw(num_trees, num_sweeps); // save predictions of each tree
    Rcpp::NumericVector split_count_sum(p, 0);             // split counts
    Rcpp::XPtr<std::vector<std::vector<tree>>> tree_pnt(trees_ps, true);

    // copy from std vector to Rcpp Numeric Matrix objects
    Matrix_to_NumericMatrix(yhats_test_xinfo, yhats_test);
    Matrix_to_NumericMatrix(prognostic_xinfo, prognostic);
    Matrix_to_NumericMatrix(treatment_xinfo, treatment);

    Matrix_to_NumericMatrix(sigma_draw_xinfo, sigma_draw);

    for (size_t i = 0; i < p; i++)
    {
        split_count_sum(i) = (int)state->split_count_all[i];
    }

    // clean memory
    // delete model;
    state.reset();
    x_struct_ps.reset();
    x_struct_trt.reset();

    // print out tree structure, for usage of BART warm-start

    Rcpp::StringVector output_tree_ps(num_sweeps);
    Rcpp::StringVector output_tree_trt(num_sweeps);

    tree_to_string(*trees_trt, output_tree_trt, num_sweeps, num_trees, p);
    tree_to_string(*trees_ps, output_tree_ps, num_sweeps, num_trees, p);

    thread_pool.stop();

    return Rcpp::List::create(
        // Rcpp::Named("yhats") = yhats,
        Rcpp::Named("yhats_test") = yhats_test,
        Rcpp::Named("mu") = prognostic,
        Rcpp::Named("tau") = treatment,
        Rcpp::Named("sigma") = sigma_draw,
        Rcpp::Named("importance") = split_count_sum,
        Rcpp::Named("model_list") = Rcpp::List::create(Rcpp::Named("tree_pnt") = tree_pnt, Rcpp::Named("y_mean") = y_mean, Rcpp::Named("p") = p),
        Rcpp::Named("treedraws") = Rcpp::List::create(Rcpp::Named("treatment") = output_tree_trt, Rcpp::Named("Prognostic") = output_tree_ps));
}
