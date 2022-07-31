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
using namespace arma;

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]]
Rcpp::List XBART_cpp(mat y, mat X, size_t num_trees, size_t num_sweeps, size_t max_depth, size_t n_min, size_t num_cutpoints, double alpha, double beta, double tau, double no_split_penality, size_t burnin = 1, size_t mtry = 0, size_t p_categorical = 0, double kap = 16, double s = 4, double tau_kap = 3, double tau_s = 0.5, bool verbose = false, bool sampling_tau = true, bool parallel = true, bool set_random_seed = false, size_t random_seed = 0, bool sample_weights = true, double nthread = 0)
{
    if (parallel)
    {
        thread_pool.start(nthread);
    }

    size_t N = X.n_rows;

    // number of total variables
    size_t p = X.n_cols;

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

    rcpp_to_std2(y, X, y_std, y_mean, X_std, Xorder_std);

    ///////////////////////////////////////////////////////////////////

    // double *ypointer = &y_std[0];
    double *Xpointer = &X_std[0];

    matrix<double> sigma_draw_xinfo;
    ini_matrix(sigma_draw_xinfo, num_trees, num_sweeps);

    // Create trees
    vector<vector<tree>> trees(num_sweeps);
    for (size_t i = 0; i < num_sweeps; i++)
    {
        trees[i].resize(num_trees);
    }

    // define model
    NormalModel *model = new NormalModel(kap, s, tau, alpha, beta, sampling_tau, tau_kap, tau_s);
    
    model->setNoSplitPenality(no_split_penality);

    // State settings
    std::vector<double> initial_theta(1, y_mean / (double)num_trees);
    NormalState state(Xpointer, Xorder_std, N, p, num_trees, p_categorical, p_continuous, set_random_seed, random_seed, n_min, num_cutpoints, mtry, Xpointer, num_sweeps, sample_weights, &y_std, 1.0, max_depth, y_mean, burnin, model->dim_residual, nthread, parallel);

    // initialize X_struct
    X_struct x_struct(Xpointer, &y_std, N, Xorder_std, p_categorical, p_continuous, &initial_theta, num_trees);

    ////////////////////////////////////////////////////////////////
    std::vector<double> resid(N * num_sweeps * num_trees);

    mcmc_loop(Xorder_std, verbose, sigma_draw_xinfo, trees, no_split_penality, state, model, x_struct, resid);

    // R Objects to Return
    Rcpp::NumericMatrix sigma_draw(num_trees, num_sweeps); // save predictions of each tree
    Rcpp::NumericVector split_count_sum(p, 0);             // split counts

    // copy from std vector to Rcpp Numeric Matrix objects
    Matrix_to_NumericMatrix(sigma_draw_xinfo, sigma_draw);

    for (size_t i = 0; i < p; i++)
    {
        split_count_sum(i) = (size_t)(*state.split_count_all)[i];
    }

    // print out tree structure, for usage of BART warm-start

    std::stringstream treess;

    Rcpp::StringVector output_tree(num_sweeps);
    tree_to_string(trees, output_tree, num_sweeps, num_trees, p);

    // return the matrix of residuals, useful for prediction by GP
    Rcpp::NumericVector resid_rcpp = Rcpp::wrap(resid);
    resid_rcpp.attr("dim") = Rcpp::Dimension(N, num_sweeps, num_trees);

    Rcpp::StringVector tree_json(1);
    json j = get_forest_json(trees, y_mean);
    tree_json[0] = j.dump(4);

    thread_pool.stop();

    return Rcpp::List::create(
        // Rcpp::Named("yhats") = yhats,
        Rcpp::Named("sigma") = sigma_draw,
        Rcpp::Named("importance") = split_count_sum,
        Rcpp::Named("model_list") = Rcpp::List::create(Rcpp::Named("y_mean") = y_mean, Rcpp::Named("p") = p),
        Rcpp::Named("treedraws") = output_tree,
        Rcpp::Named("residuals") = resid_rcpp,
        Rcpp::Named("tree_json") = tree_json);
}
