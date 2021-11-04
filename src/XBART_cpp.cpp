#include <ctime>
#include <RcppArmadillo.h>
#include "tree.h"
#include "forest.h"
#include <chrono>
#include "mcmc_loop.h"
#include "X_struct.h"
#include "omp.h"
#include "utility_rcpp.h"

using namespace std;
using namespace chrono;

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]]
Rcpp::List XBART_cpp(arma::mat y, arma::mat X, arma::mat Xtest, size_t num_trees, size_t num_sweeps, size_t max_depth, size_t n_min, size_t num_cutpoints, double alpha, double beta, double tau, double no_split_penality, size_t burnin = 1, size_t mtry = 0, size_t p_categorical = 0, double kap = 16, double s = 4, double tau_kap = 3, double tau_s = 0.5, bool verbose = false, bool sampling_tau = true, bool parallel = true, bool set_random_seed = false, size_t random_seed = 0, bool sample_weights_flag = true, double nthread = 0)
{
    if (parallel && (nthread == 0))
    {
        // if turn on parallel and do not sepicifiy number of threads
        // use max - 1, leave one out
        nthread = omp_get_max_threads() - 1;
    }

    if (parallel)
    {
        omp_set_num_threads(nthread);
        cout << "Running in parallel with " << nthread << " threads." << endl;
    }
    else
    {
        cout << "Running with single thread." << endl;
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

    // matrix<double> yhats_xinfo;
    // ini_matrix(yhats_xinfo, N, num_sweeps);

    matrix<double> yhats_test_xinfo;
    ini_matrix(yhats_test_xinfo, N_test, num_sweeps);

    matrix<double> sigma_draw_xinfo;
    ini_matrix(sigma_draw_xinfo, num_trees, num_sweeps);

    // // Create trees
    vector<vector<tree>> *trees2 = new vector<vector<tree>>(num_sweeps);
    for (size_t i = 0; i < num_sweeps; i++)
    {
        (*trees2)[i] = vector<tree>(num_trees);
    }

    // define model
    NormalModel *model = new NormalModel(kap, s, tau, alpha, beta, sampling_tau, tau_kap, tau_s);
    // cout << "after define model " << model->tau << " " << model->tau_mean << endl;
    model->setNoSplitPenality(no_split_penality);

    // State settings
    std::vector<double> initial_theta(1, y_mean / (double)num_trees);
    std::unique_ptr<State> state(new NormalState(Xpointer, Xorder_std, N, p, num_trees, p_categorical, p_continuous, set_random_seed, random_seed, n_min, num_cutpoints, mtry, Xpointer, num_sweeps, sample_weights_flag, &y_std, 1.0, max_depth, y_mean, burnin, model->dim_residual, nthread, parallel)); //last input is nthread, need update

    // state->set_Xcut(Xcutmat);

    // initialize X_struct
    std::unique_ptr<X_struct> x_struct(new X_struct(Xpointer, &y_std, N, Xorder_std, p_categorical, p_continuous, &initial_theta, num_trees));

    ////////////////////////////////////////////////////////////////
    std::vector<double> resid(N * num_sweeps * num_trees);

    mcmc_loop(Xorder_std, verbose, sigma_draw_xinfo, *trees2, no_split_penality, state, model, x_struct, resid);

    model->predict_std(Xtestpointer, N_test, p, num_trees, num_sweeps, yhats_test_xinfo, *trees2);

    // R Objects to Return
    // Rcpp::NumericMatrix yhats(N, num_sweeps);
    Rcpp::NumericMatrix yhats_test(N_test, num_sweeps);
    Rcpp::NumericMatrix sigma_draw(num_trees, num_sweeps); // save predictions of each tree
    Rcpp::NumericVector split_count_sum(p, 0);             // split counts
    Rcpp::XPtr<std::vector<std::vector<tree>>> tree_pnt(trees2, true);

    // copy from std vector to Rcpp Numeric Matrix objects
    Matrix_to_NumericMatrix(yhats_test_xinfo, yhats_test);
    Matrix_to_NumericMatrix(sigma_draw_xinfo, sigma_draw);

    for (size_t i = 0; i < p; i++)
    {
        split_count_sum(i) = (int)state->split_count_all[i];
    }

    // clean memory
    // delete model;
    state.reset();
    x_struct.reset();

    // print out tree structure, for usage of BART warm-start

    std::stringstream treess;

    Rcpp::StringVector output_tree(num_sweeps);

    for (size_t i = 0; i < num_sweeps; i++)
    {
        treess.precision(10);

        treess.str(std::string());
        treess << num_trees << " " << p << endl;

        for (size_t t = 0; t < num_trees; t++)
        {
            treess << (*trees2)[i][t];
        }

        output_tree(i) = treess.str();
    }

    Rcpp::NumericVector resid_rcpp = Rcpp::wrap(resid);
    resid_rcpp.attr("dim") = Rcpp::Dimension(N, num_sweeps, num_trees);

    return Rcpp::List::create(
        // Rcpp::Named("yhats") = yhats,
        Rcpp::Named("yhats_test") = yhats_test,
        Rcpp::Named("sigma") = sigma_draw,
        Rcpp::Named("importance") = split_count_sum,
        Rcpp::Named("model_list") = Rcpp::List::create(Rcpp::Named("tree_pnt") = tree_pnt, Rcpp::Named("y_mean") = y_mean, Rcpp::Named("p") = p),
        Rcpp::Named("treedraws") = output_tree,
        Rcpp::Named("residuals") = resid_rcpp
        );
        
}