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

// TODO: determine the appropriate set of input variables
// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]]
Rcpp::List XBART_heterosk_cpp(arma::mat y,
                              arma::mat X,
                              arma::mat Xtest,
                              size_t num_trees,
                              size_t num_sweeps,
                              size_t max_depth,
                              size_t n_min,
                              size_t num_cutpoints,
                              double alpha, double beta,
                              double tau,
                              double no_split_penality,
                              size_t burnin = 1,
                              size_t mtry = 0,
                              size_t p_categorical = 0,
                              double kap = 16, double s = 4,
                              double tau_kap = 3, double tau_s = 0.5,
                              double a_v = 2.0, double b_v = 2.0, // shape and rate
                              bool verbose = false,
                              bool sampling_tau = true,
                              bool parallel = true,
                              bool set_random_seed = false,
                              size_t random_seed = 0,
                              bool sample_weights_flag = true,
                              double nthread = 0
                              )
{
    COUT << "In source." << endl;

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

    matrix<double> yhats_test_xinfo;
    ini_matrix(yhats_test_xinfo, N, num_sweeps);

    matrix<double> mhats_test_xinfo;
    ini_matrix(mhats_test_xinfo, N_test, num_sweeps);

    matrix<double> vhats_test_xinfo;
    ini_matrix(vhats_test_xinfo, N_test, num_sweeps);

    // TODO: check and remove -- this is likely an unnecessary storage
    matrix<double> sigma_draw_xinfo;
    ini_matrix(sigma_draw_xinfo, num_trees, num_sweeps);

    // Create trees for the mean model
    vector<vector<tree>> *trees2_m = new vector<vector<tree>>(num_sweeps);
    for (size_t i = 0; i < num_sweeps; i++)
    {
        (*trees2_m)[i] = vector<tree>(num_trees);
    }

    // Create trees for the variance model
    vector<vector<tree>> *trees2_v = new vector<vector<tree>>(num_sweeps);
    for (size_t i = 0; i < num_sweeps; i++)
    {
        (*trees2_v)[i] = vector<tree>(num_trees);
    }

    COUT << "Objects init." << endl;
    // define the mean model
    hskNormalModel *model_m = new hskNormalModel(kap, s, tau, alpha, beta, sampling_tau, tau_kap, tau_s);
    // cout << "after define model " << model->tau << " " << model->tau_mean << endl;
    model_m->setNoSplitPenality(no_split_penality);

    // define the variance model
    logNormalModel *model_v = new logNormalModel(kap, s, tau, alpha, beta, sampling_tau, tau_kap, tau_s);
    // cout << "after define model " << model->tau << " " << model->tau_mean << endl;
    model_v->setNoSplitPenality(no_split_penality);

    // State settings for the mean model
    std::vector<double> sigma_vec(N, 0); // initialize vector of heterogeneous sigmas
    std::unique_ptr<State> state_m(new hskState(Xpointer, Xorder_std, N, p, num_trees, p_categorical, p_continuous, set_random_seed, random_seed, n_min, num_cutpoints, mtry, Xpointer, num_sweeps, sample_weights_flag, &y_std, 1.0, max_depth, y_mean, burnin, model_m->dim_residual, nthread, parallel, sigma_vec)); //last input is nthread, need update

    COUT << y_mean << "< varyhat | state >" << state_m->ini_var_yhat << endl;
    // State settings for the variance model
    std::unique_ptr<State> state_v(new NormalState(Xpointer, Xorder_std, N, p, num_trees, p_categorical, p_continuous, set_random_seed, random_seed, n_min, num_cutpoints, mtry, Xpointer, num_sweeps, sample_weights_flag, &y_std, 1.0, max_depth, y_mean, burnin, model_v->dim_residual, nthread, parallel)); //last input is nthread, need update


    // initialize X_struct
    // TODO: check if one object is enough for two models
    std::vector<double> initial_theta_m(1, y_mean / (double)num_trees);
    std::unique_ptr<X_struct> x_struct_m(new X_struct(Xpointer, &y_std, N, Xorder_std, p_categorical, p_continuous, &initial_theta_m, num_trees));

    std::vector<double> initial_theta_v(1, 1.0);
    std::unique_ptr<X_struct> x_struct_v(new X_struct(Xpointer, &y_std, N, Xorder_std, p_categorical, p_continuous, &initial_theta_v, num_trees));

    COUT << "Running the model." << endl;
    ////////////////////////////////////////////////////////////////
    // mcmc loop
    // TODO: determine the appropriate set of input variables
   mcmc_loop_hsk(Xorder_std,
                    verbose,
                    sigma_draw_xinfo,
                    *trees2_m,
                    state_m,
                    model_m,
                    x_struct_m,
                    *trees2_v,
                    state_v,
                    model_v,
                    x_struct_v);
/*
    COUT << "Predict." << endl;
    // TODO: check how predict function will be different
    model_m->predict_std(Xtestpointer, N_test, p, num_trees, num_sweeps, mhats_test_xinfo, *trees2_m);
    model_v->predict_std(Xtestpointer, N_test, p, num_trees, num_sweeps, vhats_test_xinfo, *trees2_v);
*/
    COUT << "Re-format objects for output." << endl;

    for(size_t i = 0; i < N_test; i++) {
        for(size_t j = 0; j < p; j++) {
            yhats_test_xinfo[j][i] = 0; //mhats_test_xinfo[j][i] + mhats_test_xinfo[j][i];
        }
    }
    // R Objects to Return
    // Rcpp::NumericMatrix yhats(N, num_sweeps);
    Rcpp::NumericMatrix yhats_test(N_test, num_sweeps);
    Rcpp::NumericMatrix sigma_draw(num_trees, num_sweeps); // save predictions of each tree
    Rcpp::NumericVector split_count_sum(p, 0);             // split counts
    Rcpp::XPtr<std::vector<std::vector<tree>>> tree_pnt(trees2_m, true);

    // copy from std vector to Rcpp Numeric Matrix objects
    Matrix_to_NumericMatrix(yhats_test_xinfo, yhats_test);
    Matrix_to_NumericMatrix(sigma_draw_xinfo, sigma_draw);

/* NK: do we need to output this? per component?
    for (size_t i = 0; i < p; i++)
    {
        split_count_sum(i) = (int)state->split_count_all[i];
    } */

    // TODO: make sure all objects are removed from memory
    state_m.reset();
    state_v.reset();
    x_struct_m.reset();
    x_struct_v.reset();

    // print out tree structure, for usage of BART warm-start

    std::stringstream treess;

    Rcpp::StringVector output_tree(num_sweeps);

    // TODO: chek if we need this for storing trees (warmstart use?)
    for (size_t i = 0; i < num_sweeps; i++)
    {
        treess.precision(10);

        treess.str(std::string());
        treess << num_trees << " " << p << endl;

        for (size_t t = 0; t < num_trees; t++)
        {
            treess << (*trees2_m)[i][t];
        }

        output_tree(i) = treess.str();
    }

    // TODO: check outputs
    return Rcpp::List::create(
        // Rcpp::Named("yhats") = yhats,
        Rcpp::Named("yhats_test") = yhats_test,
        Rcpp::Named("sigma") = sigma_draw,
        Rcpp::Named("importance") = split_count_sum,
        Rcpp::Named("model_list") = Rcpp::List::create(Rcpp::Named("tree_pnt") = tree_pnt, Rcpp::Named("y_mean") = y_mean, Rcpp::Named("p") = p),
        Rcpp::Named("treedraws") = output_tree);
}