#include <ctime>
#include <RcppArmadillo.h>
#include "tree.h"
#include "forest.h"
#include <chrono>
#include "mcmc_loop.h"
#include "X_struct.h"
#include "omp.h"

using namespace std;
using namespace chrono;

// [[Rcpp::plugins(openmp)]]

////////////////////////////////////////////////////////////////////////
//                                                                    //
//                                                                    //
//  Full function, support both continuous and categorical variables  //
//                                                                    //
//                                                                    //
////////////////////////////////////////////////////////////////////////

void rcpp_to_std2(arma::mat y, arma::mat X, arma::mat Xtest, std::vector<double> &y_std, double &y_mean, Rcpp::NumericMatrix &X_std, Rcpp::NumericMatrix &Xtest_std, matrix<size_t> &Xorder_std)
{
    // The goal of this function is to convert RCPP object to std objects

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
#pragma omp parallel for schedule(dynamic, 1) shared(X, Xorder)
    for (size_t i = 0; i < X.n_cols; i++)
    {
        Xorder.col(i) = arma::sort_index(X.col(i));
    }
// Create
#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < p; j++)
        {
            Xorder_std[j][i] = Xorder(i, j);
        }
    }

    return;
}

void rcpp_to_std2(arma::mat X, arma::mat Xtest, Rcpp::NumericMatrix &X_std, Rcpp::NumericMatrix &Xtest_std, matrix<size_t> &Xorder_std)
{
    // The goal of this function is to convert RCPP object to std objects

    // TODO: Refactor code so for loops are self contained functions
    // TODO: Why RCPP and not std?
    // TODO: inefficient Need Replacement?

    size_t N = X.n_rows;
    size_t p = X.n_cols;
    size_t N_test = Xtest.n_rows;

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

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]]
Rcpp::List XBART_cpp(arma::mat y, arma::mat X, arma::mat Xtest, size_t num_trees, size_t num_sweeps, size_t max_depth, size_t n_min, size_t num_cutpoints, double alpha, double beta, double tau, double no_split_penality, size_t burnin = 1, size_t mtry = 0, size_t p_categorical = 0, double kap = 16, double s = 4, double tau_kap = 3, double tau_s = 0.5, bool verbose = false, bool sampling_tau = true, bool parallel = true, bool set_random_seed = false, size_t random_seed = 0, bool sample_weights_flag = true, double nthread = 0)
{

    // auto start = system_clock::now();

    // double nthread = 1;

    if (parallel & (nthread == 0))
    {
        nthread = omp_get_max_threads();
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
    std::unique_ptr<State> state(new NormalState(Xpointer, Xorder_std, N, p, num_trees, p_categorical, p_continuous, set_random_seed, random_seed, n_min, num_cutpoints, mtry, Xpointer, num_sweeps, sample_weights_flag, &y_std, 1.0, max_depth, y_mean, burnin, model->dim_residual, nthread)); //last input is nthread, need update

    // state->set_Xcut(Xcutmat);

    // initialize X_struct
    std::unique_ptr<X_struct> x_struct(new X_struct(Xpointer, &y_std, N, Xorder_std, p_categorical, p_continuous, &initial_theta, num_trees));

    ////////////////////////////////////////////////////////////////
    mcmc_loop(Xorder_std, verbose, sigma_draw_xinfo, *trees2, no_split_penality, state, model, x_struct);

    model->predict_std(Xtestpointer, N_test, p, num_trees, num_sweeps, yhats_test_xinfo, *trees2);

    // R Objects to Return
    // Rcpp::NumericMatrix yhats(N, num_sweeps);
    Rcpp::NumericMatrix yhats_test(N_test, num_sweeps);
    Rcpp::NumericMatrix sigma_draw(num_trees, num_sweeps); // save predictions of each tree
    Rcpp::NumericVector split_count_sum(p, 0);             // split counts
    Rcpp::XPtr<std::vector<std::vector<tree>>> tree_pnt(trees2, true);

    // TODO: Make these functions
    // for (size_t i = 0; i < N; i++)
    // {
    //     for (size_t j = 0; j < num_sweeps; j++)
    //     {
    //         yhats(i, j) = yhats_xinfo[j][i];
    //     }
    // }
    for (size_t i = 0; i < N_test; i++)
    {
        for (size_t j = 0; j < num_sweeps; j++)
        {
            yhats_test(i, j) = yhats_test_xinfo[j][i];
        }
    }
    for (size_t i = 0; i < num_trees; i++)
    {
        for (size_t j = 0; j < num_sweeps; j++)
        {
            sigma_draw(i, j) = sigma_draw_xinfo[j][i];
        }
    }
    for (size_t i = 0; i < p; i++)
    {
        split_count_sum(i) = (int)state->split_count_all[i];
    }

    // auto end = system_clock::now();

    // auto duration = duration_cast<microseconds>(end - start);

    // COUT << "Total running time " << double(duration.count()) * microseconds::period::num / microseconds::period::den << endl;

    // COUT << "Running time of split Xorder " << run_time << endl;

    // COUT << "Count of splits for each variable " << mtry_weight_current_tree << endl;

    // clean memory
    // delete model;
    state.reset();
    x_struct.reset();

    // print out tree structure, for usage of BART package

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

    return Rcpp::List::create(
        // Rcpp::Named("yhats") = yhats,
        Rcpp::Named("yhats_test") = yhats_test,
        Rcpp::Named("sigma") = sigma_draw,
        Rcpp::Named("importance") = split_count_sum,
        Rcpp::Named("model_list") = Rcpp::List::create(Rcpp::Named("tree_pnt") = tree_pnt, Rcpp::Named("y_mean") = y_mean, Rcpp::Named("p") = p),
        Rcpp::Named("treedraws") = output_tree);
}

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]]
Rcpp::List XBART_CLT_cpp(arma::mat y, arma::mat X, arma::mat Xtest, size_t num_trees, size_t num_sweeps, size_t max_depth, size_t n_min, size_t num_cutpoints, double alpha, double beta, double tau, double no_split_penality, size_t burnin = 1, size_t mtry = 0, size_t p_categorical = 0, double kap = 16, double s = 4, bool verbose = false, bool parallel = true, bool set_random_seed = false, size_t random_seed = 0, bool sample_weights_flag = true, double nthread = 0)
{

    // auto start = system_clock::now();

    // double nthread = 1;

    if (parallel & (nthread == 0))
    {
        nthread = omp_get_max_threads();
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
    CLTClass *model = new CLTClass(kap, s, tau, alpha, beta);
    model->setNoSplitPenality(no_split_penality);

    // State settings
    std::vector<double> initial_theta(1, y_mean / (double)num_trees);
    std::unique_ptr<State> state(new State(Xpointer, Xorder_std, N, p, num_trees, p_categorical, p_continuous, set_random_seed, random_seed, n_min, num_cutpoints, mtry, Xpointer, num_sweeps, sample_weights_flag, &y_std, 1.0, max_depth, y_mean, burnin, model->dim_residual, nthread)); //last input is nthread, need update

    // initialize X_struct
    std::unique_ptr<X_struct> x_struct(new X_struct(Xpointer, &y_std, N, Xorder_std, p_categorical, p_continuous, &initial_theta, num_trees));

    /////////////////////////////////////////////////////////////////

    mcmc_loop_clt(Xorder_std, verbose, sigma_draw_xinfo, *trees2, no_split_penality, state, model, x_struct);

    model->predict_std(Xtestpointer, N_test, p, num_trees, num_sweeps, yhats_test_xinfo, *trees2);

    // R Objects to Return
    // Rcpp::NumericMatrix yhats(N, num_sweeps);
    Rcpp::NumericMatrix yhats_test(N_test, num_sweeps);
    Rcpp::NumericMatrix sigma_draw(num_trees, num_sweeps); // save predictions of each tree
    Rcpp::NumericVector split_count_sum(p);                // split counts
    Rcpp::XPtr<std::vector<std::vector<tree>>> tree_pnt(trees2, true);

    // TODO: Make these functions
    // for (size_t i = 0; i < N; i++)
    // {
    //     for (size_t j = 0; j < num_sweeps; j++)
    //     {
    //         yhats(i, j) = yhats_xinfo[j][i];
    //     }
    // }
    for (size_t i = 0; i < N_test; i++)
    {
        for (size_t j = 0; j < num_sweeps; j++)
        {
            yhats_test(i, j) = yhats_test_xinfo[j][i];
        }
    }
    for (size_t i = 0; i < num_trees; i++)
    {
        for (size_t j = 0; j < num_sweeps; j++)
        {
            sigma_draw(i, j) = sigma_draw_xinfo[j][i];
        }
    }

    for (size_t i = 0; i < p; i++)
    {
        split_count_sum(i) = (int)state->split_count_all[i];
    }

    // auto end = system_clock::now();

    // auto duration = duration_cast<microseconds>(end - start);

    // COUT << "Total running time " << double(duration.count()) * microseconds::period::num / microseconds::period::den << endl;

    // COUT << "Running time of split Xorder " << run_time << endl;

    // COUT << "Count of splits for each variable " << mtry_weight_current_tree << endl;

    // return Rcpp::List::create(Rcpp::Named("yhats") = yhats, Rcpp::Named("yhats_test") = yhats_test, Rcpp::Named("sigma") = sigma_draw, Rcpp::Named("trees") = Rcpp::CharacterVector(treess.str()));
    return Rcpp::List::create(
        // Rcpp::Named("yhats") = yhats,
        Rcpp::Named("yhats_test") = yhats_test,
        Rcpp::Named("sigma") = sigma_draw,
        Rcpp::Named("importance") = split_count_sum,
        Rcpp::Named("model_list") = Rcpp::List::create(Rcpp::Named("tree_pnt") = tree_pnt, Rcpp::Named("y_mean") = y_mean, Rcpp::Named("p") = p, Rcpp::Named("num_sweeps") = num_sweeps, Rcpp::Named("num_trees") = num_trees));
}

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]]
Rcpp::List XBART_multinomial_cpp(Rcpp::IntegerVector y, int num_class, arma::mat X, arma::mat Xtest, size_t num_trees, size_t num_sweeps, size_t max_depth, 
size_t n_min, size_t num_cutpoints, double alpha, double beta, double tau_a, double tau_b, double no_split_penality, 
size_t burnin = 1, size_t mtry = 0, size_t p_categorical = 0, bool verbose = false, bool parallel = true, bool set_random_seed = false, size_t random_seed = 0, 
bool sample_weights_flag = true, bool separate_tree = false, double weight = 1, double hmult = 1, double heps = 0, bool update_tau = false, bool update_weight = true, double nthread = 0)
{
    // temporary
    double stop_threshold = 0;
    // auto start = system_clock::now();

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

    if (parallel & (nthread == 0)) nthread = omp_get_max_threads();
    omp_set_num_threads(nthread);

    arma::umat Xorder(X.n_rows, X.n_cols);
    matrix<size_t> Xorder_std;
    ini_matrix(Xorder_std, N, p);

    std::vector<size_t> y_size_t(N);
    for (size_t i = 0; i < N; ++i)
        y_size_t[i] = y[i];

    //TODO: check if I need to carry this // Yes, for now we need it.
    std::vector<double> y_std(N);
    double y_mean = 0.0;
    for (size_t i = 0; i < N; ++i)
        y_std[i] = y[i];

    Rcpp::NumericMatrix X_std(N, p);
    Rcpp::NumericMatrix Xtest_std(N_test, p);

    //dumb little hack to make this work, should write a new one of these
    rcpp_to_std2(X, Xtest, X_std, Xtest_std, Xorder_std);

    ///////////////////////////////////////////////////////////////////

    // double *ypointer = &y_std[0];
    double *Xpointer = &X_std[0];
    double *Xtestpointer = &Xtest_std[0];

    // matrix<double> yhats_xinfo;
    // ini_matrix(yhats_xinfo, N, num_sweeps);

    matrix<double> yhats_test_xinfo;
    ini_matrix(yhats_test_xinfo, N_test, num_sweeps);

    // // Create trees
    // vector<vector<tree>> *trees2 = new vector<vector<tree>>(num_sweeps);
    // for (size_t i = 0; i < num_sweeps; i++)
    // {
    //     (*trees2)[i] = vector<tree>(num_trees);
    // }

    // State settings
    // Logit doesn't need an inherited state class at the moment
    // (see comments in the public declarations of LogitModel)
    // but we should consider moving phi and y_size_t to a LogitState
    // (y_size_t definitely belongs there, phi probably does)

    std::vector<double> initial_theta(num_class, 1);
    std::unique_ptr<State> state(new LogitState(Xpointer, Xorder_std, N, p, num_trees, p_categorical, p_continuous, set_random_seed, random_seed, n_min, num_cutpoints, mtry, Xpointer, num_sweeps, sample_weights_flag, &y_std, 1.0, max_depth, y_mean, burnin, num_class, nthread));

    // initialize X_struct
    std::unique_ptr<X_struct> x_struct(new X_struct(Xpointer, &y_std, N, Xorder_std, p_categorical, p_continuous, &initial_theta, num_trees));

    std::vector<std::vector<double>> tau_sample;
    ini_matrix(tau_sample, num_trees, num_sweeps);

    std::vector<std::vector<double>> weight_sample;
    ini_matrix(weight_sample, num_trees, num_sweeps);

    std::vector<std::vector<double>> logloss;
    ini_matrix(logloss, num_trees, num_sweeps);

    ////////////////////////////////////////////////////////////////
    size_t num_stops = 0;

    // output is in 3 dim, stacked as a vector, number of sweeps * observations * number of classes
    std::vector<double> output_vec(num_sweeps * N_test * num_class);

    ////////////////////////////////////////////////
    // for a n * p * m matrix, the (i,j,k) element is
    // i + j * n + k * n * p in the stacked vector
    // if stack by column, index starts from 0
    ////////////////////////////////////////////////

    vector<vector<tree>> *trees2 = new vector<vector<tree>>(num_sweeps);
    // separate tree
    vector<vector<vector<tree>>> *trees3 = new vector<vector<vector<tree>>>(num_class);

    if (!separate_tree)
    {
        for (size_t i = 0; i < num_sweeps; i++)
        {
            (*trees2)[i] = vector<tree>(num_trees);
        }

    std::vector<double> phi(N);
    for(size_t i=0; i<N; ++i) phi[i] = 1;

    if (!separate_tree)
    {
        LogitModel *model = new LogitModel(num_class, tau_a, tau_b, alpha, beta, &y_size_t, &phi, weight, update_tau, update_weight, hmult, heps);
        model->setNoSplitPenality(no_split_penality);

        mcmc_loop_multinomial(Xorder_std, verbose, *trees2, no_split_penality, state, model, x_struct, tau_sample, weight_sample, logloss, stop_threshold, num_stops);

        model->predict_std(Xtestpointer, N_test, p, num_trees, num_sweeps, yhats_test_xinfo, *trees2, output_vec);

        // delete model;
    }
    else
    {
        cout << "grow separate trees" << endl;
        if (stop_threshold > 0){
            cout << "early stopping is disabled for separate trees" << endl;
        }
        LogitModelSeparateTrees *model = new LogitModelSeparateTrees(num_class, tau_a, tau_b, alpha, beta, &y_size_t, &phi, weight, update_tau, update_weight, hmult, heps);

        model->setNoSplitPenality(no_split_penality);

        mcmc_loop_multinomial_sample_per_tree(Xorder_std, verbose, *trees3, no_split_penality, state, model, x_struct, tau_sample, weight_sample, stop_threshold, num_stops);

        model->predict_std(Xtestpointer, N_test, p, num_trees, num_sweeps, yhats_test_xinfo, *trees3, output_vec);

        // delete model;
    }

    // mcmc_loop_multinomial(Xorder_std, verbose, *trees2, no_split_penality, state, model, x_struct, phi_samples, tau_sample, stop_threshold, num_stops);

    // model->predict_std(Xtestpointer, N_test, p, num_trees, num_sweeps, yhats_test_xinfo, *trees2, output_vec);

    Rcpp::NumericVector output = Rcpp::wrap(output_vec);
    output.attr("dim") = Rcpp::Dimension(num_sweeps, N_test, num_class);

    // STOPPED HERE
    // TODO: Figure out how we should store and return in sample preds
    // probably add step at the end of mcmc loop to retrieve leaf pars, aggregate and
    // normalize

    // R Objects to Return
    // Rcpp::NumericMatrix yhats(N, num_sweeps);
    Rcpp::NumericMatrix yhats_test(N_test, num_sweeps);
    Rcpp::NumericVector split_count_sum(p); // split counts
    // Rcpp::XPtr<std::vector<std::vector<tree>>> tree_pnt(trees2, true);
    Rcpp::NumericMatrix weight_sample_rcpp(num_trees, num_sweeps);
    Rcpp::NumericMatrix depth_rcpp(num_trees, num_sweeps);

    for (size_t i = 0; i < num_trees; i++)
    {
        for (size_t j = 0; j < num_sweeps; j++)
        {
            weight_sample_rcpp(i, j) = weight_sample[j][i];
        }
    }
    for (size_t i = 0; i < num_trees; i++)
    {
        for (size_t j = 0; j < num_sweeps; j++)
        {
            // npv bv = (*trees2)[j][i].bots()

            depth_rcpp(i, j) = (*trees2)[j][i].getdepth();
            // cout << "depth = "<< (*trees3)[j][i].getdepth() << endl;
        }
    }
    for (size_t i = 0; i < N_test; i++)
    {
        for (size_t j = 0; j < num_sweeps; j++)
        {
            yhats_test(i, j) = yhats_test_xinfo[j][i];
        }
    }
    for (size_t i = 0; i < p; i++)
    {
        split_count_sum(i) = (int)state->split_count_all[i];
    }


    std::stringstream treess;

    // if separate trees, return length num_class object, each contains num_sweeps * num_trees trees
    // if shared trees, return length 1 object, num_sweeps * num_trees trees
    Rcpp::StringVector output_tree(0);

    if(! separate_tree)
    {
        // shared trees
        // the output is a length num_sweeps vector, each string is a sweep
        // for each sweep, put first tree of all K classes first (duplicated), then the second tree
        // still num_class * num_trees in each string, for convenience of BART initialization
        for(size_t i = 0; i < num_sweeps; i++)
        {
            treess.precision(10);
            treess.str(std::string());
            treess << (double) separate_tree << " " << num_class << " " << num_sweeps << " " << num_trees << " " << p << endl;
            for(size_t j = 0; j < num_trees; j ++)
            {
                for(size_t kk = 0; kk < num_class; kk ++ )
                {
                    treess << (*trees2)[i][j];
                }
            }
            output_tree.push_back(treess.str());    
        }

    }else{
        // separate trees
        // the output is a length num_sweeps vector, each string is a sweep
        // for each sweep, put first tree of all K classes first, then the second tree, etc
        for(size_t i = 0; i < num_sweeps; i++)
        {
            treess.precision(10);
            treess.str(std::string());
            treess << (double) separate_tree << " " << num_class << " " << num_sweeps << " " << num_trees << " " << p << endl;
            for(size_t j = 0; j < num_trees; j ++)
            {
                for(size_t kk = 0; kk < num_class; kk ++ )
                {
                    treess << (*trees3)[kk][i][j];
                }
            }
            output_tree.push_back(treess.str());
        }
    }

    // clean memory
    // // delete model;
    state.reset();
    x_struct.reset();

    // cout << "creat output list " << endl;
    Rcpp::List ret = Rcpp::List::create(
        // Rcpp::Named("yhats") = yhats,
        Rcpp::Named("num_class") = num_class,
        Rcpp::Named("yhats_test") = output,
        Rcpp::Named("weight") = weight_sample_rcpp,
        Rcpp::Named("importance") = split_count_sum,
        Rcpp::Named("depth") = depth_rcpp,
        Rcpp::Named("model_list") = Rcpp::List::create(Rcpp::Named("y_mean") = y_mean, Rcpp::Named("p") = p, Rcpp::Named("num_class") = num_class, 
        Rcpp::Named("num_sweeps") = num_sweeps, Rcpp::Named("num_trees") = num_trees));

    // cout << "export tree pointer " << endl;
    if (!separate_tree)
    {
        Rcpp::XPtr<std::vector<std::vector<tree>>> tree_pnt(trees2, true);
        ret.push_back(tree_pnt, "tree_pnt");
    }
    else
    {
        Rcpp::XPtr<std::vector<std::vector<std::vector<tree>>>> tree_pnt(trees3, true);
        ret.push_back(tree_pnt, "tree_pnt");
    }

    return ret;
}

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]]
Rcpp::List XBART_Probit_cpp(arma::mat y, arma::mat X, arma::mat Xtest, size_t num_trees, size_t num_sweeps, size_t max_depth, size_t n_min, size_t num_cutpoints, double alpha, double beta, double tau, double no_split_penality, size_t burnin = 1, size_t mtry = 0, size_t p_categorical = 0, double kap = 16, double s = 4, bool verbose = false, bool parallel = true, bool set_random_seed = false, size_t random_seed = 0, bool sample_weights_flag = true, double nthread = 0)
{

    // auto start = system_clock::now();

    // double nthread = 1.0;

    if (parallel & (nthread == 0))
    {
        nthread = omp_get_max_threads();
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

    matrix<double> yhats_std;
    ini_matrix(yhats_std, N, num_sweeps);
    matrix<double> yhats_test_std;
    ini_matrix(yhats_test_std, N_test, num_sweeps);

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
    ProbitClass *model = new ProbitClass(kap, s, tau, alpha, beta, y_std);
    model->setNoSplitPenality(no_split_penality);

    // State settings
    std::vector<double> initial_theta(1, y_mean / (double)num_trees);
    std::unique_ptr<State> state(new NormalState(Xpointer, Xorder_std, N, p, num_trees, p_categorical, p_continuous, set_random_seed, random_seed, n_min, num_cutpoints, mtry, Xpointer, num_sweeps, sample_weights_flag, &y_std, 1.0, max_depth, y_mean, burnin, model->dim_residual, nthread)); //last input is nthread, need update

    // initialize X_struct
    std::unique_ptr<X_struct> x_struct(new X_struct(Xpointer, &y_std, N, Xorder_std, p_categorical, p_continuous, &initial_theta, num_trees));

    /////////////////////////////////////////////////////////////////

    mcmc_loop_probit(Xorder_std, verbose, sigma_draw_xinfo, *trees2, no_split_penality, state, model, x_struct);

    model->predict_std(Xtestpointer, N_test, p, num_trees, num_sweeps, yhats_test_xinfo, *trees2);

    // R Objects to Return
    // Rcpp::NumericMatrix yhats(N, num_sweeps);
    Rcpp::NumericMatrix yhats_test(N_test, num_sweeps);
    Rcpp::NumericMatrix sigma_draw(num_trees, num_sweeps); // save predictions of each tree
    Rcpp::NumericVector split_count_sum(p);                // split counts
    Rcpp::XPtr<std::vector<std::vector<tree>>> tree_pnt(trees2, true);

    // TODO: Make these functions
    // for (size_t i = 0; i < N; i++)
    // {
    //     for (size_t j = 0; j < num_sweeps; j++)
    //     {
    //         yhats(i, j) = yhats_xinfo[j][i];
    //     }
    // }
    for (size_t i = 0; i < N_test; i++)
    {
        for (size_t j = 0; j < num_sweeps; j++)
        {
            yhats_test(i, j) = yhats_test_xinfo[j][i];
        }
    }
    for (size_t i = 0; i < num_trees; i++)
    {
        for (size_t j = 0; j < num_sweeps; j++)
        {
            sigma_draw(i, j) = sigma_draw_xinfo[j][i];
        }
    }
    for (size_t i = 0; i < p; i++)
    {
        split_count_sum(i) = (int)state->split_count_all[i];
    }

    // auto end = system_clock::now();

    // auto duration = duration_cast<microseconds>(end - start);

    // COUT << "Total running time " << double(duration.count()) * microseconds::period::num / microseconds::period::den << endl;

    // COUT << "Running time of split Xorder " << run_time << endl;

    // COUT << "Count of splits for each variable " << mtry_weight_current_tree << endl;

    // return Rcpp::List::create(Rcpp::Named("yhats") = yhats, Rcpp::Named("yhats_test") = yhats_test, Rcpp::Named("sigma") = sigma_draw, Rcpp::Named("trees") = Rcpp::CharacterVector(treess.str()));

    // clean memory
    // delete model;
    state.reset();
    x_struct.reset();

    return Rcpp::List::create(
        // Rcpp::Named("yhats") = yhats,
        Rcpp::Named("yhats_test") = yhats_test,
        Rcpp::Named("sigma") = sigma_draw,
        Rcpp::Named("importance") = split_count_sum,
        Rcpp::Named("model_list") = Rcpp::List::create(Rcpp::Named("tree_pnt") = tree_pnt,
                                                       Rcpp::Named("y_mean") = y_mean,
                                                       Rcpp::Named("p") = p));
}

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]]
Rcpp::List XBART_MH_cpp(arma::mat y, arma::mat X, arma::mat Xtest, size_t num_trees, size_t num_sweeps, size_t max_depth, size_t n_min, size_t num_cutpoints, double alpha, double beta, double tau, double no_split_penality, size_t burnin = 1, size_t mtry = 0, size_t p_categorical = 0, double kap = 16, double s = 4, bool verbose = false, bool parallel = true, bool set_random_seed = false, size_t random_seed = 0, bool sample_weights_flag = true, double nthread = 0)
{

    // auto start = system_clock::now();

    // double nthread = 1;

    if (parallel & (nthread == 0))
    {
        nthread = omp_get_max_threads();
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
    NormalModel *model = new NormalModel(kap, s, tau, alpha, beta);
    model->setNoSplitPenality(no_split_penality);

    // State settings
    std::vector<double> initial_theta(1, y_mean / (double)num_trees);
    std::unique_ptr<State> state(new NormalState(Xpointer, Xorder_std, N, p, num_trees, p_categorical, p_continuous, set_random_seed, random_seed, n_min, num_cutpoints, mtry, Xpointer, num_sweeps, sample_weights_flag, &y_std, 1.0, max_depth, y_mean, burnin, model->dim_residual, nthread)); //last input is nthread, need update

    // initialize X_struct
    std::unique_ptr<X_struct> x_struct(new X_struct(Xpointer, &y_std, N, Xorder_std, p_categorical, p_continuous, &initial_theta, num_trees));

    std::vector<double> accept_count;
    std::vector<double> MH_vector;
    std::vector<double> Q_ratio;
    std::vector<double> P_ratio;
    std::vector<double> prior_ratio;

    ////////////////////////////////////////////////////////////////
    //mcmc_loop_MH(Xorder_std, verbose, sigma_draw_xinfo, *trees2, no_split_penality, state, model, x_struct, accept_count, MH_vector, P_ratio, Q_ratio, prior_ratio);

    model->predict_std(Xtestpointer, N_test, p, num_trees, num_sweeps, yhats_test_xinfo, *trees2);

    // R Objects to Return
    // Rcpp::NumericMatrix yhats(N, num_sweeps);
    Rcpp::NumericMatrix yhats_test(N_test, num_sweeps);
    Rcpp::NumericMatrix sigma_draw(num_trees, num_sweeps); // save predictions of each tree
    Rcpp::NumericVector split_count_sum(p);                // split counts
    Rcpp::XPtr<std::vector<std::vector<tree>>> tree_pnt(trees2, true);

    // TODO: Make these functions
    // for (size_t i = 0; i < N; i++)
    // {
    //     for (size_t j = 0; j < num_sweeps; j++)
    //     {
    //         yhats(i, j) = yhats_xinfo[j][i];
    //     }
    // }
    for (size_t i = 0; i < N_test; i++)
    {
        for (size_t j = 0; j < num_sweeps; j++)
        {
            yhats_test(i, j) = yhats_test_xinfo[j][i];
        }
    }
    for (size_t i = 0; i < num_trees; i++)
    {
        for (size_t j = 0; j < num_sweeps; j++)
        {
            sigma_draw(i, j) = sigma_draw_xinfo[j][i];
        }
    }
    for (size_t i = 0; i < p; i++)
    {
        split_count_sum(i) = (int)state->split_count_all[i];
    }

    // auto end = system_clock::now();

    // auto duration = duration_cast<microseconds>(end - start);

    // COUT << "Total running time " << double(duration.count()) * microseconds::period::num / microseconds::period::den << endl;

    // COUT << "Running time of split Xorder " << run_time << endl;

    // COUT << "Count of splits for each variable " << mtry_weight_current_tree << endl;

    // clean memory
    // delete model;
    state.reset();
    x_struct.reset();

    return Rcpp::List::create(
        // Rcpp::Named("yhats") = yhats,
        Rcpp::Named("yhats_test") = yhats_test,
        Rcpp::Named("sigma") = sigma_draw,
        Rcpp::Named("importance") = split_count_sum,
        Rcpp::Named("model_list") = Rcpp::List::create(Rcpp::Named("tree_pnt") = tree_pnt, Rcpp::Named("y_mean") = y_mean, Rcpp::Named("p") = p));
}
