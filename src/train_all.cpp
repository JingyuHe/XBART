#include <ctime>
#include <RcppArmadillo.h>
#include "tree.h"
#include "forest.h"
#include <chrono>
#include "mcmc_loop.h"
#include "X_struct.h"

using namespace std;
using namespace chrono;

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
Rcpp::List XBART_cpp(arma::mat y, arma::mat X, arma::mat Xtest, size_t num_trees, size_t num_sweeps, size_t max_depth, size_t n_min, size_t num_cutpoints, double alpha, double beta, double tau, double no_split_penality, size_t burnin = 1, size_t mtry = 0, size_t p_categorical = 0, double kap = 16, double s = 4, bool verbose = false, bool parallel = true, bool set_random_seed = false, size_t random_seed = 0, bool sample_weights_flag = true)
{

    auto start = system_clock::now();

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
    std::unique_ptr<State> state(new NormalState(Xpointer, Xorder_std, N, p, num_trees, p_categorical, p_continuous, set_random_seed, random_seed, n_min, num_cutpoints, parallel, mtry, Xpointer, num_sweeps, sample_weights_flag, &y_std, 1.0, max_depth, y_mean, burnin, model->dim_residual));

    // initialize X_struct
    std::unique_ptr<X_struct> x_struct(new X_struct(Xpointer, &y_std, N, Xorder_std, p_categorical, p_continuous, &initial_theta, num_trees));

    ////////////////////////////////////////////////////////////////
    mcmc_loop(Xorder_std, verbose, sigma_draw_xinfo, *trees2, no_split_penality, state, model, x_struct);

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
        split_count_sum(i) = (int)state->mtry_weight_current_tree[i];
    }

    auto end = system_clock::now();

    auto duration = duration_cast<microseconds>(end - start);

    // COUT << "Total running time " << double(duration.count()) * microseconds::period::num / microseconds::period::den << endl;

    // COUT << "Running time of split Xorder " << run_time << endl;

    // COUT << "Count of splits for each variable " << mtry_weight_current_tree << endl;

    // clean memory
    delete model;
    state.reset();
    x_struct.reset();

    return Rcpp::List::create(
        // Rcpp::Named("yhats") = yhats,
        Rcpp::Named("yhats_test") = yhats_test,
        Rcpp::Named("sigma") = sigma_draw,
        Rcpp::Named("importance") = split_count_sum,
        Rcpp::Named("model_list") = Rcpp::List::create(Rcpp::Named("tree_pnt") = tree_pnt, Rcpp::Named("y_mean") = y_mean, Rcpp::Named("p") = p));

}

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]]
Rcpp::List XBART_CLT_cpp(arma::mat y, arma::mat X, arma::mat Xtest, size_t num_trees, size_t num_sweeps, size_t max_depth, size_t n_min, size_t num_cutpoints, double alpha, double beta, double tau, double no_split_penality, size_t burnin = 1, size_t mtry = 0, size_t p_categorical = 0, double kap = 16, double s = 4, bool verbose = false, bool parallel = true, bool set_random_seed = false, size_t random_seed = 0, bool sample_weights_flag = true)
{

    auto start = system_clock::now();

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
    std::unique_ptr<State> state(new State(Xpointer, Xorder_std, N, p, num_trees, p_categorical, p_continuous, set_random_seed, random_seed, n_min, num_cutpoints, parallel, mtry, Xpointer, num_sweeps, sample_weights_flag, &y_std, 1.0, max_depth, y_mean, burnin, model->dim_residual));

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
        split_count_sum(i) = (int)state->mtry_weight_current_tree[i];
    }

    auto end = system_clock::now();

    auto duration = duration_cast<microseconds>(end - start);

    // COUT << "Total running time " << double(duration.count()) * microseconds::period::num / microseconds::period::den << endl;

    // COUT << "Running time of split Xorder " << run_time << endl;

    // COUT << "Count of splits for each variable " << mtry_weight_current_tree << endl;

    // return Rcpp::List::create(Rcpp::Named("yhats") = yhats, Rcpp::Named("yhats_test") = yhats_test, Rcpp::Named("sigma") = sigma_draw, Rcpp::Named("trees") = Rcpp::CharacterVector(treess.str()));
    return Rcpp::List::create(
        // Rcpp::Named("yhats") = yhats,
        Rcpp::Named("yhats_test") = yhats_test,
        Rcpp::Named("sigma") = sigma_draw,
        Rcpp::Named("importance") = split_count_sum,
        Rcpp::Named("model_list") = Rcpp::List::create(Rcpp::Named("tree_pnt") = tree_pnt, Rcpp::Named("y_mean") = y_mean, Rcpp::Named("p") = p));
}

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]]
Rcpp::List XBART_multinomial_cpp(Rcpp::IntegerVector y, int num_class, arma::mat X, arma::mat Xtest, size_t num_trees, size_t num_sweeps, size_t max_depth, size_t n_min, size_t num_cutpoints, double alpha, double beta, double tau, double no_split_penality, Rcpp::DoubleVector weight, size_t burnin = 1, size_t mtry = 0, size_t p_categorical = 0, double kap = 16, double s = 4, bool verbose = false, bool parallel = true, bool set_random_seed = false, size_t random_seed = 0, bool sample_weights_flag = true)
{

    auto start = system_clock::now();

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

    std::vector<size_t> y_size_t(N);
    for(size_t i=0; i<N; ++i) y_size_t[i] = y[i];
    
    
    //TODO: check if I need to carry this
    std::vector<double> y_std(N);
    double y_mean = 0.0;
    for(size_t i=0; i<N; ++i) y_std[i] = y[i];

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
    vector<vector<tree>> *trees2 = new vector<vector<tree>>(num_sweeps);
    for (size_t i = 0; i < num_sweeps; i++)
    {
        (*trees2)[i] = vector<tree>(num_trees);
    }

    // define model
    double tau_a = 1/tau + 0.5;
    double tau_b = 1/tau;
    std::vector<double> phi(N);
    for(size_t i=0; i<N; ++i) phi[i] = 1;

    std::vector<double> weight_std(weight.size());
    for(size_t i=0; i<weight.size(); ++i) weight_std[i] = weight[i];
    
    LogitModel *model = new LogitModel(num_class, tau_a, tau_b, alpha, beta, &y_size_t, &phi, weight_std);
    model->setNoSplitPenality(no_split_penality);


    // State settings
    // Logit doesn't need an inherited state class at the moment
    // (see comments in the public declarations of LogitModel)
    // but we should consider moving phi and y_size_t to a LogitState
    // (y_size_t definitely belongs there, phi probably does)

    std::vector<double> initial_theta(num_class, 1);
    std::unique_ptr<State> state(new State(Xpointer, Xorder_std, N, p, num_trees, p_categorical, p_continuous, set_random_seed, random_seed, n_min, num_cutpoints, parallel, mtry, Xpointer, num_sweeps, sample_weights_flag, &y_std, 1.0, max_depth, y_mean, burnin, model->dim_residual));
    
    // initialize X_struct
    std::unique_ptr<X_struct> x_struct(new X_struct(Xpointer, &y_std, N, Xorder_std, p_categorical, p_continuous, &initial_theta, num_trees));

    std::vector< std::vector<double> > phi_samples;
    ini_matrix(phi_samples, N, num_sweeps * num_trees);

    std::vector< std::vector<double> > weight_samples;
    ini_matrix(weight_samples, num_trees, num_sweeps);
    
    ////////////////////////////////////////////////////////////////
    mcmc_loop_multinomial(Xorder_std, verbose, *trees2, no_split_penality, state, model, x_struct, phi_samples, weight_samples);
    // mcmc_loop_multinomial(Xorder_std, verbose, *trees2, no_split_penality, state, model, x_struct, phi_samples);
    // mcmc_loop_multinomial_sample_vars_per_tree(Xorder_std, verbose, *trees2, no_split_penality, state, model, x_struct, phi_samples);

    // TODO: Implement predict OOS

    // output is in 3 dim, stacked as a vector, number of sweeps * observations * number of classes
    std::vector<double> output_vec(num_sweeps * N_test * num_class);

    ////////////////////////////////////////////////
    // for a n * p * m matrix, the (i,j,k) element is
    // i + j * n + k * n * p in the stacked vector
    // if stack by column, index starts from 0
    ////////////////////////////////////////////////

    model->predict_std(Xtestpointer, N_test, p, num_trees, num_sweeps, yhats_test_xinfo, *trees2, output_vec);

    Rcpp::NumericVector output = Rcpp::wrap(output_vec);
    output.attr("dim") = Rcpp::Dimension(num_sweeps, N_test, num_class);

    // STOPPED HERE
    // TODO: Figure out how we should store and return in sample preds
    // probably add step at the end of mcmc loop to retrieve leaf pars, aggregate and
    // normalize

    // R Objects to Return
    // Rcpp::NumericMatrix yhats(N, num_sweeps);
    Rcpp::NumericMatrix yhats_test(N_test, num_sweeps);
    Rcpp::NumericVector split_count_sum(p);                // split counts
    Rcpp::XPtr<std::vector<std::vector<tree>>> tree_pnt(trees2, true);
    Rcpp::NumericMatrix phi_sample_rcpp(N, num_sweeps * num_trees);
    Rcpp::NumericMatrix weight_sample_rcpp(num_trees, num_sweeps);

    // TODO: Make these functions
    // for (size_t i = 0; i < N; i++)
    // {
    //     for (size_t j = 0; j < num_sweeps; j++)
    //     {
    //         yhats(i, j) = yhats_xinfo[j][i];
    //     }
    // }

    for(size_t i = 0; i < N; i ++ ){
        for(size_t j = 0; j < num_trees * num_sweeps; j ++ ){
            phi_sample_rcpp(i, j) = phi_samples[j][i];
        }
    }
    for(size_t i = 0; i < num_trees; i ++ ){
        for(size_t j = 0; j < num_sweeps; j ++ ){
            weight_sample_rcpp(i, j) = weight_samples[j][i];
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
        split_count_sum(i) = (int)state->mtry_weight_current_tree[i];
    }

    auto end = system_clock::now();

    auto duration = duration_cast<microseconds>(end - start);

    // COUT << "Total running time " << double(duration.count()) * microseconds::period::num / microseconds::period::den << endl;

    // COUT << "Running time of split Xorder " << run_time << endl;

    // COUT << "Count of splits for each variable " << mtry_weight_current_tree << endl;

    // clean memory
    delete model;
    state.reset();
    x_struct.reset();

    return Rcpp::List::create(
        // Rcpp::Named("yhats") = yhats,
        Rcpp::Named("num_class") = num_class,
        Rcpp::Named("yhats_test") = output,
        Rcpp::Named("phi") = phi_sample_rcpp,
        Rcpp::Named("weight") = weight_sample_rcpp,
        Rcpp::Named("importance") = split_count_sum,
        Rcpp::Named("model_list") = Rcpp::List::create(Rcpp::Named("tree_pnt") = tree_pnt, Rcpp::Named("y_mean") = y_mean, Rcpp::Named("p") = p));

}

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]]
Rcpp::List XBART_Probit_cpp(arma::mat y, arma::mat X, arma::mat Xtest, size_t num_trees, size_t num_sweeps, size_t max_depth, size_t n_min, size_t num_cutpoints, double alpha, double beta, double tau, double no_split_penality, size_t burnin = 1, size_t mtry = 0, size_t p_categorical = 0, double kap = 16, double s = 4, bool verbose = false, bool parallel = true, bool set_random_seed = false, size_t random_seed = 0, bool sample_weights_flag = true)
{

    auto start = system_clock::now();

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
    std::unique_ptr<State> state(new NormalState(Xpointer, Xorder_std, N, p, num_trees, p_categorical, p_continuous, set_random_seed, random_seed, n_min, num_cutpoints, parallel, mtry, Xpointer, num_sweeps, sample_weights_flag, &y_std, 1.0, max_depth, y_mean, burnin, model->dim_residual));

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
        split_count_sum(i) = (int)state->mtry_weight_current_tree[i];
    }

    auto end = system_clock::now();

    auto duration = duration_cast<microseconds>(end - start);

    // COUT << "Total running time " << double(duration.count()) * microseconds::period::num / microseconds::period::den << endl;

    // COUT << "Running time of split Xorder " << run_time << endl;

    // COUT << "Count of splits for each variable " << mtry_weight_current_tree << endl;

    // return Rcpp::List::create(Rcpp::Named("yhats") = yhats, Rcpp::Named("yhats_test") = yhats_test, Rcpp::Named("sigma") = sigma_draw, Rcpp::Named("trees") = Rcpp::CharacterVector(treess.str()));


    // clean memory
    delete model;
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
Rcpp::List XBART_MH_cpp(arma::mat y, arma::mat X, arma::mat Xtest, size_t num_trees, size_t num_sweeps, size_t max_depth, size_t n_min, size_t num_cutpoints, double alpha, double beta, double tau, double no_split_penality, size_t burnin = 1, size_t mtry = 0, size_t p_categorical = 0, double kap = 16, double s = 4, bool verbose = false, bool parallel = true, bool set_random_seed = false, size_t random_seed = 0, bool sample_weights_flag = true)
{

    auto start = system_clock::now();

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
    std::unique_ptr<State> state(new NormalState(Xpointer, Xorder_std, N, p, num_trees, p_categorical, p_continuous, set_random_seed, random_seed, n_min, num_cutpoints, parallel, mtry, Xpointer, num_sweeps, sample_weights_flag, &y_std, 1.0, max_depth, y_mean, burnin, model->dim_residual));

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
        split_count_sum(i) = (int)state->mtry_weight_current_tree[i];
    }

    auto end = system_clock::now();

    auto duration = duration_cast<microseconds>(end - start);

    // COUT << "Total running time " << double(duration.count()) * microseconds::period::num / microseconds::period::den << endl;

    // COUT << "Running time of split Xorder " << run_time << endl;

    // COUT << "Count of splits for each variable " << mtry_weight_current_tree << endl;

    // clean memory
    delete model;
    state.reset();
    x_struct.reset();

    return Rcpp::List::create(
        // Rcpp::Named("yhats") = yhats,
        Rcpp::Named("yhats_test") = yhats_test,
        Rcpp::Named("sigma") = sigma_draw,
        Rcpp::Named("importance") = split_count_sum,
        Rcpp::Named("model_list") = Rcpp::List::create(Rcpp::Named("tree_pnt") = tree_pnt, Rcpp::Named("y_mean") = y_mean, Rcpp::Named("p") = p));
}
