#include <ctime>
#include <RcppArmadillo.h>
#include "tree.h"
#include "forest.h"
#include <chrono>
#include "mcmc_loop.h"
#include "X_struct.h"
#include "xbcf_mcmc_loop.h"

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

// FUNCTION arma_to_std (instance 1)
// transfers data from an armadillo matrix object (column 0) to an std vector object
void arma_to_std(const arma::mat &matrix_in, std::vector<double> &vector_out)
{
    size_t dim = matrix_in.n_rows;

    for (size_t i = 0; i < dim; i++)
    {
        vector_out[i] = matrix_in(i, 0);
    }

    return;
}

// FUNCTION arma_to_rcpp (instance 1)                    ?? Rcpp matrix or std matrix ??
// transfers data from an armadillo matrix object to an Rcpp matrix object
void arma_to_rcpp(const arma::mat &matrix_in, Rcpp::NumericMatrix &matrix_out)
{
    size_t dim_x = matrix_in.n_rows;
    size_t dim_y = matrix_in.n_cols;

    for (size_t i = 0; i < dim_x; i++)
    {
        for (size_t j = 0; j < dim_y; j++)
        {
            matrix_out(i, j) = matrix_in(i, j);
        }
    }

    return;
}

// FUNCTION arma_to_std_ordered
// transfers data from an armadillo matrix object to an std matrix object with indeces [carries the pre-sorted features]
void arma_to_std_ordered(const arma::mat &matrix_in, matrix<size_t> &matrix_ordered_std)
{
    size_t dim_x = matrix_in.n_rows;
    size_t dim_y = matrix_in.n_cols;

    arma::umat matrix_ordered(dim_x, dim_y);
    for (size_t i = 0; i < dim_y; i++)
    {
        matrix_ordered.col(i) = arma::sort_index(matrix_in.col(i));
    }

    for (size_t i = 0; i < dim_x; i++)
    {
        for (size_t j = 0; j < dim_y; j++)
        {
            matrix_ordered_std[j][i] = matrix_ordered(i, j);
        }
    }

    return;
}

// FUNCTION std_to_Rcpp
// transfers data from an std matrix object to an Rcpp NumericMatrix object
void std_to_rcpp(const matrix<double> &matrix_in, Rcpp::NumericMatrix &matrix_out)
{
    size_t dim_x = matrix_in.size();
    size_t dim_y = matrix_in[0].size();
    for (size_t i = 0; i < dim_y; i++)
    {
        for (size_t j = 0; j < dim_x; j++)
        {
            matrix_out(i, j) = matrix_in[j][i];
        }
    }

    return;
}

double compute_mean(const std::vector<double> &vec)
{
    double mean = 0;
    int length = vec.size();

    for (size_t i = 0; i < length; i++)
    {
        mean = mean + vec[i];
    }
    mean = mean / (double)length;
    return mean;
}

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]]
Rcpp::List XBART(arma::mat y, arma::mat X, arma::mat Xtest, size_t num_trees, size_t num_sweeps, size_t max_depth, size_t n_min, size_t num_cutpoints, double alpha, double beta, double tau, double no_split_penality, size_t burnin = 1, size_t mtry = 0, size_t p_categorical = 0, double kap = 16, double s = 4, bool verbose = false, bool parallel = true, bool set_random_seed = false, size_t random_seed = 0, bool sample_weights_flag = true)
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

    matrix<double> yhats_xinfo;
    ini_matrix(yhats_xinfo, N, num_sweeps);

    matrix<double> yhats_test_xinfo;
    ini_matrix(yhats_test_xinfo, N, num_sweeps);

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
    mcmc_loop(Xorder_std, verbose, yhats_xinfo, sigma_draw_xinfo, *trees2, no_split_penality, state, model, x_struct);

    model->predict_std(Xtestpointer, N_test, p, num_trees, num_sweeps, yhats_test_xinfo, *trees2);

    // R Objects to Return
    Rcpp::NumericMatrix yhats(N, num_sweeps);
    Rcpp::NumericMatrix yhats_test(N_test, num_sweeps);
    Rcpp::NumericMatrix sigma_draw(num_trees, num_sweeps); // save predictions of each tree
    Rcpp::NumericVector split_count_sum(p);                // split counts
    Rcpp::XPtr<std::vector<std::vector<tree>>> tree_pnt(trees2, true);

    // TODO: Make these functions
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < num_sweeps; j++)
        {
            yhats(i, j) = yhats_xinfo[j][i];
        }
    }
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
        Rcpp::Named("yhats") = yhats,
        Rcpp::Named("yhats_test") = yhats_test,
        Rcpp::Named("sigma") = sigma_draw,
        Rcpp::Named("importance") = split_count_sum,
        Rcpp::Named("model_list") = Rcpp::List::create(Rcpp::Named("tree_pnt") = tree_pnt, Rcpp::Named("y_mean") = y_mean, Rcpp::Named("p") = p));
}

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]]
Rcpp::List XBART_CLT(arma::mat y, arma::mat X, arma::mat Xtest, size_t num_trees, size_t num_sweeps, size_t max_depth, size_t n_min, size_t num_cutpoints, double alpha, double beta, double tau, double no_split_penality, size_t burnin = 1, size_t mtry = 0, size_t p_categorical = 0, double kap = 16, double s = 4, bool verbose = false, bool parallel = true, bool set_random_seed = false, size_t random_seed = 0, bool sample_weights_flag = true)
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

    matrix<double> yhats_xinfo;
    ini_matrix(yhats_xinfo, N, num_sweeps);

    matrix<double> yhats_test_xinfo;
    ini_matrix(yhats_test_xinfo, N, num_sweeps);

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

    mcmc_loop_clt(Xorder_std, verbose, yhats_xinfo, sigma_draw_xinfo, *trees2, no_split_penality, state, model, x_struct);

    model->predict_std(Xtestpointer, N_test, p, num_trees, num_sweeps, yhats_test_xinfo, *trees2);

    // R Objects to Return
    Rcpp::NumericMatrix yhats(N, num_sweeps);
    Rcpp::NumericMatrix yhats_test(N_test, num_sweeps);
    Rcpp::NumericMatrix sigma_draw(num_trees, num_sweeps); // save predictions of each tree
    Rcpp::NumericVector split_count_sum(p);                // split counts
    Rcpp::XPtr<std::vector<std::vector<tree>>> tree_pnt(trees2, true);

    // TODO: Make these functions
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < num_sweeps; j++)
        {
            yhats(i, j) = yhats_xinfo[j][i];
        }
    }
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
        Rcpp::Named("yhats") = yhats,
        Rcpp::Named("yhats_test") = yhats_test,
        Rcpp::Named("sigma") = sigma_draw,
        Rcpp::Named("importance") = split_count_sum,
        Rcpp::Named("model_list") = Rcpp::List::create(Rcpp::Named("tree_pnt") = tree_pnt, Rcpp::Named("y_mean") = y_mean, Rcpp::Named("p") = p));
}

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]]
Rcpp::List XBART_multinomial(arma::mat y, arma::mat X, arma::mat Xtest, size_t num_trees, size_t num_sweeps, size_t max_depth, size_t n_min, size_t num_cutpoints, double alpha, double beta, double tau, double no_split_penality, size_t burnin = 1, size_t mtry = 0, size_t p_categorical = 0, double kap = 16, double s = 4, bool verbose = false, bool parallel = true, bool set_random_seed = false, size_t random_seed = 0, bool sample_weights_flag = true)
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

    matrix<double> yhats_xinfo;
    ini_matrix(yhats_xinfo, N, num_sweeps);

    matrix<double> yhats_test_xinfo;
    ini_matrix(yhats_test_xinfo, N, num_sweeps);

    matrix<double> sigma_draw_xinfo;
    ini_matrix(sigma_draw_xinfo, num_trees, num_sweeps);

    // // Create trees
    vector<vector<tree>> *trees2 = new vector<vector<tree>>(num_sweeps);
    for (size_t i = 0; i < num_sweeps; i++)
    {
        (*trees2)[i] = vector<tree>(num_trees);
    }

    /////////////////////////////////////////////////////////////////
    //
    //
    //      Need to define n_class
    //
    //
    /////////////////////////////////////////////////////////////////

    size_t n_class;

    // define model
    LogitClass *model = new LogitClass();
    model->setNoSplitPenality(no_split_penality);

    // State settings
    std::vector<double> initial_theta(1, 0);
    std::unique_ptr<State> state(new State(Xpointer, Xorder_std, N, p, num_trees, p_categorical, p_continuous, set_random_seed, random_seed, n_min, num_cutpoints, parallel, mtry, Xpointer, num_sweeps, sample_weights_flag, &y_std, 1.0, max_depth, y_mean, burnin, model->dim_residual));

    // initialize X_struct
    std::unique_ptr<X_struct> x_struct(new X_struct(Xpointer, &y_std, N, Xorder_std, p_categorical, p_continuous, &initial_theta, num_trees));

    /////////////////////////////////////////////////////////////////
    mcmc_loop_multinomial(Xorder_std, verbose, yhats_xinfo, sigma_draw_xinfo, *trees2, no_split_penality, state, model, x_struct);

    // predict_std_multinomial(Xtestpointer, N_test, p, num_trees, num_sweeps, yhats_test_xinfo, *trees2);

    // R Objects to Return
    Rcpp::NumericMatrix yhats(N, num_sweeps);
    Rcpp::NumericMatrix yhats_test(N_test, num_sweeps);
    Rcpp::NumericMatrix sigma_draw(num_trees, num_sweeps); // save predictions of each tree
    Rcpp::NumericVector split_count_sum(p);                // split counts
    Rcpp::XPtr<std::vector<std::vector<tree>>> tree_pnt(trees2, true);

    // TODO: Make these functions
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < num_sweeps; j++)
        {
            yhats(i, j) = yhats_xinfo[j][i];
        }
    }
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
        Rcpp::Named("yhats") = yhats,
        Rcpp::Named("yhats_test") = yhats_test,
        Rcpp::Named("sigma") = sigma_draw,
        Rcpp::Named("importance") = split_count_sum,
        Rcpp::Named("model_list") = Rcpp::List::create(Rcpp::Named("tree_pnt") = tree_pnt,
                                                       Rcpp::Named("y_mean") = y_mean,
                                                       Rcpp::Named("p") = p));
}

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]]
Rcpp::List XBART_Probit(arma::mat y, arma::mat X, arma::mat Xtest, size_t num_trees, size_t num_sweeps, size_t max_depth, size_t n_min, size_t num_cutpoints, double alpha, double beta, double tau, double no_split_penality, size_t burnin = 1, size_t mtry = 0, size_t p_categorical = 0, double kap = 16, double s = 4, bool verbose = false, bool parallel = true, bool set_random_seed = false, size_t random_seed = 0, bool sample_weights_flag = true)
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

    matrix<double> yhats_xinfo;
    ini_matrix(yhats_xinfo, N, num_sweeps);

    matrix<double> yhats_test_xinfo;
    ini_matrix(yhats_test_xinfo, N, num_sweeps);

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
    std::vector<double> initial_theta(1, 0);
    std::unique_ptr<State> state(new State(Xpointer, Xorder_std, N, p, num_trees, p_categorical, p_continuous, set_random_seed, random_seed, n_min, num_cutpoints, parallel, mtry, Xpointer, num_sweeps, sample_weights_flag, &y_std, 1.0, max_depth, y_mean, burnin, model->dim_residual));

    // initialize X_struct
    std::unique_ptr<X_struct> x_struct(new X_struct(Xpointer, &y_std, N, Xorder_std, p_categorical, p_continuous, &initial_theta, num_trees));

    /////////////////////////////////////////////////////////////////

    mcmc_loop_probit(Xorder_std, verbose, yhats_xinfo, sigma_draw_xinfo, *trees2, no_split_penality, state, model, x_struct);

    model->predict_std(Xtestpointer, N_test, p, num_trees, num_sweeps, yhats_test_xinfo, *trees2);

    // R Objects to Return
    Rcpp::NumericMatrix yhats(N, num_sweeps);
    Rcpp::NumericMatrix yhats_test(N_test, num_sweeps);
    Rcpp::NumericMatrix sigma_draw(num_trees, num_sweeps); // save predictions of each tree
    Rcpp::NumericVector split_count_sum(p);                // split counts
    Rcpp::XPtr<std::vector<std::vector<tree>>> tree_pnt(trees2, true);

    // TODO: Make these functions
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < num_sweeps; j++)
        {
            yhats(i, j) = yhats_xinfo[j][i];
        }
    }
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
        Rcpp::Named("yhats") = yhats,
        Rcpp::Named("yhats_test") = yhats_test,
        Rcpp::Named("sigma") = sigma_draw,
        Rcpp::Named("importance") = split_count_sum,
        Rcpp::Named("model_list") = Rcpp::List::create(Rcpp::Named("tree_pnt") = tree_pnt,
                                                       Rcpp::Named("y_mean") = y_mean,
                                                       Rcpp::Named("p") = p));
}

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]]
Rcpp::List XBART_MH(arma::mat y, arma::mat X, arma::mat Xtest, size_t num_trees, size_t num_sweeps, size_t max_depth, size_t n_min, size_t num_cutpoints, double alpha, double beta, double tau, double no_split_penality, size_t burnin = 1, size_t mtry = 0, size_t p_categorical = 0, double kap = 16, double s = 4, bool verbose = false, bool parallel = true, bool set_random_seed = false, size_t random_seed = 0, bool sample_weights_flag = true)
{

    cout << "MHMHMH" << endl;

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

    matrix<double> yhats_xinfo;
    ini_matrix(yhats_xinfo, N, num_sweeps);

    matrix<double> yhats_test_xinfo;
    ini_matrix(yhats_test_xinfo, N, num_sweeps);

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

    // State settings
    std::vector<double> initial_theta(1, 0);
    std::unique_ptr<State> state(new State(Xpointer, Xorder_std, N, p, num_trees, p_categorical, p_continuous, set_random_seed, random_seed, n_min, num_cutpoints, parallel, mtry, Xpointer, num_sweeps, sample_weights_flag, &y_std, 1.0, max_depth, y_mean, burnin, model->dim_residual));

    // initialize X_struct
    std::unique_ptr<X_struct> x_struct(new X_struct(Xpointer, &y_std, N, Xorder_std, p_categorical, p_continuous, &initial_theta, num_trees));

    /////////////////////////////////////////////////////////////////
    std::vector<double> accept_count;
    std::vector<double> MH_vector;
    std::vector<double> Q_ratio;
    std::vector<double> P_ratio;
    std::vector<double> prior_ratio;

    mcmc_loop_MH(Xorder_std, verbose, yhats_xinfo, sigma_draw_xinfo, *trees2, no_split_penality, state, model, x_struct, accept_count, MH_vector, P_ratio, Q_ratio, prior_ratio);

    model->predict_std(Xtestpointer, N_test, p, num_trees, num_sweeps, yhats_test_xinfo, *trees2);

    // R Objects to Return
    Rcpp::NumericMatrix yhats(N, num_sweeps);
    Rcpp::NumericMatrix yhats_test(N_test, num_sweeps);
    Rcpp::NumericMatrix sigma_draw(num_trees, num_sweeps); // save predictions of each tree
    Rcpp::NumericVector split_count_sum(p);                // split counts
    Rcpp::XPtr<std::vector<std::vector<tree>>> tree_pnt(trees2, true);

    // TODO: Make these functions
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < num_sweeps; j++)
        {
            yhats(i, j) = yhats_xinfo[j][i];
        }
    }
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
        Rcpp::Named("yhats") = yhats,
        Rcpp::Named("yhats_test") = yhats_test,
        Rcpp::Named("sigma") = sigma_draw,
        Rcpp::Named("importance") = split_count_sum,
        Rcpp::Named("accept_count") = accept_count,
        Rcpp::Named("MH") = MH_vector,
        Rcpp::Named("Q_ratio") = Q_ratio,
        Rcpp::Named("P_ratio") = P_ratio,
        Rcpp::Named("prior_ratio") = prior_ratio,
        Rcpp::Named("model_list") = Rcpp::List::create(Rcpp::Named("tree_pnt") = tree_pnt,
                                                       Rcpp::Named("y_mean") = y_mean,
                                                       Rcpp::Named("p") = p));
}

// FUNCTION XBCF
// preprocesses input received from R
// feeds data into main loop function 'mcmc_loop_xbcf'
// returns the list of objects to later become the output in R
// general attributes: y (vector of responses), X (matrix of covariates), z (vector of treatment assignments)
//                     num_sweeps, burnin (# of burn-in sweeps),
//                     max_depth (of a tree), n_min (minimum node size),
//                     num_cutpoints (# of adaptive cutpoints considered at each split for cont variables),
//                     no_split_penalty, mtry (# of variables considered at each split),
//                     p_categorical (# of categorical regressors)
// per forest:         alpha, beta, tau, (BART prior parameters)
//                     num_trees,
// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]]
Rcpp::List XBCF(arma::mat y, arma::mat X, arma::mat z,                              // responses vec y, covariates mat x, treatment assignment vec z
                size_t num_sweeps, size_t burnin = 1,                               // burnin is the # of burn-in sweeps
                size_t max_depth = 1, size_t n_min = 5,                             // n_min is the minimum node size
                size_t num_cutpoints = 1,                                           // # of adaptive cutpoints considered at each split for cont variables
                double no_split_penality = 0.001, size_t mtry = 0,                  // mtry is the # of variables considered at each split
                size_t p_categorical = 0,                                           // # of categorical regressors
                size_t num_trees_pr = 200,                                          // --- Prognostic term parameters start here
                double alpha_pr = 0.95, double beta_pr = 2, double tau_pr = 0.5,    // BART prior parameters
                double kap_pr = 16, double s_pr = 4,                                // prior parameters of sigma
                bool pr_scale = false,                                              // use half-Cauchy prior
                size_t num_trees_trt = 50,                                          // --- Treatment term parameters start here
                double alpha_trt = 0.25, double beta_trt = 3, double tau_trt = 0.5, // BART priot parameters
                double kap_trt = 16, double s_trt = 4,                              // prior parameters of sigma
                bool trt_scale = false,                                             // use half-Normal prior
                bool verbose = false, bool parallel = true,                         // optional parameters
                bool set_random_seed = false, size_t random_seed = 0,
                bool sample_weights_flag = true)
{

    auto start = system_clock::now();

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
    std::vector<double> z_std(N);
    std::vector<double> b(N);
    double y_mean = 0.0;

    Rcpp::NumericMatrix X_std(N, p);
    // Rcpp::NumericMatrix Xtest_std(N_test, p);

    //    rcpp_to_std2(y, X, Xtest, y_std, y_mean, X_std, Xtest_std, Xorder_std);
    arma_to_std(y, y_std);
    arma_to_std(z, z_std);
    arma_to_rcpp(X, X_std);
    arma_to_std_ordered(X, Xorder_std);
    y_mean = compute_mean(y_std);
    ///////////////////////////////////////////////////////////////////
    std::vector<double> sigma_vec; // vector of sigmas

    double bscale0 = -1;
    double bscale1 = 1;

    size_t n_trt = 0; // number of treated individuals

    // assuming we have presorted data (treated individuals first, then control group)

    for (size_t i = 0; i < N; i++)
    {
        b[i] = z[i] * bscale1 + (1 - z[i]) * bscale0;
        sigma_vec[i] = 1.0;
        if (z[i] == 1)
        {
            n_trt++;
        }
    }

    // double *ypointer = &y_std[0];

    double *Xpointer = &X_std[0];
    // double *Xtestpointer = &Xtest_std[0];

    matrix<double> tauhats_xinfo;
    ini_matrix(tauhats_xinfo, N, num_sweeps);
    // yhats_xinfo.matrix<double> yhats_test_xinfo;
    // ini_matrix(yhats_test_xinfo, N, num_sweeps);

    matrix<double> sigma_draw_xinfo_pr;
    ini_matrix(sigma_draw_xinfo_pr, num_trees_pr, num_sweeps);

    matrix<double> sigma_draw_xinfo_trt;
    ini_matrix(sigma_draw_xinfo_trt, num_trees_trt, num_sweeps);
    // // Create trees
    vector<vector<tree>> *trees_pr = new vector<vector<tree>>(num_sweeps);
    for (size_t i = 0; i < num_sweeps; i++)
    {
        (*trees_pr)[i] = vector<tree>(num_trees_pr);
    }

    // // Create trees
    vector<vector<tree>> *trees_trt = new vector<vector<tree>>(num_sweeps);
    for (size_t i = 0; i < num_sweeps; i++)
    {
        (*trees_trt)[i] = vector<tree>(num_trees_trt);
    }
    // define the model for the prognostic term
    NormalModel *model_pr = new NormalModel(kap_pr, s_pr, tau_pr, alpha_pr, beta_pr);
    model_pr->setNoSplitPenality(no_split_penality);

    // define the model for the treatment term
    xbcfModel *model_trt = new xbcfModel(kap_trt, s_trt, tau_trt, alpha_trt, beta_trt);
    model_trt->setNoSplitPenality(no_split_penality);

    // State settings for the prognostic term
    std::vector<double> initial_theta_pr(1, y_mean / (double)num_trees_pr);
    std::unique_ptr<State> state_pr(new NormalState(Xpointer, Xorder_std, N, p, num_trees_pr, p_categorical, p_continuous, set_random_seed, random_seed, n_min, num_cutpoints, parallel, mtry, Xpointer, num_sweeps, sample_weights_flag, &y_std, 1.0, max_depth, y_mean, burnin, model_pr->dim_residual));

    // State settings for the treatment term
    std::vector<double> initial_theta_trt(1, y_mean / (double)num_trees_trt);
    std::unique_ptr<State> state_trt(new xbcfState(Xpointer, Xorder_std, N, n_trt, p, num_trees_trt, p_categorical, p_continuous, set_random_seed, random_seed, n_min, num_cutpoints, parallel, mtry, Xpointer, num_sweeps, sample_weights_flag, &y_std, &b, sigma_vec, max_depth, y_mean, burnin, model_trt->dim_residual));

    // initialize X_struct for the prognostic term
    std::unique_ptr<X_struct> x_struct_pr(new X_struct(Xpointer, &y_std, N, Xorder_std, p_categorical, p_continuous, &initial_theta_pr, num_trees_pr));

    // initialize X_struct for the treatment term
    std::unique_ptr<X_struct> x_struct_trt(new X_struct(Xpointer, &y_std, N, Xorder_std, p_categorical, p_continuous, &initial_theta_trt, num_trees_trt));

    // mcmc_loop returns tauhat [N x sweeps] matrix
    mcmc_loop_xbcf(Xorder_std, verbose, tauhats_xinfo, sigma_draw_xinfo_pr, sigma_draw_xinfo_trt, *trees_pr, *trees_trt, no_split_penality,
                   state_pr, state_trt, model_pr, model_trt, x_struct_pr, x_struct_trt);

    // R Objects to Return
    Rcpp::NumericMatrix tauhats(N, num_sweeps);

    std_to_rcpp(tauhats_xinfo, tauhats);
    // TODO: Make these functions
    /*     for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < num_sweeps; j++)
        {
            yhats(i, j) = yhats_xinfo[j][i];
        }
    }
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
*/
    auto end = system_clock::now();

    auto duration = duration_cast<microseconds>(end - start);

    // COUT << "Total running time " << double(duration.count()) * microseconds::period::num / microseconds::period::den << endl;

    // COUT << "Running time of split Xorder " << run_time << endl;

    // COUT << "Count of splits for each variable " << mtry_weight_current_tree << endl;

    // clean memory
    delete model_pr;
    delete model_trt;
    state_pr.reset();
    state_trt.reset();
    x_struct_pr.reset();
    x_struct_trt.reset();

    return Rcpp::List::create(
        Rcpp::Named("tauhats") = tauhats);
}