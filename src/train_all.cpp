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

// transfers data from an armadillo matrix object (column 0) to an std vector object
void arma_to_std(const arma::mat &matrix_in, std::vector<size_t> &vector_out)
{
    size_t dim = matrix_in.n_rows;

    for (size_t i = 0; i < dim; i++)
    {
        vector_out[i] = (size_t)matrix_in(i, 0);
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

    // print out tree structure, for usage of BART package

    std::stringstream treess;

    Rcpp::StringVector output_tree(num_sweeps);

    for(size_t i = 0; i < num_sweeps; i ++ ){
        treess.precision(10);

        treess.str(std::string());
        treess << num_trees << " " << p << endl;

        for (size_t t = 0; t < num_trees; t++)
        {
            treess << (*trees2)[i][t];
        }

        for (size_t t = 0; t < num_trees; t++)
        {
            cout << (*trees2)[i][t].treesize() << endl;
        }

        output_tree(i) = treess.str();
    }

    return Rcpp::List::create(
        // Rcpp::Named("yhats") = yhats,
        Rcpp::Named("yhats_test") = yhats_test,
        Rcpp::Named("sigma") = sigma_draw,
        Rcpp::Named("importance") = split_count_sum,
        Rcpp::Named("model_list") = Rcpp::List::create(Rcpp::Named("tree_pnt") = tree_pnt, Rcpp::Named("y_mean") = y_mean, Rcpp::Named("p") = p),
        Rcpp::Named("treedraws") = output_tree
        );
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
Rcpp::List XBART_multinomial_cpp(Rcpp::IntegerVector y, int num_class, arma::mat X, arma::mat Xtest, size_t num_trees, size_t num_sweeps, size_t max_depth, size_t n_min, size_t num_cutpoints, double alpha, double beta, double tau, double no_split_penality, size_t burnin = 1, size_t mtry = 0, size_t p_categorical = 0, double kap = 16, double s = 4, bool verbose = false, bool parallel = true, bool set_random_seed = false, size_t random_seed = 0, bool sample_weights_flag = true)
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
    for (size_t i = 0; i < N; ++i)
        y_size_t[i] = y[i];

    //TODO: check if I need to carry this
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
    vector<vector<tree>> *trees2 = new vector<vector<tree>>(num_sweeps);
    for (size_t i = 0; i < num_sweeps; i++)
    {
        (*trees2)[i] = vector<tree>(num_trees);
    }

    // define model
    double tau_a = 1 / tau + 0.5;
    double tau_b = 1 / tau;
    std::vector<double> phi(N);
    for (size_t i = 0; i < N; ++i)
        phi[i] = 1;

    LogitModel *model = new LogitModel(num_class, tau_a, tau_b, alpha, beta, &y_size_t, &phi);
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

    std::vector<std::vector<double>> phi_samples;
    ini_matrix(phi_samples, N, num_sweeps * num_trees);

    ////////////////////////////////////////////////////////////////
    mcmc_loop_multinomial(Xorder_std, verbose, *trees2, no_split_penality, state, model, x_struct, phi_samples);

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
    Rcpp::NumericVector split_count_sum(p); // split counts
    Rcpp::XPtr<std::vector<std::vector<tree>>> tree_pnt(trees2, true);
    Rcpp::NumericMatrix phi_sample_rcpp(N, num_sweeps * num_trees);

    // TODO: Make these functions
    // for (size_t i = 0; i < N; i++)
    // {
    //     for (size_t j = 0; j < num_sweeps; j++)
    //     {
    //         yhats(i, j) = yhats_xinfo[j][i];
    //     }
    // }

    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < num_trees * num_sweeps; j++)
        {
            phi_sample_rcpp(i, j) = phi_samples[j][i];
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
Rcpp::List XBCF_cpp(arma::mat y, arma::mat X, arma::mat X_tau, arma::mat z,             // responses vec y, covariates mat x, treatment assignment vec z
                size_t num_sweeps, size_t burnin = 1,                               // burnin is the # of burn-in sweeps
                size_t max_depth = 1, size_t n_min = 5,                             // n_min is the minimum node size
                size_t num_cutpoints = 1,                                           // # of adaptive cutpoints considered at each split for cont variables
                double no_split_penality = 0.001,                                   // penalty for not splitting
                size_t mtry_pr = 0, size_t mtry_trt = 0,                            // mtry is the # of variables considered at each split
                size_t p_categorical_pr = 0,                                        // # of categorical regressors
                size_t p_categorical_trt = 0,                                       // # of categorical regressors
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
                bool sample_weights_flag = true, bool a_scaling = true, bool b_scaling = true)
{

    auto start = system_clock::now();

    size_t N = X.n_rows;

    // number of total variables
    size_t p_pr = X.n_cols;
    size_t p_trt = X_tau.n_cols;

    // number of continuous variables
    size_t p_continuous_pr = p_pr - p_categorical_pr;
    size_t p_continuous_trt = p_trt - p_categorical_trt;

    // suppose first p_continuous variables are continuous, then categorical

    assert(mtry_pr <= p_pr);
    assert(mtry_trt <= p_trt);

    assert(burnin <= num_sweeps);

    if (mtry_pr == 0)
    {
        mtry_pr = p_pr;
    }

    if (mtry_pr != p_pr)
    {
        COUT << "Sample " << mtry_pr << " out of " << p_pr << " variables when grow each tree." << endl;
    }

    if (mtry_trt == 0)
    {
        mtry_trt = p_trt;
    }

    if (mtry_trt != p_trt)
    {
        COUT << "Sample " << mtry_trt << " out of " << p_trt << " variables when grow each tree." << endl;
    }

    arma::umat Xorder(X.n_rows, X.n_cols);
    matrix<size_t> Xorder_std;
    ini_matrix(Xorder_std, N, p_pr);

    arma::umat Xorder_tau(X_tau.n_rows, X_tau.n_cols);
    matrix<size_t> Xorder_tau_std;
    ini_matrix(Xorder_tau_std, N, p_trt);

    std::vector<double> y_std(N);
    std::vector<size_t> z_std(N);
    std::vector<double> b(N);
    double y_mean = 0.0;

    Rcpp::NumericMatrix X_std(N, p_pr);
    Rcpp::NumericMatrix X_tau_std(N, p_trt);
    // Rcpp::NumericMatrix Xtest_std(N_test, p);

    //    rcpp_to_std2(y, X, Xtest, y_std, y_mean, X_std, Xtest_std, Xorder_std);
    arma_to_std(y, y_std);
    arma_to_std(z, z_std);
    arma_to_rcpp(X, X_std);
    arma_to_std_ordered(X, Xorder_std);
    arma_to_rcpp(X_tau, X_tau_std);
    arma_to_std_ordered(X_tau, Xorder_tau_std);
    y_mean = compute_mean(y_std);
    ///////////////////////////////////////////////////////////////////
    std::vector<double> sigma_vec(2); // vector of sigma0, sigma1
    sigma_vec[0] = 1.0;
    sigma_vec[1] = 1.0;

    double bscale0 = -0.5;
    double bscale1 = 0.5;

    std::vector<double> b_vec(2); // vector of sigma0, sigma1
    b_vec[0] = bscale0;
    b_vec[1] = bscale1;

    std::vector<size_t> num_trees(2); // vector of tree number for each of mu and tau
    num_trees[0] = num_trees_pr;
    num_trees[1] = num_trees_trt;

    size_t n_trt = 0; // number of treated individuals TODO: remove from here and from constructor as well

    // assuming we have presorted data (treated individuals first, then control group)

    for (size_t i = 0; i < N; i++)
    {
        b[i] = z[i] * bscale1 + (1 - z[i]) * bscale0;
        if (z[i] == 1)
            n_trt++;
    }

    // double *ypointer = &y_std[0];

    double *Xpointer = &X_std[0];
    double *Xpointer_tau = &X_tau_std[0];
    // double *Xtestpointer = &Xtest_std[0];

    matrix<double> tauhats_xinfo;
    ini_matrix(tauhats_xinfo, N, num_sweeps);
    matrix<double> muhats_xinfo;
    ini_matrix(muhats_xinfo, N, num_sweeps);
    matrix<double> total_fit;
    ini_matrix(total_fit, N, num_sweeps);

    matrix<double> sigma0_draw_xinfo;
    ini_matrix(sigma0_draw_xinfo, num_trees_trt, num_sweeps);

    matrix<double> sigma1_draw_xinfo;
    ini_matrix(sigma1_draw_xinfo, num_trees_trt, num_sweeps);

    matrix<double> b0_draw_xinfo;
    ini_matrix(b0_draw_xinfo, num_trees_trt, num_sweeps);

    matrix<double> b1_draw_xinfo;
    ini_matrix(b1_draw_xinfo, num_trees_trt, num_sweeps);

    matrix<double> b_xinfo;
    ini_matrix(b_xinfo, num_sweeps, 2);

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
    xbcfModel *model_pr = new xbcfModel(kap_pr, s_pr, tau_pr, alpha_pr, beta_pr);
    model_pr->setNoSplitPenality(no_split_penality);

    // define the model for the treatment term
    xbcfModel *model_trt = new xbcfModel(kap_trt, s_trt, tau_trt, alpha_trt, beta_trt);
    model_trt->setNoSplitPenality(no_split_penality);

    // State settings for the prognostic term
    std::unique_ptr<State> state(new xbcfState(Xpointer, Xorder_std, N, n_trt, p_pr, p_trt, num_trees, p_categorical_pr, p_categorical_trt, p_continuous_pr, p_continuous_trt, set_random_seed, random_seed, n_min, num_cutpoints, parallel, mtry_pr, mtry_trt, Xpointer, num_sweeps, sample_weights_flag, &y_std, b, z_std, sigma_vec, b_vec, max_depth, y_mean, burnin, model_trt->dim_residual));

    // initialize X_struct for the prognostic term
    std::vector<double> initial_theta_pr(1, y_mean / (double)num_trees_pr);
    std::unique_ptr<X_struct> x_struct_pr(new X_struct(Xpointer, &y_std, N, Xorder_std, p_categorical_pr, p_continuous_pr, &initial_theta_pr, num_trees_pr));

    // initialize X_struct for the treatment term
    std::vector<double> initial_theta_trt(1, 0);
    std::unique_ptr<X_struct> x_struct_trt(new X_struct(Xpointer_tau, &y_std, N, Xorder_tau_std, p_categorical_trt, p_continuous_trt, &initial_theta_trt, num_trees_trt));

    // mcmc_loop returns tauhat [N x sweeps] matrix
    mcmc_loop_xbcf(Xorder_std, Xorder_tau_std, Xpointer, Xpointer_tau, verbose, sigma0_draw_xinfo, sigma1_draw_xinfo, b_xinfo, b0_draw_xinfo, b1_draw_xinfo, total_fit, *trees_pr, *trees_trt, no_split_penality,
                   state, model_pr, model_trt, x_struct_pr, x_struct_trt, a_scaling, b_scaling);

    //predict tauhats and muhats
    model_trt->predict_std(Xpointer_tau, N, p_trt, num_trees_trt, num_sweeps, tauhats_xinfo, *trees_trt);
    model_pr->predict_std(Xpointer, N, p_pr, num_trees_pr, num_sweeps, muhats_xinfo, *trees_pr);

    // R Objects to Return
    Rcpp::NumericMatrix tauhats(N, num_sweeps);
    Rcpp::NumericMatrix muhats(N, num_sweeps);
    Rcpp::NumericMatrix b_tau(N, num_sweeps);
    Rcpp::NumericMatrix sigma0_draws(num_trees_trt, num_sweeps);
    Rcpp::NumericMatrix sigma1_draws(num_trees_trt, num_sweeps);
    Rcpp::NumericMatrix b0_draws(num_trees_trt, num_sweeps);
    Rcpp::NumericMatrix b1_draws(num_trees_trt, num_sweeps);
    Rcpp::NumericMatrix b_draws(num_sweeps, 2);

    std_to_rcpp(tauhats_xinfo, tauhats);
    std_to_rcpp(muhats_xinfo, muhats);
    std_to_rcpp(total_fit, b_tau);
    std_to_rcpp(sigma0_draw_xinfo, sigma0_draws);
    std_to_rcpp(sigma1_draw_xinfo, sigma1_draws);
    std_to_rcpp(b0_draw_xinfo, b0_draws);
    std_to_rcpp(b1_draw_xinfo, b1_draws);
    std_to_rcpp(b_xinfo, b_draws);

    auto end = system_clock::now();

    auto duration = duration_cast<microseconds>(end - start);

    // COUT << "Total running time " << double(duration.count()) * microseconds::period::num / microseconds::period::den << endl;

    // COUT << "Running time of split Xorder " << run_time << endl;

    // COUT << "Count of splits for each variable " << mtry_weight_current_tree << endl;

    // clean memory
    delete model_pr;
    delete model_trt;
    state.reset();
    x_struct_pr.reset();
    x_struct_trt.reset();

    // R Objects to Return
    return Rcpp::List::create(
        Rcpp::Named("tauhats") = tauhats,
        Rcpp::Named("muhats") = muhats,
        Rcpp::Named("b_tau") = b_tau,
        Rcpp::Named("sigma0_draws") = sigma0_draws,
        Rcpp::Named("sigma1_draws") = sigma1_draws,
        Rcpp::Named("b0_draws") = b0_draws,
        Rcpp::Named("b1_draws") = b1_draws,
        Rcpp::Named("b_draws") = b_draws);
}