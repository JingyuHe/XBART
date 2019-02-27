#include <ctime>
#include <RcppArmadillo.h>
#include "tree.h"
#include "treefuns.h"
#include "forest.h"
#include <chrono>
#include "fit_std_main_loop.h"

using namespace std;

using namespace chrono;

// Helpers
struct AbarthParams
{
    size_t M;
    size_t L;
    size_t N_sweeps;
    size_t Nmin;
    size_t Ncutpoints;
    size_t burnin;
    size_t mtry;
    size_t max_depth_num;
    double alpha;
    double beta;
    double tau;
    double kap;
    double s;
    bool draw_sigma;
    bool verbose;
    bool m_update_sigma;
    bool draw_mu;
    bool parallel;
};

void rcpp_to_std(arma::mat y, arma::mat X, arma::mat Xtest, arma::mat max_depth,
                 std::vector<double> &y_std, double &y_mean,
                 Rcpp::NumericMatrix &X_std, Rcpp::NumericMatrix &Xtest_std,
                 xinfo_sizet &Xorder_std, xinfo_sizet &max_depth_std)
{
    // The goal of this function is to convert RCPP object to std objects

    // TODO: Refactor to remove N,p,N_test
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

    //max_depth_std_test
    for (size_t i = 0; i < max_depth.n_rows; i++)
    {
        for (size_t j = 0; j < max_depth.n_cols; j++)
        {
            max_depth_std[j][i] = max_depth(i, j);
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

Rcpp::List abarth_train(arma::mat y, arma::mat X, arma::mat Xtest,
                        size_t M, size_t L, size_t N_sweeps, arma::mat max_depth,
                        size_t Nmin, size_t Ncutpoints, double alpha, double beta,
                        double tau, size_t burnin = 1, size_t mtry = 0,
                        bool draw_sigma = false, double kap = 16, double s = 4,
                        bool verbose = false, bool m_update_sigma = false,
                        bool draw_mu = false, bool parallel = true)
{

    auto start = system_clock::now();
    // RCPP -> STD
    // Container for std types to "return"
    size_t N = X.n_rows;
    size_t p = X.n_cols;
    size_t N_test = Xtest.n_rows;
    // y containers
    std::vector<double> y_std(N);
    double y_mean = 0.0;
    // x containers
    Rcpp::NumericMatrix X_std(N, p);
    Rcpp::NumericMatrix Xtest_std(N_test, p);

    // xorder containers
    xinfo_sizet Xorder_std;
    ini_xinfo_sizet(Xorder_std, N, p);

    //max_depth_std container
    xinfo_sizet max_depth_std;
    ini_xinfo_sizet(max_depth_std, max_depth.n_rows, max_depth.n_cols);

    // convert rcpp to std
    rcpp_to_std(y, X, Xtest, max_depth, y_std, y_mean, X_std, Xtest_std, Xorder_std, max_depth_std);

    // TEMP!!!!!!
    //   std::vector<double> x_temp(N*p);
    //     for(size_t i = 0 ;i<N;i++){
    //           for(size_t j = 0;j<p;j++){
    //           size_t index = j*N + i;
    //           x_temp[index] = X_std(i,j);
    //   }
    // }

    // Assertions
    assert(mtry <= p);
    assert(burnin <= N_sweeps);

    if (mtry == 0)
    {
        mtry = p;
    }

    if (mtry != p)
    {
        cout << "Sample " << mtry << " out of " << p << " variables when grow each tree." << endl;
    }

    // Pointers for data
    double *ypointer = &y_std[0];
    double *Xpointer = &X_std[0]; // &x_temp[0];
    double *Xtestpointer = &Xtest_std[0];

    // Cpp native objects to return
    xinfo yhats_xinfo;
    ini_xinfo(yhats_xinfo, N, N_sweeps);

    xinfo yhats_test_xinfo;
    ini_xinfo(yhats_test_xinfo, N, N_sweeps);

    xinfo sigma_draw_xinfo;
    ini_xinfo(sigma_draw_xinfo, M, N_sweeps);

    // main fit_std
    fit_std_main_loop(Xpointer, y_std, y_mean, Xtestpointer, Xorder_std,
                      N, p, N_test,
                      M, L, N_sweeps, max_depth_std, // NEED TO CHANGE "max_depth"
                      Nmin, Ncutpoints, alpha, beta,
                      tau, burnin, mtry,
                      draw_sigma, kap, s,
                      verbose, m_update_sigma,
                      draw_mu, parallel,
                      yhats_xinfo, yhats_test_xinfo, sigma_draw_xinfo);

    // Convert STD -> RCPP outputs

    // R Objects to Return
    Rcpp::NumericMatrix yhats(N, N_sweeps);
    Rcpp::NumericMatrix yhats_test(N_test, N_sweeps);
    Rcpp::NumericMatrix sigma_draw(M, N_sweeps); // save predictions of each tree

    // TODO: Make these functions
    for (size_t i = 0; i < N_test; i++)
    {
        for (size_t j = 0; j < N_sweeps; j++)
        {
            yhats(i, j) = yhats_xinfo[j][i];
        }
    }
    for (size_t i = 0; i < N_test; i++)
    {
        for (size_t j = 0; j < N_sweeps; j++)
        {
            yhats_test(i, j) = yhats_test_xinfo[j][i];
        }
    }
    for (size_t i = 0; i < N_test; i++)
    {
        for (size_t j = 0; j < M; j++)
        {
            sigma_draw(i, j) = sigma_draw_xinfo[j][i];
        }
    }

    auto end = system_clock::now();

    auto duration = duration_cast<microseconds>(end - start);

    cout << "Total running time " << double(duration.count()) * microseconds::period::num / microseconds::period::den << endl;

    // cout << "Running time of split Xorder " << run_time << endl;

    // cout << "Count of splits for each variable " << mtry_weight_current_tree << endl;

    // return Rcpp::List::create(Rcpp::Named("yhats") = yhats, Rcpp::Named("yhats_test") = yhats_test, Rcpp::Named("sigma") = sigma_draw, Rcpp::Named("trees") = Rcpp::CharacterVector(treess.str()));
    return Rcpp::List::create(Rcpp::Named("yhats") = yhats, Rcpp::Named("yhats_test") = yhats_test, Rcpp::Named("sigma") = sigma_draw);
}
