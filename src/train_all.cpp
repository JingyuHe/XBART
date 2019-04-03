#include <ctime>
#include <RcppArmadillo.h>
#include "tree.h"
#include "treefuns.h"
#include "forest.h"
#include <chrono>
#include "fit_std_main_loop.h"


using namespace std;
using namespace chrono;

////////////////////////////////////////////////////////////////////////
//                                                                    //
//                                                                    //
//  Full function, support both continuous and categorical variables  //
//                                                                    //
//                                                                    //
////////////////////////////////////////////////////////////////////////

void rcpp_to_std2(
    arma::mat y, arma::mat X, arma::mat Xtest, arma::mat max_depth,
    std::vector<double> &y_std, double &y_mean, Rcpp::NumericMatrix &X_std, Rcpp::NumericMatrix &Xtest_std,
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

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]]
Rcpp::List XBART(arma::mat y, arma::mat X, arma::mat Xtest,
                            size_t M, size_t L, size_t N_sweeps, arma::mat max_depth,
                            size_t Nmin, size_t Ncutpoints, double alpha, double beta,
                            double tau, size_t burnin = 1, size_t mtry = 0, size_t p_categorical = 0,
                            bool draw_sigma = false, double kap = 16, double s = 4, bool verbose = false,
                            bool m_update_sigma = false, bool draw_mu = false, bool parallel = true, bool set_random_seed = false, size_t random_seed = 0)
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
    assert(burnin <= N_sweeps);

    if (mtry == 0)
    {
        mtry = p;
    }

    if (mtry != p)
    {
        cout << "Sample " << mtry << " out of " << p << " variables when grow each tree." << endl;
    }

    arma::umat Xorder(X.n_rows, X.n_cols);
    xinfo_sizet Xorder_std;
    ini_xinfo_sizet(Xorder_std, N, p);

    std::vector<double> y_std(N);
    double y_mean = 0.0;

    Rcpp::NumericMatrix X_std(N, p);
    Rcpp::NumericMatrix Xtest_std(N_test, p);

    xinfo_sizet max_depth_std;
    ini_xinfo_sizet(max_depth_std, max_depth.n_rows, max_depth.n_cols);

    rcpp_to_std2(y, X, Xtest, max_depth, y_std, y_mean, X_std, Xtest_std, Xorder_std, max_depth_std);

    ///////////////////////////////////////////////////////////////////

    // double *ypointer = &y_std[0];
    double *Xpointer = &X_std[0];
    double *Xtestpointer = &Xtest_std[0];

    xinfo yhats_std;
    ini_xinfo(yhats_std, N, N_sweeps);
    xinfo yhats_test_std;
    ini_xinfo(yhats_test_std, N_test, N_sweeps);

    xinfo yhats_xinfo;
    ini_xinfo(yhats_xinfo, N, N_sweeps);

    xinfo yhats_test_xinfo;
    ini_xinfo(yhats_test_xinfo, N, N_sweeps);

    xinfo sigma_draw_xinfo;
    ini_xinfo(sigma_draw_xinfo, M, N_sweeps);

    xinfo split_count_all_tree;
    ini_xinfo(split_count_all_tree, p, M); // initialize at 0

    // // Create trees
    vector<vector<tree>>* trees2 = new vector<vector<tree>>(N_sweeps);
    for (size_t i = 0; i < N_sweeps; i++)
    {
        (*trees2)[i] = vector<tree>(M);
    }

    // Forests *forests = new Forests(y_mean,p,L,M,N_sweeps);




    /////////////////////////////////////////////////////////////////
    fit_std_main_loop_all(Xpointer, y_std, y_mean, Xtestpointer, Xorder_std,
                          N, p, N_test, M, L, N_sweeps, max_depth_std, // NEED TO CHANGE "max_depth"
                          Nmin, Ncutpoints, alpha, beta, tau, burnin, mtry,
                          draw_sigma, kap, s, verbose, m_update_sigma, draw_mu, parallel,
                          yhats_xinfo, yhats_test_xinfo, sigma_draw_xinfo, split_count_all_tree,
                          p_categorical, p_continuous, *trees2, set_random_seed, random_seed);
   

    
    // R Objects to Return    
    Rcpp::NumericMatrix yhats(N, N_sweeps);
    Rcpp::NumericMatrix yhats_test(N_test, N_sweeps);
    Rcpp::NumericMatrix sigma_draw(M, N_sweeps); // save predictions of each tree
    Rcpp::XPtr<std::vector<std::vector<tree>>> tree_pnt(trees2,true);






    // TODO: Make these functions
    for (size_t i = 0; i < N; i++)
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
    for (size_t i = 0; i < M; i++)
    {
        for (size_t j = 0; j < N_sweeps; j++)
        {
            sigma_draw(i, j) = sigma_draw_xinfo[j][i];
        }
    }

    auto end = system_clock::now();

    auto duration = duration_cast<microseconds>(end - start);

    // cout << "Total running time " << double(duration.count()) * microseconds::period::num / microseconds::period::den << endl;

    // cout << "Running time of split Xorder " << run_time << endl;

    // cout << "Count of splits for each variable " << mtry_weight_current_tree << endl;

    // return Rcpp::List::create(Rcpp::Named("yhats") = yhats, Rcpp::Named("yhats_test") = yhats_test, Rcpp::Named("sigma") = sigma_draw, Rcpp::Named("trees") = Rcpp::CharacterVector(treess.str()));
    return Rcpp::List::create(Rcpp::Named("yhats") = yhats, Rcpp::Named("yhats_test") = yhats_test, 
        Rcpp::Named("sigma") = sigma_draw
        ,Rcpp::Named("model_list") = Rcpp::List::create(Rcpp::Named("tree_pnt")= tree_pnt, 
                                                       Rcpp::Named("y_mean") = y_mean,
                                                       Rcpp::Named("p")=p,
                                                       Rcpp::Named("L")=L)
        );
}

Rcpp::List XBARTlp(arma::mat y, arma::mat X, arma::mat Xtest,
                            size_t M, size_t L, size_t N_sweeps, arma::mat max_depth,
                            size_t Nmin, size_t Ncutpoints, double alpha, double beta,
                            double tau, size_t burnin = 1, size_t mtry = 0, size_t p_categorical = 0,
                            bool draw_sigma = false, double kap = 16, double s = 4, bool verbose = false,
                            bool m_update_sigma = false, bool draw_mu = false, bool parallel = true, 
                            bool set_random_seed = false, size_t random_seed = 0,double a = 1/50, double b = 1/50)
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
    assert(burnin <= N_sweeps);

    if (mtry == 0)
    {
        mtry = p;
    }

    if (mtry != p)
    {
        cout << "Sample " << mtry << " out of " << p << " variables when grow each tree." << endl;
    }

    arma::umat Xorder(X.n_rows, X.n_cols);
    xinfo_sizet Xorder_std;
    ini_xinfo_sizet(Xorder_std, N, p);

    std::vector<double> y_std(N);
    double y_mean = 0.0;

    Rcpp::NumericMatrix X_std(N, p);
    Rcpp::NumericMatrix Xtest_std(N_test, p);

    xinfo_sizet max_depth_std;
    ini_xinfo_sizet(max_depth_std, max_depth.n_rows, max_depth.n_cols);

    rcpp_to_std2(y, X, Xtest, max_depth, y_std, y_mean, X_std, Xtest_std, Xorder_std, max_depth_std);

    ///////////////////////////////////////////////////////////////////

    // double *ypointer = &y_std[0];
    double *Xpointer = &X_std[0];
    double *Xtestpointer = &Xtest_std[0];

    xinfo yhats_std;
    ini_xinfo(yhats_std, N, N_sweeps);
    xinfo yhats_test_std;
    ini_xinfo(yhats_test_std, N_test, N_sweeps);

    xinfo yhats_xinfo;
    ini_xinfo(yhats_xinfo, N, N_sweeps);

    xinfo yhats_test_xinfo;
    ini_xinfo(yhats_test_xinfo, N, N_sweeps);

    xinfo sigma_draw_xinfo;
    ini_xinfo(sigma_draw_xinfo, M, N_sweeps);

    xinfo split_count_all_tree;
    ini_xinfo(split_count_all_tree, p, M); // initialize at 0

    // // Create trees
    vector<vector<tree>>* trees2 = new vector<vector<tree>>(N_sweeps);
    for (size_t i = 0; i < N_sweeps; i++)
    {
        (*trees2)[i] = vector<tree>(M);
    }

    // Forests *forests = new Forests(y_mean,p,L,M,N_sweeps);




    /////////////////////////////////////////////////////////////////
    // fit_std_main_loop_all(Xpointer, y_std, y_mean, Xtestpointer, Xorder_std,
    //                       N, p, N_test, M, L, N_sweeps, max_depth_std, // NEED TO CHANGE "max_depth"
    //                       Nmin, Ncutpoints, alpha, beta, tau, burnin, mtry,
    //                       draw_sigma, kap, s, verbose, m_update_sigma, draw_mu, parallel,
    //                       yhats_xinfo, yhats_test_xinfo, sigma_draw_xinfo, split_count_all_tree,
    //                       p_categorical, p_continuous, *trees2, set_random_seed, random_seed);




        fit_std_poisson_classification(Xpointer, y_std, y_mean, Xtestpointer, Xorder_std,
                    N, p, N_test, M, L, N_sweeps, max_depth_std, // NEED TO CHANGE "max_depth"
                    Nmin, Ncutpoints, alpha, beta, tau, burnin, mtry,
                    draw_sigma, kap, s, verbose, m_update_sigma, draw_mu, parallel,
                    yhats_xinfo, yhats_test_xinfo, sigma_draw_xinfo, split_count_all_tree,
                     p_categorical, p_continuous, *trees2, set_random_seed, random_seed,a,b);




   
    // R Objects to Return    
    Rcpp::NumericMatrix yhats(N, N_sweeps);
    Rcpp::NumericMatrix yhats_test(N_test, N_sweeps);
    Rcpp::NumericMatrix sigma_draw(M, N_sweeps); // save predictions of each tree
    Rcpp::XPtr<std::vector<std::vector<tree>>> tree_pnt(trees2,true);






    // TODO: Make these functions
    for (size_t i = 0; i < N; i++)
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
    for (size_t i = 0; i < M; i++)
    {
        for (size_t j = 0; j < N_sweeps; j++)
        {
            sigma_draw(i, j) = sigma_draw_xinfo[j][i];
        }
    }

    auto end = system_clock::now();

    auto duration = duration_cast<microseconds>(end - start);

    // cout << "Total running time " << double(duration.count()) * microseconds::period::num / microseconds::period::den << endl;

    // cout << "Running time of split Xorder " << run_time << endl;

    // cout << "Count of splits for each variable " << mtry_weight_current_tree << endl;

    // return Rcpp::List::create(Rcpp::Named("yhats") = yhats, Rcpp::Named("yhats_test") = yhats_test, Rcpp::Named("sigma") = sigma_draw, Rcpp::Named("trees") = Rcpp::CharacterVector(treess.str()));
    return Rcpp::List::create(
        Rcpp::Named("yhats") = yhats, Rcpp::Named("yhats_test") = yhats_test, 
        Rcpp::Named("sigma") = sigma_draw
        ,Rcpp::Named("model_list") = Rcpp::List::create(Rcpp::Named("tree_pnt")= tree_pnt, 
                                                       Rcpp::Named("y_mean") = y_mean,
                                                       Rcpp::Named("p")=p,
                                                       Rcpp::Named("L")=L)
        );
}
