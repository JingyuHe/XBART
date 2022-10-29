#include <ctime>
// #include <RcppArmadillo.h>
#include "Rcpp.h"
#include <armadillo>
#include "tree.h"
#include <chrono>
#include "mcmc_loop.h"
#include "X_struct.h"
#include "utility_rcpp.h"

using namespace std;
using namespace chrono;
using namespace arma;

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]]
Rcpp::List XBART_multinomial_cpp(Rcpp::IntegerVector y, size_t num_class, mat X, size_t num_trees, size_t num_sweeps, size_t max_depth, size_t n_min, size_t num_cutpoints, double alpha, double beta, double tau_a, double tau_b, double no_split_penalty, size_t burnin = 1, size_t mtry = 0, size_t p_categorical = 0, bool verbose = false, bool parallel = true, bool set_random_seed = false, size_t random_seed = 0, bool sample_weights = true, bool separate_tree = false, double weight = 1, bool update_weight = true, bool update_tau = true, bool update_phi = true, double nthread = 0, double hmult = 1, double heps = 0.1, double a = 0.0001, size_t weight_exponent = 4)
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

    umat Xorder(X.n_rows, X.n_cols);
    matrix<size_t> Xorder_std;
    ini_matrix(Xorder_std, N, p);

    std::vector<size_t> y_size_t(N);
    for (size_t i = 0; i < N; ++i)
        y_size_t[i] = y[i];

    // TODO: check if I need to carry this // Yes, for now we need it.
    std::vector<double> y_std(N);
    double y_mean = 0.0;
    for (size_t i = 0; i < N; ++i)
        y_std[i] = y[i];

    Rcpp::NumericMatrix X_std(N, p);

    // dumb little hack to make this work, should write a new one of these
    rcpp_to_std2(X, X_std, Xorder_std);

    ///////////////////////////////////////////////////////////////////

    // double *ypointer = &y_std[0];
    double *Xpointer = &X_std[0];

    // matrix<double> yhats_xinfo;
    // ini_matrix(yhats_xinfo, N, num_sweeps);
    matrix<double> yhats_train_xinfo;
    ini_matrix(yhats_train_xinfo, N, num_sweeps);

    // State settings
    // Logit doesn't need an inherited state class at the moment
    // (see comments in the public declarations of LogitModel)
    // but we should consider moving phi and y_size_t to a LogitState
    // (y_size_t definitely belongs there, phi probably does)

    std::vector<double> initial_theta(num_class, 1);
    LogitState state(Xpointer, Xorder_std, N, p, num_trees, p_categorical, p_continuous, set_random_seed, random_seed, n_min, num_cutpoints, mtry, Xpointer, num_sweeps, sample_weights, &y_std, 1.0, max_depth, y_mean, burnin, num_class, nthread, a, weight_exponent);

    // initialize X_struct
    X_struct x_struct(Xpointer, &y_std, N, Xorder_std, p_categorical, p_continuous, &initial_theta, num_trees);

    cout << "number of data " << N << " " << x_struct.n_y << endl;

    std::vector<std::vector<double>> weight_samples;
    ini_matrix(weight_samples, num_trees, num_sweeps);
    std::vector<std::vector<double>> phi_samples;
    ini_matrix(phi_samples, num_trees, num_sweeps);
    std::vector<std::vector<double>> logloss;
    ini_matrix(logloss, num_trees, num_sweeps);
    std::vector<std::vector<double>> tree_size;
    ini_matrix(tree_size, num_trees, num_sweeps);
    std::vector<double> lambda_samples;

    // output is in 3 dim, stacked as a vector, number of sweeps * observations * number of classes
    std::vector<double> output_train(num_sweeps * N * num_class);

    ////////////////////////////////////////////////
    // for a n * p * m matrix, the (i,j,k) element is
    // i + j * n + k * n * p in the stacked vector
    // if stack by column, index starts from 0
    ////////////////////////////////////////////////

    vector<vector<tree>> *trees2 = new vector<vector<tree>>(num_sweeps);
    // separate tree
    vector<vector<vector<tree>>> *trees3 = new vector<vector<vector<tree>>>(num_class);

    std::vector<double> phi(N, 0);

    if (!separate_tree)
    {
        for (size_t i = 0; i < num_sweeps; i++)
        {
            (*trees2)[i] = vector<tree>(num_trees);
        }

        LogitModel *model = new LogitModel(num_class, tau_a, tau_b, alpha, beta, &y_size_t, &phi, weight, update_weight, update_tau, update_phi, hmult, heps);
        model->setNoSplitPenality(no_split_penalty);

        mcmc_loop_multinomial(Xorder_std, verbose, *trees2, no_split_penalty, state, model, x_struct, weight_samples, lambda_samples, phi_samples, logloss, tree_size);

        model->predict_std(Xpointer, N, p, num_trees, num_sweeps, yhats_train_xinfo, *trees2, output_train);

        // delete model;
    }
    else
    {
        for (size_t i = 0; i < num_class; i++)
        {
            (*trees3)[i] = vector<vector<tree>>(num_sweeps);
            for (size_t j = 0; j < num_sweeps; j++)
            {
                (*trees3)[i][j] = vector<tree>(num_trees);
            }
        }

        LogitModelSeparateTrees *model = new LogitModelSeparateTrees(num_class, tau_a, tau_b, alpha, beta, &y_size_t, &phi, weight, update_weight, update_tau, update_phi);

        model->setNoSplitPenality(no_split_penalty);

        mcmc_loop_multinomial_sample_per_tree(Xorder_std, verbose, *trees3, no_split_penalty, state, model, x_struct, weight_samples, phi_samples, logloss, tree_size);

        model->predict_std(Xpointer, N, p, num_trees, num_sweeps, yhats_train_xinfo, *trees3, output_train);

        // delete model;
    }

    Rcpp::NumericVector output_tr = Rcpp::wrap(output_train);
    output_tr.attr("dim") = Rcpp::Dimension(num_sweeps, N, num_class);
    Rcpp::NumericVector lambda_samples_rcpp = Rcpp::wrap(lambda_samples);

    // STOPPED HERE
    // TODO: Figure out how we should store and return in sample preds
    // probably add step at the end of mcmc loop to retrieve leaf pars, aggregate and
    // normalize

    // R Objects to Return
    // Rcpp::NumericMatrix yhats(N, num_sweeps);
    Rcpp::NumericVector split_count_sum(p); // split counts
    // Rcpp::XPtr<std::vector<std::vector<tree>>> tree_pnt(trees2, true);
    Rcpp::NumericMatrix weight_sample_rcpp(num_trees, num_sweeps);
    Rcpp::NumericMatrix phi_sample_rcpp(num_trees, num_sweeps);
    Rcpp::NumericMatrix logloss_rcpp(num_trees, num_sweeps);
    Rcpp::NumericMatrix tree_size_rcpp(num_trees, num_sweeps);
    Rcpp::NumericMatrix depth_rcpp(num_trees, num_sweeps);

    for (size_t i = 0; i < num_trees; i++)
    {
        for (size_t j = 0; j < num_sweeps; j++)
        {
            weight_sample_rcpp(i, j) = weight_samples[j][i];
        }
    }
    for (size_t i = 0; i < num_trees; i++)
    {
        for (size_t j = 0; j < num_sweeps; j++)
        {
            phi_sample_rcpp(i, j) = phi_samples[j][i];
        }
    }
    for (size_t i = 0; i < num_trees; i++)
    {
        for (size_t j = 0; j < num_sweeps; j++)
        {
            logloss_rcpp(i, j) = logloss[j][i];
        }
    }
    for (size_t i = 0; i < num_trees; i++)
    {
        for (size_t j = 0; j < num_sweeps; j++)
        {
            tree_size_rcpp(i, j) = tree_size[j][i];
        }
    }
    for (size_t i = 0; i < num_trees; i++)
    {
        for (size_t j = 0; j < num_sweeps; j++)
        {
            if (!separate_tree)
            {
                depth_rcpp(i, j) = (*trees2)[j][i].getdepth();
            }
            else
            {
                depth_rcpp(i, j) = (*trees3)[0][j][i].getdepth();
            }
        }
    }
    for (size_t i = 0; i < p; i++)
    {
        split_count_sum(i) = (size_t)(*state.split_count_all)[i];
    }

    std::stringstream treess;

    // if separate trees, return length num_class object, each contains num_sweeps * num_trees trees
    // if shared trees, return length 1 object, num_sweeps * num_trees trees
    Rcpp::StringVector output_tree(0);

    if (!separate_tree)
    {
        // shared trees
        // the output is a length num_sweeps vector, each string is a sweep
        // for each sweep, put first tree of all K classes first (duplicated), then the second tree
        // still num_class * num_trees in each string, for convenience of BART initialization
        for (size_t i = 0; i < num_sweeps; i++)
        {
            treess.precision(10);
            treess.str(std::string());
            treess << (double)separate_tree << " " << num_class << " " << num_sweeps << " " << num_trees << " " << p << endl;
            for (size_t j = 0; j < num_trees; j++)
            {
                for (size_t kk = 0; kk < num_class; kk++)
                {
                    treess << (*trees2)[i][j];
                }
            }
            output_tree.push_back(treess.str());
        }
    }
    else
    {
        // separate trees
        // the output is a length num_sweeps vector, each string is a sweep
        // for each sweep, put first tree of all K classes first, then the second tree, etc
        for (size_t i = 0; i < num_sweeps; i++)
        {
            treess.precision(10);
            treess.str(std::string());
            treess << (double)separate_tree << " " << num_class << " " << num_sweeps << " " << num_trees << " " << p << endl;
            for (size_t j = 0; j < num_trees; j++)
            {
                for (size_t kk = 0; kk < num_class; kk++)
                {
                    treess << (*trees3)[kk][i][j];
                }
            }
            output_tree.push_back(treess.str());
        }
    }

    // stop parallel computing
    thread_pool.stop();

    Rcpp::List ret = Rcpp::List::create(
        // Rcpp::Named("yhats") = yhats,
        Rcpp::Named("num_class") = num_class,
        Rcpp::Named("yhats_train") = output_tr,
        Rcpp::Named("weight") = weight_sample_rcpp,
        Rcpp::Named("phi") = phi_sample_rcpp,
        Rcpp::Named("logloss") = logloss_rcpp,
        Rcpp::Named("tree_size") = tree_size_rcpp,
        Rcpp::Named("lambda") = lambda_samples_rcpp,
        Rcpp::Named("importance") = split_count_sum,
        Rcpp::Named("depth") = depth_rcpp,
        Rcpp::Named("treedraws") = output_tree,
        Rcpp::Named("separate_tree") = separate_tree,
        Rcpp::Named("model_list") = Rcpp::List::create(Rcpp::Named("y_mean") = y_mean, Rcpp::Named("p") = p, Rcpp::Named("num_class") = num_class,
                                                       Rcpp::Named("num_sweeps") = num_sweeps, Rcpp::Named("num_trees") = num_trees));

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
