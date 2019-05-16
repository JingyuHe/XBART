#include <ctime>
#include <RcppArmadillo.h>
#include "tree.h"
#include "forest.h"
#include <chrono>

using namespace std;
using namespace chrono;

////////////////////////////////////////////////////////////////////////
//                                                                    //
//                                                                    //
//  Full function, support both continuous and categorical variables  //
//                                                                    //
//                                                                    //
////////////////////////////////////////////////////////////////////////

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]]
Rcpp::List train_forest_root_std_all(arma::mat y, arma::mat X, arma::mat Xtest, size_t num_trees, size_t num_sweeps, arma::mat max_depth_num, size_t n_min, size_t num_cutpoints, double alpha, double beta, double tau, double no_split_penality, size_t burnin = 1, size_t mtry = 0, size_t p_categorical = 0, double kap = 16, double s = 4, bool verbose = false, bool parallel = true, bool set_random_seed = false, size_t random_seed = 0)
{
    bool draw_mu = true;
    bool categorical_variables = false;
    if (p_categorical > 0)
    {
        categorical_variables = true;
    }

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
    for (size_t i = 0; i < X.n_cols; i++)
    {
        Xorder.col(i) = arma::sort_index(X.col(i));
    }

    // std::default_random_engine(generator);

    ///////////////////////////////////////////////////////////////////
    // inefficient! Need replacement
    std::vector<double> y_std(N);
    double y_mean = 0.0;
    for (size_t i = 0; i < N; i++)
    {
        y_std[i] = y(i, 0);
        y_mean = y_mean + y_std[i];
    }
    y_mean = y_mean / (double)N;

    Rcpp::NumericMatrix X_std(N, p);
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < p; j++)
        {
            X_std(i, j) = X(i, j);
        }
    }

    Rcpp::NumericMatrix Xtest_std(N_test, p);
    for (size_t i = 0; i < N_test; i++)
    {
        for (size_t j = 0; j < p; j++)
        {
            Xtest_std(i, j) = Xtest(i, j);
        }
    }

    xinfo_sizet Xorder_std;
    ini_xinfo_sizet(Xorder_std, N, p);
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < p; j++)
        {
            Xorder_std[j][i] = Xorder(i, j);
        }
    }

    ///////////////////////////////////////////////////////////////////

    double *ypointer = &y_std[0];
    double *Xpointer = &X_std[0];
    double *Xtestpointer = &Xtest_std[0];

    std::vector<double> X_values;
    std::vector<size_t> X_counts;
    std::vector<size_t> variable_ind(p_categorical + 1);

    size_t total_points;

    std::vector<size_t> X_num_unique(p_categorical);

    unique_value_count2(Xpointer, Xorder_std, X_values, X_counts, variable_ind, total_points, X_num_unique, p_categorical, p_continuous);

    COUT << "X_values" << X_values << endl;
    COUT << "X_counts" << X_counts << endl;
    COUT << "variable_ind " << variable_ind << endl;
    COUT << "X_num_unique " << X_num_unique << endl;

    xinfo yhats_std;
    ini_xinfo(yhats_std, N, num_sweeps);
    xinfo yhats_test_std;
    ini_xinfo(yhats_test_std, N_test, num_sweeps);

    // save predictions of each tree
    std::vector<std::vector<double>> predictions_std;
    ini_xinfo(predictions_std, N, num_trees);

    xinfo predictions_test_std;
    ini_xinfo(predictions_test_std, N_test, num_trees);

    std::vector<double> yhat_std(N);
    row_sum(predictions_std, yhat_std);
    std::vector<double> yhat_test_std(N_test);
    row_sum(predictions_test_std, yhat_test_std);

    // current residual
    std::vector<double> residual_std(N);

    xinfo sigma_draw_std;
    ini_xinfo(sigma_draw_std, num_trees, num_sweeps);

    forest trees_std(num_trees);

    std::vector<double> reshat_std;
    std::vector<double> reshat_test_std;

    ///////////////////////////////////////////////////////////////////

    Rcpp::NumericMatrix yhats(N, num_sweeps);
    Rcpp::NumericMatrix yhats_test(N_test, num_sweeps);

    // save predictions of each tree
    Rcpp::NumericMatrix sigma_draw(num_trees, num_sweeps);

    double sigma;
    // double tau;
    forest trees(num_trees);
    std::vector<double> prob(2, 0.5);
    std::random_device rd;
    std::mt19937 gen(rd());
    if (set_random_seed)
    {
        gen.seed(random_seed);
    }
    std::discrete_distribution<> d(prob.begin(), prob.end());
    // // sample one index of split point
    size_t prune;

    xinfo split_count_all_tree;
    ini_xinfo(split_count_all_tree, p, num_trees); // initialize at 0
    // split_count_all_tree = split_count_all_tree + 1; // initialize at 1
    std::vector<double> split_count_current_tree(p, 1);
    std::vector<double> mtry_weight_current_tree(p, 1);

    // in the burnin samples, use all variables
    std::vector<size_t> subset_vars(p);
    std::iota(subset_vars.begin() + 1, subset_vars.end(), 1);

    double run_time = 0.0;

    // num_trees, number of trees

    bool use_all = true;

    NormalModel model;
    model.setNoSplitPenality(no_split_penality);

    // initialize a matrix to save pointers to node for each data point

    matrix<tree::tree_p> data_pointers;
    ini_matrix(data_pointers, N, num_trees);

    // initialize predcitions and predictions_test
    for (size_t ii = 0; ii < num_trees; ii++)
    {
        std::fill(predictions_std[ii].begin(), predictions_std[ii].end(), y_mean / (double)num_trees);
        std::fill(predictions_test_std[ii].begin(), predictions_test_std[ii].end(), y_mean / (double)num_trees);
    }

    row_sum(predictions_std, yhat_std);
    row_sum(predictions_test_std, yhat_test_std);

    residual_std = y_std - yhat_std;

    for (size_t sweeps = 0; sweeps < num_sweeps; sweeps++)
    {

        if (verbose == true)
        {
            COUT << "--------------------------------" << endl;
            COUT << "number of sweeps " << sweeps << endl;
            COUT << "--------------------------------" << endl;
        }

        for (size_t tree_ind = 0; tree_ind < num_trees; tree_ind++)
        {

            // if update sigma based on residual of all m trees
            // if (m_update_sigma == true)
            // {

            std::gamma_distribution<double> gamma_samp((N + kap) / 2.0, 2.0 / (sum_squared(residual_std) + s));

            sigma = 1.0 / sqrt(gamma_samp(gen));

            sigma_draw(tree_ind, sweeps) = sigma;
            // }

            // save sigma
            sigma_draw(tree_ind, sweeps) = sigma;

            // add prediction of current tree back to residual
            // then it's m - 1 trees residual

            residual_std = residual_std + predictions_std[tree_ind];

            yhat_std = yhat_std - predictions_std[tree_ind];

            yhat_test_std = yhat_test_std - predictions_test_std[tree_ind];

            if (use_all && (sweeps > burnin) && (mtry != p))
            {
                // subset_vars = Rcpp::as<std::vector<size_t>>(sample(var_index_candidate, mtry, false, split_var_count));

                // subset_vars = sample_int_crank2(p, mtry, split_var_count);

                use_all = false;
            }

            // clear counts of splits for one tree
            std::fill(split_count_current_tree.begin(), split_count_current_tree.end(), 0.0);

            mtry_weight_current_tree = mtry_weight_current_tree - split_count_all_tree[tree_ind];

            trees.t[tree_ind].grow_tree_adaptive_std_all(sum_vec(residual_std) / (double)N, 0, max_depth_num(tree_ind, sweeps), n_min, num_cutpoints, tau, sigma, alpha, beta, draw_mu, parallel, residual_std, Xorder_std, Xpointer, mtry, use_all, split_count_all_tree, mtry_weight_current_tree, split_count_current_tree, categorical_variables, p_categorical, p_continuous, X_values, X_counts, variable_ind, X_num_unique, &model, data_pointers, tree_ind, gen,true);

            mtry_weight_current_tree = mtry_weight_current_tree + split_count_current_tree;

            split_count_all_tree[tree_ind] = split_count_current_tree;

            if (verbose == true)
            {
                COUT << "tree " << tree_ind << " size is " << trees.t[tree_ind].treesize() << endl;
            }

            // update prediction of current tree
            // fit_new_std(trees.t[tree_ind], Xpointer, N, p, predictions_std[tree_ind]);
            fit_new_std_datapointers(Xpointer, N, tree_ind, predictions_std[tree_ind], data_pointers);

            // update prediction of current tree, test set
            fit_new_std(trees.t[tree_ind], Xtestpointer, N_test, p, predictions_test_std[tree_ind]);

            // if (m_update_sigma == false)
            // {
            //     std::gamma_distribution<double> gamma_samp((N + kap) / 2.0, 2.0 / (sum_squared(residual_std) + s));

            //     sigma = 1.0 / sqrt(gamma_samp(gen));

            //     sigma_draw(tree_ind, sweeps) = sigma;
            // }

            // update residual, now it's residual of m trees

            residual_std = residual_std - predictions_std[tree_ind];

            yhat_std = yhat_std + predictions_std[tree_ind];
            yhat_test_std = yhat_test_std + predictions_test_std[tree_ind];
        }

        // save predictions to output matrix

        for (size_t kk = 0; kk < N; kk++)
        {
            yhats(kk, sweeps) = yhat_std[kk];
        }
        for (size_t kk = 0; kk < N_test; kk++)
        {
            yhats_test(kk, sweeps) = yhat_test_std[kk];
        }
    }

    auto end = system_clock::now();

    auto duration = duration_cast<microseconds>(end - start);

    COUT << "Total running time " << double(duration.count()) * microseconds::period::num / microseconds::period::den << endl;

    COUT << "Running time of split Xorder " << run_time << endl;

    COUT << "Count of splits for each variable " << mtry_weight_current_tree << endl;

    return Rcpp::List::create(Rcpp::Named("yhats") = yhats, Rcpp::Named("yhats_test") = yhats_test, Rcpp::Named("sigma") = sigma_draw);
}