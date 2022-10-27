//////////////////////////////////////////////////////////////////////////////////////
// predict from tree structure
//////////////////////////////////////////////////////////////////////////////////////

#include <ctime>
#include "Rcpp.h"
#include <armadillo>
#include "tree.h"
#include <chrono>
#include "mcmc_loop.h"
#include "utility.h"
#include "json_io.h"
#include "utility_rcpp.h"

using namespace arma;

// [[Rcpp::export]]
Rcpp::List xbart_predict(mat X, double y_mean, Rcpp::XPtr<std::vector<std::vector<tree>>> tree_pnt)
{
    // predict for XBART normal regression model

    // Size of data
    size_t N = X.n_rows;
    size_t p = X.n_cols;

    // Init X_std matrix
    Rcpp::NumericMatrix X_std(N, p);
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < p; j++)
        {
            X_std(i, j) = X(i, j);
        }
    }
    double *Xpointer = &X_std[0];

    // Trees
    std::vector<std::vector<tree>> *trees = tree_pnt;

    // Result Container
    matrix<double> yhats_test_xinfo;
    size_t N_sweeps = (*trees).size();
    size_t M = (*trees)[0].size();
    ini_xinfo(yhats_test_xinfo, N, N_sweeps);

    NormalModel *model = new NormalModel();

    // Predict
    model->predict_std(Xpointer, N, p, M, N_sweeps, yhats_test_xinfo, *trees);

    // Convert back to Rcpp
    Rcpp::NumericMatrix yhats(N, N_sweeps);
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < N_sweeps; j++)
        {
            yhats(i, j) = yhats_test_xinfo[j][i];
        }
    }

    return Rcpp::List::create(Rcpp::Named("yhats") = yhats);
}

// [[Rcpp::export]]
Rcpp::List xbcf_predict(mat X_con, mat X_mod, mat Z, Rcpp::XPtr<std::vector<std::vector<tree>>> tree_con, Rcpp::XPtr<std::vector<std::vector<tree>>> tree_mod)
{
    // size of data
    size_t N = X_con.n_rows;
    size_t p_con = X_con.n_cols;
    size_t p_mod = X_mod.n_cols;
    size_t p_z = Z.n_cols;
    assert(X_con.n_rows == X_mod.n_rows);

    // Init X_std matrix
    Rcpp::NumericMatrix X_std_con(N, p_con);
    Rcpp::NumericMatrix X_std_mod(N, p_mod);

    matrix<double> Ztest_std;
    ini_matrix(Ztest_std, N, p_z);

    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < p_con; j++)
        {
            X_std_con(i, j) = X_con(i, j);
        }

        for (size_t j = 0; j < p_mod; j++)
        {
            X_std_mod(i, j) = X_mod(i, j);
        }

        for (size_t j = 0; j < p_z; j++)
        {
            Ztest_std[j][i] = Z(i, j);
        }
    }
    double *Xpointer_con = &X_std_con[0];
    double *Xpointer_mod = &X_std_mod[0];

    // Trees
    std::vector<std::vector<tree>> *trees_con = tree_con;
    std::vector<std::vector<tree>> *trees_mod = tree_mod;

    // Result Container
    size_t num_sweeps = (*trees_con).size();
    size_t num_trees_con = (*trees_con)[0].size();
    size_t num_trees_mod = (*trees_mod)[0].size();

    COUT << "number of trees " << num_trees_con << " " << num_trees_mod << endl;

    matrix<double> prognostic_xinfo;
    ini_matrix(prognostic_xinfo, N, num_sweeps);

    matrix<double> treatment_xinfo;
    ini_matrix(treatment_xinfo, N, num_sweeps);

    matrix<double> yhats_test_xinfo;
    ini_xinfo(yhats_test_xinfo, N, num_sweeps);
    XBCFContinuousModel *model = new XBCFContinuousModel();
    // Predict

    model->predict_std(Ztest_std, Xpointer_con, Xpointer_mod, N, p_con, p_mod, num_trees_con, num_trees_mod, num_sweeps, yhats_test_xinfo, prognostic_xinfo, treatment_xinfo, *trees_con, *trees_mod);

    // Convert back to Rcpp
    Rcpp::NumericMatrix yhats(N, num_sweeps);
    Rcpp::NumericMatrix prognostic(N, num_sweeps);
    Rcpp::NumericMatrix treatment(N, num_sweeps);
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < num_sweeps; j++)
        {
            yhats(i, j) = yhats_test_xinfo[j][i];
            prognostic(i, j) = prognostic_xinfo[j][i];
            treatment(i, j) = treatment_xinfo[j][i];
        }
    }

    return Rcpp::List::create(Rcpp::Named("mu") = prognostic, Rcpp::Named("tau") = treatment, Rcpp::Named("yhats") = yhats);
}

// [[Rcpp::export]]
Rcpp::List xbart_predict_full(mat X, double y_mean, Rcpp::XPtr<std::vector<std::vector<tree>>> tree_pnt)
{

    // Size of data
    size_t N = X.n_rows;
    size_t p = X.n_cols;

    // Init X_std matrix
    Rcpp::NumericMatrix X_std(N, p);
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < p; j++)
        {
            X_std(i, j) = X(i, j);
        }
    }
    double *Xpointer = &X_std[0];

    // Trees
    std::vector<std::vector<tree>> *trees = tree_pnt;

    size_t N_sweeps = (*trees).size();
    size_t M = (*trees)[0].size();

    std::vector<double> output_vec(N * N_sweeps * M);

    NormalModel *model = new NormalModel();

    // Predict
    model->predict_whole_std(Xpointer, N, p, M, N_sweeps, output_vec, *trees);

    Rcpp::NumericVector output = Rcpp::wrap(output_vec);
    output.attr("dim") = Rcpp::Dimension(N, N_sweeps, M);

    return Rcpp::List::create(Rcpp::Named("yhats") = output);
}

// [[Rcpp::export]]
Rcpp::List gp_predict(mat y, mat X, mat Xtest, Rcpp::XPtr<std::vector<std::vector<tree>>> tree_pnt, Rcpp::NumericVector resid, mat sigma, double theta, double tau, size_t p_categorical = 0)
{
    // should be able to run in parallel
    COUT << "predict with gaussian process" << endl;

    // Size of data
    size_t N = X.n_rows;
    size_t p = X.n_cols;
    size_t N_test = Xtest.n_rows;
    // number of continuous variables
    size_t p_continuous = p - p_categorical; // only work for continuous for now

    matrix<size_t> Xorder_std;
    ini_matrix(Xorder_std, N, p);

    std::vector<double> y_std(N);
    double y_mean = 0.0;

    Rcpp::NumericMatrix X_std(N, p);
    Rcpp::NumericMatrix Xtest_std(N_test, p);

    rcpp_to_std2(y, X, Xtest, y_std, y_mean, X_std, Xtest_std, Xorder_std);

    matrix<size_t> Xtestorder_std;
    ini_matrix(Xtestorder_std, N_test, p);

    // Create Xtestorder
    umat Xtestorder(Xtest.n_rows, Xtest.n_cols);
    for (size_t i = 0; i < Xtest.n_cols; i++)
    {
        Xtestorder.col(i) = sort_index(Xtest.col(i));
    }
    for (size_t i = 0; i < N_test; i++)
    {
        for (size_t j = 0; j < p; j++)
        {
            Xtestorder_std[j][i] = Xtestorder(i, j);
        }
    }

    // double *ypointer = &y_std[0];
    double *Xpointer = &X_std[0];
    double *Xtestpointer = &Xtest_std[0];

    // Trees
    std::vector<std::vector<tree>> *trees = tree_pnt;
    size_t num_sweeps = (*trees).size();
    size_t num_trees = (*trees)[0].size();

    std::vector<double> sigma_std(num_sweeps);
    for (size_t i = 0; i < num_sweeps; i++)
    {
        sigma_std[i] = sigma(i);
    }

    // initialize X_struct
    std::vector<double> initial_theta(1, y_mean / (double)num_trees);

    gp_struct x_struct(Xpointer, &y_std, N, Xorder_std, p_categorical, p_continuous, &initial_theta, sigma_std, num_trees);
    gp_struct xtest_struct(Xtestpointer, &y_std, N_test, Xtestorder_std, p_categorical, p_continuous, &initial_theta, sigma_std, num_trees);
    x_struct.n_y = N;
    xtest_struct.n_y = N_test;

    matrix<double> yhats_test_xinfo;
    ini_matrix(yhats_test_xinfo, N_test, num_sweeps);
    for (size_t i = 0; i < num_sweeps; i++)
    {
        std::fill(yhats_test_xinfo[i].begin(), yhats_test_xinfo[i].end(), 0.0);
    }

    std::vector<bool> active_var(p);
    std::fill(active_var.begin(), active_var.end(), false);

    // get residuals
    matrix<std::vector<double>> residuals;
    ini_matrix(residuals, num_trees, num_sweeps);
    for (size_t i = 0; i < num_sweeps; i++)
    {
        for (size_t j = 0; j < num_trees; j++)
        {
            residuals[i][j].resize(N);
            for (size_t k = 0; k < N; k++)
            {
                residuals[i][j][k] = resid(k + i * N + j * num_sweeps * N);
            }
        }
    }
    x_struct.set_resid(residuals);

    // mcmc loop
    for (size_t sweeps = 0; sweeps < num_sweeps; sweeps++)
    {
        for (size_t tree_ind = 0; tree_ind < num_trees; tree_ind++)
        {
            (*trees)[sweeps][tree_ind].gp_predict_from_root(Xorder_std, x_struct, x_struct.X_counts, x_struct.X_num_unique,
                                                            Xtestorder_std, xtest_struct, xtest_struct.X_counts, xtest_struct.X_num_unique,
                                                            yhats_test_xinfo, active_var, p_categorical, sweeps, tree_ind, theta, tau);
        }
    }

    Rcpp::NumericMatrix yhats_test(N_test, num_sweeps);
    Matrix_to_NumericMatrix(yhats_test_xinfo, yhats_test);

    return Rcpp::List::create(Rcpp::Named("yhats_test") = yhats_test);
}

// [[Rcpp::export]]
Rcpp::List xbart_multinomial_predict(mat X, double y_mean, size_t num_class, Rcpp::XPtr<std::vector<std::vector<tree>>> tree_pnt)
{

    // Size of data
    size_t N = X.n_rows;
    size_t p = X.n_cols;

    // Init X_std matrix
    Rcpp::NumericMatrix X_std(N, p);
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < p; j++)
        {
            X_std(i, j) = X(i, j);
        }
    }
    double *Xpointer = &X_std[0];

    // Trees
    std::vector<std::vector<tree>> *trees = tree_pnt;

    // Result Container
    matrix<double> yhats_test_xinfo;
    size_t N_sweeps = (*trees).size();
    size_t N_trees = (*trees)[0].size();
    ini_xinfo(yhats_test_xinfo, N, N_sweeps);

    std::vector<double> output_vec(N_sweeps * N * num_class);

    LogitModel *model = new LogitModel();

    model->dim_residual = num_class;

    // Predict
    model->predict_std(Xpointer, N, p, N_trees, N_sweeps, yhats_test_xinfo, *trees, output_vec);

    Rcpp::NumericVector output = Rcpp::wrap(output_vec);
    output.attr("dim") = Rcpp::Dimension(N_sweeps, N, num_class);

    return Rcpp::List::create(Rcpp::Named("yhats") = output);
}

// [[Rcpp::export]]
Rcpp::List xbart_multinomial_predict_separatetrees(mat X, double y_mean, size_t num_class, Rcpp::XPtr<std::vector<std::vector<std::vector<tree>>>> tree_pnt)
{

    // Size of data
    size_t N = X.n_rows;
    size_t p = X.n_cols;

    // Init X_std matrix
    Rcpp::NumericMatrix X_std(N, p);
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < p; j++)
        {
            X_std(i, j) = X(i, j);
        }
    }
    double *Xpointer = &X_std[0];

    // Trees
    std::vector<std::vector<std::vector<tree>>> *trees = tree_pnt;

    // Result Container
    matrix<double> yhats_test_xinfo;
    size_t N_sweeps = (*trees).size();
    size_t N_trees = (*trees)[0].size();
    ini_xinfo(yhats_test_xinfo, N, N_sweeps);

    std::vector<double> output_vec(N_sweeps * N * num_class);

    LogitModelSeparateTrees *model = new LogitModelSeparateTrees();

    model->dim_residual = num_class;

    // Predict
    model->predict_std(Xpointer, N, p, N_trees, N_sweeps, yhats_test_xinfo, *trees, output_vec);

    Rcpp::NumericVector output = Rcpp::wrap(output_vec);
    output.attr("dim") = Rcpp::Dimension(N_sweeps, N, num_class);

    return Rcpp::List::create(Rcpp::Named("yhats") = output);
}

// [[Rcpp::export]]
Rcpp::StringVector r_to_json(double y_mean, Rcpp::XPtr<std::vector<std::vector<tree>>> tree_pnt)
{
    // push two dimensional matrix of trees to json
    Rcpp::StringVector result(1);
    std::vector<std::vector<tree>> *trees = tree_pnt;
    json j = get_forest_json(*trees, y_mean);
    result[0] = j.dump(4);
    return result;
}

// [[Rcpp::export]]
Rcpp::List json_to_r(Rcpp::StringVector json_string_r)
{
    // load json to a two dimensional matrix of trees
    std::vector<std::string> json_string(json_string_r.size());
    // std::string json_string = json_string_r(0);
    json_string[0] = json_string_r(0);
    double y_mean;

    // Create trees
    vector<vector<tree>> *trees2 = new std::vector<vector<tree>>();

    // Load
    from_json_to_forest(json_string[0], *trees2, y_mean);

    // Define External Pointer
    Rcpp::XPtr<std::vector<std::vector<tree>>> tree_pnt(trees2, true);

    return Rcpp::List::create(Rcpp::Named("model_list") = Rcpp::List::create(Rcpp::Named("tree_pnt") = tree_pnt, Rcpp::Named("y_mean") = y_mean));
}

// [[Rcpp::export]]
Rcpp::StringVector r_to_json_3D(Rcpp::XPtr<std::vector<std::vector<std::vector<tree>>>> tree_pnt)
{
    // push 3 dimensional matrix of trees to json, used for separate trees in multinomial classification
    Rcpp::StringVector result(1);
    std::vector<std::vector<std::vector<tree>>> *trees = tree_pnt;
    json j = get_forest_json_3D(*trees);
    result[0] = j.dump(4);
    return result;
}

// [[Rcpp::export]]
Rcpp::List json_to_r_3D(Rcpp::StringVector json_string_r)
{
    // push json to 3 dimensional matrix of trees, used for separate trees in multinomial classification
    std::vector<std::string> json_string(json_string_r.size());
    // std::string json_string = json_string_r(0);
    json_string[0] = json_string_r(0);

    // Create trees
    vector<vector<vector<tree>>> *trees2 = new std::vector<std::vector<vector<tree>>>();

    // Load
    from_json_to_forest_3D(json_string[0], *trees2);

    // Define External Pointer
    Rcpp::XPtr<std::vector<std::vector<std::vector<tree>>>> tree_pnt(trees2, true);

    return Rcpp::List::create(Rcpp::Named("model_list") = Rcpp::List::create(Rcpp::Named("tree_pnt") = tree_pnt));
}
