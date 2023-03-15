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
Rcpp::List XBCF_continuous_predict(mat X_con, mat X_mod, mat Z, Rcpp::XPtr<std::vector<std::vector<tree>>> tree_con, Rcpp::XPtr<std::vector<std::vector<tree>>> tree_mod)
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
Rcpp::List XBCF_discrete_predict(mat X_con, mat X_mod, mat Z, Rcpp::XPtr<std::vector<std::vector<tree>>> tree_con, Rcpp::XPtr<std::vector<std::vector<tree>>> tree_mod)
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
Rcpp::List XBCF_rd_predict(mat Xpred_con, mat Xpred_mod, mat Zpred, mat Xtr_con, mat Xtr_mod, mat Ztr,
                            Rcpp::XPtr<std::vector<std::vector<tree>>> tree_con, Rcpp::XPtr<std::vector<std::vector<tree>>> tree_mod,
                            Rcpp::NumericMatrix res_indicator_con, Rcpp::NumericMatrix valid_residuals_con, Rcpp::NumericMatrix resid_mean_con,
                            Rcpp::NumericMatrix res_indicator_mod, Rcpp::NumericMatrix valid_residuals_mod, Rcpp::NumericMatrix resid_mean_mod,
                            Rcpp::NumericMatrix sigma0, Rcpp::NumericMatrix sigma1,
                            double cutoff, double theta, double tau)
{
    // size of data
    size_t Ntr = Xtr_con.n_rows;
    size_t Npred = Xpred_con.n_rows;
    size_t p_con = Xpred_con.n_cols;
    size_t p_mod = Xpred_mod.n_cols;
    size_t p_z = Zpred.n_cols;
    assert(Xpred_con.n_rows == Xpred_mod.n_rows);
    assert(Xpred_con.n_cols == Xtr_con.n_cols);
    assert(Xpred_mod.n_cols == Xtr_mod.n_cols);


    // Init X_std matrix
    Rcpp::NumericMatrix Xpred_std_con(Npred, p_con);
    Rcpp::NumericMatrix Xpred_std_mod(Npred, p_mod);

    matrix<double> Ztest_std;
    ini_matrix(Ztest_std, Npred, p_z);

    for (size_t i = 0; i < Npred; i++)
    {
        for (size_t j = 0; j < p_con; j++)
        {
            Xpred_std_con(i, j) = Xpred_con(i, j);
        }

        for (size_t j = 0; j < p_mod; j++)
        {
            Xpred_std_mod(i, j) = Xpred_mod(i, j);
        }

        for (size_t j = 0; j < p_z; j++)
        {
            Ztest_std[j][i] = Zpred(i, j);
        }
    }

    double *Xpointer_con = &Xpred_std_con[0];
    double *Xpointer_mod = &Xpred_std_mod[0];

    // Trees
    std::vector<std::vector<tree>> *trees_con = tree_con;
    std::vector<std::vector<tree>> *trees_mod = tree_mod;

    // Result Container
    size_t num_sweeps = (*trees_con).size();
    size_t num_trees_con = (*trees_con)[0].size();
    size_t num_trees_mod = (*trees_mod)[0].size();

    COUT << "number of trees " << num_trees_con << " " << num_trees_mod << endl;

    matrix<double> prognostic_xinfo;
    ini_matrix(prognostic_xinfo, Npred, num_sweeps);

    matrix<double> treatment_xinfo;
    ini_matrix(treatment_xinfo, Npred, num_sweeps);

    matrix<double> yhats_test_xinfo;
    ini_xinfo(yhats_test_xinfo, Npred, num_sweeps);

    XBCFContinuousModel *model = new XBCFContinuousModel();
    // Predict

    model->predict_std(Ztest_std, Xpointer_con, Xpointer_mod, Npred, p_con, p_mod, num_trees_con, num_trees_mod, num_sweeps, yhats_test_xinfo, prognostic_xinfo, treatment_xinfo, *trees_con, *trees_mod);
    
    // get covariance matrix for predict

    // combine X matrix using only running variable
    mat X_con(Ntr + Npred, p_con);
    mat X_mod(Ntr + Npred, p_mod);

    // get X_range
    matrix<double> X_lim_con;
    matrix<double> X_lim_mod;
    ini_matrix(X_lim_con, 2, p_con);
    ini_matrix(X_lim_mod, 2, p_mod);
    std::vector<double> X_range_con(p_con);
    std::vector<double> X_range_mod(p_mod);

    for (size_t i = 0; i < Ntr; i++){
        for (size_t j = 0; j < p_con; j++){
            X_con(i, j) = Xtr_con(i, j);

            // check X limit for X_range
            if (X_con(i, j) < X_lim_con[j][0]){
                X_lim_con[j][0] = X_con(i, j);
            } else if (X_con(i, j) > X_lim_con[j][1]){
                X_lim_con[j][1] = X_con(i, j);
            }
        }
        for (size_t j = 0; j < p_mod; j++){
            X_mod(i, j) = Xtr_mod(i, j);

            // check X limit for X_range
            if (X_mod(i, j) < X_lim_mod[j][0]){
                X_lim_mod[j][0] = X_mod(i, j);
            } else if (X_mod(i, j) > X_lim_mod[j][1]){
                X_lim_mod[j][1] = X_mod(i, j);
            }
        }
    }
    for (size_t i = 0; i < Npred; i++){
        for (size_t j = 0; j < p_con; j++){
            X_con(i + Ntr, j) = Xpred_con(i, j);
        }
        for (size_t j = 0; j < p_mod; j++){
            X_mod(i + Ntr, j) = Xpred_mod(i, j);
        }
    }

    for (size_t j = 0; j < p_con; j++){
        X_range_con[j] = X_lim_con[j][1] - X_lim_con[j][0];
    }

    for (size_t j = 0; j < p_mod; j++){
        X_range_mod[j] = X_lim_mod[j][1] - X_lim_mod[j][0];
    }

    
    mat cov_con(Ntr + Npred, Ntr + Npred);
    mat cov_mod(Ntr + Npred, Ntr + Npred);

    get_rel_covariance(cov_con, X_con, X_range_con, theta, tau);
    get_rel_covariance(cov_mod, X_mod, X_range_mod, theta, tau);

    // get l2 distance on running variable to the cutoff for weighted residual mean 
    // (l2 is the same as absolute distance in 1d)
    mat resid_dist(Ntr, 1);
    for (size_t i = 0; i < Ntr; i++){
        // !! Assuming the last column is running variable
        resid_dist(i, 0) = abs(Xtr_con(i, p_con - 1) - cutoff);
    }

    for (size_t sweeps = 0; sweeps < num_sweeps; sweeps++)
    {
        for (size_t tree_ind = 0; tree_ind < num_trees_con; tree_ind++)
        {
            // get valid residuals for each tree
            // count valid residuals
            size_t this_tree = sweeps * num_trees_con  + tree_ind;
            size_t N_valid = 0;
            mat resid(Ntr, 1);
            double weighted_res = 0;
            double sum_weight = 0;
            for (size_t k = 0; k < Ntr; k++){
                if (res_indicator_con(this_tree, k) == 1){
                    resid(N_valid, 0) = valid_residuals_con(this_tree, k);

                    weighted_res += resid(N_valid, 0) * resid_dist(N_valid, 0);
                    sum_weight += resid_dist(N_valid, 0);

                    N_valid += 1;
                }
            }
            resid.resize(N_valid);
            weighted_res = weighted_res / sum_weight;

            // add sigma to covairnace diagnal for predict data
            for (size_t i = 0; i < Npred; i++)
            {
                if (Zpred(i, 0) == 0){
                    cov_con(i + Ntr, i + Ntr) += pow(sigma0(tree_ind, sweeps), 2) / num_trees_con;
                } else {
                    cov_con(i + Ntr, i + Ntr) += pow(sigma1(tree_ind, sweeps), 2) / num_trees_con;
                }
            }

            // mat cov(N + Ntest, N + Ntest);
            // get_rel_covariance(cov, X, x_range, theta, tau);
            // for (size_t i = 0; i < N; i++)
            // {
            //     cov(i, i) += pow(x_struct.sigma[tree_ind], 2) / x_struct.num_trees;
            // }

        //     mat mu(Ntest, 1);
        //     mat Sig(Ntest, Ntest);
        //     if (N > 0)
        //     {
        //         mat k = cov.submat(N, 0, N + Ntest - 1, N - 1);
        //         mat Kinv = pinv(cov.submat(0, 0, N - 1, N - 1));
        //         mu = k * Kinv * resid;
        //         Sig = cov.submat(N, N, N + Ntest - 1, N + Ntest - 1) - k * Kinv * trans(k);
        //     }
        //     else
        //     {
        //         // prior
        //         mu.zeros(Ntest, 1);
        //         Sig = cov.submat(0, 0, Ntest - 1, Ntest - 1);
        //     }
        //     mat U;
        //     vec S;
        //     mat V;
        //     svd(U, S, V, Sig);

        //     std::normal_distribution<double> normal_samp(0.0, 1.0);
        //     mat samp(Ntest, 1);
        //     for (size_t i = 0; i < Ntest; i++)
        //         samp(i, 0) = normal_samp(x_struct.gen);
        //     mat draws = mu + U * diagmat(sqrt(S)) * samp;
        //     for (size_t i = 0; i < Ntest; i++)
        //         yhats_test_xinfo[sweeps][test_ind[i]] += draws(i, 0);

            // remove sigma from covairnace diagnal for predict data
            for (size_t i = 0; i < Npred; i++)
            {
                if (Zpred(i, 0) == 0){
                    cov_con(i + Ntr, i + Ntr) -= pow(sigma0(tree_ind, sweeps), 2) / num_trees_con;
                } else {
                    cov_con(i + Ntr, i + Ntr) += pow(sigma1(tree_ind, sweeps), 2) / num_trees_con;
                }
            }
        }
        //     }
        //     for (size_t tree_ind = 0; tree_ind < num_trees_mod; tree_ind++)
        //     {
            
        //     }
    }

    // Convert back to Rcpp
    Rcpp::NumericMatrix yhats(Npred, num_sweeps);
    Rcpp::NumericMatrix prognostic(Npred, num_sweeps);
    Rcpp::NumericMatrix treatment(Npred, num_sweeps);
    for (size_t i = 0; i < Npred; i++)
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

// [[Rcpp::export]]
Rcpp::List xbart_heteroskedastic_predict(mat X,
                                         Rcpp::XPtr<std::vector<std::vector<tree>>> tree_m,
                                         Rcpp::XPtr<std::vector<std::vector<tree>>> tree_v)
{
    // size of data
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
    std::vector<std::vector<tree>> *trees_m = tree_m;
    std::vector<std::vector<tree>> *trees_v = tree_v;

    // Result Container
    size_t num_sweeps = (*trees_m).size();
    size_t num_trees_m = (*trees_m)[0].size();
    size_t num_trees_v = (*trees_v)[0].size();

    COUT << "number of trees " << num_trees_m << " " << num_trees_v << endl;

    matrix<double> mhats_test_xinfo;
    ini_matrix(mhats_test_xinfo, N, num_sweeps);

    matrix<double> vhats_test_xinfo;
    ini_matrix(vhats_test_xinfo, N, num_sweeps);

    hskNormalModel *model_m = new hskNormalModel();
    logNormalModel *model_v = new logNormalModel();

    // Predict
    model_m->predict_std(Xpointer, N, p, num_trees_m, num_sweeps, mhats_test_xinfo, *trees_m);
    model_v->predict_std(Xpointer, N, p, num_trees_v, num_sweeps, vhats_test_xinfo, *trees_v);

    // Convert back to Rcpp
    Rcpp::NumericMatrix mhats(N, num_sweeps);
    Rcpp::NumericMatrix vhats(N, num_sweeps);
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < num_sweeps; j++)
        {
            mhats(i, j) = mhats_test_xinfo[j][i];
            vhats(i, j) = 1.0 / vhats_test_xinfo[j][i];
        }
    }

    return Rcpp::List::create(Rcpp::Named("mhats") = mhats, Rcpp::Named("vhats") = vhats);
}