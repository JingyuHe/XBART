#include <ctime>
// #include <RcppArmadillo.h>
#include "tree.h"
#include <chrono>
#include "mcmc_loop.h"
#include "X_struct.h"
#include "utility_rcpp.h"
#include "json_io.h"

using namespace std;
using namespace chrono;
using namespace arma;

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]]
Rcpp::List XBCF_discrete_heterosk_vary_variance_cpp2(arma::mat y,
                                                     arma::mat Z,
                                                     arma::mat X_con,
                                                     arma::mat X_mod,
                                                     size_t num_trees_con,
                                                     size_t num_trees_mod,
                                                     size_t num_trees_v,
                                                     size_t num_sweeps,
                                                     size_t max_depth,
                                                     size_t max_depth_v,
                                                     size_t n_min,
                                                     size_t n_min_v,
                                                     size_t num_cutpoints,
                                                     size_t num_cutpoints_v,
                                                     double alpha_con,
                                                     double beta_con,
                                                     double alpha_mod,
                                                     double beta_mod,
                                                     double alpha_v,
                                                     double beta_v,
                                                     double tau_con,
                                                     double tau_mod,
                                                     double a_v,
                                                     double b_v,
                                                     double ini_var,
                                                     double no_split_penalty,
                                                     double no_split_penalty_v,
                                                     size_t burnin = 1,
                                                     size_t mtry_con = 0,
                                                     size_t mtry_mod = 0,
                                                     size_t mtry_v = 0,
                                                     size_t p_categorical_con = 0,
                                                     size_t p_categorical_mod = 0,
                                                     double kap = 16,
                                                     double s = 4,
                                                     double tau_con_kap = 3,
                                                     double tau_con_s = 0.5,
                                                     double tau_mod_kap = 3,
                                                     double tau_mod_s = 0.5,
                                                     bool a_scaling = true,
                                                     bool b_scaling = true,
                                                     bool verbose = false,
                                                     bool sampling_tau = true,
                                                     bool parallel = true,
                                                     bool set_random_seed = false,
                                                     size_t random_seed = 0,
                                                     bool sample_weights = true,
                                                     double nthread = 0)
{
    if (parallel)
    {
        thread_pool.start(nthread);
        COUT << "Running in parallel with " << nthread << " threads." << endl;
    }
    else
    {
        COUT << "Running with single thread." << endl;
    }

    size_t N = X_con.n_rows;

    // number of total variables
    size_t p_con = X_con.n_cols;
    size_t p_mod = X_mod.n_cols;

    COUT << "size of X_con " << X_con.n_rows << " " << X_con.n_cols << endl;
    COUT << "size of X_mod " << X_mod.n_rows << " " << X_mod.n_cols << endl;

    // number of basis functions (1 in the case of the OG bcf)
    size_t p_z = Z.n_cols;

    // number of continuous variables
    size_t p_continuous_con = p_con - p_categorical_con;
    size_t p_continuous_mod = p_mod - p_categorical_mod;

    // suppose first p_continuous variables are continuous, then categorical
    assert(mtry_con <= p_con);

    assert(mtry_mod <= p_mod);

    assert(burnin <= num_sweeps);

    if (mtry_con == 0)
    {
        mtry_con = p_con;
    }

    if (mtry_mod == 0)
    {
        mtry_mod = p_mod;
    }

    if (mtry_v == 0)
    {
        mtry_v = p_con;
    }

    if (mtry_con != p_con)
    {
        COUT << "Sample " << mtry_con << " out of " << p_con << " variables when grow each prognostic tree." << endl;
    }

    if (mtry_mod != p_mod)
    {
        COUT << "Sample " << mtry_mod << " out of " << p_mod << " variables when grow each treatment tree." << endl;
    }

    if (mtry_v != p_con)
    {
        COUT << "Sample " << mtry_v << " out of " << p_con << " variables when grow each variance tree." << endl;
    }

    arma::umat Xorder_con(X_con.n_rows, X_con.n_cols);
    matrix<size_t> Xorder_std_con;
    ini_matrix(Xorder_std_con, N, p_con);

    arma::umat Xorder_mod(X_mod.n_rows, X_mod.n_cols);
    matrix<size_t> Xorder_std_mod;
    ini_matrix(Xorder_std_mod, N, p_mod);

    cout << "size of Xorder con and mode " << Xorder_std_con.size() << " " << Xorder_std_con[0].size() << " " << Xorder_std_mod.size() << " " << Xorder_std_mod[0].size() << endl;

    std::vector<double> y_std(N);

    double y_mean = 0.0;

    size_t N_trt = 0;  // number of treated in training
    size_t N_ctrl = 0; // number of control in training

    for (size_t i = 0; i < N; i++)
    {
        y_mean += y[i];

        // count number of treated and control data
        if (Z[i] == 1)
        {
            N_trt++;
        }
        else
        {
            N_ctrl++;
        }
    }
    y_mean = y_mean / N;

    Rcpp::NumericMatrix X_std_con(N, p_con);
    Rcpp::NumericMatrix X_std_mod(N, p_mod);

    matrix<double> Z_std;
    ini_matrix(Z_std, N, p_z);

    rcpp_to_std2(y, Z, X_con, X_mod, y_std, y_mean, Z_std, X_std_con, X_std_mod, Xorder_std_con, Xorder_std_mod);

    ///////////////////////////////////////////////////////////////////

    double *Xpointer_con = &X_std_con[0];
    double *Xpointer_mod = &X_std_mod[0];

    matrix<double> sigma0_draw_xinfo;
    ini_matrix(sigma0_draw_xinfo, num_trees_con + num_trees_mod, num_sweeps);

    matrix<double> sigma1_draw_xinfo;
    ini_matrix(sigma1_draw_xinfo, num_trees_con + num_trees_mod, num_sweeps);

    matrix<double> a_xinfo;
    ini_matrix(a_xinfo, num_sweeps, 1);

    matrix<double> b_xinfo;
    ini_matrix(b_xinfo, num_sweeps, 2);

    // create trees
    vector<vector<tree>> trees_con(num_sweeps);
    vector<vector<tree>> trees_mod(num_sweeps);

    for (size_t i = 0; i < num_sweeps; i++)
    {
        trees_con[i].resize(num_trees_con);
        trees_mod[i].resize(num_trees_mod);
    }

    // define the mean model
    hskXBCFDiscreteModel *model = new hskXBCFDiscreteModel(kap, s, tau_con, tau_mod, alpha_con, beta_con, alpha_mod, beta_mod, sampling_tau, tau_con_kap, tau_con_s, tau_mod_kap, tau_mod_s);
    model->setNoSplitPenalty(no_split_penalty);

    // create trees for variance
    // treated and control group fits different trees
    vector<vector<tree>> trees_v_con(num_sweeps);
    vector<vector<tree>> trees_v_mod(num_sweeps);

    for (size_t i = 0; i < num_sweeps; i++)
    {
        trees_v_con[i].resize(num_trees_con);
        trees_v_mod[i].resize(num_trees_mod);
    }

    // define the variance model
    logNormalXBCFModel2 *model_v = new logNormalXBCFModel2(a_v, b_v, kap, s, 0, alpha_v, beta_v);
    model_v->setNoSplitPenalty(no_split_penalty_v);

    // initialize X_struct
    std::vector<double> initial_theta_v_con(1, exp(log(1.0 / ini_var) / (double)num_trees_con));
    X_struct x_struct_v_con(Xpointer_con, &y_std, N, Xorder_std_con, p_categorical_con, p_continuous_con, &initial_theta_v_con, num_trees_con);

    std::vector<double> initial_theta_v_mod(1, exp(log(1.0 / ini_var) / (double)num_trees_mod));
    X_struct x_struct_v_mod(Xpointer_con, &y_std, N, Xorder_std_con, p_categorical_con, p_continuous_con, &initial_theta_v_mod, num_trees_mod);

    // State settings
    std::vector<double> sigma_vec(N, 1.0);
    XBCFdiscreteHeteroskedasticState2 state(&Z_std, Xpointer_con, Xpointer_mod,
                                            Xorder_std_con, Xorder_std_mod,
                                            N, p_con, p_mod, num_trees_con,
                                            num_trees_mod, num_trees_v,
                                            p_categorical_con, p_categorical_mod,
                                            p_continuous_con, p_continuous_mod,
                                            set_random_seed, random_seed,
                                            n_min, n_min_v,
                                            num_cutpoints, num_cutpoints_v,
                                            mtry_con, mtry_mod, mtry_v,
                                            num_sweeps, sample_weights,
                                            &y_std, 1.0, max_depth, max_depth_v,
                                            y_mean, burnin,
                                            model->dim_residual, nthread, parallel,
                                            a_scaling, b_scaling, N_trt, N_ctrl, sigma_vec);

    // initialize X_struct for mean trees
    std::vector<double> initial_theta_con(1, 0);
    X_struct x_struct_con(Xpointer_con, &y_std, N, Xorder_std_con, p_categorical_con, p_continuous_con, &initial_theta_con, num_trees_con);

    std::vector<double> initial_theta_mod(1, y_mean / (double)num_trees_mod);
    X_struct x_struct_mod(Xpointer_mod, &y_std, N, Xorder_std_mod, p_categorical_mod, p_continuous_mod, &initial_theta_mod, num_trees_mod);

    ////////////////////////////////////////////////////////////////
    mcmc_loop_xbcf_discrete_heteroskedastic_vary_variance2(Xorder_std_con, Xorder_std_mod,
                                                           verbose, sigma0_draw_xinfo, sigma1_draw_xinfo,
                                                           a_xinfo, b_xinfo,
                                                           trees_con, trees_mod, trees_v_con, trees_v_mod,
                                                           no_split_penalty, state, model, model_v,
                                                           x_struct_con, x_struct_mod, x_struct_v_con, x_struct_v_mod);

    cout << "finish mcmc loop" << endl;

    // R Objects to Return
    Rcpp::NumericMatrix sigma0_draw(num_trees_con + num_trees_mod, num_sweeps); // save predictions of each tree

    Rcpp::NumericMatrix sigma1_draw(num_trees_con + num_trees_mod, num_sweeps); // save predictions of each tree

    Rcpp::NumericMatrix a_draw(num_sweeps, 1);

    Rcpp::NumericMatrix b_draw(num_sweeps, 2);

    Rcpp::NumericVector split_count_sum_con(p_con, 0); // split counts

    Rcpp::NumericVector split_count_sum_mod(p_mod, 0);

    Rcpp::NumericVector split_count_sum_v_con(p_con, 0);

    Rcpp::NumericVector split_count_sum_v_mod(p_con, 0);

    // copy from std vector to Rcpp Numeric Matrix objects
    Matrix_to_NumericMatrix(sigma0_draw_xinfo, sigma0_draw);
    Matrix_to_NumericMatrix(sigma1_draw_xinfo, sigma1_draw);
    Matrix_to_NumericMatrix(a_xinfo, a_draw);
    Matrix_to_NumericMatrix(b_xinfo, b_draw);

    for (size_t i = 0; i < p_con; i++)
    {
        split_count_sum_con(i) = (int)(*state.split_count_all_con)[i];
    }

    for (size_t i = 0; i < p_mod; i++)
    {
        split_count_sum_mod(i) = (int)(*state.split_count_all_mod)[i];
    }

    for (size_t i = 0; i < p_con; i++)
    {
        split_count_sum_v_con(i) = (int)(*state.split_count_all_v_con)[i];
    }

    for (size_t i = 0; i < p_mod; i++)
    {
        split_count_sum_v_mod(i) = (int)(*state.split_count_all_v_mod)[i];
    }

    // print out tree structure, for usage of BART warm-start
    Rcpp::StringVector output_tree_con(num_sweeps);
    Rcpp::StringVector output_tree_mod(num_sweeps);
    Rcpp::StringVector output_tree_v_con(num_sweeps);
    Rcpp::StringVector output_tree_v_mod(num_sweeps);

    tree_to_string(trees_mod, output_tree_mod, num_sweeps, num_trees_mod, p_mod);
    tree_to_string(trees_con, output_tree_con, num_sweeps, num_trees_con, p_con);
    tree_to_string(trees_v_mod, output_tree_v_mod, num_sweeps, num_trees_mod, p_mod);
    tree_to_string(trees_v_con, output_tree_v_con, num_sweeps, num_trees_con, p_con);

    Rcpp::StringVector tree_json_mod(1);
    Rcpp::StringVector tree_json_con(1);
    Rcpp::StringVector tree_json_v_con(1);
    Rcpp::StringVector tree_json_v_mod(1);

    json j = get_forest_json(trees_mod, y_mean);
    json j2 = get_forest_json(trees_con, y_mean);
    json j3 = get_forest_json(trees_v_con, y_mean);
    json j4 = get_forest_json(trees_v_mod, y_mean);

    tree_json_mod[0] = j.dump(4);
    tree_json_con[0] = j2.dump(4);
    tree_json_v_con[0] = j3.dump(4);
    tree_json_v_mod[0] = j4.dump(4);

    thread_pool.stop();

    return Rcpp::List::create(
        Rcpp::Named("a") = a_draw,
        Rcpp::Named("b") = b_draw,
        Rcpp::Named("sigma0") = sigma0_draw,
        Rcpp::Named("sigma1") = sigma1_draw,
        Rcpp::Named("importance_prognostic") = split_count_sum_con,
        Rcpp::Named("importance_treatment") = split_count_sum_mod,
        Rcpp::Named("importance_precision_prognostic") = split_count_sum_v_con,
        Rcpp::Named("importance_precision_treatment") = split_count_sum_v_mod,
        Rcpp::Named("model_list") = Rcpp::List::create(Rcpp::Named("y_mean") = y_mean, Rcpp::Named("p_con") = p_con, Rcpp::Named("p_mod") = p_mod),
        Rcpp::Named("tree_json_mod") = tree_json_mod,
        Rcpp::Named("tree_json_con") = tree_json_con,
        Rcpp::Named("tree_json_v_con") = tree_json_v_con,
        Rcpp::Named("tree_json_v_mod") = tree_json_v_mod,
        Rcpp::Named("tree_string_mod") = output_tree_mod,
        Rcpp::Named("tree_string_con") = output_tree_con,
        Rcpp::Named("tree_string_v_con") = output_tree_v_con,
        Rcpp::Named("tree_string_v_mod") = output_tree_v_mod);
}
