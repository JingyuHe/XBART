#include "fit_std_main_loop.h"

void fit_std_main_loop(const double *Xpointer, std::vector<double> &y_std, double &y_mean,
                       const double *Xtestpointer, xinfo_sizet &Xorder_std,
                       size_t N, size_t p, size_t N_test,
                       size_t M, size_t L, size_t N_sweeps, xinfo_sizet &max_depth_std,
                       size_t Nmin, size_t Ncutpoints, double alpha, double beta,
                       double tau, size_t burnin, size_t mtry,
                       bool draw_sigma, double kap, double s,
                       bool verbose, bool m_update_sigma,
                       bool draw_mu, bool parallel,
                       xinfo &yhats_xinfo, xinfo &yhats_test_xinfo, xinfo &sigma_draw_xinfo)
{
    // //Set Random Generator
    // std::default_random_engine(generator);

    // std::vector<std::vector<double>> predictions_std;
    // ini_xinfo(predictions_std, N, M);

    // xinfo predictions_test_std;
    // ini_xinfo(predictions_test_std, N_test, M);

    // std::vector<double> yhat_std(N);
    // row_sum(predictions_std, yhat_std);
    // std::vector<double> yhat_test_std(N_test);
    // row_sum(predictions_test_std, yhat_test_std);

    // xinfo sigma_draw_std;
    // ini_xinfo(sigma_draw_std, M, N_sweeps);

    // xinfo split_count_all_tree;  // initialize at 0
    // ini_xinfo(split_count_all_tree, p, M);

    // // split_count_all_tree = split_count_all_tree + 1; // initialize at 1
    // std::vector<double> split_count_current_tree(p, 1.0);
    // std::vector<double> mtry_weight_current_tree(p, 1.0);

    // // Rcpp::NumericVector split_count_current_tree(p, 1);
    // // Rcpp::NumericVector mtry_weight_current_tree(p, 1);

    // double sigma;
    // // double tau;
    // forest trees(M);
    // std::vector<double> prob(2, 0.5);
    // std::random_device rd;
    // std::mt19937 gen(rd());
    // std::discrete_distribution<> d(prob.begin(), prob.end());
    // // // sample one index of split point
    // size_t prune;

    // // current residual
    // std::vector<double> residual_std(N);

    // // std::vector<double> split_var_count(p);
    // // std::fill(split_var_count.begin(), split_var_count.end(), 1);
    // // Rcpp::NumericVector split_var_count(p, 1);

    // // double *split_var_count_pointer = &split_var_count[0];

    // // in the burnin samples, use all variables
    // std::vector<size_t> subset_vars(p);
    // std::iota(subset_vars.begin() + 1, subset_vars.end(), 1);

    // // cout << subset_vars << endl;

    // double run_time = 0.0;

    // // save tree objects to strings
    // // std::stringstream treess;
    // // treess.precision(10);
    // // treess << L << " " << M << " " << p << endl;

    // // L, number of samples
    // // M, number of trees

    // bool use_all = true;

    // for (size_t mc = 0; mc < L; mc++)
    // {

    //     // initialize predcitions and predictions_test
    //     for (size_t ii = 0; ii < M; ii++)
    //     {
    //         std::fill(predictions_std[ii].begin(), predictions_std[ii].end(), y_mean / (double)M);
    //         std::fill(predictions_test_std[ii].begin(), predictions_test_std[ii].end(), y_mean / (double)M);
    //     }

    //     row_sum(predictions_std, yhat_std);
    //     row_sum(predictions_test_std, yhat_test_std);

    //     residual_std = y_std - yhat_std;

    //     for (size_t sweeps = 0; sweeps < N_sweeps; sweeps++)
    //     {

    //         if (verbose == true)
    //         {
    //             // cout << "--------------------------------" << endl;
    //             // cout << "number of sweeps " << sweeps << endl;
    //             // cout << "--------------------------------" << endl;
    //         }

    //         for (size_t tree_ind = 0; tree_ind < M; tree_ind++)
    //         {

    //             // if update sigma based on residual of all m trees
    //             if (m_update_sigma == true)
    //             {

    //                 std::gamma_distribution<double> gamma_samp((N + kap) / 2.0, 2.0 / (sum_squared(residual_std) + s));

    //                 sigma = 1.0 / sqrt(gamma_samp(generator));

    //                 sigma_draw_xinfo[sweeps][tree_ind] = sigma;
    //             }

    //             // save sigma
    //             sigma_draw_xinfo[sweeps][tree_ind] = sigma;

    //             // add prediction of current tree back to residual
    //             // then it's m - 1 trees residual

    //             residual_std = residual_std + predictions_std[tree_ind];

    //             // do the samething for residual_theta_noise, residual of m - 1 trees

    //             yhat_std = yhat_std - predictions_std[tree_ind];

    //             yhat_test_std = yhat_test_std - predictions_test_std[tree_ind];

    //             if (use_all && (sweeps > burnin) && (mtry != p))
    //             {
    //                 // subset_vars = Rcpp::as<std::vector<size_t>>(sample(var_index_candidate, mtry, false, split_var_count));
    //                 use_all = false;
    //             }

    //             // cout << "variables used " << subset_vars << endl;
    //             // cout << "------------------" << endl;

    //             // clear counts of splits for one tree
    //             std::fill(split_count_current_tree.begin(), split_count_current_tree.end(), 0.0);

    //             for(int i=0;i<p;i++){
    //                 mtry_weight_current_tree[i] = mtry_weight_current_tree[i] - split_count_all_tree[tree_ind][i];
    //             }
    //             // mtry_weight_current_tree = mtry_weight_current_tree - split_count_all_tree(Rcpp::_, tree_ind);

    //             // cout << "before " << mtry_weight_current_tree << endl;

    //             trees.t[tree_ind].grow_tree_adaptive_XBART_train(sum_vec(residual_std) / (double)N, 0, max_depth_std[sweeps][tree_ind], Nmin, Ncutpoints, tau, sigma, alpha, beta, draw_sigma, draw_mu, parallel, residual_std, Xorder_std, Xpointer, mtry, run_time, use_all, mtry_weight_current_tree, split_count_current_tree);

    //             mtry_weight_current_tree = mtry_weight_current_tree + split_count_current_tree;

    //             // cout << "after " << mtry_weight_current_tree << endl;

    //             for(int i = 0;i<p;i++){
    //                 split_count_all_tree[tree_ind][i] = split_count_current_tree[i];
    //             }
    //             // split_count_all_tree(Rcpp::_, tree_ind) = split_count_current_tree;

    //             if (verbose == true)
    //             {
    //                 // cout << "tree " << tree_ind << " size is " << trees.t[tree_ind].treesize() << endl;
    //             }

    //             // update prediction of current tree
    //             fit_new_std(trees.t[tree_ind], Xpointer, N, p, predictions_std[tree_ind]);

    //             // update prediction of current tree, test set
    //             fit_new_std(trees.t[tree_ind], Xtestpointer, N_test, p, predictions_test_std[tree_ind]);

    //             // update sigma based on residual of m - 1 trees, residual_theta_noise
    //             if (m_update_sigma == false)
    //             {

    //                 std::gamma_distribution<double> gamma_samp((N + kap) / 2.0, 2.0 / (sum_squared(residual_std) + s));

    //                 sigma = 1.0 / sqrt(gamma_samp(generator));

    //                 sigma_draw_xinfo[sweeps][tree_ind] = sigma;
    //             }

    //             // update residual, now it's residual of m trees

    //             residual_std = residual_std - predictions_std[tree_ind];

    //             yhat_std = yhat_std + predictions_std[tree_ind];
    //             yhat_test_std = yhat_test_std + predictions_test_std[tree_ind];

    //             // treess << trees.t[tree_ind];
    //         }

    //         // save predictions to output matrix
    //         yhats_xinfo[sweeps] = yhat_std;
    //         yhats_test_xinfo[sweeps] = yhat_test_std;

    //     }
    // }
}

void fit_std_main_loop_all(const double *Xpointer, std::vector<double> &y_std, double &y_mean, const double *Xtestpointer, xinfo_sizet &Xorder_std,
                           size_t N, size_t p, size_t N_test,
                           size_t M, size_t L, size_t N_sweeps, xinfo_sizet &max_depth_std,
                           size_t Nmin, size_t Ncutpoints, double alpha, double beta,
                           double tau, size_t burnin, size_t mtry,
                           bool draw_sigma, double kap, double s,
                           bool verbose, bool m_update_sigma,
                           bool draw_mu, bool parallel,
                           xinfo &yhats_xinfo, xinfo &yhats_test_xinfo,
                           xinfo &sigma_draw_xinfo, xinfo &split_count_all_tree,
                           size_t p_categorical, size_t p_continuous, vector<vector<tree>> &trees, bool set_random_seed, size_t random_seed)
{

    fit_std(Xpointer, y_std, y_mean, Xorder_std,
            N, p,
            M, L, N_sweeps, max_depth_std,
            Nmin, Ncutpoints, alpha, beta,
            tau, burnin, mtry,
            draw_sigma, kap, s,
            verbose, m_update_sigma,
            draw_mu, parallel,
            yhats_xinfo, sigma_draw_xinfo,
            p_categorical, p_continuous, trees, set_random_seed, random_seed);

    predict_std(Xtestpointer, N_test, p, M, L,
                N_sweeps, yhats_test_xinfo, trees, y_mean);
    // bool categorical_variables = false;
    // if(p_categorical > 0){
    //     categorical_variables = true;
    // }

    // std::default_random_engine(generator);

    // //std::vector<size_t> X_values;
    // std::vector<double> X_values;
    // std::vector<size_t> X_counts;
    // std::vector<size_t> variable_ind(p_categorical + 1);

    // size_t total_points;

    // std::vector<size_t> X_num_unique(p_categorical);

    // unique_value_count2(Xpointer, Xorder_std, X_values, X_counts,
    //     variable_ind, total_points, X_num_unique, p_categorical, p_continuous);

    // // save predictions of each tree
    // std::vector<std::vector<double>> predictions_std;
    // ini_xinfo(predictions_std, N, M);

    // xinfo predictions_test_std;
    // ini_xinfo(predictions_test_std, N_test, M);

    // std::vector<double> yhat_std(N);
    // row_sum(predictions_std, yhat_std);
    // std::vector<double> yhat_test_std(N_test);
    // row_sum(predictions_test_std, yhat_test_std);

    // // current residual
    // std::vector<double> residual_std(N);

    // ///////////////////////////////////////////////////////////////////

    // // Rcpp::NumericMatrix yhats(N, N_sweeps);
    // // Rcpp::NumericMatrix yhats_test(N_test, N_sweeps);

    // // // save predictions of each tree
    // // Rcpp::NumericMatrix sigma_draw(M, N_sweeps);

    // double sigma;
    // // double tau;
    // // forest trees(M);
    // std::vector<double> prob(2, 0.5);
    // std::random_device rd;
    // std::mt19937 gen(rd());
    // std::discrete_distribution<> d(prob.begin(), prob.end());
    // // // sample one index of split point
    // size_t prune;

    // // std::vector<double> split_var_count(p);
    // // std::fill(split_var_count.begin(), split_var_count.end(), 1);
    // // Rcpp::NumericVector split_var_count(p, 1);

    // // xinfo split_count_all_tree;
    // // ini_xinfo(split_count_all_tree, p, M); // initialize at 0

    // // split_count_all_tree = split_count_all_tree + 1; // initialize at 1
    // std::vector<double> split_count_current_tree(p, 1);
    // std::vector<double> mtry_weight_current_tree(p, 1);

    // // double *split_var_count_pointer = &split_var_count[0];

    // // in the burnin samples, use all variables
    // std::vector<size_t> subset_vars(p);
    // std::iota(subset_vars.begin() + 1, subset_vars.end(), 1);

    // double run_time = 0.0;

    // // save tree objects to strings
    // // std::stringstream treess;
    // // treess.precision(10);
    // // treess << L << " " << M << " " << p << endl;

    // // L, number of samples
    // // M, number of trees

    // bool use_all = true;

    // NormalModel model;

    // for (size_t mc = 0; mc < L; mc++)
    // {

    //     // initialize predcitions and predictions_test
    //     for (size_t ii = 0; ii < M; ii++)
    //     {
    //         std::fill(predictions_std[ii].begin(), predictions_std[ii].end(), y_mean / (double)M);
    //         std::fill(predictions_test_std[ii].begin(), predictions_test_std[ii].end(), y_mean / (double)M);
    //     }

    //     row_sum(predictions_std, yhat_std);
    //     row_sum(predictions_test_std, yhat_test_std);

    //     residual_std = y_std - yhat_std;

    //     for (size_t sweeps = 0; sweeps < N_sweeps; sweeps++)
    //     {

    //         if (verbose == true)
    //         {
    //             cout << "--------------------------------" << endl;
    //             cout << "number of sweeps " << sweeps << endl;
    //             cout << "--------------------------------" << endl;
    //         }

    //         for (size_t tree_ind = 0; tree_ind < M; tree_ind++)
    //         {

    //             // if update sigma based on residual of all m trees
    //             if (m_update_sigma == true)
    //             {

    //                 std::gamma_distribution<double> gamma_samp((N + kap) / 2.0, 2.0 / (sum_squared(residual_std) + s));

    //                 sigma = 1.0 / sqrt(gamma_samp(generator));

    //                 sigma_draw_xinfo[sweeps][tree_ind] = sigma;
    //             }

    //             // save sigma
    //             sigma_draw_xinfo[sweeps][tree_ind] = sigma;

    //             // add prediction of current tree back to residual
    //             // then it's m - 1 trees residual

    //             residual_std = residual_std + predictions_std[tree_ind];

    //             // do the samething for residual_theta_noise, residual of m - 1 trees

    //             yhat_std = yhat_std - predictions_std[tree_ind];

    //             yhat_test_std = yhat_test_std - predictions_test_std[tree_ind];

    //             if (use_all && (sweeps > burnin) && (mtry != p))
    //             {
    //                 // subset_vars = Rcpp::as<std::vector<size_t>>(sample(var_index_candidate, mtry, false, split_var_count));

    //                 // subset_vars = sample_int_crank2(p, mtry, split_var_count);

    //                 use_all = false;
    //             }

    //             // cout << "variables used " << subset_vars << endl;
    //             // cout << "------------------" << endl;

    //             // clear counts of splits for one tree
    //             std::fill(split_count_current_tree.begin(), split_count_current_tree.end(), 0.0);

    //             mtry_weight_current_tree = mtry_weight_current_tree - split_count_all_tree[tree_ind];

    //             // cout << "before " << mtry_weight_current_tree << endl;

    //             // trees.t[tree_ind].grow_tree_adaptive_std_all(sum_vec(residual_std) / (double)N, 0, max_depth_std[sweeps][tree_ind], Nmin, Ncutpoints, tau, sigma, alpha, beta, draw_sigma, draw_mu, parallel, residual_std, Xorder_std, Xpointer, mtry, use_all, split_count_all_tree, mtry_weight_current_tree, split_count_current_tree, categorical_variables, p_categorical, p_continuous,
    //             //     X_values, X_counts, variable_ind, X_num_unique);

    //             trees[sweeps][tree_ind].grow_tree_adaptive_std_all(sum_vec(residual_std) / (double)N, 0, max_depth_std[sweeps][tree_ind], Nmin, Ncutpoints, tau, sigma, alpha, beta, draw_sigma, draw_mu, parallel, residual_std, Xorder_std, Xpointer, mtry, use_all, split_count_all_tree, mtry_weight_current_tree, split_count_current_tree, categorical_variables, p_categorical, p_continuous,
    //                 X_values, X_counts, variable_ind, X_num_unique, &model);

    //             mtry_weight_current_tree = mtry_weight_current_tree + split_count_current_tree;

    //             // cout << "after " << mtry_weight_current_tree << endl;

    //             split_count_all_tree[tree_ind] = split_count_current_tree;

    //             if (verbose == true)
    //             {
    //                 //cout << "tree " << tree_ind << " size is " << trees.t[tree_ind].treesize() << endl;
    //                 cout << "tree " << tree_ind << " size is " << trees[sweeps][tree_ind].treesize() << endl;

    //             }

    //             // // update prediction of current tree
    //             // fit_new_std(trees.t[tree_ind], Xpointer, N, p, predictions_std[tree_ind]);

    //             // // update prediction of current tree, test set
    //             // fit_new_std(trees.t[tree_ind], Xtestpointer, N_test, p, predictions_test_std[tree_ind]);

    //               // update prediction of current tree
    //             fit_new_std(trees[sweeps][tree_ind], Xpointer, N, p, predictions_std[tree_ind]);

    //             // update prediction of current tree, test set
    //             fit_new_std(trees[sweeps][tree_ind], Xtestpointer, N_test, p, predictions_test_std[tree_ind]);

    //             // update sigma based on residual of m - 1 trees, residual_theta_noise
    //             if (m_update_sigma == false)
    //             {

    //                 std::gamma_distribution<double> gamma_samp((N + kap) / 2.0, 2.0 / (sum_squared(residual_std) + s));

    //                 sigma = 1.0 / sqrt(gamma_samp(generator));

    //                 sigma_draw_xinfo[sweeps][tree_ind] = sigma;
    //             }

    //             // update residual, now it's residual of m trees

    //             residual_std = residual_std - predictions_std[tree_ind];

    //             yhat_std = yhat_std + predictions_std[tree_ind];
    //             yhat_test_std = yhat_test_std + predictions_test_std[tree_ind];

    //             // treess << trees.t[tree_ind];
    //         }

    //         // save predictions to output matrix
    //         // save predictions to output matrix
    //         yhats_xinfo[sweeps] = yhat_std;
    //         yhats_test_xinfo[sweeps] = yhat_test_std;

    //         // for (size_t kk = 0; kk < N; kk++)
    //         // {
    //         //     yhats(kk, sweeps) = yhat_std[kk];
    //         // }
    //         // for (size_t kk = 0; kk < N_test; kk++)
    //         // {
    //         //     yhats_test(kk, sweeps) = yhat_test_std[kk];
    //         // }
    //     }
    // }
}

void fit_std(const double *Xpointer, std::vector<double> &y_std, double y_mean, xinfo_sizet &Xorder_std,
             size_t N, size_t p,
             size_t M, size_t L, size_t N_sweeps, xinfo_sizet &max_depth_std,
             size_t Nmin, size_t Ncutpoints, double alpha, double beta,
             double tau, size_t burnin, size_t mtry,
             bool draw_sigma, double kap, double s,
             bool verbose, bool m_update_sigma,
             bool draw_mu, bool parallel,
             xinfo &yhats_xinfo, xinfo &sigma_draw_xinfo,
             size_t p_categorical, size_t p_continuous, vector<vector<tree>> &trees, bool set_random_seed, size_t random_seed)
{
    bool categorical_variables = false;
    if (p_categorical > 0)
    {
        categorical_variables = true;
    }

    // std::default_random_engine(generator);

    //std::vector<size_t> X_values;
    std::vector<double> X_values;
    std::vector<size_t> X_counts;
    std::vector<size_t> variable_ind(p_categorical + 1);

    size_t total_points;

    std::vector<size_t> X_num_unique(p_categorical);

    if (parallel)
        thread_pool.start();

    unique_value_count2(Xpointer, Xorder_std, X_values, X_counts,
                        variable_ind, total_points, X_num_unique, p_categorical, p_continuous);

    // cout << "X_values" << X_values << endl;
    // cout << "X_counts" << X_counts << endl;
    // cout << "variable_ind " << variable_ind << endl;
    // cout << "X_num_unique " << X_num_unique << endl;

    NormalModel model;
    model.suff_stat_init();

    // save predictions of each tree
    xinfo predictions_std;
    ini_xinfo(predictions_std, N, M);

    std::vector<double> yhat_std(N);
    row_sum(predictions_std, yhat_std);

    // current residual
    std::vector<double> residual_std(N);

    ///////////////////////////////////////////////////////////////////

    // Rcpp::NumericMatrix yhats(N, N_sweeps);
    // Rcpp::NumericMatrix yhats_test(N_test, N_sweeps);

    // // save predictions of each tree
    // Rcpp::NumericMatrix sigma_draw(M, N_sweeps);

    double sigma;
    // double tau;
    //forest trees(M);
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

    // std::vector<double> split_var_count(p);
    // std::fill(split_var_count.begin(), split_var_count.end(), 1);
    // Rcpp::NumericVector split_var_count(p, 1);

    xinfo split_count_all_tree;
    ini_xinfo(split_count_all_tree, p, M); // initialize at 0
    // split_count_all_tree = split_count_all_tree + 1; // initialize at 1
    std::vector<double> split_count_current_tree(p, 1);
    std::vector<double> mtry_weight_current_tree(p, 1);

    // in the burnin samples, use all variables
    std::vector<size_t> subset_vars(p);
    std::iota(subset_vars.begin() + 1, subset_vars.end(), 1);

    double run_time = 0.0;

    // save tree objects to strings
    // std::stringstream treess;
    // treess.precision(10);
    // treess << L << " " << M << " " << p << endl;

    // L, number of samples
    // M, number of trees

    matrix<tree::tree_p> data_pointers;
    ini_matrix(data_pointers, N, M);

    bool use_all = true;
    for (size_t mc = 0; mc < L; mc++)
    {

        // initialize predcitions and predictions_test
        for (size_t ii = 0; ii < M; ii++)
        {
            std::fill(predictions_std[ii].begin(), predictions_std[ii].end(), y_mean / (double)M);
        }

        // Set yhat_std to mean 
        row_sum(predictions_std, yhat_std);

        // Residual for 0th tree 
        residual_std = y_std - yhat_std + predictions_std[0];

        for (size_t sweeps = 0; sweeps < N_sweeps; sweeps++)
        {

            if (verbose == true)
            {
                cout << "--------------------------------" << endl;
                cout << "number of sweeps " << sweeps << endl;
                cout << "--------------------------------" << endl;
            }

            for (size_t tree_ind = 0; tree_ind < M; tree_ind++)
            {

                // if update sigma based on residual of all m trees
                if (m_update_sigma == true)
                {

                    std::vector<double> residual_std_full = residual_std - predictions_std[tree_ind];

                    std::gamma_distribution<double> gamma_samp((N + kap) / 2.0, 2.0 / (sum_squared(residual_std_full) + s));

                    sigma = 1.0 / sqrt(gamma_samp(gen));

                    sigma_draw_xinfo[sweeps][tree_ind] = sigma;
                }

                // save sigma
                sigma_draw_xinfo[sweeps][tree_ind] = sigma;

                // add prediction of current tree back to residual
                // then it's m - 1 trees residual

                // do the samething for residual_theta_noise, residual of m - 1 trees

                yhat_std = yhat_std - predictions_std[tree_ind];

                if (use_all && (sweeps > burnin) && (mtry != p))
                {
                    // subset_vars = Rcpp::as<std::vector<size_t>>(sample(var_index_candidate, mtry, false, split_var_count));

                    // subset_vars = sample_int_crank2(p, mtry, split_var_count);

                    use_all = false;
                }

                // cout << "variables used " << subset_vars << endl;
                // cout << "------------------" << endl;

                // clear counts of splits for one tree
                std::fill(split_count_current_tree.begin(), split_count_current_tree.end(), 0.0);

                mtry_weight_current_tree = mtry_weight_current_tree - split_count_all_tree[tree_ind];

                trees[sweeps][tree_ind].grow_tree_adaptive_std_all(sum_vec(residual_std) / (double)N, 0, max_depth_std[sweeps][tree_ind], Nmin, Ncutpoints, tau, sigma, alpha, beta, draw_sigma, draw_mu, parallel, residual_std, Xorder_std, Xpointer, mtry, use_all, split_count_all_tree, mtry_weight_current_tree, split_count_current_tree, categorical_variables, p_categorical, p_continuous, X_values, X_counts, variable_ind, X_num_unique, &model, data_pointers, tree_ind, gen);

                mtry_weight_current_tree = mtry_weight_current_tree + split_count_current_tree;

                split_count_all_tree[tree_ind] = split_count_current_tree;

                // update prediction of current tree

                fit_new_std(trees[sweeps][tree_ind], Xpointer, N, p, predictions_std[tree_ind]);

                // update prediction of current tree, test set

                // update sigma based on residual of m - 1 trees, residual_theta_noise
                if (m_update_sigma == false)
                {

                    
                    std::gamma_distribution<double> gamma_samp((N + kap) / 2.0, 2.0 / (sum_squared(residual_std) + s));

                    sigma = 1.0 / sqrt(gamma_samp(gen));

                    sigma_draw_xinfo[sweeps][tree_ind] = sigma;
                }

                // update residual, now it's residual of m trees
                model.updateResidual(predictions_std, tree_ind, M, residual_std);

                yhat_std = yhat_std + predictions_std[tree_ind];
            }
            // save predictions to output matrix
            yhats_xinfo[sweeps] = yhat_std;

            // for (size_t kk = 0; kk < N; kk++)
            // {
            //     yhats(kk, sweeps) = yhat_std[kk];
            // }
            // for (size_t kk = 0; kk < N_test; kk++)
            // {
            //     yhats_test(kk, sweeps) = yhat_test_std[kk];
            // }
        }
    }
    thread_pool.stop();
}

void predict_std(const double *Xtestpointer, size_t N_test, size_t p, size_t M, size_t L,
                 size_t N_sweeps, xinfo &yhats_test_xinfo, vector<vector<tree>> &trees, double y_mean)
{

    xinfo predictions_test_std;
    ini_xinfo(predictions_test_std, N_test, M);

    std::vector<double> yhat_test_std(N_test);
    row_sum(predictions_test_std, yhat_test_std);

    for (size_t mc = 0; mc < L; mc++)
    {
        // initialize predcitions and predictions_test
        for (size_t ii = 0; ii < M; ii++)
        {
            std::fill(predictions_test_std[ii].begin(), predictions_test_std[ii].end(), y_mean / (double)M);
        }
        row_sum(predictions_test_std, yhat_test_std);


        for (size_t sweeps = 0; sweeps < N_sweeps; sweeps++)
        {
            for (size_t tree_ind = 0; tree_ind < M; tree_ind++)
            {
                
                yhat_test_std = yhat_test_std - predictions_test_std[tree_ind];
                fit_new_std(trees[sweeps][tree_ind], Xtestpointer, N_test, p, predictions_test_std[tree_ind]);
                yhat_test_std = yhat_test_std + predictions_test_std[tree_ind];


            }
            yhats_test_xinfo[sweeps] = yhat_test_std;

        }
    }
}


void fit_std_poisson_classification(const double *Xpointer, std::vector<double> &y_std, double &y_mean, const double *Xtestpointer, xinfo_sizet &Xorder_std, size_t N, size_t p, size_t N_test, size_t M, size_t L, size_t N_sweeps, xinfo_sizet &max_depth_std, size_t Nmin, size_t Ncutpoints, double alpha, double beta, double tau, size_t burnin, size_t mtry, bool draw_sigma, double kap, double s, bool verbose, bool m_update_sigma, bool draw_mu, bool parallel, xinfo &yhats_xinfo, xinfo &yhats_test_xinfo, xinfo &sigma_draw_xinfo, xinfo &split_count_all_tree, size_t p_categorical, size_t p_continuous, vector<vector<tree>> &trees, bool set_random_seed, size_t random_seed,double a, double b)
{
    bool categorical_variables = false;
    if (p_categorical > 0)
    {
        categorical_variables = true;
    }

    // std::default_random_engine(generator);

    //std::vector<size_t> X_values;
    std::vector<double> X_values;
    std::vector<size_t> X_counts;
    std::vector<size_t> variable_ind(p_categorical + 1);

    size_t total_points;

    std::vector<size_t> X_num_unique(p_categorical);

    if (parallel)
        thread_pool.start();

    unique_value_count2(Xpointer, Xorder_std, X_values, X_counts,
                        variable_ind, total_points, X_num_unique, p_categorical, p_continuous);

    // cout << "X_values" << X_values << endl;
    // cout << "X_counts" << X_counts << endl;
    // cout << "variable_ind " << variable_ind << endl;
    // cout << "X_num_unique " << X_num_unique << endl;

    PoissonClassifcationModel model;
    model.suff_stat_init();

    // save predictions of each tree
    xinfo predictions_std;
    ini_xinfo(predictions_std, N, M);

    std::vector<double> yhat_std(N);
    row_sum(predictions_std, yhat_std);

    // current residual
    std::vector<double> residual_std(N);

    ///////////////////////////////////////////////////////////////////

    // Rcpp::NumericMatrix yhats(N, N_sweeps);
    // Rcpp::NumericMatrix yhats_test(N_test, N_sweeps);

    // // save predictions of each tree
    // Rcpp::NumericMatrix sigma_draw(M, N_sweeps);

    // double tau;
    //forest trees(M);
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

    // std::vector<double> split_var_count(p);
    // std::fill(split_var_count.begin(), split_var_count.end(), 1);
    // Rcpp::NumericVector split_var_count(p, 1);

    // xinfo split_count_all_tree;
    // ini_xinfo(split_count_all_tree, p, M); // initialize at 0
    // split_count_all_tree = split_count_all_tree + 1; // initialize at 1
    std::vector<double> split_count_current_tree(p, 1);
    std::vector<double> mtry_weight_current_tree(p, 1);

    // in the burnin samples, use all variables
    std::vector<size_t> subset_vars(p);
    std::iota(subset_vars.begin() + 1, subset_vars.end(), 1);

    double run_time = 0.0;

    // save tree objects to strings
    // std::stringstream treess;
    // treess.precision(10);
    // treess << L << " " << M << " " << p << endl;

    // L, number of samples
    // M, number of trees

    // lambda in fit
    // auto get_lambda = [](double p) { return -std::log(2*std::max(p,1-p)-1)/2.0 ;};



    // initialize_trees  
    double ini_p = 1/((double)M)+0.000001; // 1/2 -> get_lambda = -inf
    std::cout << "first: " << ini_p <<"  second  " <<  -std::log(2*std::max(ini_p,1-ini_p)-1)/2.0 << endl;
    std::vector<double> init_theta_vector = {ini_p ,-std::log(2*std::max(ini_p,1-ini_p)-1)/2.0 ,0};
    for(size_t i=0; i< trees.size();i++){
        std::vector<tree> &tree_vec = trees[i];
        for(size_t j=0; j< tree_vec.size(); j++){
            tree_vec[j].settheta(init_theta_vector);
        }
    }
    
    // Initialize data pointer
    tree temp_tree((size_t)3); // to be safe if first tree doesn't grow
    tree::tree_p first_tree = &temp_tree; 
    first_tree->settheta(init_theta_vector); // set temp to 
    matrix<tree::tree_p> data_pointers; // Init data points
    ini_matrix(data_pointers, N, M);
    for(size_t i =0;i<M;i++){
        std::vector<tree::tree_p> &tree_vec = data_pointers[i];
        for(size_t j =0;j<N;j++){
            tree_vec[j] = &temp_tree;
        }
    }

 
    // Initialize Partial Fit as (lambs = (m-1)lambda,0)
    std::vector<double> lambs(N,(M-1.0)*init_theta_vector[1]); // Lambda
    std::vector<double> ks(N,0);
    matrix<double> partial_fit(2);
    partial_fit[0] =  lambs;
    partial_fit[1] = ks;

    // std::cout << "Before loop " << (M-1.0)*init_theta_vector[1] << "  " << (double)get_lambda(1/(double)M) << std::endl;
    // std::cout << "Before loop " << partial_fit[0][0] << "  " << partial_fit[0][1]  << std::endl;
    

    bool use_all = true;
    for (size_t mc = 0; mc < L; mc++)
    {
        for (size_t sweeps = 0; sweeps < N_sweeps; sweeps++)
        {

            if (verbose == true)
            {
                cout << "--------------------------------" << endl;
                cout << "number of sweeps " << sweeps << endl;
                cout << "--------------------------------" << endl;
            }

            for (size_t tree_ind = 0; tree_ind < M; tree_ind++)
            {
                ////// draw residual //////
                // draw_residual - residual =\tilde{y_j}  = prob.odd(partial_fit)
                // TODO: write as function 
                std::cout <<"Before draw" << std::endl;
                model.draw_residual(partial_fit[0],partial_fit[1],y_std,residual_std,gen);
                std::cout <<"after draw" << std::endl;

                if (use_all && (sweeps > burnin) && (mtry != p))
                {
                    use_all = false;
                }

                // // clear counts of splits for one tree
                std::fill(split_count_current_tree.begin(), split_count_current_tree.end(), 0.0);

                mtry_weight_current_tree = mtry_weight_current_tree - split_count_all_tree[tree_ind];

                //what to put for y_mean?
                std::cout << "before gfr: " << sum_vec(residual_std) / (double)N << std::endl;
                trees[sweeps][tree_ind].grow_tree_adaptive_std_all(sum_vec(residual_std) / (double)N, 0, max_depth_std[sweeps][tree_ind], Nmin, Ncutpoints, a, b, alpha, beta, false, false, parallel, residual_std, Xorder_std, Xpointer, mtry, use_all, split_count_all_tree, mtry_weight_current_tree, split_count_current_tree, categorical_variables, p_categorical, p_continuous, X_values, X_counts, variable_ind, X_num_unique, &model, data_pointers, tree_ind, gen);
                std::cout << "after gfr: " << sum_vec(residual_std) / (double)N << std::endl;
                mtry_weight_current_tree = mtry_weight_current_tree + split_count_current_tree;
                std::cout << "sweep: " << sweeps << "tree: " << tree_ind <<endl;
                split_count_all_tree[tree_ind] = split_count_current_tree;

                //get_params

                //update residual, now it's residual of m trees
                std::cout << "Set values" << endl;
                std::cout << "Pre Partial fit 0,0: " << partial_fit[0][0] << "  " << partial_fit[0][1]  << std::endl;
                std::vector<double>& lambs = partial_fit[0];
                std::vector<double>& ks = partial_fit[1];
                std::vector<tree::tree_p>& current_data_pointers = data_pointers[tree_ind];
                std::vector<tree::tree_p>& next_data_pointers = data_pointers[(tree_ind+1)%M]; 

                std::cout << "Mem of current tree " << current_data_pointers[0] ;
                std::cout << "  Mem of next tree " << next_data_pointers[0]<< "  "  << std::endl;
                for(size_t i = 0;i < N;i++)
                {
				    std::vector<double> thetas_current = current_data_pointers[i]->theta_vector;
				    std::vector<double> thetas_next = next_data_pointers[i]->theta_vector;
				    partial_fit[0][i]  = partial_fit[0][i]  - thetas_next[1] + thetas_current[1];
				    partial_fit[1][i]  = partial_fit[1][i]  - thetas_next[2] + thetas_current[2];
			    }
                std::cout << "Post Partial fit 0,0: " << partial_fit[0][0] << "   " << partial_fit[0][1]  << std::endl;
                
            }
            // save predictions to output matrix
            //yhats_xinfo[sweeps] = yhat_std;

            // for (size_t kk = 0; kk < N; kk++)
            // {
            //     yhats(kk, sweeps) = yhat_std[kk];
            // }
            // for (size_t kk = 0; kk < N_test; kk++)
            // {
            //     yhats_test(kk, sweeps) = yhat_test_std[kk];
            // }
        }
    }
    thread_pool.stop();
}

void predict_std_poisson_classification(const double *Xtestpointer, size_t N_test, size_t p, size_t M, size_t L,
                 size_t N_sweeps, xinfo &yhats_test_xinfo, vector<vector<tree>> &trees, double y_mean)
{

    xinfo predictions_test_std;
    ini_xinfo(predictions_test_std, N_test, M);

    std::vector<double> yhat_test_std(N_test);
    row_sum(predictions_test_std, yhat_test_std);

    for (size_t mc = 0; mc < L; mc++)
    {
        // initialize predcitions and predictions_test
        for (size_t ii = 0; ii < M; ii++)
        {
            std::fill(predictions_test_std[ii].begin(), predictions_test_std[ii].end(), y_mean / (double)M);
        }
        row_sum(predictions_test_std, yhat_test_std);

        std::vector<double> sum_lambs(N_test); 
        std::vector<double> sum_ks(N_test); 
        for (size_t sweeps = 0; sweeps < N_sweeps; sweeps++)
        {
            
            std::vector<double> sum_lambs(N_test);
            std::vector<double> sum_ks(N_test);

            for (size_t tree_ind = 0; tree_ind < M; tree_ind++)
            {
                
                //yhat_test_std = yhat_test_std - predictions_test_std[tree_ind];
                tree::tree_p bn;
                for (size_t i = 0; i < N_test; i++)
                {
                    bn = trees[sweeps][tree_ind].search_bottom_std(Xtestpointer, i, p, N_test);
                    sum_lambs[i] += bn->theta_vector[1];
                    sum_ks[i] += bn->theta_vector[2];

                }
                //yhat_test_std = yhat_test_std + predictions_test_std[tree_ind];


            }
            std::vector<double>& current_sweep_result = yhats_test_xinfo[sweeps];
            for (size_t i = 0; i < N_test; i++)
            {
                current_sweep_result[i] = 0.5*(1 + std::pow(-1,1-(size_t)sum_ks[i]%2) * std::exp(-2*sum_lambs[i]));
            }
            //yhats_test_xinfo[sweeps] = yhat_test_std;

        }
    }
}