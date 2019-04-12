#include "fit_std_main_loop.h"

void fit_std_main_loop_all(const double *Xpointer, std::vector<double> &y_std, double &y_mean, const double *Xtestpointer, xinfo_sizet &Xorder_std,
                           size_t N, size_t p, size_t N_test,
                           size_t num_trees, size_t num_sweeps, xinfo_sizet &max_depth_std,
                           size_t n_min, size_t Ncutpoints, double alpha, double beta,
                           double tau, size_t burnin, size_t mtry,
                           double kap, double s,
                           bool verbose,
                           bool draw_mu, bool parallel,
                           xinfo &yhats_xinfo, xinfo &yhats_test_xinfo,
                           xinfo &sigma_draw_xinfo, xinfo &split_count_all_tree,
                           size_t p_categorical, size_t p_continuous, vector<vector<tree>> &trees, bool set_random_seed, size_t random_seed)
{

    fit_std(Xpointer, y_std, y_mean, Xorder_std,
            N, p,
            num_trees, num_sweeps, max_depth_std,
            n_min, Ncutpoints, alpha, beta,
            tau, burnin, mtry,
            kap, s,
            verbose,
            draw_mu, parallel,
            yhats_xinfo, sigma_draw_xinfo,
            p_categorical, p_continuous, trees, set_random_seed, random_seed);

    predict_std(Xtestpointer, N_test, p, num_trees, num_sweeps, yhats_test_xinfo, trees, y_mean);
    return;
}

void fit_std(const double *Xpointer, std::vector<double> &y_std, double y_mean, xinfo_sizet &Xorder_std,
             size_t N, size_t p,
             size_t num_trees, size_t num_sweeps, xinfo_sizet &max_depth_std,
             size_t n_min, size_t Ncutpoints, double alpha, double beta,
             double tau, size_t burnin, size_t mtry,
             double kap, double s,
             bool verbose,
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

    NormalModel model;
    model.suff_stat_init();

    // save predictions of each tree
    xinfo predictions_std;
    ini_xinfo(predictions_std, N, num_trees);

    std::vector<double> yhat_std(N);
    row_sum(predictions_std, yhat_std);

    // current residual
    std::vector<double> residual_std(N);

    double sigma = 1.0;
    // double tau;
    //forest trees(num_trees);
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

    // Initalize Pointer Matrix
    tree temp_tree((size_t)1); // to be safe if first tree doesn't grow
    tree::tree_p first_tree = &temp_tree; 
    matrix<tree::tree_p> data_pointers; // Init data points
    ini_matrix(data_pointers, N, num_trees);
    for(size_t i =0;i<num_trees;i++){
        std::vector<tree::tree_p> &tree_vec = data_pointers[i];
        for(size_t j =0;j<N;j++){
            tree_vec[j] = &temp_tree;
        }
    }

    bool use_all = true;

    std::vector<double> residual_std_full;

    // initialize predcitions and predictions_test
    for (size_t ii = 0; ii < num_trees; ii++)
    {
        std::fill(predictions_std[ii].begin(), predictions_std[ii].end(), y_mean / (double)num_trees);
    }

    // Set yhat_std to mean 
    row_sum(predictions_std, yhat_std);

    // Residual for 0th tree 
    residual_std = y_std - yhat_std + predictions_std[0];

    for (size_t sweeps = 0; sweeps < num_sweeps; sweeps++)
    {

        if (verbose == true)
        {
            cout << "--------------------------------" << endl;
            cout << "number of sweeps " << sweeps << endl;
            cout << "--------------------------------" << endl;
        }

        for (size_t tree_ind = 0; tree_ind < num_trees; tree_ind++)
        {

            // if update sigma based on residual of all m trees
            // if (m_update_sigma == true)
            // {

                residual_std_full = residual_std - predictions_std[tree_ind];

                std::gamma_distribution<double> gamma_samp((N + kap) / 2.0, 2.0 / (sum_squared(residual_std_full) + s));

                sigma = 1.0 / sqrt(gamma_samp(gen));

                sigma_draw_xinfo[sweeps][tree_ind] = sigma;
            // }

            // save sigma
            sigma_draw_xinfo[sweeps][tree_ind] = sigma;


            // add prediction of current tree back to residual
            // then it's m - 1 trees residual

            yhat_std = yhat_std - predictions_std[tree_ind];

            if (use_all && (sweeps > burnin) && (mtry != p))
            {
                // subset_vars = Rcpp::as<std::vector<size_t>>(sample(var_index_candidate, mtry, false, split_var_count));

                // subset_vars = sample_int_crank2(p, mtry, split_var_count);

                use_all = false;
            }

            // clear counts of splits for one tree
            std::fill(split_count_current_tree.begin(), split_count_current_tree.end(), 0.0);

            mtry_weight_current_tree = mtry_weight_current_tree - split_count_all_tree[tree_ind];

            trees[sweeps][tree_ind].grow_tree_adaptive_std_all(sum_vec(residual_std) / (double)N, 0, max_depth_std[sweeps][tree_ind], n_min, Ncutpoints, tau, sigma, alpha, beta, draw_mu, parallel, residual_std, Xorder_std, Xpointer, mtry, use_all, split_count_all_tree, mtry_weight_current_tree, split_count_current_tree, categorical_variables, p_categorical, p_continuous, X_values, X_counts, variable_ind, X_num_unique, &model, data_pointers, tree_ind, gen);

            mtry_weight_current_tree = mtry_weight_current_tree + split_count_current_tree;

            split_count_all_tree[tree_ind] = split_count_current_tree;

            // update prediction of current tree

            // fit_new_std(trees[sweeps][tree_ind], Xpointer, N, p, predictions_std[tree_ind]);
            fit_new_std_datapointers(Xpointer, N, tree_ind, predictions_std[tree_ind], data_pointers);

            // update prediction of current tree, test set

            // if (m_update_sigma == false)
            // {
            //     std::gamma_distribution<double> gamma_samp((N + kap) / 2.0, 2.0 / (sum_squared(residual_std) + s));

            //     sigma = 1.0 / sqrt(gamma_samp(gen));

            //     sigma_draw_xinfo[sweeps][tree_ind] = sigma;

            // }

            // update residual, now it's residual of m trees
            model.updateResidual(predictions_std, tree_ind, num_trees, residual_std);

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
    
    thread_pool.stop();
}

void predict_std(const double *Xtestpointer, size_t N_test, size_t p, size_t num_trees, 
                 size_t num_sweeps, xinfo &yhats_test_xinfo, vector<vector<tree>> &trees, double y_mean)
{

    xinfo predictions_test_std;
    ini_xinfo(predictions_test_std, N_test, num_trees);

    std::vector<double> yhat_test_std(N_test);
    row_sum(predictions_test_std, yhat_test_std);

    // initialize predcitions and predictions_test
    for (size_t ii = 0; ii < num_trees; ii++)
    {
        std::fill(predictions_test_std[ii].begin(), predictions_test_std[ii].end(), y_mean / (double)num_trees);
    }
    row_sum(predictions_test_std, yhat_test_std);


    for (size_t sweeps = 0; sweeps < num_sweeps; sweeps++)
    {
        for (size_t tree_ind = 0; tree_ind < num_trees; tree_ind++)
        {
            
            yhat_test_std = yhat_test_std - predictions_test_std[tree_ind];
            fit_new_std(trees[sweeps][tree_ind], Xtestpointer, N_test, p, predictions_test_std[tree_ind]);
            yhat_test_std = yhat_test_std + predictions_test_std[tree_ind];


        }
        yhats_test_xinfo[sweeps] = yhat_test_std;

    }
    
    return;
}

void fit_std_probit(const double *Xpointer, std::vector<double> &y_std, double y_mean, xinfo_sizet &Xorder_std,
             size_t N, size_t p,
             size_t num_trees, size_t num_sweeps, xinfo_sizet &max_depth_std,
             size_t n_min, size_t Ncutpoints, double alpha, double beta,
             double tau, size_t burnin, size_t mtry,
             double kap, double s,
             bool verbose,
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

    CLTClass model;
    model.suff_stat_init();

    // save predictions of each tree
    xinfo predictions_std;
    ini_xinfo(predictions_std, N, num_trees);

    std::vector<double> yhat_std(N);
    row_sum(predictions_std, yhat_std);

    // current residual
    std::vector<double> residual_std(N);

    double sigma = 1.0;
    // double tau;
    //forest trees(num_trees);
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

    // Initalize Pointer Matrix
    tree temp_tree((size_t)1); // to be safe if first tree doesn't grow
    tree::tree_p first_tree = &temp_tree; 
    matrix<tree::tree_p> data_pointers; // Init data points
    ini_matrix(data_pointers, N, num_trees);
    for(size_t i =0;i<num_trees;i++){
        std::vector<tree::tree_p> &tree_vec = data_pointers[i];
        for(size_t j =0;j<N;j++){
            tree_vec[j] = &temp_tree;
        }
    }

    bool use_all = true;

    std::vector<double> residual_std_full;

    // initialize predcitions and predictions_test
    for (size_t ii = 0; ii < num_trees; ii++)
    {
        std::fill(predictions_std[ii].begin(), predictions_std[ii].end(), y_mean / (double)num_trees);
    }

    // Set yhat_std to mean 
    row_sum(predictions_std, yhat_std);

    // Residual for 0th tree 
    residual_std = y_std - yhat_std + predictions_std[0];

    for (size_t sweeps = 0; sweeps < num_sweeps; sweeps++)
    {

        if (verbose == true)
        {
            cout << "--------------------------------" << endl;
            cout << "number of sweeps " << sweeps << endl;
            cout << "--------------------------------" << endl;
        }

        for (size_t tree_ind = 0; tree_ind < num_trees; tree_ind++)
        {

            // if update sigma based on residual of all m trees
            // if (m_update_sigma == true)
            // {

                residual_std_full = residual_std - predictions_std[tree_ind];

                std::gamma_distribution<double> gamma_samp((N + kap) / 2.0, 2.0 / (sum_squared(residual_std_full) + s));

                sigma = 1.0 / sqrt(gamma_samp(gen));

                sigma_draw_xinfo[sweeps][tree_ind] = sigma;
            // }

            // save sigma
            sigma_draw_xinfo[sweeps][tree_ind] = sigma;

            // Add total_fit
            model.total_fit = yhat_std;

            // add prediction of current tree back to residual
            // then it's m - 1 trees residual

            yhat_std = yhat_std - predictions_std[tree_ind];

            if (use_all && (sweeps > burnin) && (mtry != p))
            {
                // subset_vars = Rcpp::as<std::vector<size_t>>(sample(var_index_candidate, mtry, false, split_var_count));

                // subset_vars = sample_int_crank2(p, mtry, split_var_count);

                use_all = false;
            }

            // clear counts of splits for one tree
            std::fill(split_count_current_tree.begin(), split_count_current_tree.end(), 0.0);

            mtry_weight_current_tree = mtry_weight_current_tree - split_count_all_tree[tree_ind];

            trees[sweeps][tree_ind].grow_tree_adaptive_std_all(sum_vec(residual_std) / (double)N, 0, max_depth_std[sweeps][tree_ind], n_min, Ncutpoints, tau, sigma, alpha, beta, draw_mu, parallel, residual_std, Xorder_std, Xpointer, mtry, use_all, split_count_all_tree, mtry_weight_current_tree, split_count_current_tree, categorical_variables, p_categorical, p_continuous, X_values, X_counts, variable_ind, X_num_unique, &model, data_pointers, tree_ind, gen);

            mtry_weight_current_tree = mtry_weight_current_tree + split_count_current_tree;

            split_count_all_tree[tree_ind] = split_count_current_tree;

            // update prediction of current tree

            // fit_new_std(trees[sweeps][tree_ind], Xpointer, N, p, predictions_std[tree_ind]);
            fit_new_std_datapointers(Xpointer, N, tree_ind, predictions_std[tree_ind], data_pointers);

            // update prediction of current tree, test set

            // if (m_update_sigma == false)
            // {
            //     std::gamma_distribution<double> gamma_samp((N + kap) / 2.0, 2.0 / (sum_squared(residual_std) + s));

            //     sigma = 1.0 / sqrt(gamma_samp(gen));

            //     sigma_draw_xinfo[sweeps][tree_ind] = sigma;

            // }

            // update residual, now it's residual of m trees
            model.updateResidual(predictions_std, tree_ind, num_trees, residual_std);

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