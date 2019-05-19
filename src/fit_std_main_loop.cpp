#include "fit_std_main_loop.h"

void fit_std(const double *Xpointer, std::vector<double> &y_std, double y_mean, xinfo_sizet &Xorder_std,
             size_t N, size_t p,
             size_t num_trees, size_t num_sweeps, xinfo_sizet &max_depth_std,
             size_t n_min, size_t Ncutpoints, double alpha, double beta,
             double tau, size_t burnin, size_t mtry,
             double kap, double s,
             bool verbose,
             bool draw_mu, bool parallel,
             xinfo &yhats_xinfo, xinfo &sigma_draw_xinfo, vec_d &mtry_weight_current_tree,
             size_t p_categorical, size_t p_continuous, vector<vector<tree>> &trees, 
             bool set_random_seed, size_t random_seed, double no_split_penality,bool sample_weights_flag)
{

    std::vector<double> initial_theta(1,0);
    std::unique_ptr<FitInfo> fit_info (new FitInfo(Xpointer, Xorder_std, N, p, num_trees, p_categorical, p_continuous, set_random_seed, random_seed, &initial_theta));


    if (parallel)
        thread_pool.start();

    //std::unique_ptr<NormalModel> model (new NormalModel);
    NormalModel *model = new NormalModel();
    model->setNoSplitPenality(no_split_penality);

    // initialize predcitions
    for (size_t ii = 0; ii < num_trees; ii++)
    {
        std::fill(fit_info->predictions_std[ii].begin(), fit_info->predictions_std[ii].end(), y_mean / (double)num_trees);
    }

    // Set yhat_std to mean
    row_sum(fit_info->predictions_std, fit_info->yhat_std);

    // Residual for 0th tree
    fit_info->residual_std = y_std - fit_info->yhat_std + fit_info->predictions_std[0];

    double sigma = 1.0;

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

            // Draw Sigma
            fit_info->residual_std_full = fit_info->residual_std - fit_info->predictions_std[tree_ind];
            std::gamma_distribution<double> gamma_samp((N + kap) / 2.0, 2.0 / (sum_squared(fit_info->residual_std_full) + s));
            sigma = 1.0 / sqrt(gamma_samp(fit_info->gen));
            sigma_draw_xinfo[sweeps][tree_ind] = sigma;

            // add prediction of current tree back to residual
            // then it's m - 1 trees residual
            fit_info->yhat_std = fit_info->yhat_std - fit_info->predictions_std[tree_ind];

            if (fit_info->use_all && (sweeps > burnin) && (mtry != p))
            {
                fit_info->use_all = false;
            }

            // clear counts of splits for one tree
            std::fill(fit_info->split_count_current_tree.begin(), fit_info->split_count_current_tree.end(), 0.0);

            // subtract old tree for sampling case
            if(sample_weights_flag){
                mtry_weight_current_tree = mtry_weight_current_tree - fit_info->split_count_all_tree[tree_ind];
            }

            trees[sweeps][tree_ind].grow_tree_adaptive_std_all(sum_vec(fit_info->residual_std) / (double)N, 0, max_depth_std[sweeps][tree_ind], n_min, Ncutpoints, tau, sigma, alpha, beta, draw_mu, parallel, fit_info->residual_std, Xorder_std, Xpointer, mtry, fit_info->use_all, fit_info->split_count_all_tree, mtry_weight_current_tree, fit_info->split_count_current_tree, fit_info->categorical_variables, p_categorical, p_continuous, fit_info->X_values, fit_info->X_counts, fit_info->variable_ind, fit_info->X_num_unique, model, fit_info->data_pointers, tree_ind, fit_info->gen,sample_weights_flag);

            // Add split counts
            mtry_weight_current_tree = mtry_weight_current_tree + fit_info->split_count_current_tree;
            fit_info->split_count_all_tree[tree_ind] = fit_info->split_count_current_tree;

            // Update Predict
            fit_new_std_datapointers(Xpointer, N, tree_ind, fit_info->predictions_std[tree_ind], fit_info->data_pointers);

            // update residual, now it's residual of m trees
            model->updateResidual(fit_info->predictions_std, tree_ind, num_trees, fit_info->residual_std);

            fit_info->yhat_std = fit_info->yhat_std + fit_info->predictions_std[tree_ind];
        }
        // save predictions to output matrix
        yhats_xinfo[sweeps] = fit_info->yhat_std;
    }
    thread_pool.stop();

    delete model;
}

void predict_std(const double *Xtestpointer, size_t N_test, size_t p, size_t num_trees,
                 size_t num_sweeps, xinfo &yhats_test_xinfo,
                 vector<vector<tree>> &trees, double y_mean)
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

void fit_std_clt(const double *Xpointer, std::vector<double> &y_std, double y_mean, xinfo_sizet &Xorder_std,
                 size_t N, size_t p,
                 size_t num_trees, size_t num_sweeps, xinfo_sizet &max_depth_std,
                 size_t n_min, size_t Ncutpoints, double alpha, double beta,
                 double tau, size_t burnin, size_t mtry,
                 double kap, double s,
                 bool verbose,
                 bool draw_mu, bool parallel,
                 xinfo &yhats_xinfo, xinfo &sigma_draw_xinfo, vec_d &mtry_weight_current_tree,
                 size_t p_categorical, size_t p_continuous, vector<vector<tree>> &trees, bool set_random_seed, size_t random_seed, double no_split_penality, bool sample_weights_flag)
{

    std::vector<double> initial_theta(1,0);
    std::unique_ptr<FitInfo> fit_info (new FitInfo(Xpointer, Xorder_std, N, p, num_trees, p_categorical, p_continuous, set_random_seed, random_seed, &initial_theta));

    if (parallel)
        thread_pool.start();

    CLTClass *model = new CLTClass();
    model->setNoSplitPenality(no_split_penality);

    // initialize predcitions and predictions_test
    for (size_t ii = 0; ii < num_trees; ii++)
    {
        std::fill(fit_info->predictions_std[ii].begin(), fit_info->predictions_std[ii].end(), y_mean / (double)num_trees);
    }

    // Residual for 0th tree
    fit_info->residual_std = y_std - fit_info->yhat_std + fit_info->predictions_std[0];

    double sigma = 0.0;

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
            std::cout << "Tree " << tree_ind << std::endl;
            fit_info->yhat_std = fit_info->yhat_std - fit_info->predictions_std[tree_ind];

            model->total_fit = fit_info->yhat_std;

            if ((sweeps > burnin) && (mtry < p))
            {
                fit_info->use_all = false;
            }

            // clear counts of splits for one tree
            std::fill(fit_info->split_count_current_tree.begin(), fit_info->split_count_current_tree.end(), 0.0);

            //COUT << fit_info->split_count_current_tree << endl;

            // subtract old tree for sampling case
            if(sample_weights_flag){
                mtry_weight_current_tree = mtry_weight_current_tree - fit_info->split_count_all_tree[tree_ind];
            }


            trees[sweeps][tree_ind].grow_tree_adaptive_std_all(sum_vec(fit_info->residual_std) / (double)N, 0, max_depth_std[sweeps][tree_ind], n_min, Ncutpoints, tau, sigma, alpha, beta, draw_mu, parallel, fit_info->residual_std, Xorder_std, Xpointer, mtry, fit_info->use_all, fit_info->split_count_all_tree, mtry_weight_current_tree, fit_info->split_count_current_tree, fit_info->categorical_variables, p_categorical, p_continuous, fit_info->X_values, fit_info->X_counts, fit_info->variable_ind, fit_info->X_num_unique, model, fit_info->data_pointers, tree_ind, fit_info->gen,sample_weights_flag);

            mtry_weight_current_tree = mtry_weight_current_tree + fit_info->split_count_current_tree;

            fit_info->split_count_all_tree[tree_ind] = fit_info->split_count_current_tree;

            // fit_new_std(trees[sweeps][tree_ind], Xpointer, N, p, predictions_std[tree_ind]);
            fit_new_std_datapointers(Xpointer, N, tree_ind, fit_info->predictions_std[tree_ind], fit_info->data_pointers);

            // update residual, now it's residual of m trees
            model->updateResidual(fit_info->predictions_std, tree_ind, num_trees, fit_info->residual_std);

            fit_info->yhat_std = fit_info->yhat_std + fit_info->predictions_std[tree_ind];

            std::cout << "stuff stat" << model->suff_stat_total << std::endl;
        }
        // save predictions to output matrix
        yhats_xinfo[sweeps] = fit_info->yhat_std;
    }
    thread_pool.stop();
    delete model;
}

void fit_std_probit(const double *Xpointer, std::vector<double> &y_std, double y_mean, xinfo_sizet &Xorder_std,
                    size_t N, size_t p,
                    size_t num_trees, size_t num_sweeps, xinfo_sizet &max_depth_std,
                    size_t n_min, size_t Ncutpoints, double alpha, double beta,
                    double tau, size_t burnin, size_t mtry,
                    double kap, double s,
                    bool verbose,
                    bool draw_mu, bool parallel,
                    xinfo &yhats_xinfo, xinfo &sigma_draw_xinfo, vec_d &mtry_weight_current_tree,
                    size_t p_categorical, size_t p_continuous, vector<vector<tree>> &trees, bool set_random_seed, size_t random_seed, double no_split_penality, bool sample_weights_flag)
{

    std::vector<double> initial_theta(1,0);
    std::unique_ptr<FitInfo> fit_info (new FitInfo(Xpointer, Xorder_std, N, p, num_trees, p_categorical, p_continuous, set_random_seed, random_seed, &initial_theta));


    if (parallel)
        thread_pool.start();

    NormalModel *model = new NormalModel();
    model->setNoSplitPenality(no_split_penality);

    // initialize predcitions
    for (size_t ii = 0; ii < num_trees; ii++)
    {
        std::fill(fit_info->predictions_std[ii].begin(), fit_info->predictions_std[ii].end(), y_mean / (double)num_trees);
    }

    // Set yhat_std to mean
    row_sum(fit_info->predictions_std, fit_info->yhat_std);

    // Residual for 0th tree
    fit_info->residual_std = y_std - fit_info->yhat_std + fit_info->predictions_std[0];

    double sigma = 1.0;

    // Probit
    std::vector<double> z = y_std;
    std::vector<double> z_prev(N);

    double a = 0;
    double b = 1;
    double mu_temp;
    double u;

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

            // Update Z
            if (verbose){
                cout << "Tree "<< tree_ind << endl;
                cout << "Updating Z" << endl;
            }
            z_prev = z;
            for (size_t i = 0; i < N; i++)
            {
                a = 0;
                b = 1;

                mu_temp = normCDF(z_prev[i]);

                // Draw from truncated normal via inverse CDF methods
                if (y_std[i] > 0)
                {
                    a = std::min(mu_temp, 0.999);
                }
                else
                {
                    b = std::max(mu_temp, 0.001);
                }

                std::uniform_real_distribution<double> unif(a, b);
                u = unif(fit_info->gen);
                z[i] = normCDFInv(u) + mu_temp;
            }

            // add prediction of current tree back to residual
            // then it's m - 1 trees residual
            fit_info->yhat_std = fit_info->yhat_std - fit_info->predictions_std[tree_ind];

            if (fit_info->use_all && (sweeps > burnin) && (mtry != p))
            {
                fit_info->use_all = false;
            }

            // clear counts of splits for one tree
            std::fill(fit_info->split_count_current_tree.begin(), fit_info->split_count_current_tree.end(), 0.0);

            if (verbose){
                cout << "Grow from root" << endl;
            }

            if(sample_weights_flag){
                mtry_weight_current_tree = mtry_weight_current_tree - fit_info->split_count_all_tree[tree_ind];
            }

            trees[sweeps][tree_ind].grow_tree_adaptive_std_all(sum_vec(fit_info->residual_std) / (double)N, 0, max_depth_std[sweeps][tree_ind], n_min, Ncutpoints, tau, sigma, alpha, beta, draw_mu, parallel, fit_info->residual_std, Xorder_std, Xpointer, mtry, fit_info->use_all, fit_info->split_count_all_tree, mtry_weight_current_tree, fit_info->split_count_current_tree, fit_info->categorical_variables, p_categorical, p_continuous, fit_info->X_values, fit_info->X_counts, fit_info->variable_ind, fit_info->X_num_unique, model, fit_info->data_pointers, tree_ind, fit_info->gen,sample_weights_flag);

            // Add split counts
            //            fit_info->mtry_weight_current_tree = fit_info->mtry_weight_current_tree - fit_info->split_count_all_tree[tree_ind];

            mtry_weight_current_tree = mtry_weight_current_tree + fit_info->split_count_current_tree;
            fit_info->split_count_all_tree[tree_ind] = fit_info->split_count_current_tree;

            //	COUT << "outer loop split_count" << fit_info->split_count_current_tree << endl;
            //	COUT << "outer loop weights" << fit_info->mtry_weight_current_tree << endl;

            // Update Predict
            fit_new_std_datapointers(Xpointer, N, tree_ind, fit_info->predictions_std[tree_ind], fit_info->data_pointers);

            // update residual, now it's residual of m trees
            model->updateResidual(fit_info->predictions_std, tree_ind, num_trees, fit_info->residual_std);
            for (size_t i = 0; i < N; i++)
            {
                fit_info->residual_std[i] = fit_info->residual_std[i] - z_prev[i] + z[i];
            }

            fit_info->yhat_std = fit_info->yhat_std + fit_info->predictions_std[tree_ind];
        }
        // save predictions to output matrix
        yhats_xinfo[sweeps] = fit_info->yhat_std;
    }

    thread_pool.stop();
    delete model;
}