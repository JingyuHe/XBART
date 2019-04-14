#include "fit_std_main_loop.h"
#include "fit.h"

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

    tree first_tree((size_t)1); // to be safe if first tree doesn't grow
     
    Fit *fit = new Fit(Xpointer, Xorder_std,N,p,num_trees,p_categorical,p_continuous,set_random_seed,random_seed,&first_tree);
    
    if (parallel)
        thread_pool.start();

    NormalModel *model = new NormalModel();
    model->suff_stat_init();

    // initialize predcitions 
    for (size_t ii = 0; ii < num_trees; ii++)
    {
        std::fill(fit->predictions_std[ii].begin(), fit->predictions_std[ii].end(), y_mean / (double)num_trees);
    }

    // Set yhat_std to mean 
    row_sum(fit->predictions_std, fit->yhat_std);

    // Residual for 0th tree 
    fit->residual_std = y_std - fit->yhat_std + fit->predictions_std[0];

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

            // Draw Sigma
            fit->residual_std_full = fit->residual_std - fit->predictions_std[tree_ind];
            std::gamma_distribution<double> gamma_samp((N + kap) / 2.0, 2.0 / (sum_squared(fit->residual_std_full) + s));
            fit->sigma = 1.0 / sqrt(gamma_samp(fit->gen));
            sigma_draw_xinfo[sweeps][tree_ind] = fit->sigma;


            // save sigma
            sigma_draw_xinfo[sweeps][tree_ind] = fit->sigma;


            // add prediction of current tree back to residual
            // then it's m - 1 trees residual
            fit->yhat_std = fit->yhat_std - fit->predictions_std[tree_ind];

            if (fit->use_all && (sweeps > burnin) && (mtry != p))
            {

                fit->use_all = false;
            }

            // clear counts of splits for one tree
            std::fill(fit->split_count_current_tree.begin(), fit->split_count_current_tree.end(), 0.0);

            fit->mtry_weight_current_tree = fit->mtry_weight_current_tree - fit->split_count_all_tree[tree_ind];

            trees[sweeps][tree_ind].grow_tree_adaptive_std_all(sum_vec(fit->residual_std) / (double)N, 0, max_depth_std[sweeps][tree_ind], n_min, Ncutpoints, tau, fit->sigma, alpha, beta, draw_mu, parallel, fit->residual_std, Xorder_std, Xpointer, mtry, fit->use_all, fit->split_count_all_tree, fit->mtry_weight_current_tree, fit->split_count_current_tree, fit->categorical_variables, p_categorical, p_continuous, fit->X_values, fit->X_counts, fit->variable_ind, fit->X_num_unique, model, fit->data_pointers, tree_ind, fit->gen);

            fit->mtry_weight_current_tree = fit->mtry_weight_current_tree + fit->split_count_current_tree;

            fit->split_count_all_tree[tree_ind] = fit->split_count_current_tree;

            // fit_new_std(trees[sweeps][tree_ind], Xpointer, N, p, predictions_std[tree_ind]);
            fit_new_std_datapointers(Xpointer, N, tree_ind, fit->predictions_std[tree_ind], fit->data_pointers);

            // update residual, now it's residual of m trees
            model->updateResidual(fit->predictions_std, tree_ind, num_trees, fit->residual_std);

            fit->yhat_std = fit->yhat_std + fit->predictions_std[tree_ind];
        }
        // save predictions to output matrix
        yhats_xinfo[sweeps] = fit->yhat_std;

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

void fit_std_clt(const double *Xpointer, std::vector<double> &y_std, double y_mean, xinfo_sizet &Xorder_std,
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
    
     tree first_tree((size_t)1); // to be safe if first tree doesn't grow

     Fit *fit = new Fit(Xpointer, Xorder_std,N,p,num_trees,p_categorical,p_continuous,set_random_seed,random_seed,&first_tree);
    
    if (parallel)
        thread_pool.start();

    CLTClass *model = new CLTClass();
    model->suff_stat_init();

    // num_trees, number of trees

    // initialize predcitions and predictions_test
    for (size_t ii = 0; ii < num_trees; ii++)
    {
        std::fill(fit->predictions_std[ii].begin(), fit->predictions_std[ii].end(), y_mean / (double)num_trees);
    }

    // Set yhat_std to mean 
    row_sum(fit->predictions_std, fit->yhat_std);

    // Residual for 0th tree 
    fit->residual_std = y_std - fit->yhat_std + fit->predictions_std[0];

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

            // Update Sigma
            fit->residual_std_full = fit->residual_std - fit->predictions_std[tree_ind];
            std::gamma_distribution<double> gamma_samp((N + kap) / 2.0, 2.0 / (sum_squared(fit->residual_std_full) + s));
            fit->sigma = 1.0 / sqrt(gamma_samp(fit->gen));
            sigma_draw_xinfo[sweeps][tree_ind] = fit->sigma;

            // Add total_fit
            model->total_fit = fit->yhat_std;

            // Update sum(1/psi) and sum(log(1/psi)) in model class
            model->updateFullSuffStat();

            // add prediction of current tree back to residual
            // then it's m - 1 trees residual
            fit->yhat_std = fit->yhat_std - fit->predictions_std[tree_ind];

            if (fit->use_all && (sweeps > burnin) && (mtry != p))
            {
                fit->use_all = false;
            }

            // clear counts of splits for one tree
            std::fill(fit->split_count_current_tree.begin(), fit->split_count_current_tree.end(), 0.0);

            fit->mtry_weight_current_tree = fit->mtry_weight_current_tree - fit->split_count_all_tree[tree_ind];

            trees[sweeps][tree_ind].grow_tree_adaptive_std_all(sum_vec(fit->residual_std) / (double)N, 0, max_depth_std[sweeps][tree_ind], n_min, Ncutpoints, tau, fit->sigma, alpha, beta, draw_mu, parallel, fit->residual_std, Xorder_std, Xpointer, mtry, fit->use_all, fit->split_count_all_tree, fit->mtry_weight_current_tree, fit->split_count_current_tree, fit->categorical_variables, p_categorical, p_continuous, fit->X_values, fit->X_counts, fit->variable_ind, fit->X_num_unique, model, fit->data_pointers, tree_ind, fit->gen);

            fit->mtry_weight_current_tree = fit->mtry_weight_current_tree + fit->split_count_current_tree;

            fit->split_count_all_tree[tree_ind] = fit->split_count_current_tree;

            // update prediction of current tree

            // fit_new_std(trees[sweeps][tree_ind], Xpointer, N, p, predictions_std[tree_ind]);
            fit_new_std_datapointers(Xpointer, N, tree_ind, fit->predictions_std[tree_ind], fit->data_pointers);


            // update residual, now it's residual of m trees
            model->updateResidual(fit->predictions_std, tree_ind, num_trees, fit->residual_std);

            fit->yhat_std = fit->yhat_std + fit->predictions_std[tree_ind];
        }
        // save predictions to output matrix
        yhats_xinfo[sweeps] = fit->yhat_std;


    }
}