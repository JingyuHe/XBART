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

    tree first_tree((size_t)1); // to be safe if first tree doesn't grow
     
    FitInfo *fit_info = new FitInfo(Xpointer, Xorder_std,N,p,num_trees,p_categorical,p_continuous,set_random_seed,random_seed,&first_tree);
    
    if (parallel)
        thread_pool.start();

    NormalModel *model = new NormalModel();
    model->suff_stat_init();

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
            cout << "--------------------------------" << endl;
            cout << "number of sweeps " << sweeps << endl;
            cout << "--------------------------------" << endl;
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

            if (fit_info->use_all && (sweeps > burnin) && (mtry != p)){fit_info->use_all = false; }

            // clear counts of splits for one tree
            std::fill(fit_info->split_count_current_tree.begin(), fit_info->split_count_current_tree.end(), 0.0);
            fit_info->mtry_weight_current_tree = fit_info->mtry_weight_current_tree - fit_info->split_count_all_tree[tree_ind];

            trees[sweeps][tree_ind].grow_tree_adaptive_std_all(sum_vec(fit_info->residual_std) / (double)N, 0, max_depth_std[sweeps][tree_ind], n_min, Ncutpoints, tau, sigma, alpha, beta, draw_mu, parallel, fit_info->residual_std, Xorder_std, Xpointer, mtry, fit_info->use_all, fit_info->split_count_all_tree, fit_info->mtry_weight_current_tree, fit_info->split_count_current_tree, fit_info->categorical_variables, p_categorical, p_continuous, fit_info->X_values, fit_info->X_counts, fit_info->variable_ind, fit_info->X_num_unique, model, fit_info->data_pointers, tree_ind, fit_info->gen);

            // Add split counts    
            fit_info->mtry_weight_current_tree = fit_info->mtry_weight_current_tree + fit_info->split_count_current_tree;
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

     FitInfo *fit_info = new FitInfo(Xpointer, Xorder_std,N,p,num_trees,p_categorical,p_continuous,set_random_seed,random_seed,&first_tree);
    
    if (parallel)
        thread_pool.start();

    CLTClass *model = new CLTClass();
    model->suff_stat_init();

    // initialize predcitions and predictions_test
    for (size_t ii = 0; ii < num_trees; ii++)
    {
        std::fill(fit_info->predictions_std[ii].begin(), fit_info->predictions_std[ii].end(), y_mean / (double)num_trees);
    }

    // Set yhat_std to mean 
    fit_info->yhat_std = std::vector<double>(N,y_mean);

    // Residual for 0th tree 
    fit_info->residual_std = y_std - fit_info->yhat_std + fit_info->predictions_std[0];

    double sigma = 2000.0;

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
            // // Add total_fit
            // if( sweeps <= burnin){
            //    model->total_fit = std::vector<double>(N,1); // set psi = 0
            // }else{
            //     model->total_fit = fit_info->yhat_std;
            // }
            model->total_fit = fit_info->yhat_std;
            
            
            // Update sum(1/psi) and sum(log(1/psi)) in model class
            model->updateFullSuffStat();

            std::cout << "Mean psi : " << model->mean_psi ;

            // add prediction of current tree back to residual
            // then it's m - 1 trees residual
            fit_info->yhat_std = fit_info->yhat_std - fit_info->predictions_std[tree_ind];

            if (fit_info->use_all && (sweeps > burnin) && (mtry != p))
            {
                fit_info->use_all = false;
            }

            // clear counts of splits for one tree
            std::fill(fit_info->split_count_current_tree.begin(), fit_info->split_count_current_tree.end(), 0.0);
            fit_info->mtry_weight_current_tree = fit_info->mtry_weight_current_tree - fit_info->split_count_all_tree[tree_ind];

            trees[sweeps][tree_ind].grow_tree_adaptive_std_all(sum_vec(fit_info->residual_std) / (double)N, 0, max_depth_std[sweeps][tree_ind], n_min, Ncutpoints, tau, sigma, alpha, beta, draw_mu, parallel, fit_info->residual_std, Xorder_std, Xpointer, mtry, fit_info->use_all, fit_info->split_count_all_tree, fit_info->mtry_weight_current_tree, fit_info->split_count_current_tree, fit_info->categorical_variables, p_categorical, p_continuous, fit_info->X_values, fit_info->X_counts, fit_info->variable_ind, fit_info->X_num_unique, model, fit_info->data_pointers, tree_ind, fit_info->gen);

            fit_info->mtry_weight_current_tree = fit_info->mtry_weight_current_tree + fit_info->split_count_current_tree;
            fit_info->split_count_all_tree[tree_ind] = fit_info->split_count_current_tree;

            // update prediction of current tree

            // fit_new_std(trees[sweeps][tree_ind], Xpointer, N, p, predictions_std[tree_ind]);
            fit_new_std_datapointers(Xpointer, N, tree_ind, fit_info->predictions_std[tree_ind], fit_info->data_pointers);


            // update residual, now it's residual of m trees
            model->updateResidual(fit_info->predictions_std, tree_ind, num_trees, fit_info->residual_std);

            fit_info->yhat_std = fit_info->yhat_std + fit_info->predictions_std[tree_ind];
        }
        // save predictions to output matrix
        yhats_xinfo[sweeps] = fit_info->yhat_std;


    }
}

// void fit_std_probit(const double *Xpointer, std::vector<double> &y_std, double y_mean, xinfo_sizet &Xorder_std,
//              size_t N, size_t p,
//              size_t num_trees, size_t num_sweeps, xinfo_sizet &max_depth_std,
//              size_t n_min, size_t Ncutpoints, double alpha, double beta,
//              double tau, size_t burnin, size_t mtry,
//              double kap, double s,
//              bool verbose,
//              bool draw_mu, bool parallel,
//              xinfo &yhats_xinfo, xinfo &sigma_draw_xinfo,
//              size_t p_categorical, size_t p_continuous, vector<vector<tree>> &trees, bool set_random_seed, size_t random_seed)
// {

//     tree first_tree((size_t)1); // to be safe if first tree doesn't grow
     
//     FitInfo *fit_info = new FitInfo(Xpointer, Xorder_std,N,p,num_trees,p_categorical,p_continuous,
//                                     set_random_seed,random_seed,&first_tree);
    
//     if (parallel)
//         thread_pool.start();

//     NormalModel *model = new NormalModel();
//     model->suff_stat_init();

//     // initialize predcitions 
//     for (size_t ii = 0; ii < num_trees; ii++)
//     {
//         std::fill(fit_info->predictions_std[ii].begin(), fit_info->predictions_std[ii].end(), y_mean / (double)num_trees);
//     }

//     // Set yhat_std to mean 
//     row_sum(fit_info->predictions_std, fit_info->yhat_std);

//     // Residual for 0th tree 
//     fit_info->residual_std = y_std - fit_info->yhat_std + fit_info->predictions_std[0];

//     double sigma = 1.0;

//     std::vector<double> z(N);
//     for(size_t i; i < N;i++){
//         z[i] = y_std[i]*2 -1;
//     }

//     for (size_t sweeps = 0; sweeps < num_sweeps; sweeps++)
//     {

//         if (verbose == true)
//         {
//             cout << "--------------------------------" << endl;
//             cout << "number of sweeps " << sweeps << endl;
//             cout << "--------------------------------" << endl;
//         }

//         for (size_t tree_ind = 0; tree_ind < num_trees; tree_ind++)
//         {
            
            
//             // Update Z
//             double mu_temp;
//             for(size_t i =0; i < N; i++){
//                 double a = 0; 
//                 double b = 1;

//                 std::normal_distribution<double> normal_samp(fit_info->yhat_std [i], 1.0);
//                 mu_temp = normal_samp(fit_info->gen);
                
//                 if(y_std[i] == 0){
//                     a = mu_temp;
//                 }else{
//                     b = mu_temp;
//                 }
//                 std::uniform_real_distribution<double> unif(a,b);
//                 double u = unif(fit_info->gen);
//                 z[i] = NormalCDFInverse(u) + mu_temp;
                
//             }
            
//              // add prediction of current tree back to residual
//             // then it's m - 1 trees residual
//             fit_info->yhat_std = fit_info->yhat_std - fit_info->predictions_std[tree_ind];

//             if (fit_info->use_all && (sweeps > burnin) && (mtry != p)){fit_info->use_all = false; }

//             // clear counts of splits for one tree
//             std::fill(fit_info->split_count_current_tree.begin(), fit_info->split_count_current_tree.end(), 0.0);
//             fit_info->mtry_weight_current_tree = fit_info->mtry_weight_current_tree - fit_info->split_count_all_tree[tree_ind];

//             trees[sweeps][tree_ind].grow_tree_adaptive_std_all(sum_vec(fit_info->residual_std) / (double)N, 0, max_depth_std[sweeps][tree_ind], n_min, Ncutpoints, tau, sigma, alpha, beta, draw_mu, parallel, fit_info->residual_std, Xorder_std, Xpointer, mtry, fit_info->use_all, fit_info->split_count_all_tree, fit_info->mtry_weight_current_tree, fit_info->split_count_current_tree, fit_info->categorical_variables, p_categorical, p_continuous, fit_info->X_values, fit_info->X_counts, fit_info->variable_ind, fit_info->X_num_unique, model, fit_info->data_pointers, tree_ind, fit_info->gen);

//             // Add split counts    
//             fit_info->mtry_weight_current_tree = fit_info->mtry_weight_current_tree + fit_info->split_count_current_tree;
//             fit_info->split_count_all_tree[tree_ind] = fit_info->split_count_current_tree;

//             // Update Predict
//             fit_new_std_datapointers(Xpointer, N, tree_ind, fit_info->predictions_std[tree_ind], fit_info->data_pointers);

//             // update residual, now it's residual of m trees
//             model->updateResidual(fit_info->predictions_std, tree_ind, num_trees, fit_info->residual_std);

//             fit_info->yhat_std = fit_info->yhat_std + fit_info->predictions_std[tree_ind];
//         }
//         // save predictions to output matrix
//         yhats_xinfo[sweeps] = fit_info->yhat_std;

//     }
    
//     thread_pool.stop();
// }
