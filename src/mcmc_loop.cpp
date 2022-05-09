#include "mcmc_loop.h"
#include "omp.h"

void mcmc_loop(matrix<size_t> &Xorder_std, bool verbose, matrix<double> &sigma_draw_xinfo, vector<vector<tree>> &trees, double no_split_penalty, std::unique_ptr<State> &state, NormalModel *model, std::unique_ptr<X_struct> &x_struct)
{

    // if (state->parallel)
    //     thread_pool.start();

    // Residual for 0th tree
    // state->residual_std = *state->y_std - state->yhat_std + state->predictions_std[0];
    model->ini_residual_std(state);

    for (size_t sweeps = 0; sweeps < state->num_sweeps; sweeps++)
    {

        if (verbose == true)
        {
            COUT << "--------------------------------" << endl;
            COUT << "number of sweeps " << sweeps << endl;
            COUT << "--------------------------------" << endl;
        }

        for (size_t tree_ind = 0; tree_ind < state->num_trees; tree_ind++)
        {

            if (verbose)
            {
                cout << "sweep " << sweeps << " tree " << tree_ind << endl;
            }
            // Draw Sigma

            model->update_state(state, tree_ind, x_struct);

            sigma_draw_xinfo[sweeps][tree_ind] = state->sigma;

            if (state->use_all && (sweeps > state->burnin) && (state->mtry != state->p))
            {
                state->use_all = false;
            }

            // clear counts of splits for one tree
            std::fill(state->split_count_current_tree.begin(), state->split_count_current_tree.end(), 0.0);

            // subtract old tree for sampling case
            if (state->sample_weights_flag)
            {
                state->mtry_weight_current_tree = state->mtry_weight_current_tree - state->split_count_all_tree[tree_ind];
            }

            model->initialize_root_suffstat(state, trees[sweeps][tree_ind].suff_stat);

            if (state->parallel)
            {
                // run parallel
                // omp_set_max_active_levels(state->nthread);
// #pragma omp parallel default(none) shared(trees, sweeps, state, Xorder_std, x_struct, model, tree_ind)
                // {
// #pragma omp sections
                    // {
// #pragma omp section
                        // {
                            trees[sweeps][tree_ind].grow_from_root(state, Xorder_std, x_struct->X_counts, x_struct->X_num_unique, model, x_struct, sweeps, tree_ind, true, false, true);
                        // }
                    // }
                // }
            }
            else
            {
                // single core
                trees[sweeps][tree_ind].grow_from_root(state, Xorder_std, x_struct->X_counts, x_struct->X_num_unique, model, x_struct, sweeps, tree_ind, true, false, true);
            }

            // update tau after sampling the tree
            // model->update_tau(state, tree_ind, sweeps, trees);

            state->update_split_counts(tree_ind);

            // update partial residual for the next tree to fit
            model->state_sweep(tree_ind, state->num_trees, state->residual_std, x_struct);
        }

        if (model->sampling_tau)
        {
            model->update_tau_per_forest(state, sweeps, trees);
        }
    }
    // thread_pool.stop();

    return;
}

// void predict_std_multinomial(const double *Xtestpointer, size_t N_test, size_t p, size_t num_trees,
//                              size_t num_sweeps, matrix<double> &yhats_test_xinfo,
//                              vector<vector<tree>> &trees)
// {

//     NormalModel *model = new NormalModel();
//     matrix<double> predictions_test_std;
//     ini_xinfo(predictions_test_std, N_test, num_trees);

//     std::vector<double> yhat_test_std(N_test);
//     row_sum(predictions_test_std, yhat_test_std);

//     // // initialize predcitions and predictions_test
//     // for (size_t ii = 0; ii < num_trees; ii++)
//     // {
//     //     std::fill(predictions_test_std[ii].begin(), predictions_test_std[ii].end(), y_mean / (double)num_trees);
//     // }
//     row_sum(predictions_test_std, yhat_test_std);

//     for (size_t sweeps = 0; sweeps < num_sweeps; sweeps++)
//     {
//         for (size_t tree_ind = 0; tree_ind < num_trees; tree_ind++)
//         {

//             yhat_test_std = yhat_test_std - predictions_test_std[tree_ind];
//             predict_from_tree(trees[sweeps][tree_ind], Xtestpointer, N_test, p, predictions_test_std[tree_ind], model);
//             yhat_test_std = yhat_test_std + predictions_test_std[tree_ind];
//         }
//         yhats_test_xinfo[sweeps] = yhat_test_std;
//     }

//     delete model;
//     return;
// }

void mcmc_loop_clt(matrix<size_t> &Xorder_std, bool verbose, matrix<double> &sigma_draw_xinfo, vector<vector<tree>> &trees, double no_split_penalty, std::unique_ptr<State> &state, CLTClass *model, std::unique_ptr<X_struct> &x_struct)
{
    // if (state->parallel)
    //     thread_pool.start();

    // // Residual for 0th tree
    // state->residual_std = *state->y_std - state->yhat_std + state->predictions_std[0];
    // model->ini_residual_std(state);

    // for (size_t sweeps = 0; sweeps < state->num_sweeps; sweeps++)
    // {

    //     if (verbose == true)
    //     {
    //         COUT << "--------------------------------" << endl;
    //         COUT << "number of sweeps " << sweeps << endl;
    //         COUT << "--------------------------------" << endl;
    //     }

    //     for (size_t tree_ind = 0; tree_ind < state->num_trees; tree_ind++)
    //     {
    //         std::cout << "Tree " << tree_ind << std::endl;
    //         model->update_state(state, tree_ind, x_struct);
    //         model->total_fit = state->yhat_std;

    //         if ((sweeps > state->burnin) && (state->mtry < state->p))
    //         {
    //             state->use_all = false;
    //         }

    //         // clear counts of splits for one tree
    //         std::fill(state->split_count_current_tree.begin(), state->split_count_current_tree.end(), 0.0);

    //         //COUT << state->split_count_current_tree << endl;

    //         // subtract old tree for sampling case
    //         if (state->sample_weights_flag)
    //         {
    //             state->mtry_weight_current_tree = state->mtry_weight_current_tree - state->split_count_all_tree[tree_ind];
    //         }

    //         // set sufficient statistics at root node first
    //         model->initialize_root_suffstat(state, trees[sweeps][tree_ind].suff_stat);

    //         trees[sweeps][tree_ind].grow_from_root(state, Xorder_std, x_struct->X_counts, x_struct->X_num_unique, model, x_struct, sweeps, tree_ind, true, false, true);

    //         state->update_split_counts(tree_ind);

    //         // fit_new_std(trees[sweeps][tree_ind], Xpointer, N, p, predictions_std[tree_ind]);
    //         // predict_from_datapointers(state->X_std, state->n_y, tree_ind, state->predictions_std[tree_ind], state->data_pointers, model);
    //         predict_from_datapointers(tree_ind, model, state, x_struct);

    //         // update residual, now it's residual of m trees
    //         model->state_sweep(tree_ind, state->num_trees, state->residual_std, x_struct);

    //         state->yhat_std = state->yhat_std + state->predictions_std[tree_ind];

    //         std::cout << "stuff stat" << model->suff_stat_total << std::endl;
    //     }
    //     // save predictions to output matrix
    //     yhats_xinfo[sweeps] = state->yhat_std;
    // }
    // thread_pool.stop();
    // delete model;
}

void mcmc_loop_multinomial(matrix<size_t> &Xorder_std, bool verbose, vector<vector<tree>> &trees, double no_split_penalty,
                           std::unique_ptr<State> &state, LogitModel *model, std::unique_ptr<X_struct> &x_struct,
                           std::vector<std::vector<double>> &weight_samples, std::vector<double> &lambda_samples, std::vector<std::vector<double>> &tau_samples)
{
    // if (state->parallel)
    //     thread_pool.start();

    // Residual for 0th tree
    // state->residual_std = *state->y_std - state->yhat_std + state->predictions_std[0];
    model->ini_residual_std(state);

    for (size_t sweeps = 0; sweeps < state->num_sweeps; sweeps++)
    {

        if (verbose == true)
        {
            COUT << "--------------------------------" << endl;
            COUT << "number of sweeps " << sweeps << endl;
            COUT << "--------------------------------" << endl;
        }

        for (size_t tree_ind = 0; tree_ind < state->num_trees; tree_ind++)
        {

            if (verbose)
            {
                cout << "sweep " << sweeps << " tree " << tree_ind << endl;
            }
            // Draw latents -- do last?

            //Rcpp::Rcout << "Updating state";

            if (state->use_all && (sweeps >= state->burnin)) // && (state->mtry != state->p) // If mtry = p, it will all be sampled anyway. Now use_all can be an indication of burnin period.
            {
                state->use_all = false;
            }

            // clear counts of splits for one tree
            std::fill(state->split_count_current_tree.begin(), state->split_count_current_tree.end(), 0.0);

            // subtract old tree for sampling case
            if (state->sample_weights_flag)
            {
                state->mtry_weight_current_tree = state->mtry_weight_current_tree - state->split_count_all_tree[tree_ind];
            }

            model->initialize_root_suffstat(state, trees[sweeps][tree_ind].suff_stat);
            for (size_t k = 0; k < trees[sweeps][tree_ind].suff_stat.size(); k++)
            {
                if (std::isnan(trees[sweeps][tree_ind].suff_stat[k]))
                {
                    cout << "unidentified error: suffstat " << k << " initialized as nan" << endl;
                    exit(1);
                }
            }

            trees[sweeps][tree_ind].theta_vector.resize(model->dim_residual);
            state->lambdas[tree_ind].clear();

            // set nthread based on number of observations * mtry
            /*  double fake_p = (state->use_all) ? state->p : state->mtry;
            if (state->n_y * fake_p < 1e5) { omp_set_num_threads( std::min(4, int(state->nthread)) ); }
            else if (state->n_y * fake_p < 5e5 ) { omp_set_num_threads( std::min(6, int(state->nthread) ) ); }
            else {omp_set_num_threads(state->nthread); }*/

            omp_set_max_active_levels(3);
#pragma omp parallel default(none) shared(trees, sweeps, state, Xorder_std, x_struct, model, tree_ind)
            {
#pragma omp sections
                {
#pragma omp section
                    {
                        trees[sweeps][tree_ind].grow_from_root_entropy(state, Xorder_std, x_struct->X_counts, x_struct->X_num_unique, model, x_struct, sweeps, tree_ind, true, false, true);
                    }
                }
            }

            state->update_split_counts(tree_ind);
            if (sweeps >= state->burnin)
            {
                for (size_t i = 0; i < state->split_count_all.size(); i++)
                {
                    state->split_count_all[i] += state->split_count_current_tree[i];
                }
            }
            // update partial fits for the next tree
            model->update_state(state, tree_ind, x_struct);

            model->state_sweep(tree_ind, state->num_trees, state->residual_std, x_struct);

            weight_samples[sweeps][tree_ind] = model->weight;
            tau_samples[sweeps][tree_ind] = model->tau_a;

            for (size_t j = 0; j < state->lambdas[tree_ind].size(); j++)
            {
                for (size_t k = 0; k < state->lambdas[tree_ind][j].size(); k++)
                {
                    lambda_samples.push_back(state->lambdas[tree_ind][j][k]);
                }
            }
        }

        // if (sweeps <= state->burnin){
        //     model->stop = false;
        // }
        // if (sweeps > state->burnin & model->stop){
        //     state->num_sweeps = sweeps + 1;
        //     break;
        // }
    }
    // thread_pool.stop();
}

void mcmc_loop_multinomial_sample_per_tree(matrix<size_t> &Xorder_std, bool verbose, vector<vector<vector<tree>>> &trees, double no_split_penality, std::unique_ptr<State> &state, LogitModelSeparateTrees *model,
                                           std::unique_ptr<X_struct> &x_struct, std::vector<std::vector<double>> &weight_samples)
{
    // Residual for 0th tree
    // state->residual_std = *state->y_std - state->yhat_std + state->predictions_std[0];
    model->ini_residual_std(state);
    size_t p = Xorder_std.size();
    std::vector<size_t> subset_vars(p);
    std::vector<double> weight_samp(p);
    // double weight_sum;

    for (size_t sweeps = 0; sweeps < state->num_sweeps; sweeps++)
    {

        if (verbose == true)
        {
            COUT << "--------------------------------" << endl;
            COUT << "number of sweeps " << sweeps << endl;
            COUT << "--------------------------------" << endl;
        }

        for (size_t tree_ind = 0; tree_ind < state->num_trees; tree_ind++)
        {

            if (verbose)
            {
                cout << "sweep " << sweeps << " tree " << tree_ind << endl;
            }
            // Draw latents -- do last?

            //Rcpp::Rcout << "Updating state";

            if (state->use_all && (sweeps >= state->burnin)) // && (state->mtry != state->p) // If mtry = p, it will all be sampled anyway. Now use_all can be an indication of burnin period.
            {
                state->use_all = false;
            }

            // clear counts of splits for one tree
            std::fill(state->split_count_current_tree.begin(), state->split_count_current_tree.end(), 0.0);

            // subtract old tree for sampling case
            if (state->sample_weights_flag)
            {
                state->mtry_weight_current_tree = state->mtry_weight_current_tree - state->split_count_all_tree[tree_ind];
            }

            omp_set_max_active_levels(3);
#pragma omp parallel default(none) shared(trees, sweeps, state, Xorder_std, x_struct, model, tree_ind)
            {
#pragma omp sections
                {
#pragma omp section
                    {
                        for (size_t class_ind = 0; class_ind < model->dim_residual; class_ind++)
                        {
                            // cout << "class_ind " << class_ind << endl;
                            model->set_class_operating(class_ind);

                            state->lambdas_separate[tree_ind][class_ind].clear();

                            model->initialize_root_suffstat(state, trees[class_ind][sweeps][tree_ind].suff_stat);

                            trees[class_ind][sweeps][tree_ind].theta_vector.resize(model->dim_residual);

                            trees[class_ind][sweeps][tree_ind].grow_from_root_separate_tree(state, Xorder_std, x_struct->X_counts, x_struct->X_num_unique, model, x_struct, sweeps, tree_ind, true, false, true);
                        }
                    }
                }
            }

            state->update_split_counts(tree_ind);

            if (sweeps >= state->burnin)
            {
                for (size_t i = 0; i < state->split_count_all.size(); i++)
                {
                    state->split_count_all[i] += state->split_count_current_tree[i];
                }
            }

            model->update_state(state, tree_ind, x_struct);

            model->state_sweep(tree_ind, state->num_trees, state->residual_std, x_struct);

            weight_samples[sweeps][tree_ind] = model->weight;
        }
    }

    return;
}

void mcmc_loop_probit(matrix<size_t> &Xorder_std, bool verbose, matrix<double> &sigma_draw_xinfo, vector<vector<tree>> &trees, double no_split_penalty, std::unique_ptr<State> &state, ProbitClass *model, std::unique_ptr<X_struct> &x_struct)
{

    // if (state->parallel)
    // thread_pool.start();

    // Residual for 0th tree
    // state->residual_std = *state->y_std - state->yhat_std + state->predictions_std[0];
    model->ini_residual_std(state);

    for (size_t sweeps = 0; sweeps < state->num_sweeps; sweeps++)
    {
        if (verbose == true)
        {
            COUT << "--------------------------------" << endl;
            COUT << "number of sweeps " << sweeps << endl;
            COUT << "--------------------------------" << endl;
        }

        for (size_t tree_ind = 0; tree_ind < state->num_trees; tree_ind++)
        {

            model->update_state(state, tree_ind, x_struct);

            if (state->use_all && (sweeps > state->burnin) && (state->mtry != state->p))
            {
                state->use_all = false;
            }

            // clear counts of splits for one tree
            std::fill(state->split_count_current_tree.begin(), state->split_count_current_tree.end(), 0.0);

            // subtract old tree for sampling case
            if (state->sample_weights_flag)
            {
                state->mtry_weight_current_tree = state->mtry_weight_current_tree - state->split_count_all_tree[tree_ind];
            }

            model->initialize_root_suffstat(state, trees[sweeps][tree_ind].suff_stat);

            omp_set_max_active_levels(3);
#pragma omp parallel default(none) shared(trees, sweeps, state, Xorder_std, x_struct, model, tree_ind)
            {
#pragma omp sections
                {
#pragma omp section
                    {
                        trees[sweeps][tree_ind].grow_from_root(state, Xorder_std, x_struct->X_counts, x_struct->X_num_unique, model, x_struct, sweeps, tree_ind, true, false, true);
                    }
                }
            }
            state->update_split_counts(tree_ind);

            // update partial residual for the next tree to fit
            model->state_sweep(tree_ind, state->num_trees, state->residual_std, x_struct);
        }
    }

    // thread_pool.stop();
}

// void mcmc_loop_MH(matrix<size_t> &Xorder_std, bool verbose, matrix<double> &sigma_draw_xinfo, vector<vector<tree>> &trees, double no_split_penalty, std::unique_ptr<State> &state, NormalModel *model, std::unique_ptr<X_struct> &x_struct, std::vector<double> &accept_count, std::vector<double> &MH_vector, std::vector<double> &P_ratio, std::vector<double> &Q_ratio, std::vector<double> &prior_ratio)
// {

//     if (state->parallel)
//         thread_pool.start();

//     // Residual for 0th tree
//     // state->residual_std = *state->y_std - state->yhat_std + state->predictions_std[0];
//     model->ini_residual_std(state);

//     double MH_ratio = 0.0;
//     double P_new;
//     double P_old;
//     double Q_new;
//     double Q_old;
//     double prior_new;
//     double prior_old;

//     std::uniform_real_distribution<> unif_dist(0, 1);

//     tree temp_treetree = tree();

//     std::vector<double> temp_vec_proposal(state->n_y);
//     std::vector<double> temp_vec(state->n_y);
//     std::vector<double> temp_vec2(state->n_y);
//     std::vector<double> temp_vec3(state->n_y);
//     std::vector<double> temp_vec4(state->n_y);

//     bool accept_flag = true;

//     for (size_t sweeps = 0; sweeps < state->num_sweeps; sweeps++)
//     {

//         if (verbose == true)
//         {
//             COUT << "--------------------------------" << endl;
//             COUT << "number of sweeps " << sweeps << endl;
//             COUT << "--------------------------------" << endl;
//         }

//         for (size_t tree_ind = 0; tree_ind < state->num_trees; tree_ind++)
//         {
//             // Draw Sigma

//             model->update_state(state, tree_ind, x_struct);

//             sigma_draw_xinfo[sweeps][tree_ind] = state->sigma;

//             if (state->use_all && (sweeps > state->burnin) && (state->mtry != state->p))
//             {
//                 state->use_all = false;
//             }

//             // clear counts of splits for one tree
//             std::fill(state->split_count_current_tree.begin(), state->split_count_current_tree.end(), 0.0);

//             // subtract old tree for sampling case
//             if (state->sample_weights_flag)
//             {
//                 state->mtry_weight_current_tree = state->mtry_weight_current_tree - state->split_count_all_tree[tree_ind];
//             }

//             //         ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//             //         // X_counts and X_num_unique should not be in state because they depend on node
//             //         // but they are initialized in state object
//             //         // so I'll pass x_struct->X_counts to root node, then create X_counts_left, X_counts_right for other nodes
//             //         ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//             if (sweeps < 10)
//             {

//                 // The first several sweeps are used as initialization
//                 // state->data_pointers is calculated in this function
//                 // trees[sweeps][tree_ind].tonull();

//                 model->initialize_root_suffstat(state, trees[sweeps][tree_ind].suff_stat);

//                 trees[sweeps][tree_ind].grow_from_root(state, Xorder_std, x_struct->X_counts, x_struct->X_num_unique, model, x_struct, sweeps, tree_ind, true, false, true);

//                 // accept_count.push_back(0);
//                 // MH_vector.push_back(0);
//             }
//             else
//             {
//                 //     // fit a proposal

//                 /*

//                     BE CAREFUL! Growing proposal update data_pointers in state object implictly
//                     need to creat a backup, copy from the backup if the proposal is rejected

//                 */

//                 //             // set sufficient statistics at root node first
//                 model->initialize_root_suffstat(state, trees[sweeps][tree_ind].suff_stat);

//                 trees[sweeps][tree_ind].grow_from_root(state, Xorder_std, x_struct->X_counts, x_struct->X_num_unique, model, x_struct, sweeps, tree_ind, true, false, true);

//                 //model->initialize_root_suffstat(state, trees[sweeps - 1][tree_ind].suff_stat);

//                 //trees[sweeps - 1][tree_ind].grow_from_root(state, Xorder_std, x_struct->X_counts, x_struct->X_num_unique, model, x_struct, sweeps, tree_ind, false, true, false);

//                 // Q_old = transition_prob(trees[sweeps - 1][tree_ind]);
//                 // P_old = tree_likelihood(trees[sweeps - 1][tree_ind]);

//                 // // // proposal
//                 // Q_new = transition_prob(trees[sweeps][tree_ind]);
//                 // P_new = tree_likelihood(trees[sweeps][tree_ind]);

//                 // MH_ratio = P_new + Q_old - P_old - Q_new;

//                 // cout << exp(MH_ratio) << endl;

//                 // if (MH_ratio > 0)
//                 // {
//                 //     MH_ratio = 1;
//                 // }
//                 // else
//                 // {
//                 //     MH_ratio = exp(MH_ratio);
//                 // }
//                 // MH_vector.push_back(MH_ratio);

//                 // Q_ratio.push_back(Q_old - Q_new);
//                 // P_ratio.push_back(P_new - P_old);

//                 // cout << "ratio is fine " << endl;

//                 //if (unif_dist(state->gen) <= MH_ratio)
//                 // if (true)
//                 // {
//                 //     // accept
//                 //     // do nothing
//                 //     // cout << "accept " << endl;
//                 //     accept_flag = true;
//                 //     accept_count.push_back(1);
//                 // }
//                 // else
//                 // {
//                 //     // reject
//                 //     // cout << "reject " << endl;
//                 //     accept_flag = false;
//                 //     accept_count.push_back(0);

//                 //     // // // keep the old tree

//                 //     // predict_from_tree(trees[sweeps - 1][tree_ind], Xpointer, N, p, temp_vec2, model);

//                 //     trees[sweeps][tree_ind].copy_only_root(&trees[sweeps - 1][tree_ind]);

//                 //     // predict_from_tree(trees[sweeps][tree_ind], Xpointer, N, p, temp_vec3, model);

//                 //     // // update theta
//                 //     /*
//                 //         update_theta() not only update leaf parameters, but also state->data_pointers

//                 //     */

//                 //     // update_theta = true, update_split_prob = true
//                 //     // resample leaf parameters

//                 //     trees[sweeps][tree_ind].grow_from_root(state, Xorder_std, x_struct->X_counts, x_struct->X_num_unique, model, x_struct, sweeps, tree_ind, true, true, false);

//                 //     //                 // predict_from_tree(trees[sweeps][tree_ind], Xpointer, N, p, temp_vec4, model);

//                 //     // // keep the old tree, need to update state object properly
//                 //     //                 // state->data_pointers[tree_ind] = state->data_pointers_copy[tree_ind];
//                 //     x_struct->restore_data_pointers(tree_ind);
//                 // }

//                 // cout << "copy is ok" << endl;
//             }

//             // if (accept_flag)
//             // {
//                 state->update_split_counts(tree_ind);
//             // }

//             // update residual
//             model->state_sweep(tree_ind, state->num_trees, state->residual_std, x_struct);
//         }

//         // after loop over all trees, backup the data_pointers matrix
//         // data_pointers_copy save result of previous sweep
//         // x_struct->data_pointers_copy = x_struct->data_pointers;
//         // state->create_backup_data_pointers();

//         // double average = accumulate(accept_count.end() - state->num_trees, accept_count.end(), 0.0) / state->num_trees;
//         // double MH_average = accumulate(MH_vector.end() - state->num_trees, MH_vector.end(), 0.0) / state->num_trees;
//         // // cout << "size of MH " << accept_count.size() << "  " << MH_vector.size() << endl;

//         // cout << "percentage of proposal acceptance " << average << endl;
//         // cout << "average MH ratio " << MH_average << endl;
//     }
//     thread_pool.stop();

//     delete model;
// }

// void mcmc_loop_MH(matrix<size_t> &Xorder_std, bool verbose, matrix<double> &sigma_draw_xinfo, vector<vector<tree>> &trees, double no_split_penalty, std::unique_ptr<State> &state, NormalModel *model, std::unique_ptr<X_struct> &x_struct, std::vector<double> &accept_count, std::vector<double> &MH_vector, std::vector<double> &P_ratio, std::vector<double> &Q_ratio, std::vector<double> &prior_ratio)
// {

//     if (state->parallel)
//         thread_pool.start();

//     // Residual for 0th tree
//     // state->residual_std = *state->y_std - state->yhat_std + state->predictions_std[0];
//     model->ini_residual_std(state);

//     std::uniform_real_distribution<> unif_dist(0, 1);

//     double MH_ratio = 0.0;
//     double P_new;
//     double P_old;
//     double Q_new;
//     double Q_old;
//     double prior_new;
//     double prior_old;
//     double logdetA_new;
//     double logdetA_old;
//     double val_new;
//     double val_old;
//     double sign_new;
//     double sign_old;

//     bool accept_flag = true;

//     for (size_t sweeps = 0; sweeps < state->num_sweeps; sweeps++)
//     {

//         if (verbose == true)
//         {
//             COUT << "--------------------------------" << endl;
//             COUT << "number of sweeps " << sweeps << endl;
//             COUT << "--------------------------------" << endl;
//         }

//         for (size_t tree_ind = 0; tree_ind < state->num_trees; tree_ind++)
//         {
//             // Draw Sigma

//             model->update_state(state, tree_ind, x_struct);

//             sigma_draw_xinfo[sweeps][tree_ind] = state->sigma;

//             if (state->use_all && (sweeps > state->burnin) && (state->mtry != state->p))
//             {
//                 state->use_all = false;
//             }

//             // clear counts of splits for one tree
//             std::fill(state->split_count_current_tree.begin(), state->split_count_current_tree.end(), 0.0);

//             // subtract old tree for sampling case
//             if (state->sample_weights_flag)
//             {
//                 state->mtry_weight_current_tree = state->mtry_weight_current_tree - state->split_count_all_tree[tree_ind];
//             }

//             if (sweeps < 10)
//             {
//                 model->initialize_root_suffstat(state, trees[sweeps][tree_ind].suff_stat);

//                 trees[sweeps][tree_ind].grow_from_root(state, Xorder_std, x_struct->X_counts, x_struct->X_num_unique, model, x_struct, sweeps, tree_ind, true, false, true);
//             }
//             else
//             {
//                 // fit proposal
//                 model->initialize_root_suffstat(state, trees[sweeps][tree_ind].suff_stat);

//                 trees[sweeps][tree_ind].grow_from_root(state, Xorder_std, x_struct->X_counts, x_struct->X_num_unique, model, x_struct, sweeps, tree_ind, true, false, true);

//                 // update old tree on new data
//                 model->initialize_root_suffstat(state, trees[sweeps - 1][tree_ind].suff_stat);
//                 // cout << "likelihood on old data " <<  transition_prob(trees[sweeps - 1][tree_ind]) << endl;
//                 // update leaf node likelihood of old tree on new data
//                 trees[sweeps - 1][tree_ind].grow_from_root(state, Xorder_std, x_struct->X_counts, x_struct->X_num_unique, model, x_struct, sweeps, tree_ind, false, true, false);
//                 // cout << "likelihood on new  data " <<  transition_prob(trees[sweeps - 1][tree_ind]) << endl;

//                 Q_old = transition_prob(trees[sweeps - 1][tree_ind]);
//                 P_old = tree_likelihood(trees[sweeps - 1][tree_ind]);

//                 // // proposal
//                 Q_new = transition_prob(trees[sweeps][tree_ind]);
//                 P_new = tree_likelihood(trees[sweeps][tree_ind]);

//                 // cout << MH_ratio << endl;

//                 // cout << "P_new " << P_new << " P_old " << P_old << " Q_new " << Q_new << " Q_old " << Q_old << endl;

//                 // cout <<  P_new + Q_old - P_old - Q_new << endl;

//                 determinant_precision(trees[sweeps][tree_ind], model->tau, state->sigma2, state, val_new, sign_new, logdetA_new);

//                 determinant_precision(trees[sweeps - 1][tree_ind], model->tau, state->sigma2, state, val_old, sign_old, logdetA_old);

//                 // cout << " ----- " << endl;

//                 // cout << logdetA_old << "  " << logdetA_new << endl;
//                 MH_ratio = P_new + Q_old - P_old - Q_new + logdetA_old - logdetA_new + val_old - val_new;

//                 // MH_ratio = P_new - P_old;

//                 // MH_ratio = 1;

//                 // cout << P_new - P_old << "   " << logdetA_old - logdetA_new << "   " << Q_old - Q_new << "   " << val_old - val_new << "   " << MH_ratio << endl;

//                 cout << MH_ratio << endl;

//                 if (MH_ratio > 0)
//                 {
//                     MH_ratio = 1;
//                 }
//                 else
//                 {
//                     MH_ratio = exp(MH_ratio) * sign_old / sign_new;
//                 }

//                 MH_vector.push_back(MH_ratio);

//                 Q_ratio.push_back(Q_old - Q_new);

//                 P_ratio.push_back(P_new - P_old);

//                 if (unif_dist(state->gen) <= MH_ratio)
//                 {
//                     // accept
//                     // do nothing
//                     cout << "accept " << endl;
//                     accept_flag = true;
//                     accept_count.push_back(1);
//                 }
//                 else
//                 {
//                     // reject
//                     cout << "reject " << endl;
//                     accept_flag = false;
//                     accept_count.push_back(0);

//                     // // // keep the old tree

//                     // predict_from_tree(trees[sweeps - 1][tree_ind], Xpointer, N, p, temp_vec2, model);

//                     trees[sweeps][tree_ind].copy_only_root(&trees[sweeps - 1][tree_ind]);

//                     // predict_from_tree(trees[sweeps][tree_ind], Xpointer, N, p, temp_vec3, model);

//                     // // update theta
//                     /*
//                         update_theta() not only update leaf parameters, but also state->data_pointers

//                     */

//                     // update_theta = true, update_split_prob = true
//                     // resample leaf parameters

//                     trees[sweeps][tree_ind].grow_from_root(state, Xorder_std, x_struct->X_counts, x_struct->X_num_unique, model, x_struct, sweeps, tree_ind, true, true, false);

//                     //                 // predict_from_tree(trees[sweeps][tree_ind], Xpointer, N, p, temp_vec4, model);

//                     // // keep the old tree, need to update state object properly
//                     //                 // state->data_pointers[tree_ind] = state->data_pointers_copy[tree_ind];
//                     x_struct->restore_data_pointers(tree_ind);
//                 }
//             }

//             if (accept_flag)
//             {
//                 state->update_split_counts(tree_ind);
//             }

//             // update partial residual for the next tree to fit
//             model->state_sweep(tree_ind, state->num_trees, state->residual_std, x_struct);
//         }

//         // create a backup of data_pointers for useage of the next sweep
//         x_struct->create_backup_data_pointers();
//     }
//     thread_pool.stop();

//     return;
// }
