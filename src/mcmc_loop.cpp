#include "mcmc_loop.h"
#include "omp.h"

void mcmc_loop(matrix<size_t> &Xorder_std, bool verbose, matrix<double> &sigma_draw_xinfo, vector<vector<tree>> &trees, double no_split_penalty, std::unique_ptr<State> &state, NormalModel *model, std::unique_ptr<X_struct> &x_struct, std::vector<double> &resid)
{

    // if (state->parallel)
    //     thread_pool.start();

    size_t N = state->residual_std[0].size();
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
                trees[sweeps][tree_ind].settau(model->tau_prior, model->tau); // initiate tau
                trees[sweeps][tree_ind].grow_from_root(state, Xorder_std, x_struct->X_counts, x_struct->X_num_unique, model, x_struct, sweeps, tree_ind, true, false, true);
            }
            else
            {
                // single core
                trees[sweeps][tree_ind].grow_from_root(state, Xorder_std, x_struct->X_counts, x_struct->X_num_unique, model, x_struct, sweeps, tree_ind, true, false, true);
            }

            // set id for bottom nodes
            tree::npv bv;
            trees[sweeps][tree_ind].getbots(bv); // get bottom nodes
            for (size_t i = 0; i < bv.size(); i++)
            {
                bv[i]->setID(i + 1);
                // cout << bv[i]->getID() << " " << endl;
            }

            // store residuals:
            for (size_t data_ind = 0; data_ind < state->residual_std[0].size(); data_ind++)
            {
                resid[data_ind + sweeps * N + tree_ind * state->num_sweeps * N] = state->residual_std[0][data_ind];
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

            // Rcpp::Rcout << "Updating state";

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
                if (isnan(trees[sweeps][tree_ind].suff_stat[k]))
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

            // omp_set_max_active_levels(3);
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

            // Rcpp::Rcout << "Updating state";

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

            // omp_set_max_active_levels(3);
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
