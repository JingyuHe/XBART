//////////////////////////////////////////////////////////////////////////////////////
// main function of the Bayesian backfitting algorithm
//////////////////////////////////////////////////////////////////////////////////////

#include "mcmc_loop.h"

void mcmc_loop(matrix<size_t> &Xorder_std, bool verbose, matrix<double> &sigma_draw_xinfo, vector<vector<tree>> &trees, double no_split_penalty, std::unique_ptr<State> &state, NormalModel *model, std::unique_ptr<X_struct> &x_struct, std::vector<double> &resid)
{
    size_t N = state->residual_std[0].size();

    // initialize the matrix of residuals
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
                COUT << "sweep " << sweeps << " tree " << tree_ind << endl;
            }

            // draw Sigma
            model->update_state(state, tree_ind, x_struct);

            sigma_draw_xinfo[sweeps][tree_ind] = state->sigma;

            if (state->use_all && (sweeps > state->burnin) && (state->mtry != state->p))
            {
                state->use_all = false;
            }

            // clear counts of splits for one tree
            std::fill(state->split_count_current_tree.begin(), state->split_count_current_tree.end(), 0.0);

            // subtract old tree for sampling case
            if (state->sample_weights)
            {
                state->mtry_weight_current_tree = state->mtry_weight_current_tree - state->split_count_all_tree[tree_ind];
            }

            // initialize sufficient statistics of the current tree to be updated
            model->initialize_root_suffstat(state, trees[sweeps][tree_ind].suff_stat);

            if (state->parallel)
            {
                trees[sweeps][tree_ind].settau(model->tau_prior, model->tau); // initiate tau
            }

            // main function to grow the tree from root
            trees[sweeps][tree_ind].grow_from_root(state, Xorder_std, x_struct->X_counts, x_struct->X_num_unique, model, x_struct, sweeps, tree_ind);

            // set id for bottom nodes
            tree::npv bv;
            trees[sweeps][tree_ind].getbots(bv); // get bottom nodes
            for (size_t i = 0; i < bv.size(); i++)
            {
                bv[i]->setID(i + 1);
            }

            // store residuals:
            for (size_t data_ind = 0; data_ind < state->residual_std[0].size(); data_ind++)
            {
                resid[data_ind + sweeps * N + tree_ind * state->num_sweeps * N] = state->residual_std[0][data_ind];
            }

            // count number of splits at each variable
            state->update_split_counts(tree_ind);

            // update partial residual for the next tree to fit
            model->state_sweep(tree_ind, state->num_trees, state->residual_std, x_struct);
        }

        if (model->sampling_tau)
        {
            // update tau per sweep (after drawing a forest)
            model->update_tau_per_forest(state, sweeps, trees);
        }
    }
    return;
}

void mcmc_loop_multinomial(matrix<size_t> &Xorder_std, bool verbose, vector<vector<tree>> &trees, double no_split_penalty, std::unique_ptr<State> &state, LogitModel *model, std::unique_ptr<X_struct> &x_struct, std::vector<std::vector<double>> &weight_samples, std::vector<double> &lambda_samples, std::vector<std::vector<double>> &tau_samples)
{
    // Residual for 0th tree
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
                COUT << "sweep " << sweeps << " tree " << tree_ind << endl;
            }
            // Draw latents -- do last?

            if (state->use_all && (sweeps >= state->burnin)) // && (state->mtry != state->p) // If mtry = p, it will all be sampled anyway. Now use_all can be an indication of burnin period.
            {
                state->use_all = false;
            }

            // clear counts of splits for one tree
            std::fill(state->split_count_current_tree.begin(), state->split_count_current_tree.end(), 0.0);

            // subtract old tree for sampling case
            if (state->sample_weights)
            {
                state->mtry_weight_current_tree = state->mtry_weight_current_tree - state->split_count_all_tree[tree_ind];
            }

            model->initialize_root_suffstat(state, trees[sweeps][tree_ind].suff_stat);
            for (size_t k = 0; k < trees[sweeps][tree_ind].suff_stat.size(); k++)
            {
                if (std::isnan(trees[sweeps][tree_ind].suff_stat[k]))
                {
                    COUT << "unidentified error: suffstat " << k << " initialized as nan" << endl;
                    exit(1);
                }
            }

            trees[sweeps][tree_ind].theta_vector.resize(model->dim_residual);
            state->lambdas[tree_ind].clear();

            trees[sweeps][tree_ind].grow_from_root_entropy(state, Xorder_std, x_struct->X_counts, x_struct->X_num_unique, model, x_struct, sweeps, tree_ind);

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
    }
}

void mcmc_loop_multinomial_sample_per_tree(matrix<size_t> &Xorder_std, bool verbose, vector<vector<vector<tree>>> &trees, double no_split_penality, std::unique_ptr<State> &state, LogitModelSeparateTrees *model, std::unique_ptr<X_struct> &x_struct, std::vector<std::vector<double>> &weight_samples)
{

    // Residual for 0th tree
    model->ini_residual_std(state);
    size_t p = Xorder_std.size();
    std::vector<size_t> subset_vars(p);
    std::vector<double> weight_samp(p);

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
                COUT << "sweep " << sweeps << " tree " << tree_ind << endl;
            }
            // Draw latents -- do last?

            if (state->use_all && (sweeps >= state->burnin)) // && (state->mtry != state->p) // If mtry = p, it will all be sampled anyway. Now use_all can be an indication of burnin period.
            {
                state->use_all = false;
            }

            // clear counts of splits for one tree
            std::fill(state->split_count_current_tree.begin(), state->split_count_current_tree.end(), 0.0);

            // subtract old tree for sampling case
            if (state->sample_weights)
            {
                state->mtry_weight_current_tree = state->mtry_weight_current_tree - state->split_count_all_tree[tree_ind];
            }

            for (size_t class_ind = 0; class_ind < model->dim_residual; class_ind++)
            {
                model->set_class_operating(class_ind);

                state->lambdas_separate[tree_ind][class_ind].clear();

                model->initialize_root_suffstat(state, trees[class_ind][sweeps][tree_ind].suff_stat);

                trees[class_ind][sweeps][tree_ind].theta_vector.resize(model->dim_residual);

                trees[class_ind][sweeps][tree_ind].grow_from_root_separate_tree(state, Xorder_std, x_struct->X_counts, x_struct->X_num_unique, model, x_struct, sweeps, tree_ind);
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

void mcmc_loop_linear(matrix<size_t> &Xorder_std, bool verbose, matrix<double> &sigma_draw_xinfo, vector<vector<tree>> &trees_ps, vector<vector<tree>> &trees_trt, double no_split_penalty, std::unique_ptr<State> &state, NormalLinearModel *model, std::unique_ptr<X_struct> &x_struct_ps, std::unique_ptr<X_struct> &x_struct_trt)
{
    model->ini_residual_std(state);

    for (size_t sweeps = 0; sweeps < state->num_sweeps; sweeps++)
    {
        if (verbose == true)
        {
            COUT << "--------------------------------" << endl;
            COUT << "number of sweeps " << sweeps << endl;
            COUT << "--------------------------------" << endl;
        }

        model->set_treatmentflag(state, 1);

        for (size_t tree_ind = 0; tree_ind < state->num_trees; tree_ind++)
        {
            if (verbose)
            {
                cout << "sweep " << sweeps << " tree " << tree_ind << endl;
            }

            // Draw Sigma
            model->update_state(state, tree_ind, x_struct_trt);

            sigma_draw_xinfo[sweeps][tree_ind] = state->sigma;

            if (state->use_all && (sweeps > state->burnin) && (state->mtry != state->p))
            {
                state->use_all = false;
            }

            // clear counts of splits for one tree
            std::fill(state->split_count_current_tree.begin(), state->split_count_current_tree.end(), 0.0);

            // subtract old tree for sampling case
            if (state->sample_weights)
            {
                state->mtry_weight_current_tree = state->mtry_weight_current_tree - state->split_count_all_tree[tree_ind];
            }

            // update tau_fit from full fit to partial fit
            model->subtract_old_tree_fit(tree_ind, state, x_struct_trt);

            // calculate partial residuals based on partial fit
            model->update_partial_residuals(tree_ind, state, x_struct_trt);

            model->initialize_root_suffstat(state, trees_trt[sweeps][tree_ind].suff_stat);

            trees_trt[sweeps][tree_ind].grow_from_root(state, Xorder_std, x_struct_trt->X_counts, x_struct_trt->X_num_unique, model, x_struct_trt, sweeps, tree_ind);

            // update tau_fit from partial fit to full fit
            model->add_new_tree_fit(tree_ind, state, x_struct_trt);

            state->update_split_counts(tree_ind);
        }

        if (model->sampling_tau)
        {
            model->update_tau_per_forest(state, sweeps, trees_trt);
        }
    }
    return;
}
