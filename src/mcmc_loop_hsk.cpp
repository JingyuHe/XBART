#include "mcmc_loop.h"
#include "omp.h"

// full heteroskedastic XBART (crashing atm)
void mcmc_loop_hsk(matrix<size_t> &Xorder_std,
                    bool verbose,
                    matrix<double> &sigma_draw_xinfo,
                    vector<vector<tree>> &mean_trees,
                    std::unique_ptr<State> &mean_state,
                    hskNormalModel *mean_model,
                    std::unique_ptr<X_struct> &mean_x_struct,
                    vector<vector<tree>> &var_trees,
                    std::unique_ptr<State> &var_state,
                    logNormalModel *var_model,
                    std::unique_ptr<X_struct> &var_x_struct)
{

    mean_model->ini_residual_std(mean_state);

    // NK: the question is still one state or two states?
    // NK: if two, does storing num_sweeps in the state object make sense?
    // NK: if not, we'll just pass num_sweeps as a aseparate variable
    for (size_t sweeps = 0; sweeps < mean_state->num_sweeps; sweeps++)
    {
/*
        if (verbose == true)
        {
            COUT << "--------------------------------" << endl;
            COUT << "number of sweeps " << sweeps << endl;
            COUT << "--------------------------------" << endl;
        }

        // NK: loop for the mean model forest
        for (size_t tree_ind = 0; tree_ind < mean_state->num_trees; tree_ind++)
        {

            if (verbose)
            {
                cout << "sweep " << sweeps << " tree " << tree_ind << endl;
            }
            // Draw Sigma
            // NK:(here we don't since sigma is fit by a separate model)
            // model->update_state(state, tree_ind, x_struct);
            // NK: and then we don't store it here either
            // sigma_draw_xinfo[sweeps][tree_ind] = state->sigma;

            if (mean_state->use_all && (sweeps > mean_state->burnin) && (mean_state->mtry != mean_state->p))
            {
                mean_state->use_all = false;
            }

            // clear counts of splits for one tree
            std::fill(mean_state->split_count_current_tree.begin(), mean_state->split_count_current_tree.end(), 0.0);

            // subtract old tree for sampling case
            if (mean_state->sample_weights_flag)
            {
                mean_state->mtry_weight_current_tree = mean_state->mtry_weight_current_tree - mean_state->split_count_all_tree[tree_ind];
            }

            mean_model->initialize_root_suffstat(mean_state, mean_trees[sweeps][tree_ind].suff_stat);
            // single core
            mean_trees[sweeps][tree_ind].grow_from_root(mean_state, Xorder_std, mean_x_struct->X_counts, mean_x_struct->X_num_unique, mean_model, mean_x_struct, sweeps, tree_ind, true, false, true);

            // update tau after sampling the tree
            // model->update_tau(state, tree_ind, sweeps, trees);

            mean_state->update_split_counts(tree_ind);

            // update partial residual for the next tree to fit
            mean_model->state_sweep(tree_ind, mean_state->num_trees, mean_state->residual_std, mean_x_struct);
        }

        // NK: currently the additional parameters are off, so we don't update them
        //if (mean_model->sampling_tau)
        //{
        //    mean_model->update_tau_per_forest(mean_state, sweeps, trees);
        //}
*/
        var_model->ini_residual_std(var_state, mean_state->residual_std, var_x_struct);

        // NK: loop for the variance model forest
        for (size_t tree_ind = 0; tree_ind < var_state->num_trees; tree_ind++)
        {

            if (verbose)
            {
                cout << "sweep " << sweeps << " tree " << tree_ind << endl;
            }
            // Draw Sigma
            // NK:(here we don't since sigma as a parameter is not a part of the model)
            // model->update_state(state, tree_ind, x_struct);
            // NK: and we don't need to store it
            // sigma_draw_xinfo[sweeps][tree_ind] = state->sigma;

            if (var_state->use_all && (sweeps > var_state->burnin) && (var_state->mtry != var_state->p))
            {
                var_state->use_all = false;
            }

            // clear counts of splits for one tree
            std::fill(var_state->split_count_current_tree.begin(), var_state->split_count_current_tree.end(), 0.0);

            // subtract old tree for sampling case
            if (var_state->sample_weights_flag)
            {
                var_state->mtry_weight_current_tree = var_state->mtry_weight_current_tree - var_state->split_count_all_tree[tree_ind];
            }

            var_model->initialize_root_suffstat(var_state, var_trees[sweeps][tree_ind].suff_stat);

            // single core
            var_trees[sweeps][tree_ind].grow_from_root(var_state, Xorder_std, var_x_struct->X_counts, var_x_struct->X_num_unique, var_model, var_x_struct, sweeps, tree_ind, true, false, true);

            // update tau after sampling the tree
            // model->update_tau(state, tree_ind, sweeps, trees);

            var_state->update_split_counts(tree_ind);

            // update partial residual for the next tree to fit
            var_model->state_sweep(tree_ind, var_state->num_trees, var_state->residual_std, var_state->var_fit, var_x_struct);
        }

        // pass fitted values for sigmas to the mean model
        var_model->update_sigmas(mean_state->residual_std, var_state->var_fit);

        // NK: currently the additional parameters are off, so we don't update them
        //if (var_model->sampling_tau)
        //{
        //    var_model->update_tau_per_forest(var_state, sweeps, trees);
        //}

    }
    // thread_pool.stop();

    return;
}

// partial heteroskedastic XBART for testing
/*
void mcmc_loop_hsk_test(matrix<size_t> &Xorder_std,
                    bool verbose,
                    matrix<double> &sigma_draw_xinfo,
                    vector<vector<tree>> &mean_trees,
                    std::unique_ptr<State> &mean_state,
                    hskNormalModel *mean_model,
                    std::unique_ptr<X_struct> &mean_x_struct)
{

    mean_model->ini_residual_std(mean_state);

    // NK: the question is still one state or two states?
    // NK: if two, does storing num_sweeps in the state object make sense?
    // NK: if not, we'll just pass num_sweeps as a aseparate variable
    for (size_t sweeps = 0; sweeps < mean_state->num_sweeps; sweeps++)
    {

        if (verbose == true)
        {
            COUT << "--------------------------------" << endl;
            COUT << "number of sweeps " << sweeps << endl;
            COUT << "--------------------------------" << endl;
        }

        // NK: loop for the mean model forest
        for (size_t tree_ind = 0; tree_ind < mean_state->num_trees; tree_ind++)
        {

            if (verbose)
            {
                cout << "sweep " << sweeps << " tree " << tree_ind << endl;
            }
            // Draw Sigma
            // NK:(here we don't since sigma is fit by a separate model)
            // model->update_state(state, tree_ind, x_struct);
            // NK: and then we don't store it here either
            // sigma_draw_xinfo[sweeps][tree_ind] = state->sigma;

            if (mean_state->use_all && (sweeps > mean_state->burnin) && (mean_state->mtry != mean_state->p))
            {
                mean_state->use_all = false;
            }

            // clear counts of splits for one tree
            std::fill(mean_state->split_count_current_tree.begin(), mean_state->split_count_current_tree.end(), 0.0);

            // subtract old tree for sampling case
            if (mean_state->sample_weights_flag)
            {
                mean_state->mtry_weight_current_tree = mean_state->mtry_weight_current_tree - mean_state->split_count_all_tree[tree_ind];
            }

            mean_model->initialize_root_suffstat(mean_state, mean_trees[sweeps][tree_ind].suff_stat);

            mean_trees[sweeps][tree_ind].grow_from_root(mean_state, Xorder_std, mean_x_struct->X_counts, mean_x_struct->X_num_unique, mean_model, mean_x_struct, sweeps, tree_ind, true, false, true);

            // update tau after sampling the tree
            // model->update_tau(state, tree_ind, sweeps, trees);

            mean_state->update_split_counts(tree_ind);

            // update partial residual for the next tree to fit
            mean_model->state_sweep(tree_ind, mean_state->num_trees, mean_state->residual_std, mean_x_struct);
        }

    }
    // thread_pool.stop();

    return;
}
*/