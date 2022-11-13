#include "mcmc_loop.h"

void mcmc_loop_heteroskedastic(matrix<size_t> &Xorder_std,
                    bool verbose,
                    State &state,
                    hskNormalModel *mean_model,
                    vector<vector<tree>> &mean_trees,
                    X_struct &mean_x_struct,
                    logNormalModel *var_model,
                    vector<vector<tree>> &var_trees,
                    X_struct &var_x_struct
                    )
{
    mean_model->ini_residual_std(state);

    for (size_t sweeps = 0; sweeps < state.num_sweeps; sweeps++)
    {

        if (verbose == true)
        {
            COUT << "--------------------------------" << endl;
            COUT << "number of sweeps " << sweeps << endl;
            COUT << "--------------------------------" << endl;
        }
        mean_model->switch_state_params(state);

        // loop for the mean model forest
        for (size_t tree_ind = 0; tree_ind < state.num_trees; tree_ind++)
        {

            if (verbose)
            {
                cout << "sweep " << sweeps << " tree " << tree_ind << endl;
            }

            if (state.use_all && (sweeps > state.burnin) && (state.mtry != state.p))
            {
                state.use_all = false;
            }

            // clear counts of splits for one tree
            std::fill((*state.split_count_current_tree).begin(), (*state.split_count_current_tree).end(), 0.0);

            // subtract old tree for sampling case
            if (state.sample_weights)
            {
                (*state.mtry_weight_current_tree_m) = (*state.mtry_weight_current_tree_m) - (*state.split_count_all_tree_m)[tree_ind];
                (*state.mtry_weight_current_tree) = (*state.mtry_weight_current_tree_m);
            }

            mean_model->initialize_root_suffstat(state, mean_trees[sweeps][tree_ind].suff_stat);

            // single core
            mean_trees[sweeps][tree_ind].grow_from_root(state, Xorder_std, mean_x_struct.X_counts, mean_x_struct.X_num_unique, mean_model, mean_x_struct, sweeps, tree_ind);

            // update tau after sampling the tree
            // model->update_tau(state, tree_ind, sweeps, trees);

            state.update_split_counts(tree_ind);

            // update partial residual for the next tree to fit
            mean_model->state_sweep(tree_ind, state.num_trees, state, mean_x_struct);
        }

        if (mean_model->sampling_tau)
        {
            mean_model->update_tau_per_forest(state, sweeps, mean_trees);
        }

        mean_model->store_residual(state);
        var_model->ini_residual_std2(state, var_x_struct);
        var_model->switch_state_params(state);

        // loop for the variance model forest
        for (size_t tree_ind = 0; tree_ind < state.num_trees; tree_ind++)
        {

            if (verbose)
            {
                cout << "sweep " << sweeps << " tree " << tree_ind << endl;
            }

            if (state.use_all && (sweeps > state.burnin) && (state.mtry != state.p))
            {
                state.use_all = false;
            }

            // clear counts of splits for one tree
            std::fill((*state.split_count_current_tree).begin(), (*state.split_count_current_tree).end(), 0.0);

            // subtract old tree for sampling case
            if (state.sample_weights)
            {
                (*state.mtry_weight_current_tree_v) = (*state.mtry_weight_current_tree_v) - (*state.split_count_all_tree_v)[tree_ind];
                (*state.mtry_weight_current_tree) = (*state.mtry_weight_current_tree_v);
            }

            var_model->initialize_root_suffstat(state, var_trees[sweeps][tree_ind].suff_stat);

            // single core
            var_trees[sweeps][tree_ind].grow_from_root(state, Xorder_std, var_x_struct.X_counts, var_x_struct.X_num_unique, var_model, var_x_struct, sweeps, tree_ind);

            state.update_split_counts(tree_ind);

            // update partial residual for the next tree to fit
            var_model->state_sweep(tree_ind, state.num_trees, (*state.residual_std), var_x_struct);
        }

        // pass fitted values for sigmas to the mean model
        var_model->update_state(state, state.num_trees, var_x_struct);

    }
    // thread_pool.stop();

    return;
}
