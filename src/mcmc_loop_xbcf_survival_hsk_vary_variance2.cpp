#include "mcmc_loop.h"

void mcmc_loop_xbcf_survival_discrete_heteroskedastic_vary_variance2(matrix<size_t> &Xorder_std_con,
                                                                     matrix<size_t> &Xorder_std_mod,
                                                                     bool verbose,
                                                                     matrix<double> &sigma0_draw_xinfo,
                                                                     matrix<double> &sigma1_draw_xinfo,
                                                                     matrix<double> &a_xinfo,
                                                                     matrix<double> &b_xinfo,
                                                                     vector<vector<tree>> &trees_con,
                                                                     vector<vector<tree>> &trees_mod,
                                                                     vector<vector<tree>> &var_trees_con,
                                                                     vector<vector<tree>> &var_trees_mod,
                                                                     double no_split_penalty,
                                                                     State &state,
                                                                     hskXBCFDiscreteModel *model,
                                                                     logNormalXBCFModel2 *var_model,
                                                                     X_struct &x_struct_con,
                                                                     X_struct &x_struct_mod,
                                                                     X_struct &var_x_struct_con,
                                                                     X_struct &var_x_struct_mod)
{
    model->ini_tau_mu_fit(state);

    for (size_t sweeps = 0; sweeps < state.num_sweeps; sweeps++)
    {
        if (verbose == true)
        {
            COUT << "--------------------------------" << endl;
            COUT << "number of sweeps " << sweeps << endl;
            COUT << "--------------------------------" << endl;
        }

        /////////////////////////////////////////////////////////////////////////
        //
        // sampling mean forest, prognostic and treatment forests
        //
        /////////////////////////////////////////////////////////////////////////

        // prognostic forest
        model->set_treatmentflag(state, 0); // switch params (from treatment forest)
        // model->switch_state_params(state);  // switch params (from precision forest)

        for (size_t tree_ind = 0; tree_ind < state.num_trees; tree_ind++)
        {
            if (verbose)
            {
                COUT << "sweep " << sweeps << " tree " << tree_ind << endl;
            }

            if (state.use_all && (sweeps > state.burnin) && (state.mtry != state.p))
            {
                state.use_all = false;
            }

            // clear counts of splits for one tree
            (*state.split_count_current_tree).resize(state.p_con);
            std::fill((*state.split_count_current_tree).begin(), (*state.split_count_current_tree).end(), 0.0);

            // subtract old tree for sampling case
            if (state.sample_weights)
            {
                (*state.mtry_weight_current_tree_con) = (*state.mtry_weight_current_tree_con) - (*state.split_count_all_tree_con)[tree_ind];
                (*state.mtry_weight_current_tree) = (*state.mtry_weight_current_tree_con);
            }

            // update tau_fit from full fit to partial fit
            model->subtract_old_tree_fit(tree_ind, state, x_struct_con);

            // calculate partial residuals based on partial fit
            model->update_partial_residuals(tree_ind, state, x_struct_con);

            model->initialize_root_suffstat(state, trees_con[sweeps][tree_ind].suff_stat);

            trees_con[sweeps][tree_ind].grow_from_root(state, Xorder_std_con, x_struct_con.X_counts, x_struct_con.X_num_unique, model, x_struct_con, sweeps, tree_ind);

            // update tau_fit from partial fit to full fit
            model->add_new_tree_fit(tree_ind, state, x_struct_con);

            model->update_split_counts(state, tree_ind);

            if (sweeps != 0)
            {
                if (state.a_scaling)
                {
                    model->update_a(state);
                }
                if (state.b_scaling)
                {
                    model->update_b(state);
                }
            }

            if (sweeps >= state.burnin)
            {
                for (size_t i = 0; i < (*state.split_count_all_con).size(); i++)
                {
                    (*state.split_count_all_con)[i] += (*state.split_count_current_tree)[i];
                }
            }
        }

        if (model->sampling_tau)
        {
            model->update_tau_per_forest(state, sweeps, trees_con);
        }

        // treatment forest
        model->set_treatmentflag(state, 1);

        for (size_t tree_ind = 0; tree_ind < state.num_trees; tree_ind++)
        {
            if (verbose)
            {
                COUT << "sweep " << sweeps << " tree " << tree_ind << endl;
            }

            if (state.use_all && (sweeps > state.burnin) && (state.mtry != state.p))
            {
                state.use_all = false;
            }

            // clear counts of splits for one tree
            (*state.split_count_current_tree).resize(state.p_mod);
            std::fill((*state.split_count_current_tree).begin(), (*state.split_count_current_tree).end(), 0.0);

            // subtract old tree for sampling case
            if (state.sample_weights)
            {
                (*state.mtry_weight_current_tree_mod) = (*state.mtry_weight_current_tree_mod) - (*state.split_count_all_tree_mod)[tree_ind];
                (*state.mtry_weight_current_tree) = (*state.mtry_weight_current_tree_mod);
            }

            // update tau_fit from full fit to partial fit
            model->subtract_old_tree_fit(tree_ind, state, x_struct_mod);

            // calculate partial residuals based on partial fit
            model->update_partial_residuals(tree_ind, state, x_struct_mod);

            model->initialize_root_suffstat(state, trees_mod[sweeps][tree_ind].suff_stat);

            trees_mod[sweeps][tree_ind].grow_from_root(state, Xorder_std_mod, x_struct_mod.X_counts, x_struct_mod.X_num_unique, model, x_struct_mod, sweeps, tree_ind);

            // update tau_fit from partial fit to full fit
            model->add_new_tree_fit(tree_ind, state, x_struct_mod);

            model->update_split_counts(state, tree_ind);

            if (sweeps >= state.burnin)
            {
                for (size_t i = 0; i < (*state.split_count_all_mod).size(); i++)
                {
                    (*state.split_count_all_mod)[i] += (*state.split_count_current_tree)[i];
                }
            }
        }

        if (model->sampling_tau)
        {
            model->update_tau_per_forest(state, sweeps, trees_mod);
        }

        b_xinfo[0][sweeps] = state.b_vec[0];
        b_xinfo[1][sweeps] = state.b_vec[1];
        a_xinfo[0][sweeps] = state.a;

        model->update_state(state); // update residual to full, switch some parameters

        /////////////////////////////////////////////////////////////////////////
        //
        // sampling variance forest, prognostic and treatment forests
        //
        /////////////////////////////////////////////////////////////////////////

        // prognostic forest
        model->set_treatmentflag(state, 0); // switch params
        // model->switch_state_params(state);  // switch params (from precision forest)

        var_model->ini_residual_std2(state, var_x_struct_con, var_x_struct_mod);

        // var_model->switch_state_params(state);

        // loop for the variance model forest, prognostic forest
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
                (*state.mtry_weight_current_tree_v_con) = (*state.mtry_weight_current_tree_v_con) - (*state.split_count_all_tree_v_con)[tree_ind];
                (*state.mtry_weight_current_tree) = (*state.mtry_weight_current_tree_v_con);
            }

            var_model->initialize_root_suffstat(state, var_trees_con[sweeps][tree_ind].suff_stat);

            // single core
            var_trees_con[sweeps][tree_ind].grow_from_root(state, Xorder_std_con, var_x_struct_con.X_counts, var_x_struct_con.X_num_unique, var_model, var_x_struct_con, sweeps, tree_ind);

            state.update_split_counts(tree_ind);

            // update partial residual for the next tree to fit
            var_model->state_sweep(state, tree_ind, state.num_trees, (*state.residual_std), var_x_struct_con, var_x_struct_con);
        }

        var_model->update_state(state, var_x_struct_con, var_x_struct_mod);

        // treatment forest
        model->set_treatmentflag(state, 1);
        var_model->ini_residual_std2(state, var_x_struct_con, var_x_struct_mod);

        // loop for the variance model forest, treatment forest
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
                (*state.mtry_weight_current_tree_v_mod) = (*state.mtry_weight_current_tree_v_mod) - (*state.split_count_all_tree_v_mod)[tree_ind];
                (*state.mtry_weight_current_tree) = (*state.mtry_weight_current_tree_v_mod);
            }

            var_model->initialize_root_suffstat(state, var_trees_mod[sweeps][tree_ind].suff_stat);

            // single core
            var_trees_mod[sweeps][tree_ind].grow_from_root(state, Xorder_std_mod, var_x_struct_mod.X_counts, var_x_struct_mod.X_num_unique, var_model, var_x_struct_mod, sweeps, tree_ind);

            state.update_split_counts(tree_ind);

            // update partial residual for the next tree to fit
            var_model->state_sweep(state, tree_ind, state.num_trees, (*state.residual_std), var_x_struct_mod, var_x_struct_mod);
        }

        // pass fitted values for sigmas to the mean model
        var_model->update_state(state, var_x_struct_con, var_x_struct_mod);

        // sample missing values in the truncated survival data
        // precision is saved in (*state.precision)
        // impute missing values from the truncated normal, should be larger than T_obs

        for (size_t ii = 0; ii < (*state.delta_std).size(); ii++)
        {
            // loop over all the data, check tau
            if ((*state.delta_std)[ii] == 1)
            {
                // T_obs = T, copy
                (*state.y_imputed)[ii] = (*state.y_std)[ii];
            }
            else
            {
                // T_obs = L, impute T from truncated normal
                (*state.y_imputed)[ii] = sample_truncated_normal(state.gen, (*state.y_imputed_save)[ii] - (*state.residual_std)[0][ii], (*state.precision)[ii], (*state.y_std)[ii], true);
            }
        }

        // update partial residual for the mean forest, fitted values does not change, but the data changes
        for (size_t ii = 0; ii < (*state.delta_std).size(); ii++)
        {
            (*state.residual_std)[0][ii] = (*state.residual_std)[0][ii] - (*state.y_imputed_save)[ii] + (*state.y_imputed)[ii];

            (*state.y_imputed_save)[ii] = (*state.y_imputed)[ii];

            (*state.res_x_precision)[ii] = (*state.residual_std)[0][ii] * (*state.precision)[ii];
        }
    }
    return;
}