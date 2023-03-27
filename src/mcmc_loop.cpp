//////////////////////////////////////////////////////////////////////////////////////
// main function of the Bayesian backfitting algorithm
//////////////////////////////////////////////////////////////////////////////////////

#include "mcmc_loop.h"

void mcmc_loop(matrix<size_t> &Xorder_std, bool verbose, matrix<double> &sigma_draw_xinfo, vector<vector<tree>> &trees, double no_split_penalty, State &state, NormalModel *model, X_struct &x_struct, std::vector<double> &resid)
{
    size_t N = (*state.residual_std)[0].size();

    // initialize the matrix of residuals
    model->ini_residual_std(state);

    for (size_t sweeps = 0; sweeps < state.num_sweeps; sweeps++)
    {

        if (verbose == true)
        {
            COUT << "--------------------------------" << endl;
            COUT << "number of sweeps " << sweeps << endl;
            COUT << "--------------------------------" << endl;
        }

        for (size_t tree_ind = 0; tree_ind < state.num_trees; tree_ind++)
        {

            if (verbose)
            {
                COUT << "sweep " << sweeps << " tree " << tree_ind << endl;
            }

            // draw Sigma
            model->update_state(state, tree_ind, x_struct);

            sigma_draw_xinfo[sweeps][tree_ind] = state.sigma;

            if (state.use_all && (sweeps > state.burnin) && (state.mtry != state.p))
            {
                state.use_all = false;
            }

            // clear counts of splits for one tree
            std::fill((*state.split_count_current_tree).begin(), (*state.split_count_current_tree).end(), 0.0);

            // subtract old tree for sampling case
            if (state.sample_weights)
            {
                (*state.mtry_weight_current_tree) = (*state.mtry_weight_current_tree) - (*state.split_count_all_tree)[tree_ind];
            }

            // initialize sufficient statistics of the current tree to be updated
            model->initialize_root_suffstat(state, trees[sweeps][tree_ind].suff_stat);

            if (state.parallel)
            {
                trees[sweeps][tree_ind].settau(model->tau_prior, model->tau); // initiate tau
            }

            // main function to grow the tree from root
            trees[sweeps][tree_ind].grow_from_root(state, Xorder_std, x_struct.X_counts, x_struct.X_num_unique, model, x_struct, sweeps, tree_ind);

            // set id for bottom nodes
            tree::npv bv;
            trees[sweeps][tree_ind].getbots(bv); // get bottom nodes
            for (size_t i = 0; i < bv.size(); i++)
            {
                bv[i]->setID(i + 1);
            }

            // store residuals:
            for (size_t data_ind = 0; data_ind < (*state.residual_std)[0].size(); data_ind++)
            {
                resid[data_ind + sweeps * N + tree_ind * state.num_sweeps * N] = (*state.residual_std)[0][data_ind];
            }

            if (sweeps >= state.burnin)
            {
                for (size_t i = 0; i < (*state.split_count_all).size(); i++)
                {
                    (*state.split_count_all)[i] += (*state.split_count_current_tree)[i];
                }
            }

            // count number of splits at each variable
            state.update_split_counts(tree_ind);

            // update partial residual for the next tree to fit
            model->state_sweep(tree_ind, state.num_trees, (*state.residual_std), x_struct);
        }

        if (model->sampling_tau)
        {
            // update tau per sweep (after drawing a forest)
            model->update_tau_per_forest(state, sweeps, trees);
        }
    }
    return;
}

void mcmc_loop_multinomial(matrix<size_t> &Xorder_std, bool verbose, vector<vector<tree>> &trees, double no_split_penalty, State &state, LogitModel *model, X_struct &x_struct,
                           std::vector<std::vector<double>> &weight_samples, std::vector<double> &lambda_samples, std::vector<std::vector<double>> &phi_samples, std::vector<std::vector<double>> &logloss,
                           std::vector<std::vector<double>> &tree_size)
{
    // Residual for 0th tree
    model->ini_residual_std(state);

    // keep track of mean of leaf parameters
    double mean_lambda = 1;
    size_t count_lambda = (state.num_trees - 1) * model->dim_residual; // less the lambdas in the first tree
    std::vector<double> var_lambda(state.num_trees, 0.0);

    for (size_t sweeps = 0; sweeps < state.num_sweeps; sweeps++)
    {

        if (verbose == true)
        {
            COUT << "--------------------------------" << endl;
            COUT << "number of sweeps " << sweeps << endl;
            COUT << "--------------------------------" << endl;
        }

        for (size_t tree_ind = 0; tree_ind < state.num_trees; tree_ind++)
        {
            // Draw latents -- do last?

            if (state.use_all && (sweeps >= state.burnin)) // && (state.mtry != state.p) // If mtry = p, it will all be sampled anyway. Now use_all can be an indication of burnin period.
            {
                state.use_all = false;
            }

            // clear counts of splits for one tree
            std::fill((*state.split_count_current_tree).begin(), (*state.split_count_current_tree).end(), 0.0);

            // subtract old tree for sampling case
            if (state.sample_weights)
            {
                (*state.mtry_weight_current_tree) = (*state.mtry_weight_current_tree) - (*state.split_count_all_tree)[tree_ind];
            }

            model->initialize_root_suffstat(state, trees[sweeps][tree_ind].suff_stat);

            trees[sweeps][tree_ind].theta_vector.resize(model->dim_residual);
            (*state.lambdas)[tree_ind].clear();

            trees[sweeps][tree_ind].grow_from_root_entropy(state, Xorder_std, x_struct.X_counts, x_struct.X_num_unique, model, x_struct, sweeps, tree_ind);

            state.update_split_counts(tree_ind);

            if (sweeps >= state.burnin)
            {
                for (size_t i = 0; i < (*state.split_count_all).size(); i++)
                {
                    (*state.split_count_all)[i] += (*state.split_count_current_tree)[i];
                }
            }
            // update partial fits for the next tree
            model->update_state(state, tree_ind, x_struct, mean_lambda, var_lambda, count_lambda);

            model->state_sweep(tree_ind, state.num_trees, (*state.residual_std), x_struct);

            weight_samples[sweeps][tree_ind] = model->weight;
            phi_samples[sweeps][tree_ind] = exp((*model->phi)[0]);
            logloss[sweeps][tree_ind] = model->logloss;
            tree_size[sweeps][tree_ind] = trees[sweeps][tree_ind].treesize();

            if (verbose)
            {
                if (sweeps > 0)
                {
                    COUT << " --- --- --- " << endl;
                    COUT << "tree " << tree_ind << " old size = " << trees[sweeps - 1][tree_ind].treesize() << ", new size = " << trees[sweeps][tree_ind].treesize() << endl;
                    COUT << " logloss " << model->logloss << " acc " << model->accuracy << endl;
                }
            }

            for (size_t j = 0; j < (*state.lambdas)[tree_ind].size(); j++)
            {
                for (size_t k = 0; k < (*state.lambdas)[tree_ind][j].size(); k++)
                {
                    lambda_samples.push_back((*state.lambdas)[tree_ind][j][k]);
                }
            }
        }
        model->update_weights(state, x_struct, mean_lambda, var_lambda, count_lambda);
    }
}

void mcmc_loop_multinomial_sample_per_tree(matrix<size_t> &Xorder_std, bool verbose, vector<vector<vector<tree>>> &trees, double no_split_penalty, State &state,
                                           LogitModelSeparateTrees *model, X_struct &x_struct,
                                           std::vector<std::vector<double>> &weight_samples, std::vector<std::vector<double>> &tau_samples,
                                           std::vector<std::vector<double>> &logloss, std::vector<std::vector<double>> &tree_size)
{

    // Residual for 0th tree
    model->ini_residual_std(state);
    size_t p = Xorder_std.size();
    std::vector<size_t> subset_vars(p);
    std::vector<double> weight_samp(p);

    // keep track of mean of leaf parameters
    double mean_lambda = 1;
    size_t count_lambda = (state.num_trees - 1) * model->dim_residual; // less the lambdas in the first tree
    std::vector<double> var_lambda(state.num_trees, 0.0);

    for (size_t sweeps = 0; sweeps < state.num_sweeps; sweeps++)
    {

        if (verbose == true)
        {
            COUT << "--------------------------------" << endl;
            COUT << "number of sweeps " << sweeps << endl;
            COUT << "--------------------------------" << endl;
        }

        for (size_t tree_ind = 0; tree_ind < state.num_trees; tree_ind++)
        {

            // if (verbose)
            // {
            //     COUT << "sweep " << sweeps << " tree " << tree_ind << endl;
            // }
            // Draw latents -- do last?

            if (state.use_all && (sweeps >= state.burnin)) // && (state.mtry != state.p) // If mtry = p, it will all be sampled anyway. Now use_all can be an indication of burnin period.
            {
                state.use_all = false;
            }

            // clear counts of splits for one tree
            std::fill((*state.split_count_current_tree).begin(), (*state.split_count_current_tree).end(), 0.0);

            // subtract old tree for sampling case
            if (state.sample_weights)
            {
                (*state.mtry_weight_current_tree) = (*state.mtry_weight_current_tree) - (*state.split_count_all_tree)[tree_ind];
            }

            tree_size[sweeps][tree_ind] = 0; // init

            for (size_t class_ind = 0; class_ind < model->dim_residual; class_ind++)
            {
                model->set_class_operating(class_ind);

                (*state.lambdas_separate)[tree_ind][class_ind].clear();

                model->initialize_root_suffstat(state, trees[class_ind][sweeps][tree_ind].suff_stat);

                trees[class_ind][sweeps][tree_ind].theta_vector.resize(model->dim_residual);

                trees[class_ind][sweeps][tree_ind].grow_from_root_separate_tree(state, Xorder_std, x_struct.X_counts, x_struct.X_num_unique, model, x_struct, sweeps, tree_ind);

                tree_size[sweeps][tree_ind] += trees[class_ind][sweeps][tree_ind].treesize();
            }

            state.update_split_counts(tree_ind);

            if (sweeps >= state.burnin)
            {
                for (size_t i = 0; i < (*state.split_count_all).size(); i++)
                {
                    (*state.split_count_all)[i] += (*state.split_count_current_tree)[i];
                }
            }

            model->update_state(state, tree_ind, x_struct, mean_lambda, var_lambda, count_lambda);

            weight_samples[sweeps][tree_ind] = model->weight;
            tau_samples[sweeps][tree_ind] = model->tau_a;
            logloss[sweeps][tree_ind] = model->logloss;

            if (verbose)
            {
                COUT << " tree " << tree_ind << " logloss " << model->logloss << endl;
            }

            model->state_sweep(tree_ind, state.num_trees, (*state.residual_std), x_struct);
        }
    }

    return;
}

void mcmc_loop_xbcf_continuous(matrix<size_t> &Xorder_std_con, matrix<size_t> &Xorder_std_mod, bool verbose, matrix<double> &sigma_draw_xinfo, vector<vector<tree>> &trees_con, vector<vector<tree>> &trees_mod, double no_split_penalty, State &state, XBCFContinuousModel *model, X_struct &x_struct_con, X_struct &x_struct_mod)
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

        // prognostic forest
        model->set_treatmentflag(state, 0);

        for (size_t tree_ind = 0; tree_ind < state.num_trees_con; tree_ind++)
        {
            if (verbose)
            {
                COUT << "sweep " << sweeps << " tree " << tree_ind << endl;
            }

            // Draw Sigma
            model->update_state(state, tree_ind, x_struct_con);

            sigma_draw_xinfo[sweeps][tree_ind] = state.sigma;

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

        for (size_t tree_ind = 0; tree_ind < state.num_trees_mod; tree_ind++)
        {
            if (verbose)
            {
                COUT << "sweep " << sweeps << " tree " << tree_ind << endl;
            }

            // Draw Sigma
            model->update_state(state, tree_ind, x_struct_mod);

            sigma_draw_xinfo[sweeps][tree_ind + state.num_trees_con] = state.sigma;

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
    }
    return;
}

void mcmc_loop_xbcf_discrete(matrix<size_t> &Xorder_std_con,
                             matrix<size_t> &Xorder_std_mod,
                             bool verbose, matrix<double> &sigma0_draw_xinfo,
                             matrix<double> &sigma1_draw_xinfo,
                             matrix<double> &a_xinfo,
                             matrix<double> &b_xinfo,
                             matrix<double> &tau_con_xinfo,
                             matrix<double> &tau_mod_xinfo,
                             vector<vector<tree>> &trees_con,
                             vector<vector<tree>> &trees_mod,
                             double no_split_penalty,
                             State &state,
                             XBCFDiscreteModel *model,
                             X_struct &x_struct_con,
                             X_struct &x_struct_mod)
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

        // prognostic forest
        model->set_treatmentflag(state, 0);

        for (size_t tree_ind = 0; tree_ind < state.num_trees_con; tree_ind++)
        {
            if (verbose)
            {
                COUT << "sweep " << sweeps << " tree " << tree_ind << endl;
            }

            // Draw Sigma
            model->update_state(state, tree_ind, x_struct_con, 0);

            sigma0_draw_xinfo[sweeps][tree_ind] = state.sigma_vec[0];
            sigma1_draw_xinfo[sweeps][tree_ind] = state.sigma_vec[1]; // not updated

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

            if (sweeps >= state.burnin)
            {
                for (size_t i = 0; i < (*state.split_count_all_con).size(); i++)
                {
                    (*state.split_count_all_con)[i] += (*state.split_count_current_tree)[i];
                }
            }

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
        }

        if (model->sampling_tau)
        {
            model->update_tau_per_forest(state, sweeps, trees_con);
        }

        // treatment forest
        model->set_treatmentflag(state, 1);

        for (size_t tree_ind = 0; tree_ind < state.num_trees_mod; tree_ind++)
        {
            if (verbose)
            {
                COUT << "sweep " << sweeps << " tree " << tree_ind << endl;
            }

            // Draw Sigma
            model->update_state(state, tree_ind, x_struct_mod, 1);

            sigma0_draw_xinfo[sweeps][tree_ind + state.num_trees_con] = state.sigma_vec[0]; // not udated
            sigma1_draw_xinfo[sweeps][tree_ind + state.num_trees_con] = state.sigma_vec[1];

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
        }

        if (model->sampling_tau)
        {
            model->update_tau_per_forest(state, sweeps, trees_mod);
        }
        tau_con_xinfo[0][sweeps] = model->tau_con;
        tau_mod_xinfo[0][sweeps] = model->tau_mod;
        b_xinfo[0][sweeps] = state.b_vec[0];
        b_xinfo[1][sweeps] = state.b_vec[1];
        a_xinfo[0][sweeps] = state.a;
    }
    return;
}

void mcmc_loop_xbcf_rd( matrix<size_t> &Xorder_std_con,
                        matrix<size_t> &Xorder_std_mod,
                        bool verbose, matrix<double> &sigma0_draw_xinfo,
                        matrix<double> &sigma1_draw_xinfo,
                        matrix<double> &a_xinfo,
                        matrix<double> &b_xinfo,
                        matrix<double> &tau_con_xinfo,
                        matrix<double> &tau_mod_xinfo,
                        vector<vector<tree>> &trees_con,
                        vector<vector<tree>> &trees_mod,
                        double no_split_penalty,
                        State &state,
                        XBCFrdModel *model,
                        X_struct &x_struct_con,
                        X_struct &x_struct_mod,
                        matrix<std::vector<size_t>> &con_res_indicator,
                        matrix<std::vector<double>> &con_valid_residuals,
                        matrix<std::vector<double>> &con_resid_mean,
                        matrix<std::vector<size_t>> &mod_res_indicator,
                        matrix<std::vector<double>> &mod_valid_residuals,
                        matrix<std::vector<double>> &mod_resid_mean
                        )
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

        // prognostic forest
        model->set_treatmentflag(state, 0);

        for (size_t tree_ind = 0; tree_ind < state.num_trees_con; tree_ind++)
        {
            if (verbose)
            {
                COUT << "sweep " << sweeps << " tree " << tree_ind << endl;
            }

            // Draw Sigma
            model->update_state(state, tree_ind, x_struct_con, 0);

            sigma0_draw_xinfo[sweeps][tree_ind] = state.sigma_vec[0];
            sigma1_draw_xinfo[sweeps][tree_ind] = state.sigma_vec[1]; // not updated

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

            trees_con[sweeps][tree_ind].grow_from_root_rd(state, Xorder_std_con, x_struct_con.X_counts, x_struct_con.X_num_unique, model, x_struct_con, 
                                                        con_res_indicator[sweeps][tree_ind], con_valid_residuals[sweeps][tree_ind], con_resid_mean[sweeps][tree_ind], sweeps, tree_ind);

            // update tau_fit from partial fit to full fit
            model->add_new_tree_fit(tree_ind, state, x_struct_con);

            model->update_split_counts(state, tree_ind);

            if (sweeps >= state.burnin)
            {
                for (size_t i = 0; i < (*state.split_count_all_con).size(); i++)
                {
                    (*state.split_count_all_con)[i] += (*state.split_count_current_tree)[i];
                }
            }

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
        }

        if (model->sampling_tau)
        {
            model->update_tau_per_forest(state, sweeps, trees_con);
        }

        // treatment forest
        model->set_treatmentflag(state, 1);

        for (size_t tree_ind = 0; tree_ind < state.num_trees_mod; tree_ind++)
        {
            if (verbose)
            {
                COUT << "sweep " << sweeps << " tree " << tree_ind << endl;
            }

            // Draw Sigma
            model->update_state(state, tree_ind, x_struct_mod, 1);

            sigma0_draw_xinfo[sweeps][tree_ind + state.num_trees_con] = state.sigma_vec[0]; // not updated
            sigma1_draw_xinfo[sweeps][tree_ind + state.num_trees_con] = state.sigma_vec[1];

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

            trees_mod[sweeps][tree_ind].grow_from_root_rd(state, Xorder_std_mod, x_struct_mod.X_counts, x_struct_mod.X_num_unique, model, x_struct_mod, 
                                                        mod_res_indicator[sweeps][tree_ind], mod_valid_residuals[sweeps][tree_ind], mod_resid_mean[sweeps][tree_ind], sweeps, tree_ind);

            // store residuals:
            for (size_t data_ind = 0; data_ind < (*state.residual_std)[0].size(); data_ind++)
            {
                mod_valid_residuals[sweeps][tree_ind][data_ind] = ((*state.residual_std))[0][data_ind]; 
            }

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
        }

        if (model->sampling_tau)
        {
            model->update_tau_per_forest(state, sweeps, trees_mod);
        }
        tau_con_xinfo[0][sweeps] = model->tau_con;
        tau_mod_xinfo[0][sweeps] = model->tau_mod;
        b_xinfo[0][sweeps] = state.b_vec[0];
        b_xinfo[1][sweeps] = state.b_vec[1];
        a_xinfo[0][sweeps] = state.a;
    }
    // cout << "residual = " << mod_valid_residuals[2][2][4] << endl;
    return;
}

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