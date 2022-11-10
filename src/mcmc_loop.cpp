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

void mcmc_loop_multinomial(matrix<size_t> &Xorder_std, bool verbose, vector<vector<tree>> &trees, size_t latent_num_trees, size_t large_tree_size, double no_split_penalty, State &state, LogitModel *model, X_struct &x_struct,
                           std::vector<std::vector<double>> &weight_samples, std::vector<double> &lambda_samples, std::vector<std::vector<double>> &phi_samples, std::vector<std::vector<double>> &logloss,
                           std::vector<std::vector<double>> &tree_size)
{
    // Residual for 0th tree
    model->ini_residual_std(state);

    // keep track of mean of leaf parameters
    double mean_lambda = 1;
    size_t count_lambda = (state.num_trees - 1) * model->dim_residual; // less the lambdas in the first tree
    std::vector<double> var_lambda(state.num_trees, 0.0);

    size_t large_trees = 0;
    size_t num_trees_last_sweep = latent_num_trees;

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
            if (tree_ind < num_trees_last_sweep)
            {

                if (state.use_all && (sweeps >= state.burnin)) // && (state.mtry != state.p) // If mtry = p, it will all be sampled anyway. Now use_all can be an indication of burnin period.
                {
                    state.use_all = false;
                }

                if ((sweeps > 0) && (trees[sweeps-1][tree_ind].treesize() >= large_tree_size))
                {
                    // copy large tree
                    model->copy_initialization(state, x_struct, trees, sweeps, tree_ind, sweeps-1, tree_ind, Xorder_std);
                    // split_count stays the same
                } else {
                    // clear counts of splits for one tree
                    std::fill((*state.split_count_current_tree).begin(), (*state.split_count_current_tree).end(), 0.0);

                    // subtract old tree for sampling case
                    if (state.sample_weights)
                    {
                        (*state.mtry_weight_current_tree) = (*state.mtry_weight_current_tree) - (*state.split_count_all_tree)[tree_ind];
                    }
                    // grow new tree
                    model->initialize_root_suffstat(state, trees[sweeps][tree_ind].suff_stat);

                    trees[sweeps][tree_ind].theta_vector.resize(model->dim_residual);
                    (*state.lambdas)[tree_ind].clear();

                    trees[sweeps][tree_ind].grow_from_root_entropy(state, Xorder_std, x_struct.X_counts, x_struct.X_num_unique, model, x_struct, sweeps, tree_ind);

                     if (sweeps >= state.burnin)
                    {
                        for (size_t i = 0; i < (*state.split_count_all).size(); i++)
                        {
                            (*state.split_count_all)[i] += (*state.split_count_current_tree)[i];
                        }
                    }

                    state.update_split_counts(tree_ind);

                    if (trees[sweeps][tree_ind].treesize() >= large_tree_size){
                        large_trees += 1;
                        latent_num_trees += 1;
                        if (latent_num_trees > state.num_trees){
                            cout << "total number of trees exceeds memory" << endl;
                            abort();
                        }
                    }
                }
                
                // update partial fits for the next tree
                model->update_state(state, tree_ind, x_struct, mean_lambda, var_lambda, count_lambda);
                
                model->state_sweep(tree_ind, num_trees_last_sweep, (*state.residual_std), x_struct);

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
            } else {
                // initialize theta
                trees[sweeps][tree_ind].theta_vector.resize(model->dim_residual);
                std::fill(trees[sweeps][tree_ind].theta_vector.begin(), trees[sweeps][tree_ind].theta_vector.end(), 1.);
            }

            weight_samples[sweeps][tree_ind] = model->weight;
            phi_samples[sweeps][tree_ind] = exp((*model->phi)[0]);
            logloss[sweeps][tree_ind] = model->logloss;
            tree_size[sweeps][tree_ind] = trees[sweeps][tree_ind].treesize();
        }
        model->update_weights(state, x_struct, mean_lambda, var_lambda, count_lambda);

        num_trees_last_sweep = latent_num_trees;

        cout << "Total large trees = " << large_trees << " num trees " << latent_num_trees << endl;
    }
    cout << "Total large trees = " << large_trees << endl;

    //  cout << "dp = ";
    // for (size_t j = 0; j < state.num_trees; j++)
    // {
    //     cout << " " << (*(x_struct.data_pointers[j][0])) << endl;;
    // }
    // cout << "resid = ";
    // for (size_t j = 0; j < model->dim_residual; ++j)
    // {
    //     cout << " " << (*state.residual_std)[j][0] + log((*(x_struct.data_pointers[0][0]))[j]);
    // }
    // cout << endl;
                
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

void mcmc_loop_xbcf_discrete(matrix<size_t> &Xorder_std_con, matrix<size_t> &Xorder_std_mod, bool verbose, matrix<double> &sigma0_draw_xinfo, matrix<double> &sigma1_draw_xinfo, matrix<double> &a_xinfo, matrix<double> &b_xinfo, vector<vector<tree>> &trees_con, vector<vector<tree>> &trees_mod, double no_split_penalty, State &state, XBCFDiscreteModel *model, X_struct &x_struct_con, X_struct &x_struct_mod)
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

        b_xinfo[0][sweeps] = state.b_vec[0];
        b_xinfo[1][sweeps] = state.b_vec[1];
        a_xinfo[0][sweeps] = state.a;
    }
    return;
}