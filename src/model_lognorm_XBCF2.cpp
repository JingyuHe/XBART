#include "tree.h"
#include "model.h"
#include <cfenv>

//////////////////////////////////////////////////////////////////////////////////////
//
//
//  LogNormal Model, for XBCF such that the treated and control group has different trees
//
//
//////////////////////////////////////////////////////////////////////////////////////

void logNormalXBCFModel2::ini_residual_std(State &state, matrix<double> &mean_residual_std, X_struct &x_struct_v_con, X_struct &x_struct_v_mod)
{
    // initialize partial residual at the residual^2 from the mean model
    for (size_t i = 0; i < (*state.residual_std)[0].size(); i++)
    {
        if (state.treatment_flag)
        {
            // fitting treatment tree
            if ((*state.Z_std)[0][i])
            {
                (*state.residual_std)[0][i] = 2 * log(abs(mean_residual_std[0][i])) + log((*state.precision_con)[i]) + log((*state.precision_mod)[i]) - log((*(x_struct_v_mod.data_pointers[0][i]))[0]);
            }
            else
            {
                (*state.residual_std)[0][i] = 2 * log(abs(mean_residual_std[0][i])) + log((*state.precision_con)[i]);
            }
        }
        else
        {
            // fitting prognostic tree
            (*state.residual_std)[0][i] = 2 * log(abs(mean_residual_std[0][i])) + log((*state.precision_con)[i]) + log((*state.precision_mod)[i]) - log((*(x_struct_v_con.data_pointers[0][i]))[0]);
        }
    }
    return;
}

void logNormalXBCFModel2::ini_residual_std2(State &state, X_struct &x_struct_v_con, X_struct &x_struct_v_mod)
{
    // initialize partial residual at the residual^2 from the mean model

    // the full residual for lognormal model is res^2 / var

    // then initialize partial residuals for the first tree
    for (size_t i = 0; i < (*state.residual_std)[0].size(); i++)
    {
        if (state.treatment_flag)
        {
            // treatment tree
            if ((*state.Z_std)[0][i])
            {
                (*state.residual_std)[0][i] = 2 * log(abs((*state.residual_std)[0][i])) + log((*state.precision_con)[i]) + log((*state.precision_mod)[i]) - log((*(x_struct_v_mod.data_pointers[0][i]))[0]);
            }
            else
            {
                (*state.residual_std)[0][i] = 2 * log(abs((*state.residual_std)[0][i])) + log((*state.precision_con)[i]);
            }
        }
        else
        {
            // prognostic tree
            (*state.residual_std)[0][i] = 2 * log(abs((*state.residual_std)[0][i])) + log((*state.precision_mod)[i]) + log((*state.precision_con)[i]) - log((*(x_struct_v_con.data_pointers[0][i]))[0]);
        }
    }
    return;
}

void logNormalXBCFModel2::initialize_root_suffstat(State &state, std::vector<double> &suff_stat)
{
    suff_stat[0] = 0;
    suff_stat[1] = 0;
    for (size_t i = 0; i < state.n_y; i++)
    {
        incSuffStat(state, i, suff_stat);
    }
    return;
}

void logNormalXBCFModel2::incSuffStat(State &state,
                                      size_t index_next_obs,
                                      std::vector<double> &suffstats)
{
    if (state.treatment_flag)
    {
        // trees for treated effect, mod trees
        // only if the observation is treated
        if ((*state.Z_std)[0][index_next_obs])
        {
            suffstats[0] += exp((*state.residual_std)[0][index_next_obs]);
            suffstats[1] += 1;
        }
    }
    else
    {
        // trees for baseline, con trees
        // count all data
        suffstats[0] += exp((*state.residual_std)[0][index_next_obs]);
        suffstats[1] += 1;
    }
    return;
}

void logNormalXBCFModel2::updateNodeSuffStat(State &state,
                                             std::vector<double> &suff_stat,
                                             matrix<size_t> &Xorder_std,
                                             size_t &split_var,
                                             size_t row_ind)
{
    incSuffStat(state, Xorder_std[split_var][row_ind], suff_stat);
    return;
}

void logNormalXBCFModel2::samplePars(State &state,
                                     std::vector<double> &suff_stat,
                                     std::vector<double> &theta_vector,
                                     double &prob_leaf)
{
    if (state.treatment_flag)
    {
        std::gamma_distribution<double> gammadist(20 + 0.5 * suff_stat[1], 1.0);
        theta_vector[0] = gammadist(state.gen) / (20 + 0.5 * suff_stat[0]);
    }
    else
    {
        std::gamma_distribution<double> gammadist(tau_a + 0.5 * suff_stat[1], 1.0);
        theta_vector[0] = gammadist(state.gen) / (tau_b + 0.5 * suff_stat[0]);
    }
    return;
}

void logNormalXBCFModel2::calculateOtherSideSuffStat(std::vector<double> &parent_suff_stat, std::vector<double> &lchild_suff_stat, std::vector<double> &rchild_suff_stat, size_t &N_parent, size_t &N_left, size_t &N_right, bool &compute_left_side)
{
    if (compute_left_side)
    {
        rchild_suff_stat = parent_suff_stat - lchild_suff_stat;
    }
    else
    {
        lchild_suff_stat = parent_suff_stat - rchild_suff_stat;
    }
    return;
}

double logNormalXBCFModel2::likelihood(std::vector<double> &temp_suff_stat,
                                       std::vector<double> &suff_stat_all,
                                       size_t N_left,
                                       bool left_side,
                                       bool no_split,
                                       State &state) const
{
    size_t n_b;       // number of observations in node b (times 0.5)
    double res_sum_b; // sum of res^2 in node b (times 0.5)

    if (no_split)
    {
        n_b = 0.5 * suff_stat_all[1];
        res_sum_b = 0.5 * suff_stat_all[0];
    }
    else
    {
        if (left_side)
        {
            n_b = 0.5 * temp_suff_stat[1];
            res_sum_b = 0.5 * temp_suff_stat[0];
        }
        else
        {
            n_b = 0.5 * (suff_stat_all[1] - temp_suff_stat[1]);
            res_sum_b = 0.5 * (suff_stat_all[0] - temp_suff_stat[0]);
        }
    }

    return -(tau_a + n_b) * log(tau_b + res_sum_b) + lgamma(tau_a + n_b);
}

void logNormalXBCFModel2::state_sweep(State &state,
                                      size_t tree_ind,
                                      size_t M,
                                      matrix<double> &residual_std,
                                      // std::vector<double> &fit,
                                      X_struct &x_struct_v_con,
                                      X_struct &x_struct_v_mod) const
{
    size_t next_index = tree_ind + 1;

    if (next_index == M)
    {
        next_index = 0;
    }

    for (size_t i = 0; i < residual_std[0].size(); i++)
    {
        if (state.treatment_flag)
        {
            // only update treated data
            if ((*state.Z_std)[0][i])
            {
                residual_std[0][i] = residual_std[0][i] + log((*(x_struct_v_mod.data_pointers[tree_ind][i]))[0]) - log((*(x_struct_v_mod.data_pointers[next_index][i]))[0]);
            }
        }
        else
        {
            // prognostic tree, modify all data
            residual_std[0][i] = residual_std[0][i] + log((*(x_struct_v_con.data_pointers[tree_ind][i]))[0]) - log((*(x_struct_v_con.data_pointers[next_index][i]))[0]);
        }
    }
    return;
}

void logNormalXBCFModel2::predict_std(matrix<double> &Ztestpointer, const double *Xtestpointer_con, const double *Xtestpointer_mod, size_t N_test, size_t p, size_t num_trees, size_t num_sweeps, matrix<double> &yhats_test_xinfo, matrix<double> &yhats_test_con, matrix<double> &yhats_test_mod, vector<vector<tree>> &trees_con, vector<vector<tree>> &trees_mod)
{
    matrix<double> output_con;
    matrix<double> output_mod;

    // row : dimension of theta, column : number of trees
    ini_matrix(output_con, this->dim_theta, trees_con[0].size());
    ini_matrix(output_mod, this->dim_theta, trees_mod[0].size());

    for (size_t sweeps = 0; sweeps < num_sweeps; sweeps++)
    {
        for (size_t data_ind = 0; data_ind < N_test; data_ind++)
        {
            // prognostic trees
            getThetaForObs_Outsample(output_con, trees_con[sweeps], data_ind, Xtestpointer_con, N_test, p);

            for (size_t i = 0; i < trees_con[0].size(); i++)
            {
                yhats_test_xinfo[sweeps][data_ind] += log(output_con[i][0]);
                yhats_test_con[sweeps][data_ind] += log(output_con[i][0]);
            }

            // treatment tree, if treated
            if (Ztestpointer[0][data_ind])
            {
                getThetaForObs_Outsample(output_mod, trees_mod[sweeps], data_ind, Xtestpointer_mod, N_test, p);

                for (size_t i = 0; i < trees_mod[0].size(); i++)
                {
                    yhats_test_xinfo[sweeps][data_ind] += log(output_mod[i][0]);
                    yhats_test_mod[sweeps][data_ind] += log(output_mod[i][0]);
                }
            }

            yhats_test_xinfo[sweeps][data_ind] = exp(yhats_test_xinfo[sweeps][data_ind]);
            yhats_test_con[sweeps][data_ind] = exp(yhats_test_con[sweeps][data_ind]);
            yhats_test_mod[sweeps][data_ind] = exp(yhats_test_mod[sweeps][data_ind]);
        }
    }
    return;
}

void logNormalXBCFModel2::update_state(State &state,
                                       X_struct &x_struct_v_con,
                                       X_struct &x_struct_v_mod)
{
    double log_sigma2_mod;
    double log_sigma2_con;

    for (size_t i = 0; i < (*state.residual_std)[0].size(); i++)
    {
        log_sigma2_mod = 0;
        log_sigma2_con = 0;

        for (size_t j = 0; j < state.num_trees_con; j++)
        {
            log_sigma2_con += log((*(x_struct_v_con.data_pointers[j][i]))[0]);
        }

        (*state.precision_con)[i] = exp(log_sigma2_con);

        if ((*state.Z_std)[0][i])
        {
            for (size_t j = 0; j < state.num_trees_mod; j++)
            {
                log_sigma2_mod += log((*(x_struct_v_mod.data_pointers[j][i]))[0]);
            }

            (*state.precision_mod)[i] = exp(log_sigma2_mod);
        }
        else
        {
            (*state.precision_mod)[i] = 1.0;
        }

        // update fitted precision and res * precision
        if ((*state.Z_std)[0][i])
        {
            (*state.precision)[i] = (*state.precision_con)[i] * (*state.precision_mod)[i];
        }
        else
        {
            (*state.precision)[i] = (*state.precision_con)[i];
        }

        (*state.res_x_precision)[i] = (*state.residual_std)[0][i] * (*state.precision)[i];

        // copy mean_res back to residual_std, for the next mean trees
        (*state.residual_std)[0][i] = (*state.mean_res)[i];
    }
    return;
}

void logNormalXBCFModel2::switch_state_params(State &state)
{
    // update state settings to mean forest
    state.num_trees = state.num_trees_v;
    state.n_min = state.n_min_v;
    state.max_depth = state.max_depth_v;
    state.n_cutpoints = state.n_cutpoints_v;

    return;
}
