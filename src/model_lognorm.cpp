#include "tree.h"
#include "model.h"
#include <cfenv>

//////////////////////////////////////////////////////////////////////////////////////
//
//
//  LogNormal Model
//
//
//////////////////////////////////////////////////////////////////////////////////////

void logNormalModel::ini_residual_std(State &state, matrix<double> &mean_residual_std, X_struct &x_struct)
{
    // initialize partial residual at the residual^2 from the mean model
    for (size_t i = 0; i < (*state.residual_std)[0].size(); i++)
    {
        (*state.residual_std)[0][i] = 2*log(abs(mean_residual_std[0][i])) + log((*state.precision)[i]) - log((*(x_struct.data_pointers[0][i]))[0]);
    }
    return;
}

void logNormalModel::ini_residual_std2(State &state, X_struct &x_struct)
{
    // initialize partial residual at the residual^2 from the mean model
    for (size_t i = 0; i < (*state.residual_std)[0].size(); i++)
    {
        (*state.residual_std)[0][i] = 2*log(abs((*state.mean_res)[i])) + log((*state.precision)[i]) - log((*(x_struct.data_pointers[0][i]))[0]);
    }
    return;
}

void logNormalModel::initialize_root_suffstat(State &state,
                                              std::vector<double> &suff_stat)
{
    suff_stat[0] = 0;
    suff_stat[1] = 0;
    for (size_t i = 0; i < state.n_y; i++)
    {
        incSuffStat(state, i, suff_stat);
    }
    return;
}

void logNormalModel::incSuffStat(State &state,
                                 size_t index_next_obs,
                                 std::vector<double> &suffstats)
{
    suffstats[0] += exp((*state.residual_std)[0][index_next_obs]);
    suffstats[1] += 1;
    return;
}

void logNormalModel::updateNodeSuffStat(State &state,
                                        std::vector<double> &suff_stat,
                                        matrix<size_t> &Xorder_std,
                                        size_t &split_var,
                                        size_t row_ind)
{
    incSuffStat(state, Xorder_std[split_var][row_ind], suff_stat);
    return;
}

void logNormalModel::samplePars(State &state,
                                std::vector<double> &suff_stat,
                                std::vector<double> &theta_vector,
                                double &prob_leaf)
{
    std::gamma_distribution<double> gammadist(tau_a + 0.5 * suff_stat[1], 1.0);
    theta_vector[0] = gammadist(state.gen) / (tau_b + 0.5 * suff_stat[0]);

    return;
}

void logNormalModel::calculateOtherSideSuffStat(std::vector<double> &parent_suff_stat, std::vector<double> &lchild_suff_stat, std::vector<double> &rchild_suff_stat, size_t &N_parent, size_t &N_left, size_t &N_right, bool &compute_left_side)
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

double logNormalModel::likelihood(std::vector<double> &temp_suff_stat,
                                  std::vector<double> &suff_stat_all,
                                  size_t N_left,
                                  bool left_side,
                                  bool no_split,
                                  State &state) const
{
    size_t n_b; // number of observations in node b (times 0.5)
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

void logNormalModel::state_sweep(size_t tree_ind,
                                 size_t M,
                                 matrix<double> &residual_std,
                                 //std::vector<double> &fit,
                                 X_struct &x_struct) const
{
    size_t next_index = tree_ind + 1;
    if (next_index == M)
    {
        next_index = 0;
    }

    for (size_t i = 0; i < residual_std[0].size(); i++)
    {
        residual_std[0][i] = residual_std[0][i] + log((*(x_struct.data_pointers[tree_ind][i]))[0]) - log((*(x_struct.data_pointers[next_index][i]))[0]);
    }
    return;
}

void logNormalModel::predict_std(const double *Xtestpointer, size_t N_test, size_t p, size_t num_trees, size_t num_sweeps, matrix<double> &yhats_test_xinfo, vector<vector<tree>> &trees)
{

    matrix<double> output;

    // row : dimension of theta, column : number of trees
    ini_matrix(output, this->dim_theta, trees[0].size());

    for (size_t sweeps = 0; sweeps < num_sweeps; sweeps++)
    {
        for (size_t data_ind = 0; data_ind < N_test; data_ind++)
        {
            getThetaForObs_Outsample(output, trees[sweeps], data_ind, Xtestpointer, N_test, p);

            // take sum of predictions of each tree, as final prediction
            for (size_t i = 0; i < trees[0].size(); i++)
            {
                yhats_test_xinfo[sweeps][data_ind] += log(output[i][0]);
            }
            yhats_test_xinfo[sweeps][data_ind] = exp(yhats_test_xinfo[sweeps][data_ind]);
        }
    }
    return;
}


void logNormalModel::update_sigmas(State &state,
                                   size_t M,
                                   X_struct &x_struct)
{
    // update sigma2
    for (size_t i = 0; i < (*state.residual_std)[0].size(); i++)
    {
        double log_sigma2 = 0;
        for (size_t j = 0; j < M; j++)
        {
            log_sigma2 += log((*(x_struct.data_pointers[j][i]))[0]);
        }
//        (*state.residual_std)[1][i] = exp(log_sigma2);
        (*state.precision)[i] = exp(log_sigma2);
//        (*state.residual_std)[2][i] = (*state.residual_std)[0][i] * (*state.residual_std)[1][i];
        (*state.res_x_precision)[i] = (*state.residual_std)[0][i] * (*state.precision)[i];
    }
    return;
}

void logNormalModel::update_state(State &state,
                                  size_t tree_ind,
                                  X_struct &x_struct)
{
    for (size_t i = 0; i < (*state.residual_std)[0].size(); i++)
    {
        double log_sigma2 = 0;
        for (size_t j = 0; j < tree_ind; j++)
        {
            log_sigma2 += log((*(x_struct.data_pointers[j][i]))[0]);
        }
        (*state.precision)[i] = exp(log_sigma2);
//        (*state.residual_std)[0][i] = (*state.mean_res)[i];
//        (*state.res_x_precision)[i] = (*state.residual_std)[0][i] * (*state.precision)[i];
    }
    return;
}

void logNormalModel::update_state2(State &state,
                                  size_t tree_ind,
                                  X_struct &x_struct)
{
    for (size_t i = 0; i < (*state.residual_std)[0].size(); i++)
    {
        double log_sigma2 = 0;
        for (size_t j = 0; j < tree_ind; j++)
        {
            log_sigma2 += log((*(x_struct.data_pointers[j][i]))[0]);
        }
        (*state.precision)[i] = exp(log_sigma2);
        (*state.residual_std)[0][i] = (*state.mean_res)[i];
        (*state.res_x_precision)[i] = (*state.residual_std)[0][i] * (*state.precision)[i];
    }
    return;
}

void logNormalModel::switch_state_params(State &state)
{
    // update state settings to mean forest
    state.num_trees = state.num_trees_v;
    state.n_min = state.n_min_v;
    state.max_depth = state.max_depth_v;
    state.n_cutpoints = state.n_cutpoints_v;

    return;
}