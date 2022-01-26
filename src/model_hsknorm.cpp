#include "tree.h"
#include "model.h"
#include <cfenv>

//////////////////////////////////////////////////////////////////////////////////////
//
//
//  Heteroskedastic Normal Model
//
//
//////////////////////////////////////////////////////////////////////////////////////

void hskNormalModel::incSuffStat(matrix<double> &residual_std, size_t index_next_obs, std::vector<double> &suffstats)
{
    // I have to pass matrix<double> &residual_std, size_t index_next_obs
    // which allows more flexibility for multidimensional residual_std

    suffstats[0] += residual_std[2][index_next_obs];
    suffstats[1] += residual_std[1][index_next_obs];
    return;
}

void hskNormalModel::samplePars(std::unique_ptr<State> &state, std::vector<double> &suff_stat, std::vector<double> &theta_vector, double &prob_leaf)
{
    std::normal_distribution<double> normal_samp(0.0, 1.0);

    // test result should be theta
    theta_vector[0] = suff_stat[0] / (1.0 / tau + suff_stat[1])
                    + sqrt(1.0 / tau + suff_stat[1]) * normal_samp(state->gen);

    return;
}

// UNNEEDED: we don't update sigma within this model
void hskNormalModel::update_state(std::unique_ptr<State> &state, size_t tree_ind, std::unique_ptr<X_struct> &x_struct)
{
    // Draw Sigma
    // state->residual_std_full = state->residual_std - state->predictions_std[tree_ind];

    // residual_std is only 1 dimensional for regression model

    std::vector<double> full_residual(state->n_y);

    for (size_t i = 0; i < state->residual_std[0].size(); i++)
    {
        full_residual[i] = state->residual_std[0][i] - (*(x_struct->data_pointers[tree_ind][i]))[0];
    }

    std::gamma_distribution<double> gamma_samp((state->n_y + kap) / 2.0, 2.0 / (sum_squared(full_residual) + s));
    state->update_sigma(1.0 / sqrt(gamma_samp(state->gen)));
    return;
}

void hskNormalModel::initialize_root_suffstat(std::unique_ptr<State> &state, std::vector<double> &suff_stat)
{
    // sum of r
    suff_stat[0] = sum_vec(state->residual_std[2]);
    // sum of 1/sig2
    suff_stat[1] = sum_vec(state->residual_std[1]);
    return;
}

void hskNormalModel::updateNodeSuffStat(std::vector<double> &suff_stat, matrix<double> &residual_std, matrix<size_t> &Xorder_std, size_t &split_var, size_t row_ind)
{
    incSuffStat(residual_std, Xorder_std[split_var][row_ind], suff_stat);
    return;
}

void hskNormalModel::state_sweep(size_t tree_ind, size_t M, matrix<double> &residual_std, std::unique_ptr<X_struct> &x_struct) const
{
    size_t next_index = tree_ind + 1;
    if (next_index == M)
    {
        next_index = 0;
    }

    ////////////////////////////////////////////////////////
    // Be care of line 151 in train_all.cpp, initial_theta
    ////////////////////////////////////////////////////////

    for (size_t i = 0; i < residual_std[0].size(); i++)
    {
        residual_std[0][i] = residual_std[0][i] - (*(x_struct->data_pointers[tree_ind][i]))[0] + (*(x_struct->data_pointers[next_index][i]))[0];
        residual_std[2][i] = residual_std[0][i] * residual_std[1][i];
    }
    return;
}

double hskNormalModel::likelihood(std::vector<double> &temp_suff_stat, std::vector<double> &suff_stat_all, size_t N_left, bool left_side, bool no_split, std::unique_ptr<State> &state) const
{
    double res2;
    double prec;

    if (no_split)
    {
        res2 = pow(suff_stat_all[0], 2);
        prec = suff_stat_all[1];
    }
    else
    {
        if (left_side)
        {
        res2 = pow(temp_suff_stat[0], 2);
        prec = temp_suff_stat[1];
        }
        else
        {
            res2 = pow(suff_stat_all[0] - temp_suff_stat[0], 2);
            prec= suff_stat_all[1] - temp_suff_stat[1];
        }
    }

    return log(1.0 / (1.0 + tau * prec)) + res2 / (1.0 / tau + prec);
}


void hskNormalModel::ini_residual_std(std::unique_ptr<State> &state)
{
    // initialize partial residual at (num_tree - 1) / num_tree * yhat
    double value = state->ini_var_yhat * ((double)state->num_trees - 1.0) / (double)state->num_trees;
    for (size_t i = 0; i < state->residual_std[0].size(); i++)
    {
        state->residual_std[0][i] = (*state->y_std)[i] - value;
        state->residual_std[1][i] = double (1.0 / pow(state->sigma_vec[i], 2));
        state->residual_std[2][i] = state->residual_std[0][i] * state->residual_std[1][i];
    }
    return;
}