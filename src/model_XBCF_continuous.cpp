#include "tree.h"
#include "model.h"
#include <cfenv>

//////////////////////////////////////////////////////////////////////////////////////
//
//
//  Normal Linear Model for continuous treatment XBCF
//
//
//////////////////////////////////////////////////////////////////////////////////////

void XBCFContinuousModel::incSuffStat(std::unique_ptr<State> &state, size_t index_next_obs, std::vector<double> &suffstats)
{
    if (state->treatment_flag)
    {
        // treatment forest
        // sum z_i^2
        suffstats[0] += pow((*state->Z_std)[0][index_next_obs], 2);
        // sum r_i * z_i^2
        suffstats[1] += pow((*state->Z_std)[0][index_next_obs], 2) * state->residual_std[0][index_next_obs];
        // number of points
        suffstats[2] += 1;
    }
    else
    {
        // prognostic forest
        suffstats[0] += 1;
        suffstats[1] += state->residual_std[0][index_next_obs];
        suffstats[3] += 1;

    }
    return;
}

void XBCFContinuousModel::samplePars(std::unique_ptr<State> &state, std::vector<double> &suff_stat, std::vector<double> &theta_vector, double &prob_leaf)
{
    std::normal_distribution<double> normal_samp(0.0, 1.0);

    double sigma2 = pow(state->sigma, 2);

    theta_vector[0] = suff_stat[1] / sigma2 / (suff_stat[0] / sigma2 + 1.0 / tau) + sqrt(1.0 / (1.0 / tau + suff_stat[0] / sigma2)) * normal_samp(state->gen);

    return;
}

void XBCFContinuousModel::update_state(std::unique_ptr<State> &state, size_t tree_ind, std::unique_ptr<X_struct> &x_struct)
{
    // Draw Sigma
    std::vector<double> full_residual(state->n_y);

    for (size_t i = 0; i < state->n_y; i++)
    {
        full_residual[i] = (*state->y_std)[i] - (*state->mu_fit)[i] - ((*state->Z_std)[0][i]) * (*state->tau_fit)[i];

    }

    std::gamma_distribution<double> gamma_samp((state->n_y + kap) / 2.0, 2.0 / (sum_squared(full_residual) + s));
    state->update_sigma(1.0 / sqrt(gamma_samp(state->gen)));

    return;
}

void XBCFContinuousModel::update_tau(std::unique_ptr<State> &state, size_t tree_ind, size_t sweeps, vector<vector<tree>> &trees)
{
    std::vector<tree *> leaf_nodes;
    trees[sweeps][tree_ind].getbots(leaf_nodes);
    double sum_squared = 0.0;
    for (size_t i = 0; i < leaf_nodes.size(); i++)
    {
        sum_squared = sum_squared + pow(leaf_nodes[i]->theta_vector[0], 2);
    }
    double kap = this->tau_kap;
    double s = this->tau_s * this->tau_mean;

    std::gamma_distribution<double> gamma_samp((leaf_nodes.size() + kap) / 2.0, 2.0 / (sum_squared + s));
    this->tau = 1.0 / gamma_samp(state->gen);
    return;
};

void XBCFContinuousModel::update_tau_per_forest(std::unique_ptr<State> &state, size_t sweeps, vector<vector<tree>> &trees)
{
    std::vector<tree *> leaf_nodes;
    for (size_t tree_ind = 0; tree_ind < state->num_trees; tree_ind++)
    {
        trees[sweeps][tree_ind].getbots(leaf_nodes);
    }
    double sum_squared = 0.0;
    for (size_t i = 0; i < leaf_nodes.size(); i++)
    {
        sum_squared = sum_squared + pow(leaf_nodes[i]->theta_vector[0], 2);
    };
    double kap = this->tau_kap;
    double s = this->tau_s * this->tau_mean;
    std::gamma_distribution<double> gamma_samp((leaf_nodes.size() + kap) / 2.0, 2.0 / (sum_squared + s));
    this->tau = 1.0 / gamma_samp(state->gen);
    return;
}

void XBCFContinuousModel::initialize_root_suffstat(std::unique_ptr<State> &state, std::vector<double> &suff_stat)
{
    std::fill(suff_stat.begin(), suff_stat.end(), 0.0);
    for (size_t i = 0; i < state->n_y; i++)
    {
        incSuffStat(state, i, suff_stat);
    }
    return;
}

void XBCFContinuousModel::updateNodeSuffStat(std::unique_ptr<State> &state, std::vector<double> &suff_stat, matrix<size_t> &Xorder_std, size_t &split_var, size_t row_ind)
{
    if (state->treatment_flag)
    {
        // sum of z^2
        suff_stat[0] += pow(((*state->Z_std))[0][Xorder_std[split_var][row_ind]], 2);

        // sum of partial residual * z^2 (in y scale)
        suff_stat[1] += (state->residual_std[0])[Xorder_std[split_var][row_ind]] * pow(((*state->Z_std))[0][Xorder_std[split_var][row_ind]], 2);

        // number of data points
        suff_stat[2] += 1;
    }
    else
    {
        suff_stat[0] += 1;
        suff_stat[1] += (state->residual_std[0])[Xorder_std[split_var][row_ind]];
        suff_stat[2] += 1;
    }

    return;
}

void XBCFContinuousModel::calculateOtherSideSuffStat(std::vector<double> &parent_suff_stat, std::vector<double> &lchild_suff_stat, std::vector<double> &rchild_suff_stat, size_t &N_parent, size_t &N_left, size_t &N_right, bool &compute_left_side)
{
    // in function split_xorder_std_categorical, for efficiency, the function only calculates suff stat of ONE child
    // this function calculate the other side based on parent and the other child
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

// void XBCFContinuousModel::state_sweep(std::unique_ptr<State> &state, size_t tree_ind, size_t M, std::unique_ptr<X_struct> &x_struct) const
// {
//     size_t next_index = tree_ind + 1;
//     if (next_index == M)
//     {
//         next_index = 0;
//     }

//     for (size_t i = 0; i < state->residual_std[0].size(); i++)
//     {
//         state->residual_std[0][i] = state->residual_std[0][i] - (*(x_struct->data_pointers[tree_ind][i]))[0] + (*(x_struct->data_pointers[next_index][i]))[0];
//     }
//     return;
// }

double XBCFContinuousModel::likelihood(std::vector<double> &temp_suff_stat, std::vector<double> &suff_stat_all, size_t N_left, bool left_side, bool no_split, std::unique_ptr<State> &state) const
{
    // likelihood equation,

    double sigma2 = state->sigma2;

    size_t nb;
    double s0; // sum z_i^2
    double s1; // sum r_i * z_i^2

    if (no_split)
    {
        // calculate likelihood for no-split option (early stop)
        s0 = suff_stat_all[0];
        s1 = suff_stat_all[1];
        nb = suff_stat_all[2];
    }
    else
    {
        // calculate likelihood for regular split point
        if (left_side)
        {
            s0 = temp_suff_stat[0];
            s1 = temp_suff_stat[1];
            nb = N_left + 1;
        }
        else
        {
            s0 = suff_stat_all[0] - temp_suff_stat[0];
            s1 = suff_stat_all[1] - temp_suff_stat[1];
            nb = suff_stat_all[2] - N_left - 1;
        }
    }

    return 0.5 * log(1.0 / (1.0 + tau * s0 / sigma2)) + 0.5 * pow(s1 / sigma2, 2) / (s0 / sigma2 + 1.0 / tau);
}

void XBCFContinuousModel::ini_residual_std(std::unique_ptr<State> &state)
{
    // initialize the vector of full residuals
    for (size_t i = 0; i < state->residual_std[0].size(); i++)
    {
        state->residual_std[0][i] = (*state->y_std)[i] - (*state->mu_fit)[i] - ((*state->Z_std)[0][i]) * (*state->tau_fit)[i];

    }
}

void XBCFContinuousModel::predict_std(matrix<double> &Ztestpointer, const double *Xtestpointer, size_t N_test, size_t p, size_t num_trees, size_t num_sweeps, matrix<double> &yhats_test_xinfo, vector<vector<tree>> &trees_ps, vector<vector<tree>> &trees_trt)
{
    // predict the output as a matrix
    matrix<double> output_trt;

    // row : dimension of theta, column : number of trees
    ini_matrix(output_trt, this->dim_theta, trees_trt[0].size());

    matrix<double> output_ps;
    ini_matrix(output_ps, this->dim_theta, trees_ps[0].size());

    for (size_t sweeps = 0; sweeps < num_sweeps; sweeps++)
    {
        for (size_t data_ind = 0; data_ind < N_test; data_ind++)
        {
            getThetaForObs_Outsample(output_trt, trees_trt[sweeps], data_ind, Xtestpointer, N_test, p);

            getThetaForObs_Outsample(output_ps, trees_ps[sweeps], data_ind, Xtestpointer, N_test, p);

            // take sum of predictions of each tree, as final prediction
            for (size_t i = 0; i < trees_trt[0].size(); i++)

            {
                yhats_test_xinfo[sweeps][data_ind] += output_ps[i][0] + output_trt[i][0] * (Ztestpointer[0][data_ind]);
            }
        }
    }
    return;
}

void XBCFContinuousModel::ini_tau_mu_fit(std::unique_ptr<State> &state)
{
    double value = state->ini_var_yhat;
    for (size_t i = 0; i < state->residual_std[0].size(); i++)
    {
        (*state->mu_fit)[i] = 0;
        (*state->tau_fit)[i] = value;
    }
    return;
}

void XBCFContinuousModel::set_treatmentflag(std::unique_ptr<State> &state, bool value)
{
    state->treatment_flag = value;
    return;
}

void XBCFContinuousModel::subtract_old_tree_fit(size_t tree_ind, std::unique_ptr<State> &state, std::unique_ptr<X_struct> &x_struct)
{
    if (state->treatment_flag)
    {
        for (size_t i = 0; i < (*state->tau_fit).size(); i++)
        {
            (*state->tau_fit)[i] -= (*(x_struct->data_pointers[tree_ind][i]))[0];
        }
    }
    else
    {
        for (size_t i = 0; i < (*state->mu_fit).size(); i++)
        {
            (*state->mu_fit)[i] -= (*(x_struct->data_pointers[tree_ind][i]))[0];
        }
    }
    return;
}

void XBCFContinuousModel::add_new_tree_fit(size_t tree_ind, std::unique_ptr<State> &state, std::unique_ptr<X_struct> &x_struct)
{

    if (state->treatment_flag)
    {
        for (size_t i = 0; i < (*state->tau_fit).size(); i++)
        {
            (*state->tau_fit)[i] += (*(x_struct->data_pointers[tree_ind][i]))[0];
        }
    }
    else
    {
        for (size_t i = 0; i < (*state->mu_fit).size(); i++)
        {
            (*state->mu_fit)[i] += (*(x_struct->data_pointers[tree_ind][i]))[0];
        }
    }
    return;
}

void XBCFContinuousModel::update_partial_residuals(size_t tree_ind, std::unique_ptr<State> &state, std::unique_ptr<X_struct> &x_struct)
{
    if (state->treatment_flag)
    {
        // treatment forest
        // (y - mu - Z * tau) / Z
        for (size_t i = 0; i < (*state->tau_fit).size(); i++)
        {
            (state->residual_std)[0][i] = ((*state->y_std)[i] - (*state->mu_fit)[i] - ((*state->Z_std)[0][i]) * (*state->tau_fit)[i]) / ((*state->Z_std)[0][i]);
        }
    }
    else
    {
        // prognostic forest
        // (y - mu - Z * tau)
        for (size_t i = 0; i < (*state->tau_fit).size(); i++)
        {
            (state->residual_std)[0][i] = ((*state->y_std)[i] - (*state->mu_fit)[i] - ((*state->Z_std)[0][i]) * (*state->tau_fit)[i]);
        }

    }
    return;
}
