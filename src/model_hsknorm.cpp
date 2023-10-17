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

void hskNormalModel::ini_residual_std(State &state)
{
    double value = state.ini_var_yhat * ((double)state.num_trees - 1.0) / (double)state.num_trees;

    if (state.survival)
    {
        for (size_t i = 0; i < (*state.residual_std)[0].size(); i++)
        {
            (*state.residual_std)[0][i] = (*state.y_imputed)[i] - value;
            //        (*state.residual_std)[1][i] = double (1.0 / state.sigma_vec[i]);
            (*state.precision)[i] = double(1.0 / state.sigma_vec[i]);
            //        (*state.residual_std)[2][i] = (*state.residual_std)[0][i] * (*state.residual_std)[1][i];
            (*state.res_x_precision)[i] = (*state.residual_std)[0][i] * (*state.precision)[i];
        }
    }
    else
    {
        for (size_t i = 0; i < (*state.residual_std)[0].size(); i++)
        {
            (*state.residual_std)[0][i] = (*state.y_std)[i] - value;
            //        (*state.residual_std)[1][i] = double (1.0 / state.sigma_vec[i]);
            (*state.precision)[i] = double(1.0 / state.sigma_vec[i]);
            //        (*state.residual_std)[2][i] = (*state.residual_std)[0][i] * (*state.residual_std)[1][i];
            (*state.res_x_precision)[i] = (*state.residual_std)[0][i] * (*state.precision)[i];
        }
    }
    return;
}

void hskNormalModel::initialize_root_suffstat(State &state, std::vector<double> &suff_stat)
{
    // sum of r/sig2
    suff_stat[0] = sum_vec((*state.res_x_precision));
    // sum of 1/sig2
    suff_stat[1] = sum_vec((*state.precision));
    return;
}

void hskNormalModel::incSuffStat(State &state, size_t index_next_obs, std::vector<double> &suffstats)
{
    // r/sigma^2
    suffstats[0] += (*state.res_x_precision)[index_next_obs];
    // 1/sigma^2
    suffstats[1] += (*state.precision)[index_next_obs];
}

void hskNormalModel::samplePars(State &state, std::vector<double> &suff_stat, std::vector<double> &theta_vector, double &prob_leaf)
{
    std::normal_distribution<double> normal_samp(0.0, 1.0);

    // test result should be theta
    theta_vector[0] = suff_stat[0] / (1.0 / tau + suff_stat[1]) + sqrt(1.0 / (1.0 / tau + suff_stat[1])) * normal_samp(state.gen);
    return;
}

void hskNormalModel::updateNodeSuffStat(State &state,
                                        std::vector<double> &suff_stat, matrix<size_t> &Xorder_std, size_t &split_var, size_t row_ind)
{
    incSuffStat(state, Xorder_std[split_var][row_ind], suff_stat);
    return;
}

void hskNormalModel::calculateOtherSideSuffStat(std::vector<double> &parent_suff_stat, std::vector<double> &lchild_suff_stat, std::vector<double> &rchild_suff_stat, size_t &N_parent, size_t &N_left, size_t &N_right, bool &compute_left_side)
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

void hskNormalModel::state_sweep(size_t tree_ind, size_t M, State &state, X_struct &x_struct) const
{
    size_t next_index = tree_ind + 1;
    if (next_index == M)
    {
        next_index = 0;
    }

    for (size_t i = 0; i < (*state.residual_std)[0].size(); i++)
    {
        (*state.residual_std)[0][i] = (*state.residual_std)[0][i] - (*(x_struct.data_pointers[tree_ind][i]))[0] + (*(x_struct.data_pointers[next_index][i]))[0];
        //        (*state.residual_std)[2][i] = (*state.residual_std)[0][i] * (*state.residual_std)[1][i];
        (*state.res_x_precision)[i] = (*state.residual_std)[0][i] * (*state.precision)[i];
    }
    return;
}

double hskNormalModel::likelihood(std::vector<double> &temp_suff_stat, std::vector<double> &suff_stat_all, size_t N_left, bool left_side, bool no_split, State &state) const
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
            prec = suff_stat_all[1] - temp_suff_stat[1];
        }
    }

    return 0.5 * log(1.0 / (1.0 + tau * prec)) + 0.5 * res2 / (1.0 / tau + prec);
}

void hskNormalModel::predict_std(const double *Xtestpointer, size_t N_test, size_t p, size_t num_trees, size_t num_sweeps, matrix<double> &yhats_test_xinfo, vector<vector<tree>> &trees)
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
                yhats_test_xinfo[sweeps][data_ind] += output[i][0];
            }
        }
    }
    return;
}

void hskNormalModel::switch_state_params(State &state)
{
    // update state settings to mean forest
    state.num_trees = state.num_trees_m;
    state.n_min = state.n_min_m;
    state.max_depth = state.max_depth_m;
    state.n_cutpoints = state.n_cutpoints_m;

    return;
}

void hskNormalModel::store_residual(State &state, X_struct &x_struct)
{
    // once finish mean trees, copy residual_std to mean_res, saved for the next round
    // state.residual_std will be clear for the variance tree

    for (size_t i = 0; i < (*state.residual_std)[0].size(); i++)
    {
        // save results
        (*state.mean_res)[i] = (*state.residual_std)[0][i];

        // update to full residuals of mean trees
        (*state.residual_std)[0][i] -= (*(x_struct.data_pointers[0][i]))[0];
    }
    return;
}

void hskNormalModel::update_tau_per_forest(State &state, size_t sweeps, vector<vector<tree>> &trees)
{
    // this function samples tau based on all leaf parameters of the entire forest (a sweep)
    // tighter posterior, better performance

    std::vector<tree *> leaf_nodes;
    for (size_t tree_ind = 0; tree_ind < state.num_trees; tree_ind++)
    {
        trees[sweeps][tree_ind].getbots(leaf_nodes);
    }
    double sum_squared = 0.0;
    for (size_t i = 0; i < leaf_nodes.size(); i++)
    {
        sum_squared = sum_squared + pow(leaf_nodes[i]->theta_vector[0], 2);
    }
    double kap = this->tau_kap;
    double s = this->tau_s * this->tau_mean;
    std::gamma_distribution<double> gamma_samp((leaf_nodes.size() + kap) / 2.0, 2.0 / (sum_squared + s));
    this->tau = 1.0 / gamma_samp(state.gen);
    return;
}