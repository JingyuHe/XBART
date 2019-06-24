#include "model.h"

//////////////////////////////////////////////////////////////////////////////////////
//
//
//  Normal Model
//
//
//////////////////////////////////////////////////////////////////////////////////////

void NormalModel::incSuffStat(std::vector<double> &y_std, size_t ix, std::vector<double> &suffstats)
{
    suffstats[0] += y_std[ix];
    return;
}

void NormalModel::samplePars(std::unique_ptr<State> &state, std::vector<double> &suff_stat, std::vector<double> &theta_vector, double &prob_leaf)
{
    std::normal_distribution<double> normal_samp(0.0, 1.0);

    // test result should be theta
    theta_vector[0] = suff_stat[0] * suff_stat[2] / pow(state->sigma, 2) / (1.0 / tau + suff_stat[2] / pow(state->sigma, 2)) + sqrt(1.0 / (1.0 / tau + suff_stat[2] / pow(state->sigma, 2))) * normal_samp(state->gen); //Rcpp::rnorm(1, 0, 1)[0];//* as_scalar(arma::randn(1,1));

    // also update probability of leaf parameters
    prob_leaf = normal_density(theta_vector[0], suff_stat[0] * suff_stat[2] / pow(state->sigma, 2) / (1.0 / tau + suff_stat[2] / pow(state->sigma, 2)), 1.0 / (1.0 / tau + suff_stat[2] / pow(state->sigma, 2)), true);

    // cout << "prob_leaf " << prob_leaf << endl;

    return;
}

void NormalModel::update_state(std::unique_ptr<State> &state, size_t tree_ind)
{
    // Draw Sigma
    state->residual_std_full = state->residual_std - state->predictions_std[tree_ind];
    std::gamma_distribution<double> gamma_samp((state->n_y + kap) / 2.0, 2.0 / (sum_squared(state->residual_std_full) + s));
    state->update_sigma(1.0 / sqrt(gamma_samp(state->gen)));
    return;
}

void NormalModel::initialize_root_suffstat(std::unique_ptr<State> &state, std::vector<double> &suff_stat)
{
    suff_stat[0] = sum_vec(state->residual_std) / (double)state->n_y;
    suff_stat[1] = sum_squared(state->residual_std);
    suff_stat[2] = state->n_y;
    return;
}

void NormalModel::updateNodeSuffStat(std::vector<double> &suff_stat, std::vector<double> &residual_std, xinfo_sizet &Xorder_std, size_t &split_var, size_t row_ind)
{
    suff_stat[0] += residual_std[Xorder_std[split_var][row_ind]];
    suff_stat[1] += pow(residual_std[Xorder_std[split_var][row_ind]], 2);
    suff_stat[2] = suff_stat[2] + 1;
    return;
}

void NormalModel::calculateOtherSideSuffStat(std::vector<double> &parent_suff_stat, std::vector<double> &lchild_suff_stat, std::vector<double> &rchild_suff_stat, size_t &N_parent, size_t &N_left, size_t &N_right, bool &compute_left_side)
{

    // in function split_xorder_std_categorical, for efficiency, the function only calculates suff stat of ONE child
    // this function calculate the other side based on parent and the other child

    if (compute_left_side)
    {
        rchild_suff_stat[0] = (parent_suff_stat[0] * N_parent - lchild_suff_stat[0]) / N_right;

        rchild_suff_stat[1] = parent_suff_stat[1] - lchild_suff_stat[1];

        lchild_suff_stat[0] = lchild_suff_stat[0] / N_left;

        rchild_suff_stat[2] = parent_suff_stat[2] - lchild_suff_stat[2];
    }
    else
    {
        lchild_suff_stat[0] = (parent_suff_stat[0] * N_parent - rchild_suff_stat[0]) / N_left;

        lchild_suff_stat[1] = parent_suff_stat[1] - rchild_suff_stat[1];

        rchild_suff_stat[0] = rchild_suff_stat[0] / N_right;

        lchild_suff_stat[2] = parent_suff_stat[2] - rchild_suff_stat[2];
    }
    return;
}

void NormalModel::state_sweep(const xinfo &predictions_std, size_t tree_ind, size_t M, std::vector<double> &residual_std) const
{
    size_t next_index = tree_ind + 1;
    if (next_index == M)
    {
        next_index = 0;
    }
    residual_std = residual_std - predictions_std[tree_ind] + predictions_std[next_index];
    return;
}

double NormalModel::likelihood(std::vector<double> &temp_suff_stat, std::vector<double> &node_suff_stat, size_t N_left, bool left_side, std::unique_ptr<State> &state) const
{
    // likelihood equation,
    // note the difference of left_side == true / false
    // node_suff_stat is mean of y, sum of square of y, saved in tree class
    double y_sum = (double)node_suff_stat[2] * node_suff_stat[0];
    double sigma2 = pow(state->sigma, 2);
    double ntau;

    if (left_side)
    {
        ntau = (N_left + 1) * tau;
        return 0.5 * log(sigma2) - 0.5 * log(ntau + sigma2) + 0.5 * tau * pow(temp_suff_stat[0], 2) / (sigma2 * (ntau + sigma2));
    }
    else
    {
        ntau = (node_suff_stat[2] - N_left - 1) * tau;
        return 0.5 * log(sigma2) - 0.5 * log(ntau + sigma2) + 0.5 * tau * pow(y_sum - temp_suff_stat[0], 2) / (sigma2 * (ntau + sigma2));
    }
}

double NormalModel::likelihood_no_split(std::vector<double> &suff_stat, std::unique_ptr<State> &state) const
{
    // the likelihood of no-split option is a bit different from others
    // because the sufficient statistics is y_sum here
    // write a separate function, more flexibility
    double ntau = suff_stat[2] * tau;
    double sigma2 = pow(state->sigma, 2);
    double value = suff_stat[2] * suff_stat[0]; // sum of y

    return 0.5 * log(sigma2) - 0.5 * log(ntau + sigma2) + 0.5 * tau * pow(value, 2) / (sigma2 * (ntau + sigma2));
}

double NormalModel::predictFromTheta(const std::vector<double> &theta_vector) const
{
    return theta_vector[0];
}

//////////////////////////////////////////////////////////////////////////////////////
//
//
//  CLT Model
//
//
//////////////////////////////////////////////////////////////////////////////////////