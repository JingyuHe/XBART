#include "tree.h"
#include "model.h"
#include <cfenv>

//////////////////////////////////////////////////////////////////////////////////////
//
//
//  Heteroskedastic XBCF (binary) Model
//
//
//////////////////////////////////////////////////////////////////////////////////////

void hskXBCFDiscreteModel::incSuffStat(State &state, size_t index_next_obs, std::vector<double> &suffstats)
{
    if (state.treatment_flag)
    {
        // treatment forest
        if ((*state.Z_std)[0][index_next_obs] == 1)
        {
            // if treated
            suffstats[1] += (*state.res_x_precision)[index_next_obs];
            suffstats[3] += (*state.precision)[index_next_obs];
        }
        else
        {
            // if control group
            suffstats[0] += (*state.res_x_precision)[index_next_obs];
            suffstats[2] += (*state.precision)[index_next_obs];
        }
    }
    else
    {
        // prognostic forest
        if ((*state.Z_std)[0][index_next_obs] == 1)
        {
            // if treated
            suffstats[1] += (*state.res_x_precision)[index_next_obs];
            suffstats[3] += (*state.precision)[index_next_obs];
        }
        else
        {
            // if control group
            suffstats[0] += (*state.res_x_precision)[index_next_obs];
            suffstats[2] += (*state.precision)[index_next_obs];
        }
    }
    return;
}

void hskXBCFDiscreteModel::samplePars(State &state, std::vector<double> &suff_stat, std::vector<double> &theta_vector, double &prob_leaf)
{
    std::normal_distribution<double> normal_samp(0.0, 1.0);

    double tau_use;

    if (state.treatment_flag)
    {
        tau_use = tau_mod;
    }
    else
    {
        tau_use = tau_con;
    }

    double s0 = 0;
    double s1 = 0;
    double s2 = 0;
    double s3 = 0;

    if (state.treatment_flag)
    {
        s0 = suff_stat[0] * pow(state.b_vec[0], 2);
        s1 = suff_stat[1] * pow(state.b_vec[1], 2);
        s2 = suff_stat[2] * pow(state.b_vec[0], 2);
        s3 = suff_stat[3] * pow(state.b_vec[1], 2);
    }
    else
    {
        s0 = suff_stat[0] * pow(state.a, 2);
        s1 = suff_stat[1] * pow(state.a, 2);
        s2 = suff_stat[2] * pow(state.a, 2);
        s3 = suff_stat[3] * pow(state.a, 2);
    }

    // step 1 (control group)
    double denominator0 = 1.0 / tau_use + s2;
    double m0 = (s0) / denominator0;
    double v0 = 1.0 / denominator0;

    // step 2 (treatment group)
    double denominator1 = 1.0 / v0 + s3;
    double m1 = (1.0 / v0) * m0 / denominator1 + s1 / denominator1;
    double v1 = 1.0 / denominator1;

    // sample leaf parameter
    theta_vector[0] = m1 + sqrt(v1) * normal_samp(state.gen);

    // also update probability of leaf parameters
    prob_leaf = 1.0;

    return;
}

/*
void hskXBCFDiscreteModel::update_tau(State &state, size_t tree_ind, size_t sweeps, vector<vector<tree>> &trees)
{
    std::vector<tree *> leaf_nodes;
    trees[sweeps][tree_ind].getbots(leaf_nodes);
    double sum_squared = 0.0;
    for (size_t i = 0; i < leaf_nodes.size(); i++)
    {
        sum_squared = sum_squared + pow(leaf_nodes[i]->theta_vector[0], 2);
    }

    double kap = (state.treatment_flag) ? this->tau_mod_kap : this->tau_con_kap;

    double s = (state.treatment_flag) ? this->tau_mod_s * this->tau_mod_mean : this->tau_con_s * this->tau_con_mean;

    std::gamma_distribution<double> gamma_samp((leaf_nodes.size() + kap) / 2.0, 2.0 / (sum_squared + s));

    double tau_sample = 1.0 / gamma_samp(state.gen);

    if (state.treatment_flag)
    {
        this->tau_mod = tau_sample;
    }
    else
    {
        this->tau_con = tau_sample;
    }

    return;
};
*/
void hskXBCFDiscreteModel::update_tau_per_forest(State &state, size_t sweeps, vector<vector<tree>> &trees)
{
    std::vector<tree *> leaf_nodes;
    for (size_t tree_ind = 0; tree_ind < state.num_trees; tree_ind++)
    {
        trees[sweeps][tree_ind].getbots(leaf_nodes);
    }
    double sum_squared = 0.0;
    for (size_t i = 0; i < leaf_nodes.size(); i++)
    {
        sum_squared = sum_squared + pow(leaf_nodes[i]->theta_vector[0], 2);
    };

    double kap = (state.treatment_flag) ? this->tau_mod_kap : this->tau_con_kap;

    double s = (state.treatment_flag) ? this->tau_mod_s * this->tau_mod_mean : this->tau_con_s * this->tau_con_mean;

    std::gamma_distribution<double> gamma_samp((leaf_nodes.size() + kap) / 2.0, 2.0 / (sum_squared + s));
    double tau_sample = 1.0 / gamma_samp(state.gen);

    if (state.treatment_flag)
    {
        this->tau_mod = tau_sample;
    }
    else
    {
        this->tau_con = tau_sample;
    }
    return;
}

void hskXBCFDiscreteModel::initialize_root_suffstat(State &state, std::vector<double> &suff_stat)
{
    suff_stat.resize(4);
    std::fill(suff_stat.begin(), suff_stat.end(), 0.0);
    for (size_t i = 0; i < state.n_y; i++)
    {
        incSuffStat(state, i, suff_stat);
    }
    return;
}

void hskXBCFDiscreteModel::updateNodeSuffStat(State &state, std::vector<double> &suff_stat, matrix<size_t> &Xorder_std, size_t &split_var, size_t row_ind)
{

    incSuffStat(state, Xorder_std[split_var][row_ind], suff_stat);

    return;
}

void hskXBCFDiscreteModel::calculateOtherSideSuffStat(std::vector<double> &parent_suff_stat, std::vector<double> &lchild_suff_stat, std::vector<double> &rchild_suff_stat, size_t &N_parent, size_t &N_left, size_t &N_right, bool &compute_left_side)
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

double hskXBCFDiscreteModel::likelihood(std::vector<double> &temp_suff_stat, std::vector<double> &suff_stat_all, size_t N_left, bool left_side, bool no_split, State &state) const
{
    // likelihood equation for XBCF with discrete binary treatment variable Z

    double tau_use;

    if (state.treatment_flag)
    {
        tau_use = tau_mod;
    }
    else
    {
        tau_use = tau_con;
    }

    double denominator;   // the denominator (1 + tau * precision_squared) is the same for both terms
    double s_psi_squared; // (residual * precision_squared)^2

    if (state.treatment_flag)
    {
        if (no_split)
        {
            denominator = 1.0 + (suff_stat_all[2] * pow(state.b_vec[0], 2) + suff_stat_all[3] * pow(state.b_vec[1], 2)) * tau_use;
            s_psi_squared = suff_stat_all[0] * pow(state.b_vec[0], 2) + suff_stat_all[1] * pow(state.b_vec[1], 2);
        }
        else
        {
            if (left_side)
            {
                denominator = 1.0 + (temp_suff_stat[2] * pow(state.b_vec[0], 2) + temp_suff_stat[3] * pow(state.b_vec[1], 2)) * tau_use;
                s_psi_squared = temp_suff_stat[0] * pow(state.b_vec[0], 2) + temp_suff_stat[1] * pow(state.b_vec[1], 2);
            }
            else
            {
                denominator = 1.0 + ((suff_stat_all[2] - temp_suff_stat[2]) * pow(state.b_vec[0], 2) +
                                     (suff_stat_all[3] - temp_suff_stat[3]) * pow(state.b_vec[1], 2)) *
                                        tau_use;
                s_psi_squared = (suff_stat_all[0] - temp_suff_stat[0]) * pow(state.b_vec[0], 2) +
                                (suff_stat_all[1] - temp_suff_stat[1]) * pow(state.b_vec[1], 2);
            }
        }
    }
    else
    {
        if (no_split)
        {
            denominator = 1.0 + (suff_stat_all[2] + suff_stat_all[3]) * tau_use * state.a;
            s_psi_squared = (suff_stat_all[0] + suff_stat_all[1]) * state.a;
        }
        else
        {
            if (left_side)
            {
                denominator = 1.0 + (temp_suff_stat[2] + temp_suff_stat[3]) * tau_use * state.a;
                s_psi_squared = (temp_suff_stat[0] + temp_suff_stat[1]) * state.a;
            }
            else
            {
                denominator = 1.0 + ((suff_stat_all[2] - temp_suff_stat[2]) + (suff_stat_all[3] - temp_suff_stat[3])) * tau_use * state.a;
                s_psi_squared = ((suff_stat_all[0] - temp_suff_stat[0]) + (suff_stat_all[1] - temp_suff_stat[1])) * state.a;
            }
        }
    }
    return 0.5 * log(1.0 / denominator) + 0.5 * pow(s_psi_squared, 2) * tau_use / denominator;
}

void hskXBCFDiscreteModel::ini_residual_std(State &state)
{
    // initialize the vector of full residuals
    double b_value;
    for (size_t i = 0; i < (*state.residual_std)[0].size(); i++)
    {
        b_value = ((*state.Z_std)[0][i] == 1) ? state.b_vec[1] : state.b_vec[0];

        (*state.residual_std)[0][i] = (*state.y_std)[i] - (state.a) * (*state.mu_fit)[i] - b_value * (*state.tau_fit)[i];

        (*state.precision)[i] = double(1.0 / state.sigma_vec[i]);

        (*state.res_x_precision)[i] = (*state.residual_std)[0][i] * (*state.precision)[i];
    }
    return;
}

void hskXBCFDiscreteModel::predict_std(matrix<double> &Ztestpointer, const double *Xtestpointer_con, const double *Xtestpointer_mod, size_t N_test, size_t p_con, size_t p_mod, size_t num_trees_con, size_t num_trees_mod, size_t num_sweeps, matrix<double> &yhats_test_xinfo, matrix<double> &prognostic_xinfo, matrix<double> &treatment_xinfo, vector<vector<tree>> &trees_con, vector<vector<tree>> &trees_mod)
{
    // predict the output as a matrix
    matrix<double> output_mod;

    // row : dimension of theta, column : number of trees
    ini_matrix(output_mod, this->dim_theta, trees_mod[0].size());

    matrix<double> output_con;
    ini_matrix(output_con, this->dim_theta, trees_con[0].size());

    for (size_t sweeps = 0; sweeps < num_sweeps; sweeps++)
    {
        for (size_t data_ind = 0; data_ind < N_test; data_ind++)
        {
            getThetaForObs_Outsample(output_mod, trees_mod[sweeps], data_ind, Xtestpointer_mod, N_test, p_mod);

            getThetaForObs_Outsample(output_con, trees_con[sweeps], data_ind, Xtestpointer_con, N_test, p_con);

            // take sum of predictions of each tree, as final prediction
            for (size_t i = 0; i < trees_mod[0].size(); i++)
            {
                treatment_xinfo[sweeps][data_ind] += output_mod[i][0];
            }

            for (size_t i = 0; i < trees_con[0].size(); i++)
            {
                prognostic_xinfo[sweeps][data_ind] += output_con[i][0];
            }

            if (Ztestpointer[0][data_ind] == 1)
            {
                // yhats_test_xinfo[sweeps][data_ind] = (state.a) * prognostic_xinfo[sweeps][data_ind] + (state.b_vec[1]) * treatment_xinfo[sweeps][data_ind];
            }
            else
            {
                // yhats_test_xinfo[sweeps][data_ind] = (state.a) * prognostic_xinfo[sweeps][data_ind] + (state.b_vec[0]) * treatment_xinfo[sweeps][data_ind];
            }
            yhats_test_xinfo[sweeps][data_ind] = prognostic_xinfo[sweeps][data_ind] + treatment_xinfo[sweeps][data_ind];
        }
    }
    return;
}

void hskXBCFDiscreteModel::ini_tau_mu_fit(State &state)
{
    double value = state.ini_var_yhat;
    for (size_t i = 0; i < (*state.residual_std)[0].size(); i++)
    {
        (*state.mu_fit)[i] = 0;
        (*state.tau_fit)[i] = value;
    }
    return;
}

void hskXBCFDiscreteModel::set_treatmentflag(State &state, bool value)
{
    state.treatment_flag = value;
    if (value)
    {
        // if treatment forest
        state.p = state.p_mod;
        state.p_categorical = state.p_categorical_mod;
        state.p_continuous = state.p_continuous_mod;
        state.Xorder_std = state.Xorder_std_mod;
        state.mtry = state.mtry_mod;
        state.num_trees = state.num_trees_mod;
        state.X_std = state.X_std_mod;
        this->alpha = this->alpha_mod;
        this->beta = this->beta_mod;
    }
    else
    {
        state.p = state.p_con;
        state.p_categorical = state.p_categorical_con;
        state.p_continuous = state.p_continuous_con;
        state.Xorder_std = state.Xorder_std_con;
        state.mtry = state.mtry_con;
        state.num_trees = state.num_trees_con;
        state.X_std = state.X_std_con;
        this->alpha = this->alpha_con;
        this->beta = this->beta_con;
    }
    return;
}

void hskXBCFDiscreteModel::subtract_old_tree_fit(size_t tree_ind, State &state, X_struct &x_struct)
{
    if (state.treatment_flag)
    {
        for (size_t i = 0; i < (*state.tau_fit).size(); i++)
        {
            (*state.tau_fit)[i] -= (*(x_struct.data_pointers[tree_ind][i]))[0];
        }
    }
    else
    {
        for (size_t i = 0; i < (*state.mu_fit).size(); i++)
        {
            (*state.mu_fit)[i] -= (*(x_struct.data_pointers[tree_ind][i]))[0];
        }
    }
    return;
}

void hskXBCFDiscreteModel::add_new_tree_fit(size_t tree_ind, State &state, X_struct &x_struct)
{

    if (state.treatment_flag)
    {
        for (size_t i = 0; i < (*state.tau_fit).size(); i++)
        {
            (*state.tau_fit)[i] += (*(x_struct.data_pointers[tree_ind][i]))[0];
        }
    }
    else
    {
        for (size_t i = 0; i < (*state.mu_fit).size(); i++)
        {
            (*state.mu_fit)[i] += (*(x_struct.data_pointers[tree_ind][i]))[0];
        }
    }
    return;
}

void hskXBCFDiscreteModel::update_partial_residuals(size_t tree_ind, State &state, X_struct &x_struct)
{
    std::vector<double> *y_data_use;

    if (state.survival)
    {
        // survival model fit imputed y rather than the original input
        y_data_use = state.y_imputed;
    }
    else
    {
        // regular model fits y
        y_data_use = state.y_std;
    }

    if (state.treatment_flag)
    {
        // treatment forest
        // (y - a * mu - b * tau) / b
        for (size_t i = 0; i < (*state.tau_fit).size(); i++)
        {
            if ((*state.Z_std)[0][i] == 1)
            {
                ((*state.residual_std))[0][i] = ((*y_data_use)[i] - state.a * (*state.mu_fit)[i] - (state.b_vec[1]) * (*state.tau_fit)[i]) / (state.b_vec[1]);

                (*state.res_x_precision)[i] = (*state.residual_std)[0][i] * (*state.precision)[i];
                /// pow(state.b_vec[1], 2);
            }
            else
            {
                ((*state.residual_std))[0][i] = ((*y_data_use)[i] - state.a * (*state.mu_fit)[i] - (state.b_vec[0]) * (*state.tau_fit)[i]) / (state.b_vec[0]);

                (*state.res_x_precision)[i] = (*state.residual_std)[0][i] * (*state.precision)[i];
                /// pow(state.b_vec[0], 2);
            }
        }
    }
    else
    {
        // prognostic forest
        // (y - a * mu - b * tau) / a
        for (size_t i = 0; i < (*state.tau_fit).size(); i++)
        {
            if ((*state.Z_std)[0][i] == 1)
            {
                ((*state.residual_std))[0][i] = ((*y_data_use)[i] - state.a * (*state.mu_fit)[i] - (state.b_vec[1]) * (*state.tau_fit)[i]);
                /// (state.a);
            }
            else
            {
                ((*state.residual_std))[0][i] = ((*y_data_use)[i] - state.a * (*state.mu_fit)[i] - (state.b_vec[0]) * (*state.tau_fit)[i]);
                /// (state.a);
            }
            (*state.res_x_precision)[i] = (*state.residual_std)[0][i] * (*state.precision)[i] / 1;
            // pow(state.a, 2);
        }
    }
    return;
}

void hskXBCFDiscreteModel::update_split_counts(State &state, size_t tree_ind)
{
    if (state.treatment_flag)
    {
        (*state.mtry_weight_current_tree) = (*state.mtry_weight_current_tree) + (*state.split_count_current_tree);
        (*state.split_count_all_tree_mod)[tree_ind] = (*state.split_count_current_tree);
    }
    else
    {
        (*state.mtry_weight_current_tree) = (*state.mtry_weight_current_tree) + (*state.split_count_current_tree);
        (*state.split_count_all_tree_con)[tree_ind] = (*state.split_count_current_tree);
    }
    return;
}

void hskXBCFDiscreteModel::update_a(State &state)
{
    // update parameter a, y = a * mu + b_z * tau

    std::normal_distribution<double> normal_samp(0.0, 1.0);

    // X^TX
    double mu2sum = 0;
    // X^TY
    double muressum = 0;

    // compute the residual y - b * tau(x)

    std::vector<double> *y_data_use;

    if (state.survival)
    {
        // survival model fit imputed y rather than the original input
        y_data_use = state.y_imputed;
    }
    else
    {
        // regular model fits y
        y_data_use = state.y_std;
    }

    for (size_t i = 0; i < state.n_y; i++)
    {
        if ((*state.Z_std)[0][i] == 1)
        {
            // if treated
            (*state.residual_std)[0][i] = (*y_data_use)[i] - (*state.tau_fit)[i] * state.b_vec[1];
        }
        else
        {
            (*state.residual_std)[0][i] = (*y_data_use)[i] - (*state.tau_fit)[i] * state.b_vec[0];
        }
    }

    for (size_t i = 0; i < state.n_y; i++)
    {
        // X^TX, scaled by heteroskedastic variances
        mu2sum += pow((*state.mu_fit)[i], 2) * (*state.precision)[i];

        // X^TY
        muressum += (*state.mu_fit)[i] * (*state.residual_std)[0][i] * (*state.precision)[i];
    }

    // update parameters

    // prior on a is N(0,1)
    // also after reweighting, data has residual variance sigma = 1

    // mean (X^TX + A)^{-1}(X^TY);
    double m12 = 1.0 / (mu2sum)*muressum;

    // variance
    double v12 = 1.0 / (mu2sum);

    state.a = m12 + sqrt(v12) * normal_samp(state.gen);

    return;
}

void hskXBCFDiscreteModel::update_b(State &state)
{
    // update b0 and b1 for XBCF discrete treatment

    std::normal_distribution<double> normal_samp(0.0, 1.0);

    double tau2sum_ctrl = 0;
    double tau2sum_trt = 0;
    double tauressum_ctrl = 0;
    double tauressum_trt = 0;

    std::vector<double> *y_data_use;

    if (state.survival)
    {
        // survival model fit imputed y rather than the original input
        y_data_use = state.y_imputed;
    }
    else
    {
        // regular model fits y
        y_data_use = state.y_std;
    }

    // compute the residual y-a*mu(x) using state's objects y_std, mu_fit and a
    for (size_t i = 0; i < state.n_y; i++)
    {
        (*state.residual_std)[0][i] = (*y_data_use)[i] - state.a * (*state.mu_fit)[i];
    }

    for (size_t i = 0; i < state.n_y; i++)
    {
        if ((*state.Z_std)[0][i] == 1)
        {
            tau2sum_trt += pow((*state.tau_fit)[i], 2) * (*state.precision)[i];
            tauressum_trt += (*state.tau_fit)[i] * (*state.residual_std)[0][i] * (*state.precision)[i];
        }
        else
        {
            tau2sum_ctrl += pow((*state.tau_fit)[i], 2) * (*state.precision)[i];
            tauressum_ctrl += (*state.tau_fit)[i] * (*state.residual_std)[0][i] * (*state.precision)[i];
        }
    }

    // update parameters
    // mean (X^TX + A)^{-1}(X^TY);
    // standard deviation
    double v0 = 1.0 / (tau2sum_ctrl + 2.0);
    double m0 = v0 * tauressum_ctrl;
    double v1 = 1.0 / (tau2sum_trt + 2.0);
    double m1 = v1 * (tauressum_trt);

    // sample b0, b1
    double b0 = m0 + sqrt(v0) * normal_samp(state.gen);
    double b1 = m1 + sqrt(v1) * normal_samp(state.gen);

    state.b_vec[1] = b1;
    state.b_vec[0] = b0;

    return;
}

void hskXBCFDiscreteModel::switch_state_params(State &state)
{
    // update state settings to mean forest
    // state.num_trees = state.num_trees_m;
    state.n_min = state.n_min_m;
    state.max_depth = state.max_depth_m;
    state.n_cutpoints = state.n_cutpoints_m;

    return;
}

void hskXBCFDiscreteModel::update_state(State &state)
{
    state.p = state.p_con;
    state.p_categorical = state.p_categorical_con;
    state.p_continuous = state.p_continuous_con;
    state.Xorder_std = state.Xorder_std_con;
    state.mtry = state.mtry_v;
    state.num_trees = state.num_trees_v;
    state.X_std = state.X_std_con;

    std::vector<double> *y_data_use;

    if (state.survival)
    {
        // survival model fit imputed y rather than the original input
        y_data_use = state.y_imputed;
    }
    else
    {
        // regular model fits y
        y_data_use = state.y_std;
    }

    for (size_t i = 0; i < state.n_y; i++)
    {
        if ((*state.Z_std)[0][i] == 1)
        {
            // if treated
            (*state.residual_std)[0][i] = (*y_data_use)[i] - state.a * (*state.mu_fit)[i] - state.b_vec[1] * (*state.tau_fit)[i];
        }
        else
        {
            // if control group
            (*state.residual_std)[0][i] = (*y_data_use)[i] - state.a * (*state.mu_fit)[i] - state.b_vec[0] * (*state.tau_fit)[i];
        }
    }
    for (size_t i = 0; i < (*state.residual_std)[0].size(); i++)
    {
        (*state.mean_res)[i] = (*state.residual_std)[0][i];
    }
    return;
}