#include "tree.h"
#include "model.h"
#include <cfenv>

//////////////////////////////////////////////////////////////////////////////////////
//
//
//  binary treatment XBCF for regression discontinuity design extrpolation
//
//
//////////////////////////////////////////////////////////////////////////////////////


void XBCFrdModel::incSuffStat(State &state, size_t index_next_obs, std::vector<double> &suffstats)
{
    if (state.treatment_flag)
    {
        // treatment forest
        if ((*state.Z_std)[0][index_next_obs] == 1)
        {
            // if treated
            suffstats[1] += ((*state.y_std)[index_next_obs] - state.a * (*state.mu_fit)[index_next_obs] - state.b_vec[1] * (*state.tau_fit)[index_next_obs]) / state.b_vec[1];
            suffstats[3] += 1;
        }
        else
        {
            // if control group
            suffstats[0] += ((*state.y_std)[index_next_obs] - state.a * (*state.mu_fit)[index_next_obs] - state.b_vec[0] * (*state.tau_fit)[index_next_obs]) / state.b_vec[0];
            suffstats[2] += 1;
        }

        double running_value = *(state.X_std_mod + state.n_y * (state.p_continuous - 1) + index_next_obs);
        if ((running_value > cutoff - Owidth) & (running_value <= cutoff)){
            suffstats[4] += 1; // Ol
        } else if ((running_value > cutoff) & (running_value <= cutoff + Owidth)){
            suffstats[5] += 1; // Or
        }
    }
    else
    {
        // prognostic forest
        if ((*state.Z_std)[0][index_next_obs] == 1)
        {
            // if treated
            suffstats[1] += ((*state.y_std)[index_next_obs] - state.a * (*state.mu_fit)[index_next_obs] - state.b_vec[1] * (*state.tau_fit)[index_next_obs]) / state.a;
            suffstats[3] += 1;
        }
        else
        {
            // if control group
            suffstats[0] += ((*state.y_std)[index_next_obs] - state.a * (*state.mu_fit)[index_next_obs] - state.b_vec[0] * (*state.tau_fit)[index_next_obs]) / state.a;
            suffstats[2] += 1;
        }

        double running_value = *(state.X_std_con + state.n_y * (state.p_continuous - 1) + index_next_obs);
        if ((running_value > cutoff - Owidth) & (running_value <= cutoff)){
            suffstats[4] += 1; // Ol
        } else if ((running_value > cutoff) & (running_value <= cutoff + Owidth)){
            suffstats[5] += 1; // Or
        }
    }

    return;
}

void XBCFrdModel::initialize_root_suffstat(State &state, std::vector<double> &suff_stat)
{
    suff_stat.resize(dim_suffstat);
    std::fill(suff_stat.begin(), suff_stat.end(), 0.0);
    for (size_t i = 0; i < state.n_y; i++)
    {
        incSuffStat(state, i, suff_stat);
    }
    return;
}

double XBCFrdModel::likelihood(std::vector<double> &temp_suff_stat, std::vector<double> &suff_stat_all, size_t N_left, bool left_side, bool no_split, State &state) const
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

    double s0 = 0;
    double s1 = 0;
    double denominator;   // the denominator (1 + tau * precision_squared) is the same for both terms
    double s_psi_squared; // (residual * precision_squared)^2

    if (state.treatment_flag)
    {
        // if this is treatment forest
        s0 = state.sigma_vec[0] / fabs(state.b_vec[0]);
        s1 = state.sigma_vec[1] / fabs(state.b_vec[1]);
    }
    else
    {
        s0 = state.sigma_vec[0] / fabs(state.a);
        s1 = state.sigma_vec[1] / fabs(state.a);
    }

    if (no_split)
    {
        // check force split condition
        if ( (suff_stat_all[4] >= Omin) & (suff_stat_all[5] >= Omin) &  ((double (suff_stat_all[4] + suff_stat_all[5]) / (suff_stat_all[2] + suff_stat_all[3])) < Opct) ){
            // cout << "force split " << " Ol " << suff_stat_all[4] << " Or " << suff_stat_all[5] << " N " << suff_stat_all[2] + suff_stat_all[3] << endl;
            return -INFINITY;
        }
        denominator = 1 + (suff_stat_all[2] / pow(s0, 2) + suff_stat_all[3] / pow(s1, 2)) * tau_use;
        s_psi_squared = suff_stat_all[0] / pow(s0, 2) + suff_stat_all[1] / pow(s1, 2);
        cout << "No split likelihood not converted to zero; Denominator = " << denominator << " s_psi_squared = " << s_psi_squared << endl;
    }
    else
    {
        // set likelihood to 0 (-inf in log scale) if producing small leaves within bandwidth
        double Oll = temp_suff_stat[4];
        double Olr = temp_suff_stat[5];
        double Orl = suff_stat_all[4] - temp_suff_stat[4];
        double Orr = suff_stat_all[5] - temp_suff_stat[5];
        if ((Oll > 0) || (Olr > 0)) {
            if ((Oll < Omin) || (Olr < Omin)){
                return -INFINITY;
            }
        }
        if ((Orl > 0) || (Orr > 0)){
            if ((Orl < Omin) || (Orr < Omin)){
                return -INFINITY;
            }
        }

        if (left_side)
        {
            denominator = 1 + (temp_suff_stat[2] / pow(s0, 2) + temp_suff_stat[3] / pow(s1, 2)) * tau_use;
            s_psi_squared = temp_suff_stat[0] / pow(s0, 2) + temp_suff_stat[1] / pow(s1, 2);
        }
        else
        {
            denominator = 1 + ((suff_stat_all[2] - temp_suff_stat[2]) / pow(s0, 2) + (suff_stat_all[3] - temp_suff_stat[3]) / pow(s1, 2)) * tau_use;
            s_psi_squared = (suff_stat_all[0] - temp_suff_stat[0]) / pow(s0, 2) + (suff_stat_all[1] - temp_suff_stat[1]) / pow(s1, 2);
        }
    }
    return 0.5 * log(1 / denominator) + 0.5 * pow(s_psi_squared, 2) * tau_use / denominator;
}


void XBCFrdModel::predict_std(matrix<size_t> &Xorder_std, rd_struct &x_struct, std::vector<size_t> &X_counts, std::vector<size_t> &X_num_unique,
                            matrix<size_t> &Xtestorder_std, rd_struct &xtest_struct, std::vector<size_t> &Xtest_counts, std::vector<size_t> &Xtest_num_unique,
                            const double *Xtestpointer_con, const double *Xtestpointer_mod,
                            size_t N_test, size_t p_con, size_t p_mod, size_t num_trees_con, size_t num_trees_mod, size_t num_sweeps,
                            matrix<double> &prognostic_xinfo, matrix<double> &treatment_xinfo,
                            vector<vector<tree>> &trees_con, vector<vector<tree>> &trees_mod,
                            const double &theta, const double &tau)
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
            // getThetaForObs_Outsample(output_mod, trees_mod[sweeps], data_ind, Xtestpointer_mod, N_test, p_mod);

            getThetaForObs_Outsample(output_con, trees_con[sweeps], data_ind, Xtestpointer_con, N_test, p_con);

            // take sum of predictions of each tree, as final prediction
            // for (size_t i = 0; i < trees_mod[0].size(); i++)
            // {
            //     treatment_xinfo[sweeps][data_ind] += output_mod[i][0];
            // }

            for (size_t i = 0; i < trees_con[0].size(); i++)
            {
                prognostic_xinfo[sweeps][data_ind] += output_con[i][0];
            }

        }

        // get local ate
        std::vector<double> local_ate(num_trees_mod, 0.0);
        const double *run_var_x_pointer = x_struct.X_std + x_struct.n_y * (x_struct.p_continuous - 1);
        double run_var_value;
        size_t count_local = 0;
        for (size_t data_ind = 0; data_ind < x_struct.n_y; data_ind ++){
            run_var_value = *(run_var_x_pointer + data_ind);
            if ( (run_var_value <= x_struct.cutoff + x_struct.Owidth) & (run_var_value >= x_struct.cutoff - x_struct.Owidth) ){
                count_local += 1;
                getThetaForObs_Outsample(output_mod, trees_mod[sweeps], data_ind, x_struct.X_std, x_struct.n_y, p_mod);

                for (size_t tree_ind = 0; tree_ind < num_trees_mod; tree_ind++){
                    local_ate[tree_ind] += output_mod[tree_ind][0];
                }
            }
        }

        for (size_t tree_ind = 0; tree_ind < num_trees_mod; tree_ind++)
        {
            // cout << "sweeps " << sweeps << " tree " << tree_ind << " ate " << local_ate[tree_ind] / count_local << endl;
            std::vector<bool> active_var(Xorder_std.size(), false);
            trees_mod[sweeps][tree_ind].rd_predict_from_root(Xorder_std, x_struct, X_counts, X_num_unique, Xtestorder_std, xtest_struct, Xtest_counts, Xtest_num_unique,
                              treatment_xinfo, active_var, sweeps, tree_ind, theta, tau, local_ate[tree_ind] / count_local);
            // TODO: local_ate should be obtained on the tree level.
        }


    }
    return;
}
