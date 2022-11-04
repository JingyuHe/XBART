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
    //COUT << "Residual dim: " << state.residual_std.size() << endl;

    // initialize partial residual at (num_tree - 1) / num_tree * yhat
    double value = state.ini_var_yhat * ((double)state.num_trees - 1.0) / (double)state.num_trees;
    for (size_t i = 0; i < (*state.residual_std)[0].size(); i++)
    {
        (*state.residual_std)[0][i] = (*state.y_std)[i] - value;
        //COUT << (*state.residual_std)[0][i] << endl;
        (*state.residual_std)[1][i] = double (1.0 / state.sigma_vec[i]);
        //COUT << (*state.residual_std)[1][i] << endl;
        (*state.residual_std)[2][i] = (*state.residual_std)[0][i] * (*state.residual_std)[1][i];
    }

    return;
}

void hskNormalModel::initialize_root_suffstat(State &state, std::vector<double> &suff_stat)
{
    // sum of r/sig2
    suff_stat[0] = sum_vec((*state.residual_std)[2]);
    // sum of 1/sig2
    suff_stat[1] = sum_vec((*state.residual_std)[1]);
    // sum of r
    //suff_stat[2] = sum_vec((*state.residual_std)[0]);
/*
    COUT << "parent node | ss0: " << suff_stat[0] << ", ss1:" << suff_stat[1] << endl;
    for (size_t i = 0; i < (*state.residual_std)[1].size(); i++)
    {
        if((*state.residual_std)[1][i] < 0) {
            COUT << i << " <- i | res -> " << (*state.residual_std)[1][i] << endl;
        }
    }

    if(suff_stat[1] < 0)
        COUT << suff_stat[1] << "<- tmp ini root " << endl;
*/
    return;
}

void hskNormalModel::incSuffStat(matrix<double> &residual_std, size_t index_next_obs, std::vector<double> &suffstats)
{
    // I have to pass matrix<double> &residual_std, size_t index_next_obs
    // which allows more flexibility for multidimensional residual_std
    suffstats[0] += residual_std[2][index_next_obs]; // r/sigma^2
    suffstats[1] += residual_std[1][index_next_obs]; // 1/sigma^2
    //suffstats[2] += residual_std[0][index_next_obs]; // r

}

void hskNormalModel::samplePars(State &state, std::vector<double> &suff_stat, std::vector<double> &theta_vector, double &prob_leaf)
{
    std::normal_distribution<double> normal_samp(0.0, 1.0);

    // test result should be theta
    theta_vector[0] = suff_stat[0] / (1.0 / tau + suff_stat[1])
                    + sqrt(1.0 / (1.0 / tau + suff_stat[1])) * normal_samp(state.gen);
/*    if(suff_stat[1] < 0) {
        COUT << suff_stat[0] << " <- ss0 | ss1 -> " << suff_stat[1] << endl;
        COUT << theta_vector[0] << endl;
    }
*/
    return;
}

void hskNormalModel::updateNodeSuffStat(std::vector<double> &suff_stat, matrix<double> &residual_std, matrix<size_t> &Xorder_std, size_t &split_var, size_t row_ind)
{
    incSuffStat(residual_std, Xorder_std[split_var][row_ind], suff_stat);
    //COUT << "local node | ss0: " << suff_stat[0] << ", ss1:" << suff_stat[1] << endl;
    return;
}

void hskNormalModel::calculateOtherSideSuffStat(std::vector<double> &parent_suff_stat, std::vector<double> &lchild_suff_stat, std::vector<double> &rchild_suff_stat, size_t &N_parent, size_t &N_left, size_t &N_right, bool &compute_left_side)
{

    // in function split_xorder_std_categorical, for efficiency, the function only calculates suff stat of ONE child
    // this function calculate the other side based on parent and the other child
    //if(rchild_suff_stat[1] < 0 || parent_suff_stat[1] < 0 || lchild_suff_stat[1] < 0) {
    //    COUT << "right: " << rchild_suff_stat[1] << ", parent: " << parent_suff_stat[1] << ", left: " << lchild_suff_stat[1] << endl;
    //}

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

void hskNormalModel::state_sweep(size_t tree_ind, size_t M, matrix<double> &residual_std, X_struct &x_struct) const
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
        residual_std[0][i] = residual_std[0][i] - (*(x_struct.data_pointers[tree_ind][i]))[0] + (*(x_struct.data_pointers[next_index][i]))[0];
        residual_std[2][i] = residual_std[0][i] * residual_std[1][i];
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
            prec= suff_stat_all[1] - temp_suff_stat[1];
        }
    }
/*    if(temp_suff_stat[1] < 0 || suff_stat_all[1] < temp_suff_stat[1])
        COUT << temp_suff_stat[1] << "<- tmp | all -> " << suff_stat_all[1] << endl;
*/
    return log(1.0 / (1.0 + tau * prec)) + res2 / (1.0 / tau + prec);
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

// DELETE: we don't update sigma within this model
void hskNormalModel::update_state(State &state, size_t tree_ind, X_struct &x_struct)
{
    // Draw Sigma
    // state.residual_std_full = state.residual_std - state.predictions_std[tree_ind];

    // residual_std is only 1 dimensional for regression model

    std::vector<double> full_residual(state.n_y);

    for (size_t i = 0; i < (*state.residual_std)[0].size(); i++)
    {
        full_residual[i] = (*state.residual_std)[0][i] - (*(x_struct.data_pointers[tree_ind][i]))[0];
    }

    std::gamma_distribution<double> gamma_samp((state.n_y + kap) / 2.0, 2.0 / (sum_squared(full_residual) + s));
    state.update_sigma(1.0 / sqrt(gamma_samp(state.gen)));
    return;
}