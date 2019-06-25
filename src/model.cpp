#include "model.h"

//////////////////////////////////////////////////////////////////////////////////////
//
//
//  Normal Model
//
//
//////////////////////////////////////////////////////////////////////////////////////

void NormalModel::incSuffStat(double next_obs, std::vector<double> &suffstats)
{
    suffstats[0] += next_obs;
    return;
}

void NormalModel::samplePars(std::unique_ptr<State> &state, std::vector<double> &suff_stat, std::vector<double> &theta_vector, double &prob_leaf)
{
    std::normal_distribution<double> normal_samp(0.0, 1.0);

    // test result should be theta
    theta_vector[0] = suff_stat[0] / pow(state->sigma, 2) / (1.0 / tau + suff_stat[2] / pow(state->sigma, 2)) + sqrt(1.0 / (1.0 / tau + suff_stat[2] / pow(state->sigma, 2))) * normal_samp(state->gen); //Rcpp::rnorm(1, 0, 1)[0];//* as_scalar(arma::randn(1,1));

    // also update probability of leaf parameters
    prob_leaf = normal_density(theta_vector[0], suff_stat[0] / pow(state->sigma, 2) / (1.0 / tau + suff_stat[2] / pow(state->sigma, 2)), 1.0 / (1.0 / tau + suff_stat[2] / pow(state->sigma, 2)), true);

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
    // sum of y
    suff_stat[0] = sum_vec(state->residual_std);
    // sum of y squared
    suff_stat[1] = sum_squared(state->residual_std);
    // number of observations in the node
    suff_stat[2] = state->n_y;
    return;
}

void NormalModel::updateNodeSuffStat(std::vector<double> &suff_stat, std::vector<double> &residual_std, matrix<size_t> &Xorder_std, size_t &split_var, size_t row_ind)
{
    suff_stat[0] += residual_std[Xorder_std[split_var][row_ind]];
    suff_stat[1] += pow(residual_std[Xorder_std[split_var][row_ind]], 2);
    suff_stat[2] += 1;
    return;
}

void NormalModel::calculateOtherSideSuffStat(std::vector<double> &parent_suff_stat, std::vector<double> &lchild_suff_stat, std::vector<double> &rchild_suff_stat, size_t &N_parent, size_t &N_left, size_t &N_right, bool &compute_left_side)
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

void NormalModel::state_sweep(size_t tree_ind, size_t M, std::vector<double> &residual_std, std::unique_ptr<X_struct> &x_struct) const
{
    size_t next_index = tree_ind + 1;
    if (next_index == M)
    {
        next_index = 0;
    }

    ////////////////////////////////////////////////////////
    // Be care of line 151 in train_all.cpp, initial_theta
    ////////////////////////////////////////////////////////

    for(size_t i = 0; i < residual_std.size(); i++ ){
        residual_std[i] = residual_std[i] - (*(x_struct->data_pointers[tree_ind][i]))[0] + (*(x_struct->data_pointers[next_index][i]))[0];
    }

    return;
}

double NormalModel::likelihood(std::vector<double> &temp_suff_stat, std::vector<double> &suff_stat_all, size_t N_left, bool left_side, bool no_split, std::unique_ptr<State> &state) const
{
    // likelihood equation,
    // note the difference of left_side == true / false
    // node_suff_stat is mean of y, sum of square of y, saved in tree class
    // double y_sum = (double)suff_stat_all[2] * suff_stat_all[0];
    double y_sum = suff_stat_all[0];
    double sigma2 = state->sigma2;
    double ntau;
    double suff_one_side;

    /////////////////////////////////////////////////////////////////////////
    //                                                                     
    //  I know combining likelihood and likelihood_no_split looks nicer    
    //  but this is a very fundamental function, executed many times       
    //  the extra if(no_split) statement and value assignment make the code about 5% slower!!  
    //                                                                     
    /////////////////////////////////////////////////////////////////////////

    if (no_split)
    {
        ntau = suff_stat_all[2] * tau;
        suff_one_side = y_sum;
    }else{
        if (left_side)
        {
            ntau = (N_left + 1) * tau;
            suff_one_side = temp_suff_stat[0];
        }
        else
        {
            ntau = (suff_stat_all[2] - N_left - 1) * tau;
            suff_one_side = y_sum - temp_suff_stat[0];
        }
    }

    return 0.5 * log(sigma2) - 0.5 * log(ntau + sigma2) + 0.5 * tau * pow(suff_one_side, 2) / (sigma2 * (ntau + sigma2));
}

// double NormalModel::likelihood_no_split(std::vector<double> &suff_stat, std::unique_ptr<State> &state) const
// {
//     // the likelihood of no-split option is a bit different from others
//     // because the sufficient statistics is y_sum here
//     // write a separate function, more flexibility
//     double ntau = suff_stat[2] * tau;
//     // double sigma2 = pow(state->sigma, 2);
//     double sigma2 = state->sigma2;
//     double value = suff_stat[2] * suff_stat[0]; // sum of y

//     return 0.5 * log(sigma2) - 0.5 * log(ntau + sigma2) + 0.5 * tau * pow(value, 2) / (sigma2 * (ntau + sigma2));
// }

// void NormalModel::ini_residual_std(std::unique_ptr<State> &state){
//     double value = state->ini_var_yhat * ((double)state->num_trees - 1.0) / (double) state->num_trees;
//     for(size_t i=0; i < state->residual_std.size(); i ++ ){
//         state->residual_std[i] = (*state->y_std)[i] - value; 
//     }
//     return;
// }


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