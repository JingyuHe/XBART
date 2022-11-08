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
        //COUT << mean_residual_std[0][i] << endl;
        //COUT << 2*log(abs(mean_residual_std[0][i])) << " + " << log(mean_residual_std[1][i])<< "-" << log((*(x_struct.data_pointers[0][i]))[0]) << endl;
        // put residual to the log scale
        (*state.residual_std)[0][i] = 2*log(abs(mean_residual_std[0][i])) + log(mean_residual_std[1][i]) - log((*(x_struct.data_pointers[0][i]))[0]);
        //COUT << (*state.residual_std)[0][i] - log((*(x_struct.data_pointers[0][i]))[0]) << endl;
        //(*state.residual_std)[0][i] = log(pow(mean_residual_std[0][i],2)) - state.var_fit[i] + (*(x_struct.data_pointers[0][i]))[0];
        //COUT << (*state.residual_std)[0][i] << endl;
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
        //COUT << (*state.residual_std)[0][i] << " <- res | exp_res -> " << exp((*state.residual_std)[0][i]) << endl;
    }
    return;
}

void logNormalModel::incSuffStat(State &state,
                                 size_t index_next_obs,
                                 std::vector<double> &suffstats)
{
    // I have to pass matrix<double> &residual_std, size_t index_next_obs
    // which allows more flexibility for multidimensional residual_std

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
    //std::gamma_distribution<double> gammadist(tau_a + 0.5 * suff_stat[1],  1.0 / (tau_b + 0.5 * suff_stat[0]));
    //std::gamma_distribution<double> gammadist(tau_a, 1.0 / (tau_b));
    //theta_vector[0] = gammadist(state.gen);



    //theta_vector[0] = log(gammadist(state.gen));
    std::gamma_distribution<double> gammadist(tau_a + 0.5 * suff_stat[1], 1.0); // Maggie's Multinom: consider adding 1 sudo obs to prevent 0 theta value
    theta_vector[0] = gammadist(state.gen) / (tau_b + 0.5 * suff_stat[0]);
    //theta_vector[0] = log(gammadist(state.gen) / (tau_b + 0.5 * suff_stat[0]));
    //COUT << suff_stat[0] << " <- ss0 | ss1 -> " << suff_stat[1] << endl;
    //COUT << theta_vector[0] << endl;

//    while (theta_vector[0] == 0)
//    {
//        theta_vector[0] = log(gammadist(state.gen) / (tau_b + 0.5 * suff_stat[0]));
//    }

    return;
}

void logNormalModel::calculateOtherSideSuffStat(std::vector<double> &parent_suff_stat, std::vector<double> &lchild_suff_stat, std::vector<double> &rchild_suff_stat, size_t &N_parent, size_t &N_left, size_t &N_right, bool &compute_left_side)
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
//        for (size_t i = 0; i < residual_std[0].size(); i++)
//        {
//            COUT << residual_std[0][i] + log((*(x_struct.data_pointers[tree_ind][i]))[0]) << endl;
//        }
    }

    //COUT << "fitted value: " << (*(x_struct.data_pointers[tree_ind][1]))[0] << endl;

    for (size_t i = 0; i < residual_std[0].size(); i++)
    {

        //COUT << residual_std[0][i] << " + " << log((*(x_struct.data_pointers[tree_ind][i]))[0]) << " - " << log((*(x_struct.data_pointers[next_index][i]))[0]) << endl;

        residual_std[0][i] = residual_std[0][i] + log((*(x_struct.data_pointers[tree_ind][i]))[0]) - log((*(x_struct.data_pointers[next_index][i]))[0]);
        //residual_std[0][i] = residual_std[0][i] - (*(x_struct.data_pointers[tree_ind][i]))[0] + (*(x_struct.data_pointers[next_index][i]))[0];
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
                //yhats_test_xinfo[sweeps][data_ind] += output[i][0];
            }
            yhats_test_xinfo[sweeps][data_ind] = exp(yhats_test_xinfo[sweeps][data_ind]);
        }
    }
    return;
}
/*
// TODO: do not directly update the residuals for the other model here; update state instead
void logNormalModel::update_sigmas(matrix<double> &mean_residual_std, std::vector<double> &fit)
{
    if(mean_residual_std[0].size() != mean_residual_std[1].size()) {
        COUT << "SIZE MISMATCH OMG" << endl;
    }

    // initialize partial residual at the residual^2 from the mean model
    for (size_t i = 0; i < mean_residual_std[0].size(); i++)
    {
        //COUT << "fitted value is " << exp(fit[i]) << endl;
        mean_residual_std[1][i] = 1.0 / exp(fit[i]);
        if(mean_residual_std[1][i] < 0) {
            COUT << i << " <- i | res -> " << mean_residual_std[1][i] << endl;
        }
        mean_residual_std[2][i] = mean_residual_std[0][i] * mean_residual_std[1][i];
    }
    return;
}*/

void logNormalModel::update_sigmas(matrix<double> &mean_residual_std,
                                   size_t M,
                                   X_struct &x_struct)
{

    // update sigma2
    for (size_t i = 0; i < mean_residual_std[0].size(); i++)
    {
        double log_sigma2 = 0;
        for (size_t j = 0; j < M; j++)
        {
            log_sigma2 += log((*(x_struct.data_pointers[j][i]))[0]);
            //fit[i] += (*(x_struct.data_pointers[tree_ind][i]))[0];
            //COUT << i << "<- i, j -> " << j << endl;
        }
        //COUT << "fitted value is " << exp(fit[i]) << endl;
        mean_residual_std[1][i] = exp(log_sigma2);
        mean_residual_std[2][i] = mean_residual_std[0][i] * mean_residual_std[1][i];
    }
    return;
}

// DELETE: we don't draw sigma here (aat least for now)
void logNormalModel::update_state(State &state,
                                  size_t tree_ind,
                                  X_struct &x_struct)
{
    // Draw Sigma
    // (*state.residual_std)_full = (*state.residual_std) - state.predictions_std[tree_ind];

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
/*
void logNormalModel::update_tau(State &state, size_t tree_ind, size_t sweeps, vector<vector<tree>> & trees){
    std::vector<tree *> leaf_nodes;
    trees[sweeps][tree_ind].getbots(leaf_nodes);
    double sum_squared = 0.0;
    for(size_t i = 0; i < leaf_nodes.size(); i ++ ){
        sum_squared = sum_squared + pow(leaf_nodes[i]->theta_vector[0], 2);
    }
    double kap = this->tau_kap;
    double s = this->tau_s * this->tau_mean;

    std::gamma_distribution<double> gamma_samp((leaf_nodes.size() + kap) / 2.0, 2.0 / (sum_squared + s));
    this->tau = 1.0 / gamma_samp(state.gen);
    return;
};*/
/*
void logNormalModel::update_tau_per_forest(State &state, size_t sweeps, vector<vector<tree>> & trees){
    std::vector<tree *> leaf_nodes;
    for(size_t tree_ind = 0; tree_ind < state.num_trees; tree_ind ++){
        trees[sweeps][tree_ind].getbots(leaf_nodes);
    }
    double sum_squared = 0.0;
    for(size_t i = 0; i < leaf_nodes.size(); i ++ ){
        sum_squared = sum_squared + pow(leaf_nodes[i]->theta_vector[0], 2);
    }
    double kap = this->tau_kap;
    double s = this->tau_s * this->tau_mean;
    std::gamma_distribution<double> gamma_samp((leaf_nodes.size() + kap) / 2.0, 2.0 / (sum_squared + s));
    this->tau = 1.0 / gamma_samp(state.gen);
    return;
}*/