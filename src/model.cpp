//////////////////////////////////////////////////////////////////////////////////////
// Details of sufficient statistics calculation, likelihood of various models
//////////////////////////////////////////////////////////////////////////////////////

#include "tree.h"
#include "model.h"
#include <cfenv>

//////////////////////////////////////////////////////////////////////////////////////
//
//
//  Normal Model
//
//
//////////////////////////////////////////////////////////////////////////////////////

void NormalModel::incSuffStat(State &state, size_t index_next_obs, std::vector<double> &suffstats)
{
    suffstats[0] += (*state.residual_std)[0][index_next_obs];
    suffstats[1] += pow((*state.residual_std)[0][index_next_obs], 2);
    suffstats[2] += 1;
    return;
}

void NormalModel::samplePars(State &state, std::vector<double> &suff_stat, std::vector<double> &theta_vector, double &prob_leaf)
{
    std::normal_distribution<double> normal_samp(0.0, 1.0);

    // test result should be theta
    theta_vector[0] = suff_stat[0] / pow(state.sigma, 2) / (1.0 / tau + suff_stat[2] / pow(state.sigma, 2)) + sqrt(1.0 / (1.0 / tau + suff_stat[2] / pow(state.sigma, 2))) * normal_samp(state.gen);

    return;
}

void NormalModel::update_state(State &state, size_t tree_ind, X_struct &x_struct)
{
    // This function updates sigma, residual variance

    std::vector<double> full_residual(state.n_y);

    for (size_t i = 0; i < (*state.residual_std)[0].size(); i++)
    {
        full_residual[i] = (*state.residual_std)[0][i] - (*(x_struct.data_pointers[tree_ind][i]))[0];
    }

    std::gamma_distribution<double> gamma_samp((state.n_y + kap) / 2.0, 2.0 / (sum_squared(full_residual) + s));
    state.update_sigma(1.0 / sqrt(gamma_samp(state.gen)));
    return;
}

void NormalModel::update_tau(State &state, size_t tree_ind, size_t sweeps, vector<vector<tree>> &trees)
{
    // this function samples tau based on leaf parameters of a single tree
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
    this->tau = 1.0 / gamma_samp(state.gen);
    return;
};

void NormalModel::update_tau_per_forest(State &state, size_t sweeps, vector<vector<tree>> &trees)
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

void NormalModel::initialize_root_suffstat(State &state, std::vector<double> &suff_stat)
{
    // this function calculates sufficient statistics at the root node when growing a new tree
    // sum of y
    suff_stat[0] = sum_vec((*state.residual_std)[0]);
    // sum of y squared
    suff_stat[1] = sum_squared((*state.residual_std)[0]);
    // number of observations in the node
    suff_stat[2] = state.n_y;
    return;
}

void NormalModel::updateNodeSuffStat(State &state, std::vector<double> &suff_stat, matrix<size_t> &Xorder_std, size_t &split_var, size_t row_ind)
{
    // this function updates the sufficient statistics at each intermediate nodes when growing a new tree
    // sum of y
    suff_stat[0] += (*state.residual_std)[0][Xorder_std[split_var][row_ind]];
    // sum of y squared
    suff_stat[1] += pow((*state.residual_std)[0][Xorder_std[split_var][row_ind]], 2);
    // number of data observations
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

void NormalModel::state_sweep(size_t tree_ind, size_t M, matrix<double> &residual_std, X_struct &x_struct) const
{
    // this function updates the residual vector (fitting target) for the next tree when the current tree was grown
    size_t next_index = tree_ind + 1;
    if (next_index == M)
    {
        next_index = 0;
    }

    for (size_t i = 0; i < residual_std[0].size(); i++)
    {
        // residual becomes subtracting the current grown tree, add back the next tree
        residual_std[0][i] = residual_std[0][i] - (*(x_struct.data_pointers[tree_ind][i]))[0] + (*(x_struct.data_pointers[next_index][i]))[0];
    }
    return;
}

double NormalModel::likelihood(std::vector<double> &temp_suff_stat, std::vector<double> &suff_stat_all, size_t N_left, bool left_side, bool no_split, State &state) const
{
    // likelihood equation,
    // note the difference of left_side == true / false
    // node_suff_stat is mean of y, sum of square of y, saved in tree class
    double sigma2 = state.sigma2;

    /////////////////////////////////////////////////////////////////////////
    //
    //  I know combining likelihood and likelihood_no_split looks nicer
    //  but this is a very fundamental function, executed many times
    //  the extra if(no_split) statement and value assignment make the code about 5% slower!!
    //
    /////////////////////////////////////////////////////////////////////////

    size_t nb;    // number of data in the current node
    double nbtau; // n * tau
    double y_sum;
    double y_squared_sum;

    // find sufficient statistics under certain cases
    if (no_split)
    {
        // calculate likelihood for no-split option (early stop)
        nb = suff_stat_all[2];
        nbtau = nb * tau;
        y_sum = suff_stat_all[0];
        y_squared_sum = suff_stat_all[1];
    }
    else
    {
        // calculate likelihood for regular split point
        if (left_side)
        {
            nb = N_left + 1;
            nbtau = nb * tau;
            y_sum = temp_suff_stat[0];
            y_squared_sum = temp_suff_stat[1];
        }
        else
        {
            nb = suff_stat_all[2] - N_left - 1;
            nbtau = nb * tau;
            y_sum = suff_stat_all[0] - temp_suff_stat[0];
            y_squared_sum = suff_stat_all[1] - temp_suff_stat[1];
        }
    }

    // note that LTPI = log(2 * pi), defined in common.h
    return -0.5 * nb * log(sigma2) + 0.5 * log(sigma2) - 0.5 * log(nbtau + sigma2) - 0.5 * y_squared_sum / sigma2 + 0.5 * tau * pow(y_sum, 2) / (sigma2 * (nbtau + sigma2));
}

// double NormalModel::likelihood_no_split(std::vector<double> &suff_stat, State&state) const
// {
//     // the likelihood of no-split option is a bit different from others
//     // because the sufficient statistics is y_sum here
//     // write a separate function, more flexibility
//     double ntau = suff_stat[2] * tau;
//     // double sigma2 = pow(state.sigma, 2);
//     double sigma2 = state.sigma2;
//     double value = suff_stat[2] * suff_stat[0]; // sum of y

//     return 0.5 * log(sigma2) - 0.5 * log(ntau + sigma2) + 0.5 * tau * pow(value, 2) / (sigma2 * (ntau + sigma2));
// }

void NormalModel::ini_residual_std(State &state)
{
    // initialize partial residual at (num_tree - 1) / num_tree * yhat
    double value = state.ini_var_yhat * ((double)state.num_trees - 1.0) / (double)state.num_trees;
    for (size_t i = 0; i < (*state.residual_std)[0].size(); i++)
    {
        (*state.residual_std)[0][i] = (*state.y_std)[i] - value;
    }
    return;
}

void NormalModel::predict_std(const double *Xtestpointer, size_t N_test, size_t p, size_t num_trees, size_t num_sweeps, matrix<double> &yhats_test_xinfo, vector<vector<tree>> &trees)
{
    // predict the output as a matrix
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

void NormalModel::predict_whole_std(const double *Xtestpointer, size_t N_test, size_t p, size_t num_trees, size_t num_sweeps, std::vector<double> &output_vec, vector<vector<tree>> &trees)
{
    // predict the output, stack as a vector
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
                // yhats_test_xinfo[sweeps][data_ind] += output[i][0];
                output_vec[data_ind + sweeps * N_test + i * num_sweeps * N_test] = output[i][0];
            }
        }
    }
    return;
}

//////////////////////////////////////////////////////////////////////////////////////
//
//
//  Logit Model
//
//
//
//////////////////////////////////////////////////////////////////////////////////////

// incSuffStat should take a state as its first argument
void LogitModel::incSuffStat(State &state, size_t index_next_obs, std::vector<double> &suffstats)
{
    suffstats[(*y_size_t)[index_next_obs]] += weight;
    for (size_t j = 0; j < dim_theta; ++j)
    {
        // suffstats[dim_residual + j] += weight * exp((*state.residual_std)[j][index_next_obs]);
        suffstats[dim_residual + j] += weight * exp((*phi)[index_next_obs] + (*state.residual_std)[j][index_next_obs]);
    }

    return;
}

void LogitModel::samplePars(State &state, std::vector<double> &suff_stat, std::vector<double> &theta_vector, double &prob_leaf)
{
    double n, sy;
    for (size_t j = 0; j < dim_theta; j++)
    {
        if (std::isnan(suff_stat[j]))
        {
            COUT << "unidentified error: suff_stat is nan for class " << j << endl;
        }
        // std::gamma_distribution<double> gammadist(tau_a + suff_stat[j], 1.0); // consider adding 1 sudo obs to prevent 0 theta value
        // theta_vector[j] = gammadist(state.gen) / (tau_b + suff_stat[dim_theta + j]);
        // while (theta_vector[j] == 0)
        // {
        //     theta_vector[j] = gammadist(state.gen) / (tau_b + suff_stat[dim_theta + j]);
        // }
        n = round(suff_stat[j]); // integer
        sy = suff_stat[dim_theta + j];

        if (j == 0)
        {
            // set the first category as baseline. always 1 or 0 in logscale
            // theta_vector[j] = pow(1.0 / dim_theta, 1.0 / state.num_trees);

            theta_vector[j] = drawlambdafromR(n, sy, c, d, state.gen);
        }
        else
        {
            theta_vector[j] = drawlambdafromR(n, sy, c, d, state.gen); //(n, sy, c, d, gen);
        }
    }
    return;
}

void LogitModel::update_state(State &state, size_t tree_ind, X_struct &x_struct, double &mean_lambda, std::vector<double> &var_lambda, size_t &count_lambda)
{
    // update residuals
    for (size_t i = 0; i < (*state.residual_std)[0].size(); i++)
    {
        for (size_t j = 0; j < dim_theta; ++j)
        {
            (*state.residual_std)[j][i] = (*state.residual_std)[j][i] + log((*(x_struct.data_pointers[tree_ind][i]))[j]);
        }
    }

    // // check residauls
    // std::vector<double> residual_sample(dim_residual);
    // for (size_t j = 0; j < dim_residual; j++)
    // {
    //     residual_sample[j] = exp((*phi)[0] + (*state.residual_std)[j][0]);
    // }
    // cout << "phi " << (*phi)[0] << " exp(phi) " << exp((*phi)[0]) << " residuals " << residual_sample << endl;

    // track accuracy of each group
    std::fill(acc_gp.begin(), acc_gp.end(), 0.0);
    std::vector<double> count_gp(dim_residual, 0.0);
    size_t yhat = 0;
    double max_resid = -INFINITY;

    // Calculate logloss
    size_t y_i;
    double sum_fits;
    logloss = 0; // reset logloss
    double prob = 0.0;
    std::gamma_distribution<double> gammadist(1.0, 1.0);

    double temp;
    double temp_sum;

    for (size_t i = 0; i < state.n_y; i++)
    {
        sum_fits = 0;
        y_i = (size_t)(*y_size_t)[i];
        yhat = 0;
        max_resid = -INFINITY;
        for (size_t j = 0; j < dim_residual; ++j)
        {
            sum_fits += exp((*state.residual_std)[j][i]);
            if ((*state.residual_std)[j][i] > max_resid)
            {
                yhat = j;
                max_resid = (*state.residual_std)[j][i];
            }
        }

        count_gp[y_i] += 1;
        if (yhat == y_i)
        {
            acc_gp[y_i] += 1;
        }
        // Sample phi
        if (update_phi)
        {
            (*phi)[i] = log(gammadist(state.gen)) - log(1.0 * sum_fits);
        }
        // calculate logloss
        prob = exp((*state.residual_std)[y_i][i]) / sum_fits; // logloss =  - log(p_j)

        logloss += -log(prob);
    }

    // std::vector<double> resid_samples(dim_residual);
    // for (size_t i = 0; i < dim_residual; i++){
    //     resid_samples[i] = exp((*state.residual_std)[i][0]);
    // }
    // cout << " " << endl;
    // cout << "phi = " << exp((*phi)[0]) << " residuals " << resid_samples << endl;

    logloss = logloss / state.n_y;

    // calculate accuracy per group
    accuracy = std::accumulate(acc_gp.begin(), acc_gp.end(), 0.0) / std::accumulate(count_gp.begin(), count_gp.end(), 0.0);
    for (size_t i = 0; i < dim_residual; i++)
    {
        acc_gp[i] = acc_gp[i] / count_gp[i];
    }

    // // sample weight based on logloss

    // weight = 1 / logloss;

    // std::gamma_distribution<> d(10.0, 1.0);
    // weight = d(state.gen) / (10.0 * logloss * state.n_y / (double)state.n_y + 1.0); // it's like shift p down by

    // if (std::isnan(weight))
    // {
    //     COUT << "weight is nan" << endl;
    // }
    if (update_weight)
    {
        // // sampling weight by random walk
        // std::normal_distribution<> dd(0, 0.05);
        // double weight_latent_proposal = weight_latent + dd(state.gen);

        // double weight_proposal = fabs(weight_latent_proposal - 1.0) + 1.0;

        // // cout << weight_latent_proposal << " " << fabs(weight_latent_proposal - 1.0) << " " << weight_proposal << endl;

        // // notice that logloss is negative, multiply by -1 to convert to likelihood
        // double value1 = w_likelihood(state, weight, logloss);

        // double value2 = w_likelihood(state, weight_proposal, logloss);
        // // this is in log scale
        // double ratio = value2 - value1 + normal_density(weight_latent_proposal, 0, 4, true) - normal_density(weight_latent, 0, 4, true);

        // ratio = std::min(1.0, exp(ratio));

        // std::uniform_real_distribution<> dis(0.0, 1.0);
        // if (dis(state.gen) < ratio)
        // {
        //     // accept
        //     weight = weight_proposal;
        //     weight_latent = weight_latent_proposal;

        //     // cout << "accept weight " << weight << endl;
        // }else{
        //     // cout << "reject weight " << weight << endl;
        // }

        // sampling on a grid
        // std::vector<double> www(state.weight_exponent);

        // for (size_t iii = 0; iii < www.size(); iii++)
        // {
        //     www[iii] = w_likelihood(state, pow(2, iii), logloss);
        // }

        // // normalize and taking exponents
        // temp = *std::max_element(www.begin(), www.end());

        // temp_sum = 0;

        // for (size_t iii = 0; iii < www.size(); iii++)
        // {
        //     www[iii] = exp(www[iii] - temp);
        //     temp_sum += www[iii];
        // }

        // for (size_t iii = 0; iii < www.size(); iii++)
        // {
        //     www[iii] = www[iii] / temp_sum;
        // }

        // cout << www << endl;

        // std::discrete_distribution<size_t> d(www.begin(), www.end());

        // weight = pow(2, (double)d(state.gen));
    }

    // Sample tau_a
    // if (update_tau)
    // {
    //     // mean = ( mean * N + new_data ) / new_N
    //     mean_lambda = mean_lambda * count_lambda;
    //     count_lambda += (*state.lambdas)[tree_ind].size() * dim_residual;

    //     // get sum of all leaf parameters in that tree.
    //     for (size_t i = 0; i < (*state.lambdas)[tree_ind].size(); i++)
    //     {
    //         mean_lambda += std::accumulate((*state.lambdas)[tree_ind][i].begin(), (*state.lambdas)[tree_ind][i].end(), 0.0);
    //     }
    //     mean_lambda = mean_lambda / count_lambda;

    //     // variance is updated partially
    //     // var_lambda[tree_ind] = 0;
    //     for (size_t i = 0; i < (*state.lambdas)[tree_ind].size(); i++)
    //     {
    //         for (size_t j = 0; j < dim_residual; j++)
    //         {
    //             // var_lambda += pow((*state.lambdas)[i][j][k] / max_lambda - mean_lambda, 2);
    //             // var_lambda[tree_ind] += pow((*state.lambdas)[tree_ind][i][j] - mean_lambda, 2);
    //             var_lambda[tree_ind] += pow((*state.lambdas)[tree_ind][i][j] / mean_lambda - 1, 2);
    //         }
    //     }
    //     double sqrt_lambda = sqrt( std::accumulate(var_lambda.begin(), var_lambda.end(), 0.0) / (count_lambda - 1) );

    //     // std::normal_distribution<> norm(0, 1);
    //     std::normal_distribution<> norm(1, sqrt_lambda);
    //     tau_a = 0;
    //     size_t count_tau_a = 0;
    //     while (tau_a <= 0)
    //     {
    //         // tau_a = tau_b * mean_lambda + norm(state.gen) * tau_b * sqrt_lambda ;
    //         tau_a = norm(state.gen) * tau_b;
    //         count_tau_a += 1;
    //     }
    //     // cout << "tau_a mean = " << mean_lambda << ", sqrt = " << sqrt_lambda << ", attempts = " << count_tau_a << endl;

    //     // remove lambdas of the next tree
    //     size_t next_index = tree_ind + 1;
    //     if (next_index == state.num_trees)
    //     {
    //         next_index = 0;
    //     }

    //     // mean = ( mean * N - old_data ) / new_N
    //     mean_lambda = mean_lambda * count_lambda;
    //     count_lambda -= (*state.lambdas)[next_index].size() * dim_residual;

    //     // get sum of all leaf parameters in that tree.
    //     for (size_t i = 0; i < (*state.lambdas)[next_index].size(); i++)
    //     {
    //         mean_lambda -= std::accumulate((*state.lambdas)[next_index][i].begin(), (*state.lambdas)[next_index][i].end(), 0.0);
    //     }
    //     mean_lambda = mean_lambda / count_lambda;
    //     var_lambda[next_index] = 0;
    // }

    return;
}

void LogitModel::update_weights(State &state, X_struct &x_struct, double &mean_lambda, std::vector<double> &var_lambda, size_t &count_lambda)
{

    if (update_weight)
    {
        // track accuracy of each group
        std::fill(acc_gp.begin(), acc_gp.end(), 0.0);
        std::vector<double> count_gp(dim_residual, 0.0);
        size_t yhat = 0;
        double max_resid = -INFINITY;

        // Calculate logloss
        size_t y_i;
        double sum_fits;
        logloss = 0; // reset logloss
        double prob = 0.0;
        std::gamma_distribution<double> gammadist(1.0, 1.0);

        double temp;
        double temp_sum;

        for (size_t i = 0; i < state.n_y; i++)
        {
            sum_fits = 0;
            y_i = (size_t)(*y_size_t)[i];
            yhat = 0;
            max_resid = -INFINITY;
            for (size_t j = 0; j < dim_residual; ++j)
            {
                sum_fits += exp((*state.residual_std)[j][i]);
                if ((*state.residual_std)[j][i] > max_resid)
                {
                    yhat = j;
                    max_resid = (*state.residual_std)[j][i];
                }
            }

            count_gp[y_i] += 1;
            if (yhat == y_i)
            {
                acc_gp[y_i] += 1;
            }
            // Sample phi
            if (update_phi)
            {
                (*phi)[i] = log(gammadist(state.gen)) - log(1.0 * sum_fits);
            }
            // calculate logloss
            prob = exp((*state.residual_std)[y_i][i]) / sum_fits; // logloss =  - log(p_j)

            logloss += -log(prob);
        }

        logloss = logloss / state.n_y;

        double exp_logloss = exp(-1.0 * logloss);
        double exp_logloss_last_sweep = exp(-1.0 * state.logloss_last_sweep);
        // double mu = -0.251 + 4.125 * exp_logloss - 15.09 * pow(exp_logloss, 2) + 14.90 * pow(exp_logloss, 3);
        // double mu_last_sweep = -0.251 + 4.125 * exp_logloss_last_sweep - 15.09 * pow(exp_logloss_last_sweep, 2) + 14.90 * pow(exp_logloss_last_sweep, 3);

        double weight_latent_proposal;
        double weight_proposal;
        double value1;
        double value2;
        double value3;
        double value4;
        double mu;
        double mu_last_sweep;

        if (exp_logloss <= 0.6)
        {
            mu = 1.5;
        }
        else
        {
            mu = 29.55 - 119.02 * exp_logloss + 152.95 * pow(exp_logloss, 2) - 60.98 * pow(exp_logloss, 3);
        }

        // sampling weight by random walk
        // MH_step is standard deviation
        std::normal_distribution<>
            dd(0, MH_step);

        weight_latent_proposal = exp(mu + dd(state.gen)) + 1.0;

        // double weight_proposal = fabs(weight_latent_proposal - 1.0) + 1.0;

        weight_proposal = weight_latent_proposal < 10.0 ? weight_latent_proposal : 10.0;

        value3 = normal_density(log(weight_latent - 1.0), mu, MH_step, true);

        value4 = normal_density(log(weight_latent_proposal - 1.0), mu, MH_step, true);

        // notice that logloss is negative, multiply by -1 to convert to likelihood
        value1 = w_likelihood(state, weight, logloss);

        value2 = w_likelihood(state, weight_proposal, logloss);

        // this is in log scale
        double ratio = value2 - value1 + value3 - value4 + normal_density(log(weight_latent_proposal - 1.0), log(2.0), 0.5, true) - normal_density(log(weight_latent - 1.0), log(2.0), 0.5, true);

        ratio = std::min(1.0, exp(ratio));

        std::uniform_real_distribution<> dis(0.0, 1.0);

        if (dis(state.gen) < ratio)
        {
            // accept
            weight = weight_proposal;
            weight_latent = weight_latent_proposal;
            state.logloss_last_sweep = logloss;
        }
    }
    return;
}

double LogitModel::w_likelihood(State &state, double weight, double logloss)
{
    double output = state.n_y * (std::lgamma(weight + (dim_residual - 1) * state.a + 1) - std::lgamma(weight + 1) - weight * logloss);

    return output;
}

void LogitModel::initialize_root_suffstat(State &state, std::vector<double> &suff_stat)
{

    /*
    // sum of y
    suff_stat[0] = sum_vec((*state.residual_std)[0]);
    // sum of y squared
    suff_stat[1] = sum_squared((*state.residual_std)[0]);
    // number of observations in the node
    suff_stat[2] = state.n_y;
    */

    // JINGYU check -- should i always plan to resize this vector?
    // reply: use it for now. Not sure how to call constructor of tree when initialize vector<vector<tree>>, see definition of trees2 in XBART_multinomial, train_all.cpp

    // remove resizing it does not work, strange

    suff_stat.resize(2 * dim_theta);
    std::fill(suff_stat.begin(), suff_stat.end(), 0.0);
    for (size_t i = 0; i < state.n_y; i++)
    {
        // from 0
        incSuffStat(state, i, suff_stat);
    }

    return;
}

void LogitModel::updateNodeSuffStat(State &state, std::vector<double> &suff_stat, matrix<size_t> &Xorder_std, size_t &split_var, size_t row_ind)
{
    incSuffStat(state, Xorder_std[split_var][row_ind], suff_stat);

    return;
}

void LogitModel::calculateOtherSideSuffStat(std::vector<double> &parent_suff_stat, std::vector<double> &lchild_suff_stat, std::vector<double> &rchild_suff_stat, size_t &N_parent, size_t &N_left, size_t &N_right, bool &compute_left_side)
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

void LogitModel::state_sweep(size_t tree_ind, size_t M, matrix<double> &residual_std, X_struct &x_struct) const
{

    size_t next_index = tree_ind + 1;
    if (next_index == M)
    {
        next_index = 0;
    }

    // cumulative product of trees, multiply current one, divide by next one

    for (size_t i = 0; i < residual_std[0].size(); i++)
    {
        for (size_t j = 0; j < dim_theta; ++j)
        {
            residual_std[j][i] = residual_std[j][i] - log((*(x_struct.data_pointers[next_index][i]))[j]);
            // residual_std[j][i] = residual_std[j][i] + log((*(x_struct.data_pointers[tree_ind][i]))[j]) - log((*(x_struct.data_pointers[next_index][i]))[j]);
        }
    }

    return;
}

double LogitModel::likelihood(std::vector<double> &temp_suff_stat, std::vector<double> &suff_stat_all, size_t N_left, bool left_side, bool no_split, State &state) const
{
    // likelihood equation,
    // note the difference of left_side == true / false
    // node_suff_stat is mean of y, sum of square of y, saved in tree class
    // double y_sum = (double)suff_stat_all[2] * suff_stat_all[0];
    // double y_sum = suff_stat_all[0];
    // double suff_one_side;

    // could rewrite without all these local assigments if that helps...
    std::vector<double> local_suff_stat = suff_stat_all; // no split

    if (!no_split)
    {
        if (left_side)
        {
            local_suff_stat = temp_suff_stat;
        }
        else
        {
            local_suff_stat = suff_stat_all - temp_suff_stat;
        }
    }
    // TODO: check suff stat meets requirment, e.g integer
    for (size_t i = 0; i < dim_residual; i++)
        local_suff_stat[i] = round(local_suff_stat[i]);

    return (LogitLIL(local_suff_stat));
}

void LogitModel::ini_residual_std(State &state)
{
    // double value = state.ini_var_yhat * ((double)state.num_trees - 1.0) / (double)state.num_trees;
    for (size_t i = 0; i < (*state.residual_std)[0].size(); i++)
    {
        // init leaf pars are all 1, partial fits are all 1
        // save fits in log scale -> 0.0
        for (size_t j = 0; j < dim_theta; ++j)
        {
            (*state.residual_std)[j][i] = 0.0; // save resdiual_std as log(lamdas), start at 0.0.
        }
    }
    return;
}

void LogitModel::predict_std(const double *Xtestpointer, size_t N_test, size_t p, size_t num_trees, size_t num_sweeps, matrix<double> &yhats_test_xinfo, vector<vector<tree>> &trees, std::vector<double> &output_vec)
{

    // output is a 3D array (armadillo cube), nsweeps by n by number of categories

    tree::tree_p bn;

    for (size_t sweeps = 0; sweeps < num_sweeps; sweeps++)
    {

        for (size_t data_ind = 0; data_ind < N_test; data_ind++)
        {

            for (size_t i = 0; i < trees[0].size(); i++)
            {
                // search leaf
                bn = trees[sweeps][i].search_bottom_std(Xtestpointer, data_ind, p, N_test);

                for (size_t k = 0; k < dim_residual; k++)
                {
                    // add all trees

                    // product of trees, thus sum of logs

                    output_vec[sweeps + data_ind * num_sweeps + k * num_sweeps * N_test] += log(bn->theta_vector[k]);
                }
            }
        }
    }

    // normalizing probability

    double denom = 0.0;
    double max_log_prob = -INFINITY;

    for (size_t sweeps = 0; sweeps < num_sweeps; sweeps++)
    {
        for (size_t data_ind = 0; data_ind < N_test; data_ind++)
        {

            max_log_prob = -INFINITY;
            // take exp, subtract max to avoid overflow

            // this line does not work for some reason, havd to write loops manually
            // output.tube(sweeps, data_ind) = exp(output.tube(sweeps, data_ind) - output.tube(sweeps, data_ind).max());

            // find max of probability for all classes
            for (size_t k = 0; k < dim_residual; k++)
            {
                if (output_vec[sweeps + data_ind * num_sweeps + k * num_sweeps * N_test] > max_log_prob)
                {
                    max_log_prob = output_vec[sweeps + data_ind * num_sweeps + k * num_sweeps * N_test];
                }
            }

            // take exp after subtracting max to avoid overflow
            for (size_t k = 0; k < dim_residual; k++)
            {
                output_vec[sweeps + data_ind * num_sweeps + k * num_sweeps * N_test] = exp(output_vec[sweeps + data_ind * num_sweeps + k * num_sweeps * N_test] - max_log_prob);
            }

            // calculate normalizing constant
            denom = 0.0;
            for (size_t k = 0; k < dim_residual; k++)
            {
                denom += output_vec[sweeps + data_ind * num_sweeps + k * num_sweeps * N_test];
            }

            // normalizing
            for (size_t k = 0; k < dim_residual; k++)
            {
                output_vec[sweeps + data_ind * num_sweeps + k * num_sweeps * N_test] = output_vec[sweeps + data_ind * num_sweeps + k * num_sweeps * N_test] / denom;
            }
        }
    }
    return;
}

// this function is for a standalone prediction function for classification case.
// with extra input iteration, which specifies which iteration (sweep / forest) to use
void LogitModel::predict_std_standalone(const double *Xtestpointer, size_t N_test, size_t p, size_t num_trees, size_t num_sweeps, matrix<double> &yhats_test_xinfo, vector<vector<tree>> &trees, std::vector<double> &output_vec, std::vector<size_t> &iteration)
{

    // output is a 3D array (armadillo cube), nsweeps by n by number of categories

    size_t num_iterations = iteration.size();

    tree::tree_p bn;

    COUT << "number of iterations " << num_iterations << " " << num_sweeps << endl;

    size_t sweeps;

    for (size_t iter = 0; iter < num_iterations; iter++)
    {
        sweeps = iteration[iter];

        for (size_t data_ind = 0; data_ind < N_test; data_ind++)
        {

            for (size_t i = 0; i < trees[0].size(); i++)
            {
                // search leaf
                bn = trees[sweeps][i].search_bottom_std(Xtestpointer, data_ind, p, N_test);

                for (size_t k = 0; k < dim_residual; k++)
                {
                    // add all trees

                    // product of trees, thus sum of logs

                    output_vec[iter + data_ind * num_iterations + k * num_iterations * N_test] += log(bn->theta_vector[k]);
                }
            }
        }
    }
    // normalizing probability

    double denom = 0.0;
    double max_log_prob = -INFINITY;

    for (size_t iter = 0; iter < num_iterations; iter++)
    {

        sweeps = iteration[iter];

        for (size_t data_ind = 0; data_ind < N_test; data_ind++)
        {

            max_log_prob = -INFINITY;
            // take exp, subtract max to avoid overflow

            // this line does not work for some reason, havd to write loops manually
            // output.tube(sweeps, data_ind) = exp(output.tube(sweeps, data_ind) - output.tube(sweeps, data_ind).max());

            // find max of probability for all classes
            for (size_t k = 0; k < dim_residual; k++)
            {
                if (output_vec[iter + data_ind * num_iterations + k * num_iterations * N_test] > max_log_prob)
                {
                    max_log_prob = output_vec[iter + data_ind * num_iterations + k * num_iterations * N_test];
                }
            }

            // take exp after subtracting max to avoid overflow
            for (size_t k = 0; k < dim_residual; k++)
            {
                output_vec[iter + data_ind * num_iterations + k * num_iterations * N_test] = exp(output_vec[iter + data_ind * num_iterations + k * num_iterations * N_test] - max_log_prob);
            }

            // calculate normalizing constant
            denom = 0.0;
            for (size_t k = 0; k < dim_residual; k++)
            {
                denom += output_vec[iter + data_ind * num_iterations + k * num_iterations * N_test];
            }

            // normalizing
            for (size_t k = 0; k < dim_residual; k++)
            {
                output_vec[iter + data_ind * num_iterations + k * num_iterations * N_test] = output_vec[iter + data_ind * num_iterations + k * num_iterations * N_test] / denom;
            }
        }
    }
    return;
}

//////////////////////////////////////////////////////////////////////////////////////
//
//
//  Logit Model Separate Trees
//
//
//
//////////////////////////////////////////////////////////////////////////////////////

void LogitModelSeparateTrees::samplePars(State &state, std::vector<double> &suff_stat, std::vector<double> &theta_vector, double &prob_leaf)
{
    size_t j = class_operating;

    std::gamma_distribution<double> gammadist(tau_a + suff_stat[j] + 1, 1.0); // add a sudo observation to prevent theta = 0;

    theta_vector[j] = gammadist(state.gen) / (tau_b + suff_stat[dim_theta + j]);
    if (theta_vector[j] == 0)
    {
        COUT << "unidentified error, theta for class " << j << " = 0" << endl;
        COUT << "suff_stats = " << suff_stat[j] << ", " << suff_stat[dim_theta + j] << ", tau_a = " << tau_a << endl;
        // exit(1);
    }

    return;
}

void LogitModelSeparateTrees::update_state(State &state, size_t tree_ind, X_struct &x_struct, double &mean_lambda, std::vector<double> &var_lambda, size_t &count_lambda)
{

    size_t y_i;
    double sum_fits;
    logloss = 0; // reset logloss
    std::gamma_distribution<double> gammadist(weight, 1.0);

    for (size_t i = 0; i < state.n_y; i++)
    {
        sum_fits = 0;
        y_i = (size_t)(*y_size_t)[i];
        for (size_t j = 0; j < dim_residual; ++j)
        {
            sum_fits += exp((*state.residual_std)[j][i]) * (*(x_struct.data_pointers_multinomial[j][tree_ind][i]))[j]; // f_j(x_i) = \prod lambdas
        }
        // Sample phi
        (*phi)[i] = gammadist(state.gen) / (1.0 * sum_fits);
        // calculate logloss
        logloss += -log(exp((*state.residual_std)[y_i][i]) * (*(x_struct.data_pointers_multinomial[y_i][tree_ind][i]))[y_i] / sum_fits); // logloss =  - log(p_j)
    }

    logloss = logloss / state.n_y;

    // Draw weight
    if (update_weight)
    {
        std::gamma_distribution<> d(state.n_y, 1);
        weight = d(state.gen) / (hmult * logloss * state.n_y + heps * (double)state.n_y) + 1;
    }
    // Sample tau_a
    if (update_tau)
    {
        // mean = ( mean * N + new_data ) / new_N
        mean_lambda = mean_lambda * count_lambda;

        // get sum of all leaf parameters in that tree.
        for (size_t i = 0; i < (*state.lambdas_separate)[tree_ind].size(); i++)
        {
            mean_lambda += std::accumulate((*state.lambdas_separate)[tree_ind][i].begin(), (*state.lambdas_separate)[tree_ind][i].end(), 0.0);
            count_lambda += (*state.lambdas_separate)[tree_ind][i].size();
        }
        mean_lambda = mean_lambda / count_lambda;

        // variance is updated partially
        // var_lambda[tree_ind] = 0;
        for (size_t i = 0; i < (*state.lambdas_separate)[tree_ind].size(); i++)
        {
            for (size_t j = 0; j < (*state.lambdas_separate)[tree_ind][i].size(); j++)
            {
                var_lambda[tree_ind] += pow((*state.lambdas_separate)[tree_ind][i][j] / mean_lambda - 1, 2);
            }
        }
        double sqrt_lambda = sqrt(std::accumulate(var_lambda.begin(), var_lambda.end(), 0.0) / (count_lambda - 1));

        // std::normal_distribution<> norm(0, 1);
        std::normal_distribution<> norm(1, sqrt_lambda);
        tau_a = 0;
        while (tau_a <= 0)
        {
            // tau_a = tau_b * mean_lambda + norm(state.gen) * tau_b * sqrt_lambda ;
            tau_a = norm(state.gen) * tau_b;
        }
        // remove lambdas of the next tree
        size_t next_index = tree_ind + 1;
        if (next_index == state.num_trees)
        {
            next_index = 0;
        }

        // mean = ( mean * N - old_data ) / new_N
        mean_lambda = mean_lambda * count_lambda;
        for (size_t i = 0; i < (*state.lambdas_separate)[tree_ind].size(); i++)
        {
            mean_lambda -= std::accumulate((*state.lambdas_separate)[tree_ind][i].begin(), (*state.lambdas_separate)[tree_ind][i].end(), 0.0);
            count_lambda -= (*state.lambdas_separate)[tree_ind][i].size();
        }
        mean_lambda = mean_lambda / count_lambda;
        var_lambda[next_index] = 0;
    }
}

void LogitModelSeparateTrees::state_sweep(size_t tree_ind, size_t M, matrix<double> &residual_std, X_struct &x_struct) const
{

    size_t next_index = tree_ind + 1;
    if (next_index == M)
    {
        next_index = 0;
    }

    // cumulative product of trees, multiply current one, divide by next one

    for (size_t i = 0; i < residual_std[0].size(); i++)
    {
        for (size_t j = 0; j < dim_theta; ++j)
        {
            residual_std[j][i] = residual_std[j][i] + log((*(x_struct.data_pointers_multinomial[j][tree_ind][i]))[j]) - log((*(x_struct.data_pointers_multinomial[j][next_index][i]))[j]);
        }
    }

    return;
}

double LogitModelSeparateTrees::likelihood(std::vector<double> &temp_suff_stat, std::vector<double> &suff_stat_all, size_t N_left, bool left_side, bool no_split, State &state) const
{

    // could rewrite without all these local assigments if that helps...
    std::vector<double> local_suff_stat = suff_stat_all; // no split

    if (!no_split)
    {
        if (left_side)
        {
            local_suff_stat = temp_suff_stat;
        }
        else
        {
            local_suff_stat = suff_stat_all - temp_suff_stat;
        }
    }

    return (LogitLIL(local_suff_stat));
}

void LogitModelSeparateTrees::predict_std(const double *Xtestpointer, size_t N_test, size_t p, size_t num_trees, size_t num_sweeps, matrix<double> &yhats_test_xinfo, vector<vector<vector<tree>>> &trees, std::vector<double> &output_vec)
{

    // output is a 3D array (armadillo cube), nsweeps by n by number of categories

    tree::tree_p bn;

    for (size_t data_ind = 0; data_ind < N_test; data_ind++)
    { // for each data observation

        for (size_t sweeps = 0; sweeps < num_sweeps; sweeps++)
        {

            for (size_t k = 0; k < dim_residual; k++)
            { // loop over class

                for (size_t i = 0; i < trees[0][0].size(); i++)
                {
                    bn = trees[k][sweeps][i].search_bottom_std(Xtestpointer, data_ind, p, N_test);

                    // product of trees, thus sum of logs
                    output_vec[sweeps + data_ind * num_sweeps + k * num_sweeps * N_test] += log(bn->theta_vector[k]);
                }
            }
        }
    }

    // normalizing probability

    double denom = 0.0;
    double max_log_prob = -INFINITY;

    for (size_t sweeps = 0; sweeps < num_sweeps; sweeps++)
    {
        for (size_t data_ind = 0; data_ind < N_test; data_ind++)
        {

            max_log_prob = -INFINITY;
            // take exp, subtract max to avoid overflow

            // this line does not work for some reason, havd to write loops manually
            // output.tube(sweeps, data_ind) = exp(output.tube(sweeps, data_ind) - output.tube(sweeps, data_ind).max());

            // find max of probability for all classes
            for (size_t k = 0; k < dim_residual; k++)
            {
                if (output_vec[sweeps + data_ind * num_sweeps + k * num_sweeps * N_test] > max_log_prob)
                {
                    max_log_prob = output_vec[sweeps + data_ind * num_sweeps + k * num_sweeps * N_test];
                }
            }

            // take exp after subtracting max to avoid overflow
            for (size_t k = 0; k < dim_residual; k++)
            {
                output_vec[sweeps + data_ind * num_sweeps + k * num_sweeps * N_test] = exp(output_vec[sweeps + data_ind * num_sweeps + k * num_sweeps * N_test] - max_log_prob);
            }

            // calculate normalizing constant
            denom = 0.0;
            for (size_t k = 0; k < dim_residual; k++)
            {
                // denom += output_vec[sweeps + data_ind * num_sweeps + k * num_sweeps * N_test];
                denom += output_vec[sweeps + data_ind * num_sweeps + k * num_sweeps * N_test];
            }

            // normalizing
            for (size_t k = 0; k < dim_residual; k++)
            {
                output_vec[sweeps + data_ind * num_sweeps + k * num_sweeps * N_test] = output_vec[sweeps + data_ind * num_sweeps + k * num_sweeps * N_test] / denom;
            }
        }
    }
    return;
}

// this function is for a standalone prediction function for classification case.
// with extra input iteration, which specifies which iteration (sweep / forest) to use
void LogitModelSeparateTrees::predict_std_standalone(const double *Xtestpointer, size_t N_test, size_t p, size_t num_trees, size_t num_sweeps, matrix<double> &yhats_test_xinfo, vector<vector<vector<tree>>> &trees, std::vector<double> &output_vec, std::vector<size_t> &iteration, double weight)
{

    // output is a 3D array (armadillo cube), nsweeps by n by number of categories

    size_t num_iterations = iteration.size();

    tree::tree_p bn;

    COUT << "number of iterations " << num_iterations << " " << num_sweeps << endl;

    size_t sweeps;

    for (size_t iter = 0; iter < num_iterations; iter++)
    {
        sweeps = iteration[iter];

        for (size_t data_ind = 0; data_ind < N_test; data_ind++)
        {

            for (size_t i = 0; i < trees[0][0].size(); i++)
            {

                for (size_t k = 0; k < dim_residual; k++)
                {
                    // search leaf
                    bn = trees[k][sweeps][i].search_bottom_std(Xtestpointer, data_ind, p, N_test);

                    // add all trees

                    // product of trees, thus sum of logs

                    output_vec[iter + data_ind * num_iterations + k * num_iterations * N_test] += log(bn->theta_vector[k]);
                }
            }
        }
    }

    // normalizing probability

    double denom = 0.0;
    double max_log_prob = -INFINITY;

    for (size_t iter = 0; iter < num_iterations; iter++)
    {

        sweeps = iteration[iter];

        for (size_t data_ind = 0; data_ind < N_test; data_ind++)
        {

            max_log_prob = -INFINITY;
            // take exp, subtract max to avoid overflow

            // this line does not work for some reason, havd to write loops manually
            // output.tube(sweeps, data_ind) = exp(output.tube(sweeps, data_ind) - output.tube(sweeps, data_ind).max());

            // find max of probability for all classes
            for (size_t k = 0; k < dim_residual; k++)
            {
                if (output_vec[iter + data_ind * num_iterations + k * num_iterations * N_test] > max_log_prob)
                {
                    max_log_prob = output_vec[iter + data_ind * num_iterations + k * num_iterations * N_test];
                }
            }

            // take exp after subtracting max to avoid overflow
            for (size_t k = 0; k < dim_residual; k++)
            {
                output_vec[iter + data_ind * num_iterations + k * num_iterations * N_test] = exp(output_vec[iter + data_ind * num_iterations + k * num_iterations * N_test] - max_log_prob);
            }

            // calculate normalizing constant
            denom = 0.0;
            for (size_t k = 0; k < dim_residual; k++)
            {
                // denom += output_vec[iter + data_ind * num_iterations + k * num_iterations * N_test];
                denom += output_vec[iter + data_ind * num_iterations + k * num_iterations * N_test];
            }

            // normalizing
            for (size_t k = 0; k < dim_residual; k++)
            {
                output_vec[iter + data_ind * num_iterations + k * num_iterations * N_test] = output_vec[iter + data_ind * num_iterations + k * num_iterations * N_test] / denom;
            }
        }
    }
    return;
}
