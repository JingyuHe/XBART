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

void NormalModel::incSuffStat(matrix<double> &residual_std, size_t index_next_obs, std::vector<double> &suffstats)
{
    // I have to pass matrix<double> &residual_std, size_t index_next_obs
    // which allows more flexibility for multidimensional residual_std

    suffstats[0] += residual_std[0][index_next_obs];
    return;
}

void NormalModel::samplePars(std::unique_ptr<State> &state, std::vector<double> &suff_stat, std::vector<double> &theta_vector, double &prob_leaf)
{
    std::normal_distribution<double> normal_samp(0.0, 1.0);

    // test result should be theta
    theta_vector[0] = suff_stat[0] / pow(state->sigma, 2) / (1.0 / tau + suff_stat[2] / pow(state->sigma, 2)) + sqrt(1.0 / (1.0 / tau + suff_stat[2] / pow(state->sigma, 2))) * normal_samp(state->gen); //Rcpp::rnorm(1, 0, 1)[0];//* as_scalar(arma::randn(1,1));

    // also update probability of leaf parameters
    // prob_leaf = normal_density(theta_vector[0], suff_stat[0] / pow(state->sigma, 2) / (1.0 / tau + suff_stat[2] / pow(state->sigma, 2)), 1.0 / (1.0 / tau + suff_stat[2] / pow(state->sigma, 2)), true);

    return;
}

void NormalModel::update_state(std::unique_ptr<State> &state, size_t tree_ind, std::unique_ptr<X_struct> &x_struct)
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

void NormalModel::update_tau(std::unique_ptr<State> &state, size_t tree_ind, size_t sweeps, vector<vector<tree>> & trees){
    std::vector<tree *> leaf_nodes;
    trees[sweeps][tree_ind].getbots(leaf_nodes);
    double sum_squared = 0.0;
    for(size_t i = 0; i < leaf_nodes.size(); i ++ ){
        sum_squared = sum_squared + pow(leaf_nodes[i]->theta_vector[0], 2);
    }
    double kap = this->tau_kap;
    double s = this->tau_s * this->tau_mean;
    
    std::gamma_distribution<double> gamma_samp((leaf_nodes.size() + kap) / 2.0, 2.0 / (sum_squared + s));
    this->tau = 1.0 / gamma_samp(state->gen); 
    return;
};

void NormalModel::update_tau_per_forest(std::unique_ptr<State> &state, size_t sweeps, vector<vector<tree>> & trees){
    std::vector<tree *> leaf_nodes;
    for(size_t tree_ind = 0; tree_ind < state->num_trees; tree_ind ++){
        trees[sweeps][tree_ind].getbots(leaf_nodes);
    }
    double sum_squared = 0.0;
    for(size_t i = 0; i < leaf_nodes.size(); i ++ ){
        sum_squared = sum_squared + pow(leaf_nodes[i]->theta_vector[0], 2);
    }
    double kap = this->tau_kap;
    double s = this->tau_s * this->tau_mean;
    std::gamma_distribution<double> gamma_samp((leaf_nodes.size() + kap) / 2.0, 2.0 / (sum_squared + s));
    this->tau = 1.0 / gamma_samp(state->gen); 
    return;
}

void NormalModel::initialize_root_suffstat(std::unique_ptr<State> &state, std::vector<double> &suff_stat)
{
    // sum of y
    suff_stat[0] = sum_vec(state->residual_std[0]);
    // sum of y squared
    suff_stat[1] = sum_squared(state->residual_std[0]);
    // number of observations in the node
    suff_stat[2] = state->n_y;
    return;
}

void NormalModel::updateNodeSuffStat(std::vector<double> &suff_stat, matrix<double> &residual_std, matrix<size_t> &Xorder_std, size_t &split_var, size_t row_ind)
{
    suff_stat[0] += residual_std[0][Xorder_std[split_var][row_ind]];
    suff_stat[1] += pow(residual_std[0][Xorder_std[split_var][row_ind]], 2);
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

void NormalModel::state_sweep(size_t tree_ind, size_t M, matrix<double> &residual_std, std::unique_ptr<X_struct> &x_struct) const
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
    }
    return;
}

double NormalModel::likelihood(std::vector<double> &temp_suff_stat, std::vector<double> &suff_stat_all, size_t N_left, bool left_side, bool no_split, std::unique_ptr<State> &state) const
{
    // likelihood equation,
    // note the difference of left_side == true / false
    // node_suff_stat is mean of y, sum of square of y, saved in tree class
    // double y_sum = (double)suff_stat_all[2] * suff_stat_all[0];
    // double y_sum = suff_stat_all[0];
    double sigma2 = state->sigma2;
    // double ntau;
    // double suff_one_side;

    /////////////////////////////////////////////////////////////////////////
    //
    //  I know combining likelihood and likelihood_no_split looks nicer
    //  but this is a very fundamental function, executed many times
    //  the extra if(no_split) statement and value assignment make the code about 5% slower!!
    //
    /////////////////////////////////////////////////////////////////////////

    size_t nb;
    double nbtau;
    double y_sum;
    double y_squared_sum;

    if (no_split)
    {
        // ntau = suff_stat_all[2] * tau;
        // suff_one_side = y_sum;

        nb = suff_stat_all[2];
        nbtau = nb * tau;
        y_sum = suff_stat_all[0];
        y_squared_sum = suff_stat_all[1];
    }
    else
    {
        if (left_side)
        {
            nb = N_left + 1;
            nbtau = nb * tau;
            // ntau = (N_left + 1) * tau;
            y_sum = temp_suff_stat[0];
            y_squared_sum = temp_suff_stat[1];
            // suff_one_side = temp_suff_stat[0];
        }
        else
        {
            nb = suff_stat_all[2] - N_left - 1;
            nbtau = nb * tau;
            y_sum = suff_stat_all[0] - temp_suff_stat[0];
            y_squared_sum = suff_stat_all[1] - temp_suff_stat[1];

            // ntau = (suff_stat_all[2] - N_left - 1) * tau;
            // suff_one_side = y_sum - temp_suff_stat[0];
        }
    }

    // return 0.5 * log(sigma2) - 0.5 * log(nbtau + sigma2) + 0.5 * tau * pow(y_sum, 2) / (sigma2 * (nbtau + sigma2));

    return -0.5 * nb * log(2 * 3.141592653) - 0.5 * nb * log(sigma2) + 0.5 * log(sigma2) - 0.5 * log(nbtau + sigma2) - 0.5 * y_squared_sum / sigma2 + 0.5 * tau * pow(y_sum, 2) / (sigma2 * (nbtau + sigma2));
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

void NormalModel::ini_residual_std(std::unique_ptr<State> &state)
{
    // initialize partial residual at (num_tree - 1) / num_tree * yhat
    double value = state->ini_var_yhat * ((double)state->num_trees - 1.0) / (double)state->num_trees;
    for (size_t i = 0; i < state->residual_std[0].size(); i++)
    {
        state->residual_std[0][i] = (*state->y_std)[i] - value;
    }
    return;
}

void NormalModel::predict_std(const double *Xtestpointer, size_t N_test, size_t p, size_t num_trees, size_t num_sweeps, matrix<double> &yhats_test_xinfo, vector<vector<tree>> &trees)
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

//////////////////////////////////////////////////////////////////////////////////////
//
//
//  Logit Model
//
//
//
//////////////////////////////////////////////////////////////////////////////////////

//incSuffStat should take a state as its first argument
void LogitModel::incSuffStat(matrix<double> &residual_std, size_t index_next_obs, std::vector<double> &suffstats)
{
    suffstats[(*y_size_t)[index_next_obs]] += weight;
    for (size_t j = 0; j < dim_theta; ++j)
    {
        suffstats[dim_residual + j] += weight * exp(residual_std[j][index_next_obs]);
        // suffstats[dim_residual + j] += (*phi)[index_next_obs] * exp(residual_std[j][index_next_obs]);
    }

    return;
}

void LogitModel::samplePars(std::unique_ptr<State> &state, std::vector<double> &suff_stat, std::vector<double> &theta_vector, double &prob_leaf)
{
    for (size_t j = 0; j < dim_theta; j++)
    {
        if (isnan(suff_stat[j]))
        {
            cout <<"unidentified error: suff_stat is nan for class " << j << endl;
            exit(1);
        }
        std::gamma_distribution<double> gammadist(tau_a + suff_stat[j], 1.0); // consider adding 1 sudo obs to prevent 0 theta value
        theta_vector[j] = gammadist(state->gen) / (tau_b + suff_stat[dim_theta + j]);
    }
    return;
}

void LogitModel::update_state(std::unique_ptr<State>& state, size_t tree_ind, std::unique_ptr<X_struct>& x_struct)
{

    // Calculate logloss
    size_t y_i;
    double sum_fits;
    logloss = 0; // reset logloss
    std::gamma_distribution<double> gammadist(weight, 1.0);

    for (size_t i = 0; i < state->n_y; i++)
    {
        sum_fits = 0;
        y_i = (size_t)(*y_size_t)[i];
        for (size_t j = 0; j < dim_residual; ++j)
        {
            sum_fits += exp(state->residual_std[j][i]) * (*(x_struct->data_pointers[tree_ind][i]))[j]; // f_j(x_i) = \prod lambdas
        }
        // Sample phi
        // (*phi)[i] = gammadist(state->gen) / (1.0 * sum_fits);
        // calculate logloss
        logloss += -log(exp(state->residual_std[y_i][i]) * (*(x_struct->data_pointers[tree_ind][i]))[y_i] / sum_fits); // logloss =  - log(p_j) 
    }
    // sample weight based on logloss
    if (update_weight){
        std::gamma_distribution<> d(state->n_y, 1);
        weight = d(state->gen) / (hmult * logloss + heps * (double)state->n_y); // it's like shift p down by
    }

    // Sample tau_a
   if (update_tau)
   {
        size_t count_lambda = 0;
        double mean_lambda = 0;
        double var_lambda = 0;
        for(size_t i = 0; i < state->num_trees; i++)
        {
            for(size_t j = 0; j < state->lambdas[i].size(); j++)
            {
                mean_lambda += std::accumulate(state->lambdas[i][j].begin(), state->lambdas[i][j].end(), 0.0);
                count_lambda += dim_residual;
            }
        }
        mean_lambda = mean_lambda / count_lambda;

        for(size_t i = 0; i < state->num_trees; i++)
        {
            for(size_t j = 0; j < state->lambdas[i].size(); j++)
            {
                for(size_t k = 0; k < dim_residual; k++)
                {
                var_lambda += pow(state->lambdas[i][j][k] - mean_lambda, 2);
                }
            }
        }    
        var_lambda = var_lambda / count_lambda;
        // cout << "mean = " << mean_lambda << "; var = " << var_lambda << endl;

        std::normal_distribution<> norm(mean_lambda, sqrt(var_lambda / count_lambda));
        tau_a = 0;
        while (tau_a <= 0)
        {
            tau_a = norm(state->gen) * tau_b;
        }
    }

    return;
}

void LogitModel::initialize_root_suffstat(std::unique_ptr<State> &state, std::vector<double> &suff_stat)
{

    /*
    // sum of y
    suff_stat[0] = sum_vec(state->residual_std[0]);
    // sum of y squared
    suff_stat[1] = sum_squared(state->residual_std[0]);
    // number of observations in the node
    suff_stat[2] = state->n_y;
    */

    // JINGYU check -- should i always plan to resize this vector?
    // reply: use it for now. Not sure how to call constructor of tree when initialize vector<vector<tree>>, see definition of trees2 in XBART_multinomial, train_all.cpp

    // remove resizing it does not work, strange

    suff_stat.resize(2 * dim_theta);
    std::fill(suff_stat.begin(), suff_stat.end(), 0.0);
    for (size_t i = 0; i < state->n_y; i++)
    {
        // from 0
        incSuffStat(state->residual_std, i, suff_stat);
    }

    return;
}

void LogitModel::updateNodeSuffStat(std::vector<double> &suff_stat, matrix<double> &residual_std, matrix<size_t> &Xorder_std, size_t &split_var, size_t row_ind)
{
    /*
    suff_stat[0] += residual_std[0][Xorder_std[split_var][row_ind]];
    suff_stat[1] += pow(residual_std[0][Xorder_std[split_var][row_ind]], 2);
    suff_stat[2] += 1;
    */

    incSuffStat(residual_std, Xorder_std[split_var][row_ind], suff_stat);

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

void LogitModel::state_sweep(size_t tree_ind, size_t M, matrix<double> &residual_std, std::unique_ptr<X_struct> &x_struct) const
{

    size_t next_index = tree_ind + 1;
    if (next_index == M)
    {
        next_index = 0;
    }

    ////////////////////////////////////////////////////////
    // Be care of line 151 in train_all.cpp, initial_theta
    ////////////////////////////////////////////////////////

    // cumulative product of trees, multiply current one, divide by next one

    for (size_t i = 0; i < residual_std[0].size(); i++)
    {
        for (size_t j = 0; j < dim_theta; ++j)
        {
            // residual_std[j][i] = residual_std[j][i] * (*(x_struct->data_pointers[tree_ind][i]))[j] / (*(x_struct->data_pointers[next_index][i]))[j];
            residual_std[j][i] = residual_std[j][i] + log((*(x_struct->data_pointers[tree_ind][i]))[j]) - log( (*(x_struct->data_pointers[next_index][i]))[j] );
        }
    }

    return;
}

double LogitModel::likelihood(std::vector<double> &temp_suff_stat, std::vector<double> &suff_stat_all, size_t N_left, bool left_side, bool no_split, std::unique_ptr<State> &state) const
{
    // likelihood equation,
    // note the difference of left_side == true / false
    // node_suff_stat is mean of y, sum of square of y, saved in tree class
    // double y_sum = (double)suff_stat_all[2] * suff_stat_all[0];
    // double y_sum = suff_stat_all[0];
    // double suff_one_side;

    /////////////////////////////////////////////////////////////////////////
    //
    //  I know combining likelihood and likelihood_no_split looks nicer
    //  but this is a very fundamental function, executed many times
    //  the extra if(no_split) statement and value assignment make the code about 5% slower!!
    //
    /////////////////////////////////////////////////////////////////////////

    //could rewrite without all these local assigments if that helps...
    std::vector<double> local_suff_stat = suff_stat_all; // no split

    //COUT << "LIK" << endl;

    //COUT << "all suff stat dim " << suff_stat_all.size();

    if (!no_split)
    {
        if (left_side)
        {
            //COUT << "LEFTWARD HO" << endl;
            //COUT << "local suff stat dim " << local_suff_stat.size() << endl;
            //COUT << "temp suff stat dim " << temp_suff_stat.size() << endl;
            local_suff_stat = temp_suff_stat;
        }
        else
        {
            //COUT << "RIGHT HO" << endl;
            //COUT << "local suff stat dim " << local_suff_stat.size() << endl;
            //COUT << "temp suff stat dim " << temp_suff_stat.size() << endl;
            local_suff_stat = suff_stat_all - temp_suff_stat;

            // ntau = (suff_stat_all[2] - N_left - 1) * tau;
            // suff_one_side = y_sum - temp_suff_stat[0];
        }
    }

    return (LogitLIL(local_suff_stat));
}

void LogitModel::ini_residual_std(std::unique_ptr<State> &state)
{
    //double value = state->ini_var_yhat * ((double)state->num_trees - 1.0) / (double)state->num_trees;
    for (size_t i = 0; i < state->residual_std[0].size(); i++)
    {
        // init leaf pars are all 1, partial fits are all 1
        // save fits in log scale -> 0.0
        for (size_t j = 0; j < dim_theta; ++j)
        {
            state->residual_std[j][i] =  0.0; // save resdiual_std as log(lamdas), start at 0.0.
            // originally residual_std initialized at 1.0; // (*state->y_std)[i] - value;
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
                if(output_vec[sweeps + data_ind * num_sweeps + k * num_sweeps * N_test] > max_log_prob){
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
void LogitModel::predict_std_standalone(const double *Xtestpointer, size_t N_test, size_t p, size_t num_trees, size_t num_sweeps, matrix<double> &yhats_test_xinfo, vector<vector<tree>> &trees, std::vector<double> &output_vec, std::vector<size_t>& iteration)
{

    // output is a 3D array (armadillo cube), nsweeps by n by number of categories

    size_t num_iterations = iteration.size();

    tree::tree_p bn;

    cout << "number of iterations " << num_iterations << " " << num_sweeps << endl;

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
                if(output_vec[iter + data_ind * num_iterations + k * num_iterations * N_test] > max_log_prob){
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

//incSuffStat should take a state as its first argument
// void LogitModelSeparateTrees::incSuffStat(matrix<double> &residual_std, size_t index_next_obs, std::vector<double> &suffstats)
// {
//     suffstats[(*y_size_t)[index_next_obs]] += weight;

//     size_t j = class_operating;
//     // suffstats[dim_theta + j] += weight * exp (residual_std[j][index_next_obs]);
//     suffstats[dim_theta + j] += (*phi)[index_next_obs] * exp (residual_std[j][index_next_obs]);

//     return;
// }

void LogitModelSeparateTrees::samplePars(std::unique_ptr<State> &state, std::vector<double> &suff_stat, std::vector<double> &theta_vector, double &prob_leaf)
{
    size_t j = class_operating;

    std::gamma_distribution<double> gammadist(tau_a + suff_stat[j] + 1, 1.0); // add a sudo observation to prevent theta = 0;
    
    theta_vector[j] = gammadist(state->gen) / (tau_b + suff_stat[dim_theta + j]);
    if (theta_vector[j] == 0){
        cout << "unidentified error, theta for class " << j << " = 0" << endl;
        cout << "suff_stats = " << suff_stat[j] << ", " << suff_stat[dim_theta + j] << ", tau_a = " << tau_a << endl; 
        exit(1);
    }

    return;
}

void LogitModelSeparateTrees::update_state(std::unique_ptr<State>& state, size_t tree_ind, std::unique_ptr<X_struct>& x_struct)
{
    // Draw weight
    // Calculate logloss
    size_t y_i;
    double sum_fits;
    logloss = 0; // reset logloss
    std::gamma_distribution<double> gammadist(weight, 1.0);

    for (size_t i = 0; i < state->n_y; i++)
    {
        sum_fits = 0;
        y_i = (size_t)(*y_size_t)[i];
        for (size_t j = 0; j < dim_residual; ++j)
        {
            sum_fits += exp(state->residual_std[j][i]) * (*(x_struct->data_pointers_multinomial[j][tree_ind][i]))[j]; // f_j(x_i) = \prod lambdas
        }
        // Sample phi
        (*phi)[i] = gammadist(state->gen) / (1.0 * sum_fits);
        // cout << "phi " << i << " = " << (*phi)[i] << ", sum_fits = " << sum_fits << endl;
        // calculate logloss
        logloss += -log(exp(state->residual_std[y_i][i]) * (*(x_struct->data_pointers_multinomial[y_i][tree_ind][i]))[y_i] / sum_fits); // logloss =  - log(p_j) 
    }
    if (update_weight) 
    {
        std::gamma_distribution<> d(state->n_y, 1);
        weight = d(state->gen) / (hmult * logloss + heps * (double)state->n_y) + 1;
    }
    return;

}

// void LogitModelSeparateTrees::updateNodeSuffStat(std::vector<double> &suff_stat, matrix<double> &residual_std, matrix<size_t> &Xorder_std, size_t &split_var, size_t row_ind)
// {
//     /*
//     suff_stat[0] += residual_std[0][Xorder_std[split_var][row_ind]];
//     suff_stat[1] += pow(residual_std[0][Xorder_std[split_var][row_ind]], 2);
//     suff_stat[2] += 1;
//     */

//     incSuffStat(residual_std, Xorder_std[split_var][row_ind], suff_stat);

//     return;
// }

void LogitModelSeparateTrees::state_sweep(size_t tree_ind, size_t M, matrix<double> &residual_std, std::unique_ptr<X_struct> &x_struct) const
{

    size_t next_index = tree_ind + 1;
    if (next_index == M)
    {
        next_index = 0;
    }

    ////////////////////////////////////////////////////////
    // Be care of line 151 in train_all.cpp, initial_theta
    ////////////////////////////////////////////////////////

    // cumulative product of trees, multiply current one, divide by next one

    for (size_t i = 0; i < residual_std[0].size(); i++)
    {
        for (size_t j = 0; j < dim_theta; ++j)
        {
            // residual_std[j][i] = residual_std[j][i] * (*(x_struct->data_pointers_multinomial[j][tree_ind][i]))[j] / (*(x_struct->data_pointers_multinomial[j][next_index][i]))[j];
            residual_std[j][i] = residual_std[j][i] + log((*(x_struct->data_pointers_multinomial[j][tree_ind][i]))[j]) - log((*(x_struct->data_pointers_multinomial[j][next_index][i]))[j]);
        }
    }

    return;
}

double LogitModelSeparateTrees::likelihood(std::vector<double> &temp_suff_stat, std::vector<double> &suff_stat_all, size_t N_left, bool left_side, bool no_split, std::unique_ptr<State> &state) const
{
    // likelihood equation,
    // note the difference of left_side == true / false
    // node_suff_stat is mean of y, sum of square of y, saved in tree class
    // double y_sum = (double)suff_stat_all[2] * suff_stat_all[0];
    // double y_sum = suff_stat_all[0];
    // double suff_one_side;

    /////////////////////////////////////////////////////////////////////////
    //
    //  I know combining likelihood and likelihood_no_split looks nicer
    //  but this is a very fundamental function, executed many times
    //  the extra if(no_split) statement and value assignment make the code about 5% slower!!
    //
    /////////////////////////////////////////////////////////////////////////

    //could rewrite without all these local assigments if that helps...
    std::vector<double> local_suff_stat = suff_stat_all; // no split

    //COUT << "LIK" << endl;

    //COUT << "all suff stat dim " << suff_stat_all.size();

    if (!no_split)
    {
        if (left_side)
        {
            //COUT << "LEFTWARD HO" << endl;
            //COUT << "local suff stat dim " << local_suff_stat.size() << endl;
            //COUT << "temp suff stat dim " << temp_suff_stat.size() << endl;
            local_suff_stat = temp_suff_stat;
        }
        else
        {
            //COUT << "RIGHT HO" << endl;
            //COUT << "local suff stat dim " << local_suff_stat.size() << endl;
            //COUT << "temp suff stat dim " << temp_suff_stat.size() << endl;
            local_suff_stat = suff_stat_all - temp_suff_stat;

            // ntau = (suff_stat_all[2] - N_left - 1) * tau;
            // suff_one_side = y_sum - temp_suff_stat[0];
        }
    }

    // return 0.5 * log(sigma2) - 0.5 * log(nbtau + sigma2) + 0.5 * tau * pow(y_sum, 2) / (sigma2 * (nbtau + sigma2));

    //return - 0.5 * nb * log(2 * 3.141592653) -  0.5 * nb * log(sigma2) + 0.5 * log(sigma2) - 0.5 * log(nbtau + sigma2) - 0.5 * y_squared_sum / sigma2 + 0.5 * tau * pow(y_sum, 2) / (sigma2 * (nbtau + sigma2));

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

                    // cout << "one obs " << log(bn->theta_vector[k]) << "  "  << bn->theta_vector[k]  << endl;

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
                if(output_vec[sweeps + data_ind * num_sweeps + k * num_sweeps * N_test] > max_log_prob){
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
void LogitModelSeparateTrees::predict_std_standalone(const double *Xtestpointer, size_t N_test, size_t p, size_t num_trees, size_t num_sweeps, matrix<double> &yhats_test_xinfo,  vector<vector<vector<tree>>> &trees, std::vector<double> &output_vec, std::vector<size_t>& iteration, double weight)
{

    // output is a 3D array (armadillo cube), nsweeps by n by number of categories

    size_t num_iterations = iteration.size();

    tree::tree_p bn;

    cout << "number of iterations " << num_iterations << " " << num_sweeps << endl;

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
                if(output_vec[iter + data_ind * num_iterations + k * num_iterations * N_test] > max_log_prob){
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
                denom +=  output_vec[iter + data_ind * num_iterations + k * num_iterations * N_test] ;
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
//  Probit Model
//
//
//////////////////////////////////////////////////////////////////////////////////////
void ProbitClass::update_state(std::unique_ptr<State> &state, size_t tree_ind, std::unique_ptr<X_struct> &x_struct)
{
    // Draw Sigma
    // state->residual_std_full = state->residual_std - state->predictions_std[tree_ind];

    // residual_std is only 1 dimensional for regression model

    // std::vector<double> full_residual(state->n_y);

    // for (size_t i = 0; i < state->residual_std[0].size(); i++)
    // {
    //     full_residual[i] = state->residual_std[0][i] - (*(x_struct->data_pointers[tree_ind][i]))[0];
    // }

    // For probit model, do not need to sample gamma
    // std::gamma_distribution<double> gamma_samp((state->n_y + kap) / 2.0, 2.0 / (sum_squared(full_residual) + s));
    // state->update_sigma(1.0 / sqrt(gamma_samp(state->gen)));

    //update latent variable Z

    z_prev = z;

    double mu_temp;
    double u;

    for (size_t i = 0; i < state->n_y; i++)
    {
        a = 0;
        b = 1;

        mu_temp = normCDF(z_prev[i]);

        // Draw from truncated normal via inverse CDF methods
        if ((*state->y_std)[i] > 0)
        {
            a = std::min(mu_temp, 0.999);
        }
        else
        {
            b = std::max(mu_temp, 0.001);
        }

        std::uniform_real_distribution<double> unif(a, b);
        u = unif(state->gen);
        z[i] = normCDFInv(u) + mu_temp;
    }
    return;

    //NormalModel::update_state(state, tree_ind, x_struct);
}

void ProbitClass::state_sweep(size_t tree_ind, size_t M, matrix<double> &residual_std, std::unique_ptr<X_struct> &x_struct) const
{

    NormalModel::state_sweep(tree_ind, M, residual_std, x_struct);
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
        residual_std[0][i] = residual_std[0][i] - z_prev[i] + z[i];
    }

    return;
}

//////////////////////////////////////////////////////////////////////////////////////
//
//
//  CLT Model
//
//
//////////////////////////////////////////////////////////////////////////////////////
