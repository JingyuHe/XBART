#include "model.h"

//////////////////////////////////////////////////////////////////////////////////////
//
//
//  Normal model regression
//
//
//////////////////////////////////////////////////////////////////////////////////////

void NormalModel::suff_stat_fill(std::vector<double> &y_std, std::vector<size_t> &xorder)
{
    // fill the suff_stat_model with a value
    std::fill(Model::suff_stat_model.begin(), Model::suff_stat_model.end(), y_std[xorder[0]]);
    return;
}

void NormalModel::incrementSuffStat() const { return; };

void NormalModel::samplePars(bool draw_mu, double y_mean, size_t N_Xorder, double sigma, double tau,
                             std::mt19937 &generator, std::vector<double> &theta_vector, std::vector<double> &y_std, xinfo_sizet &Xorder)
{
    std::normal_distribution<double> normal_samp(0.0, 1.0);
    if (draw_mu == true)
    {

        // test result should be theta
        theta_vector[0] = y_mean * N_Xorder / pow(sigma, 2) / (1.0 / tau + N_Xorder / pow(sigma, 2)) + sqrt(1.0 / (1.0 / tau + N_Xorder / pow(sigma, 2))) * normal_samp(generator); //Rcpp::rnorm(1, 0, 1)[0];//* as_scalar(arma::randn(1,1));
    }
    else
    {
        // test result should be theta
        theta_vector[0] = y_mean * N_Xorder / pow(sigma, 2) / (1.0 / tau + N_Xorder / pow(sigma, 2));
    }
    return;
}

void NormalModel::updateResidual(const xinfo &predictions_std, size_t tree_ind, size_t M, std::vector<double> &residual_std) const
{
    size_t next_index = tree_ind + 1;
    if (next_index == M)
    {
        next_index = 0;
    }
    residual_std = residual_std - predictions_std[tree_ind] + predictions_std[next_index];
    return;
}

void NormalModel::calcSuffStat_categorical(std::vector<double> &y, xinfo_sizet &Xorder, size_t &start, size_t &end, const size_t &var)
{
    // calculate sufficient statistics for categorical variables

    // compute sum of y[Xorder[start:end, var]]
    size_t loop_count = 0;
    for (size_t i = start; i <= end; i++)
    {
        Model::suff_stat_model[0] += y[Xorder[var][i]];
        loop_count++;
    }
    return;
}

void NormalModel::calcSuffStat_continuous(std::vector<size_t> &xorder, std::vector<double> &y_std, std::vector<size_t> &candidate_index, size_t index, bool adaptive_cutpoint)
{
    // calculate sufficient statistics for continuous variables

    if (adaptive_cutpoint)
    {

        if (index == 0)
        {
            // initialize, only for the first cutpoint candidate, thus index == 0
            Model::suff_stat_model[0] = y_std[xorder[0]];
        }

        // if use adaptive number of cutpoints, calculated based on vector candidate_index
        for (size_t q = candidate_index[index] + 1; q <= candidate_index[index + 1]; q++)
        {
            Model::suff_stat_model[0] += y_std[xorder[q]];
        }
    }
    else
    {
        // use all data points as candidates
        Model::suff_stat_model[0] += y_std[xorder[index]];
    }
    return;
}

double NormalModel::likelihood(double tau, double ntau, double sigma2, double y_sum, bool left_side) const
{
    // likelihood equation,
    // note the difference of left_side == true / false

    if (left_side)
    {
        return 0.5 * log(sigma2) - 0.5 * log(ntau + sigma2) + 0.5 * tau * pow(Model::suff_stat_model[0], 2) / (sigma2 * (ntau + sigma2));
    }
    else
    {
        return 0.5 * log(sigma2) - 0.5 * log(ntau + sigma2) + 0.5 * tau * pow(y_sum - Model::suff_stat_model[0], 2) / (sigma2 * (ntau + sigma2));
    }
}

double NormalModel::likelihood_no_split(double value, double tau, double ntau, double sigma2) const
{
    // the likelihood of no-split option is a bit different from others
    // because the sufficient statistics is y_sum here
    // write a separate function, more flexibility

    return 0.5 * log(sigma2) - 0.5 * log(ntau + sigma2) + 0.5 * tau * pow(value, 2) / (sigma2 * (ntau + sigma2));
}

//////////////////////////////////////////////////////////////////////////////////////
//
//
//  CLT approximation
//
//
//////////////////////////////////////////////////////////////////////////////////////

void CLTClass::suff_stat_fill(std::vector<double> &y_std, std::vector<size_t> &xorder)
{
    // fill the suff_stat_model with a value
    // in function call, a = 0.0 to reset sufficient statistics vector
    size_t n = xorder.size();
    size_t x_order_0 = xorder[0];
    double current_fit_val = total_fit[x_order_0];

    double psi = max(current_fit_val * (1 - current_fit_val), 0.15);
    Model::suff_stat_model[0] = y_std[x_order_0] / psi;
    Model::suff_stat_model[1] = 1 / psi;
    Model::suff_stat_model[2] = std::log(1 / psi);
    // Model::suff_stat_model[3] = pow(y_std[x_order_0], 2) / psi;
    return;
}

void CLTClass::incrementSuffStat() const { return; };

void CLTClass::samplePars(bool draw_mu, double y_mean, size_t N_Xorder, double sigma, double tau,
                          std::mt19937 &generator, std::vector<double> &theta_vector, std::vector<double> &y_std, xinfo_sizet &Xorder)
{
    // Update params
    updateFullSuffStat(y_std, Xorder[0]);

    std::normal_distribution<double> normal_samp(0.0, 1.0);
    if (draw_mu == true)
    {

        // test result should be theta
        theta_vector[0] = suff_stat_total[0] / (1.0 / tau + suff_stat_total[1]) + sqrt(1.0 / (1.0 / tau + suff_stat_total[1])) * normal_samp(generator); //Rcpp::rnorm(1, 0, 1)[0];//* as_scalar(arma::randn(1,1));
    }
    else
    {
        // test result should be theta
        theta_vector[0] = suff_stat_total[0] / (1.0 / tau + suff_stat_total[1]);
    }
    return;
}

void CLTClass::updateResidual(const xinfo &predictions_std, size_t tree_ind, size_t M, std::vector<double> &residual_std) const
{
    size_t next_index = tree_ind + 1;
    if (next_index == M)
    {
        next_index = 0;
    }
    residual_std = residual_std - predictions_std[tree_ind] + predictions_std[next_index];
    return;
}

void CLTClass::calcSuffStat_categorical(std::vector<double> &y, xinfo_sizet &Xorder, size_t &start, size_t &end, const size_t &var)
{
    // calculate sufficient statistics for categorical variables

    // compute sum of y[Xorder[start:end, var]]
    size_t loop_count = 0;
    std::vector<size_t> &xorder_var = Xorder[var];
    size_t n = xorder_var.size();
    double current_fit_val;
    double psi;
    double obs;
    size_t x_order_i;
    for (size_t i = start; i <= end; i++)
    {
        x_order_i = xorder_var[i];
        current_fit_val = total_fit[x_order_i];
        obs = y[x_order_i];

        psi = std::max(current_fit_val * (1 - current_fit_val), 0.15);
        Model::suff_stat_model[0] += obs / psi;
        Model::suff_stat_model[1] += 1 / psi;
        Model::suff_stat_model[2] += std::log(1 / psi);
        //Model::suff_stat_model[3] += pow(obs, 2) / psi;
        loop_count++;
    }
    return;
}

void CLTClass::calcSuffStat_continuous(std::vector<size_t> &xorder, std::vector<double> &y_std, std::vector<size_t> &candidate_index, size_t index, bool adaptive_cutpoint)
{
    // calculate sufficient statistics for continuous variables
    size_t n = xorder.size();
    double current_fit_val;
    double psi;
    double obs;
    size_t x_order_q;

    if (adaptive_cutpoint)
    {
        // initialize
        Model::suff_stat_model[0] = y_std[xorder[0]];

        // if use adaptive number of cutpoints, calculated based on vector candidate_index
        for (size_t q = candidate_index[index] + 1; q <= candidate_index[index + 1]; q++)
        {
            x_order_q = xorder[q];
            current_fit_val = total_fit[x_order_q];
            obs = y_std[x_order_q];

            //if (current_fit_val > 1.0 || current_fit_val < -1.0){obs = 0.0;}

            psi = std::max(current_fit_val * (1 - current_fit_val), 0.15);
            //psi = 0.15;
            Model::suff_stat_model[0] += obs / psi;
            Model::suff_stat_model[1] += 1 / psi;
            Model::suff_stat_model[2] += std::log(1 / psi);
            //Model::suff_stat_model[3] += pow(obs, 2) / psi;
        }
    }
    else
    {
        // use all data points as candidates
        current_fit_val = total_fit[xorder[index]];
        obs = y_std[xorder[index]];

        psi = std::max(current_fit_val * (1 - current_fit_val), 0.15);
        Model::suff_stat_model[0] += obs / psi;
        Model::suff_stat_model[1] += 1 / psi;
        Model::suff_stat_model[2] += std::log(1 / psi);
        //Model::suff_stat_model[3] += pow(obs, 2) / psi;
    }

    return;
}

void CLTClass::updateFullSuffStat(std::vector<double> &y_std, std::vector<size_t> &x_info)
{
    size_t n = x_info.size();
    double current_fit_val;
    double psi;
    double obs;
    size_t x_order_i;
    for (size_t i = 0; i < n; i++)
    {
        x_order_i = x_info[i];
        current_fit_val = total_fit[x_order_i];
        obs = y_std[x_order_i];

        psi = std::max(current_fit_val * (1 - current_fit_val), 0.15);
        suff_stat_total[0] += obs / psi;
        suff_stat_total[1] += 1 / psi;
        suff_stat_total[2] += std::log(1 / psi);
        //suff_stat_total[3] += pow(obs, 2) / psi;
    }
    return;
}

double CLTClass::likelihood(double tau, double ntau, double sigma2, double y_sum, bool left_side) const
{
    // likelihood equation,
    // note the difference of left_side == true / false

    if (left_side)
    {
        return 0.5 * Model::suff_stat_model[2] + 0.5 * std::log((1 / tau) / ((1 / tau) + Model::suff_stat_model[1])) + 0.5 * tau / (1 + tau * Model::suff_stat_model[1]) * pow(Model::suff_stat_model[0], 2); //- 0.5 * Model::suff_stat_model[3];
        ;
    }
    else
    {
        return 0.5 * (suff_stat_total[2] - Model::suff_stat_model[2]) + 0.5 * std::log((1 / tau) / ((1 / tau) + (suff_stat_total[1] - Model::suff_stat_model[1]))) + 0.5 * tau / (1 + tau * (suff_stat_total[1] - Model::suff_stat_model[1])) * pow(suff_stat_total[0] - Model::suff_stat_model[0], 2); // - 0.5 * (suff_stat_total[3] - Model::suff_stat_model[3]);
    }
}

double CLTClass::likelihood_no_split(double value, double tau, double ntau, double sigma2) const
{
    // the likelihood of no-split option is a bit different from others
    // because the sufficient statistics is y_sum here
    // write a separate function, more flexibility

    return 0.5 * (suff_stat_total[2]) + 0.5 * std::log((1 / tau) / ((1 / tau) + (suff_stat_total[1]))) + 0.5 * tau / (1 + tau * (suff_stat_total[1])) * pow(suff_stat_total[0], 2) - 0.5 * suff_stat_total[3];
    ;
}

//////////////////////////////////////////////////////////////////////////////////////
//
//
//  FitInfo
//
//
//////////////////////////////////////////////////////////////////////////////////////
void FitInfo::init_tree_pointers(std::vector<double> *initial_theta, size_t N, size_t num_trees)
{
    ini_matrix(data_pointers, N, num_trees);
    for (size_t i = 0; i < num_trees; i++)
    {
        std::vector<std::vector<double> *> &pointer_vec = data_pointers[i];
        for (size_t j = 0; j < N; j++)
        {
            pointer_vec[j] = initial_theta;
        }
    }
}

FitInfo::FitInfo(const double *Xpointer, xinfo_sizet &Xorder_std, size_t N, size_t p,
                 size_t num_trees, size_t p_categorical, size_t p_continuous,
                 bool set_random_seed, size_t random_seed, std::vector<double> *initial_theta)
{

    // Handle Categorical
    if (p_categorical > 0)
    {
        this->categorical_variables = true;
    }
    this->variable_ind = std::vector<size_t>(p_categorical + 1);
    this->X_num_unique = std::vector<size_t>(p_categorical);
    unique_value_count2(Xpointer, Xorder_std, this->X_values, this->X_counts,
                        this->variable_ind, this->total_points, this->X_num_unique, p_categorical, p_continuous);

    // // Init containers
    ini_xinfo(this->predictions_std, N, num_trees);

    yhat_std = std::vector<double>(N);
    row_sum(this->predictions_std, this->yhat_std);
    this->residual_std = std::vector<double>(N);

    // Random
    this->prob = std::vector<double>(2, 0.5);
    this->gen = std::mt19937(rd());
    if (set_random_seed)
    {
        gen.seed(random_seed);
    }
    this->d = std::discrete_distribution<>(prob.begin(), prob.end());

    // Splits
    ini_xinfo(this->split_count_all_tree, p, num_trees);
    this->split_count_current_tree = std::vector<double>(p, 0);
    this->mtry_weight_current_tree = std::vector<double>(p, 0.1);

    init_tree_pointers(initial_theta, N, num_trees);
}

//////////////////////////////////////////////////////////////////////////////////////
//
//
//  Multinomial logistic model
//
//
//////////////////////////////////////////////////////////////////////////////////////
double LogitLIL(const vector<double> &suffstats, const double &tau_a, const double &tau_b)
{

    size_t c = suffstats.size() / 2;

    //suffstats[0] .. suffstats[c-1]is count of y's in cat 0,...,c-1, i.e. r in proposal
    //suffstats[c] .. suffstats[2c-1] is sum of phi_i*(partial fit j)'s ie s in proposal

    double ret = 0;
    for (size_t j = 0; j < c; j++)
    {
        double r = suffstats[j];
        double s = suffstats[c + j];
        ret += (tau_a + r) * log(tau_b + s) - lgamma(tau_a + r);
    }

    return ret;
}

vector<double> LogitSamplePars(vector<double> &suffstats, double &tau_a, double &tau_b, std::mt19937 &generator)
{
    //redefine these to use prior pars from Model class
    int c = suffstats.size() / 2;
    vector<double> ret(c);
    for (int j = 0; j < c; j++)
    {
        double r = suffstats[j];
        double s = suffstats[c + j];

        std::gamma_distribution<double> gammadist(tau_a + r, 1);

        ret[j] = gammadist(generator) / (tau_b + s);
    }
    return ret;
}

// Initialize the sufficient stat vector to the sufficient stat for the first obs
// when sorting by xorder
//no longer necessary
/*
    void suff_stat_fill(std::vector<double> &y_std, std::vector<size_t> &xorder)
    {
        // fill the suff_stat_model with a value
        // in function call, a = 0.0 to reset sufficient statistics vector

        size_t n = xorder.size();
        size_t x_order_0 = xorder[0];
        double current_fit_val = total_fit[x_order_0];

        double psi = max(current_fit_val * (1 - current_fit_val), 0.15);
        Model::suff_stat_model[0] = y_std[x_order_0] / psi;
        Model::suff_stat_model[1] = 1 / psi;
        Model::suff_stat_model[2] = std::log(1 / psi);
       // Model::suff_stat_model[3] = pow(y_std[x_order_0], 2) / psi;
        return;
    }
    */

// this function should ultimately take a FitInfo and DataInfo
void LogitClass::incSuffStat(std::vector<double> &y_std, size_t ix, std::vector<double> &suffstats) const
{

    for (size_t j = 0; j < num_classes; ++j)
    {
        if (abs(y_std[ix] - j) < 0.1)
            suffstats[j] += 1; //is it important that y_std be doubles?
        suffstats[num_classes + j] += (*phi)[ix] * (*slop)[ix][j];
    }

    return;
};

// This function call can be much simplified too --  should only require (maybe) theta plus a draw_mu flag?
void LogitClass::samplePars(bool draw_mu, double y_mean, size_t N_Xorder, double sigma, double tau,
                            std::mt19937 &generator, std::vector<double> &theta_vector, std::vector<double> &y_std,
                            xinfo_sizet &Xorder)
{
    // Update params
    updateFullSuffStat(y_std, Xorder[0]);

    theta_vector = LogitSamplePars(suff_stat_total, tau_a, tau_b, generator);

    /*
        std::normal_distribution<double> normal_samp(0.0, 1.0);
        if (draw_mu == true)
        {

            // test result should be theta
            theta_vector[0] = suff_stat_total[0] / (1.0 / tau + suff_stat_total[1]) + sqrt(1.0 / (1.0 / tau + suff_stat_total[1])) * normal_samp(generator); //Rcpp::rnorm(1, 0, 1)[0];// as_scalar(arma::randn(1,1));
        }
        else
        {
            // test result should be theta
            theta_vector[0] = suff_stat_total[0] / (1.0 / tau + suff_stat_total[1]);
        }
*/

    return;
}

/*

    void updateResidual(const xinfo &predictions_std, size_t tree_ind, size_t M, std::vector<double> &residual_std) const
    {
        size_t next_index = tree_ind + 1;
        if (next_index == M)
        {
            next_index = 0;
        }
        residual_std = residual_std - predictions_std[tree_ind] + predictions_std[next_index];
        return;
    }
    */

//    void updateResidualNew(size_t tree_ind, size_t M, std::unique_ptr<FitInfo> fit_info, std::vector<std::vector<double> > &slop) {
void LogitClass::updateResidual(const xinfo &predictions_std, size_t tree_ind, size_t M, std::vector<double> &residual_std) const
{
    size_t next_index = tree_ind + 1;
    if (next_index == M)
    {
        next_index = 0;
    }

    //slop is the partial fit
    for (size_t i = 0; i < slop->size(); ++i)
    {
        for (size_t j = 0; j < (*slop)[0].size(); ++j)
        {
            //output[i] = data_pointers[M][i]->theta_vector[0];
            (*slop)[i][j] *= (*fit_info->data_pointers[tree_ind][i])[j] / (*fit_info->data_pointers[next_index][i])[j];
        }
    }
}

// Once their function calls are standardized to take a FitInfo we should never have to redefine these
// in another model class
void LogitClass::calcSuffStat_categorical(std::vector<double> &y, xinfo_sizet &Xorder, size_t &start, size_t &end, const size_t &var)
{
    // calculate sufficient statistics for categorical variables

    // compute sum of y[Xorder[start:end, var]]
    for (size_t i = start; i <= end; i++)
        incSuffStat(y, Xorder[var][i], suff_stat_model);
}

void LogitClass::calcSuffStat_continuous(std::vector<size_t> &xorder, std::vector<double> &y_std,
                                         std::vector<size_t> &candidate_index, size_t index, bool adaptive_cutpoint)
{
    // calculate sufficient statistics for continuous variables
    if (adaptive_cutpoint)
    {

        // if use adaptive number of cutpoints, calculated based on vector candidate_index
        for (size_t q = candidate_index[index] + 1; q <= candidate_index[index + 1]; q++)
        {
            incSuffStat(y_std, xorder[q], suff_stat_model);
        }
    }
    else
    {
        incSuffStat(y_std, xorder[index], suff_stat_model);
    }

    return;
}

void LogitClass::updateFullSuffStat(std::vector<double> &y_std, std::vector<size_t> &x_info)
{
    for (size_t i = 0; i < x_info.size(); i++)
        incSuffStat(y_std, x_info[i], suff_stat_total);
    return;
}

double LogitClass::LIL(const std::vector<double> &suffstats) const
{
    LogitLIL(suffstats, tau_a, tau_b);
}

//this function should call a base LIL() member function that should be redefined in
double LogitClass::likelihood(double tau, double ntau, double sigma2, double y_sum, bool left_side) const
{
    // likelihood equation,
    // note the difference of left_side == true / false

    if (left_side)
    {
        return LIL(suff_stat_model);
        //return 0.5 * Model::suff_stat_model[2] + 0.5 * std::log((1 / tau) / ((1 / tau) + Model::suff_stat_model[1])) + 0.5 * tau / (1 + tau * Model::suff_stat_model[1]) * pow(Model::suff_stat_model[0], 2); //- 0.5 * Model::suff_stat_model[3];
        ;
    }
    else
    {
        return LIL(suff_stat_total - suff_stat_model);
        //return 0.5 * (suff_stat_total[2] - Model::suff_stat_model[2]) + 0.5 * std::log((1 / tau) / ((1 / tau) + (suff_stat_total[1] - Model::suff_stat_model[1]))) + 0.5 * tau / (1 + tau * (suff_stat_total[1] - Model::suff_stat_model[1])) * pow(suff_stat_total[0] - Model::suff_stat_model[0], 2) ;// - 0.5 * (suff_stat_total[3] - Model::suff_stat_model[3]);
    }
}

double LogitClass::likelihood_no_split(double value, double tau, double ntau, double sigma2) const
{
    // the likelihood of no-split option is a bit different from others
    // because the sufficient statistics is y_sum here
    // write a separate function, more flexibility

    return LIL(suff_stat_total);

    ;
}