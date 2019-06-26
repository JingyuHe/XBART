
#ifndef model_h
#define model_h

#include "common.h"
#include "utility.h"
#include <memory>
#include "state.h"

using namespace std;

class Model
{

  protected:
    size_t dim_theta;
    size_t dim_suffstat;
    size_t dim_suffstat_total;
    std::vector<double> suff_stat_model;
    std::vector<double> suff_stat_total;
    double no_split_penality;

  public:
    Model(size_t dim_theta, size_t dim_suff)
    {
        this->dim_theta = dim_theta;
        this->dim_suffstat = dim_suff;
        Model::suff_stat_model.resize(dim_suff);
    };

    // Abstract functions
    virtual void incrementSuffStat() const { return; };
    virtual void samplePars(double y_mean, size_t N_Xorder, double sigma, double tau,
                            std::mt19937 &generator, std::vector<double> &theta_vector, std::vector<double> &y_std, xinfo_sizet &Xorder) { return; };
    virtual void state_sweep(const xinfo &predictions_std, size_t tree_ind, size_t M,
                                std::vector<double> &residual_std) const { return; };
    virtual void calcSuffStat_categorical(std::vector<double> &y, xinfo_sizet &Xorder, size_t &start, size_t &end, const size_t &var) { return; };
    virtual void calcSuffStat_continuous(std::vector<size_t> &xorder, std::vector<double> &y_std, std::vector<size_t> &candidate_index, size_t index, bool adaptive_cutpoint) { return; };
    virtual double likelihood(double tau, double ntau, double sigma2, double y_sum, bool left_side) const { return 0.0; };
    virtual double likelihood_no_split(double value, double tau, double ntau, double sigma2) const { return 0.0; };
    virtual void suff_stat_fill(std::vector<double> &y_std, std::vector<size_t> &xorder) { return; };
    virtual Model *clone() { return nullptr; };

    // Getters and Setters
    // num classes
    size_t getNumClasses() const { return dim_theta; };
    void setNumClasses(size_t n_class) { dim_theta = n_class; };
    // dim suff stat
    size_t getDimSuffstat() const { return dim_suffstat; };
    void setDimSuffStat(size_t dim_suff) { dim_suffstat = dim_suff; };
    // suff stat
    std::vector<double> getSuffstat() const { return this->suff_stat_model; };
    //penality
    double getNoSplitPenality()
    {
        return no_split_penality;
        ;
    };
    void setNoSplitPenality(double pen) { this->no_split_penality = pen; };

    // Other Pre-Defined functions
    void suff_stat_fill_zero()
    {
        std::fill(Model::suff_stat_model.begin(), Model::suff_stat_model.end(), 0);
        return;
    };
    void printSuffstat() const
    {
        cout << this->suff_stat_model << endl;
        return;
    };
};

class NormalModel : public Model
{
  private:
    size_t dim_suffstat_total = 1;
    std::vector<double> suff_stat_total;

  public:
    NormalModel() : Model(1, 1)
    {
    }

    void suff_stat_fill(std::vector<double> &y_std, std::vector<size_t> &xorder)
    {
        // fill the suff_stat_model with a value
        std::fill(Model::suff_stat_model.begin(), Model::suff_stat_model.end(), y_std[xorder[0]]);
        return;
    }
    void incrementSuffStat() const { return; };
    void samplePars(double y_mean, size_t N_Xorder, double sigma, double tau,
                    std::mt19937 &generator, std::vector<double> &theta_vector, std::vector<double> &y_std, xinfo_sizet &Xorder)
    {
        std::normal_distribution<double> normal_samp(0.0, 1.0);
            // test result should be theta
            theta_vector[0] = y_mean * N_Xorder / pow(sigma, 2) / (1.0 / tau + N_Xorder / pow(sigma, 2)) + sqrt(1.0 / (1.0 / tau + N_Xorder / pow(sigma, 2))) * normal_samp(generator); //Rcpp::rnorm(1, 0, 1)[0];//* as_scalar(arma::randn(1,1));

        return;
    }

    void state_sweep(const xinfo &predictions_std, size_t tree_ind, size_t M, std::vector<double> &residual_std) const
    {
        size_t next_index = tree_ind + 1;
        if (next_index == M)
        {
            next_index = 0;
        }
        residual_std = residual_std - predictions_std[tree_ind] + predictions_std[next_index];
        return;
    }

    void calcSuffStat_categorical(std::vector<double> &y, xinfo_sizet &Xorder, size_t &start, size_t &end, const size_t &var)
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

    void calcSuffStat_continuous(std::vector<size_t> &xorder, std::vector<double> &y_std, std::vector<size_t> &candidate_index, size_t index, bool adaptive_cutpoint)
    {
        // calculate sufficient statistics for continuous variables


        if (adaptive_cutpoint)
        {

            if(index == 0)
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

    double likelihood(double tau, double ntau, double sigma2, double y_sum, bool left_side) const
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

    double likelihood_no_split(double value, double tau, double ntau, double sigma2) const
    {
        // the likelihood of no-split option is a bit different from others
        // because the sufficient statistics is y_sum here
        // write a separate function, more flexibility

        return 0.5 * log(sigma2) - 0.5 * log(ntau + sigma2) + 0.5 * tau * pow(value, 2) / (sigma2 * (ntau + sigma2));
    }

    Model *clone() { return new NormalModel(*this); }
};

class CLTClass : public Model
{
  private:
    size_t dim_suffstat_total = 4;
    //std::vector<double> suff_stat_total;

  public:
    CLTClass() : Model(1, 4)
    {
        suff_stat_total.resize(dim_suffstat_total);
    }
    std::vector<double> total_fit; // Keep public to save copies
    std::vector<double> suff_stat_total;

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
    void incrementSuffStat() const { return; };
    void samplePars(double y_mean, size_t N_Xorder, double sigma, double tau,
                    std::mt19937 &generator, std::vector<double> &theta_vector, std::vector<double> &y_std, xinfo_sizet &Xorder)
    {
        // Update params
        updateFullSuffStat(y_std, Xorder[0]);

        std::normal_distribution<double> normal_samp(0.0, 1.0);

            // test result should be theta
            theta_vector[0] = suff_stat_total[0] / (1.0 / tau + suff_stat_total[1]) + sqrt(1.0 / (1.0 / tau + suff_stat_total[1])) * normal_samp(generator); //Rcpp::rnorm(1, 0, 1)[0];//* as_scalar(arma::randn(1,1));

        return;
    }

    void state_sweep(const xinfo &predictions_std, size_t tree_ind, size_t M, std::vector<double> &residual_std) const
    {
        size_t next_index = tree_ind + 1;
        if (next_index == M)
        {
            next_index = 0;
        }
        residual_std = residual_std - predictions_std[tree_ind] + predictions_std[next_index];
        return;
    }

    void calcSuffStat_categorical(std::vector<double> &y, xinfo_sizet &Xorder, size_t &start, size_t &end, const size_t &var)
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

    void calcSuffStat_continuous(std::vector<size_t> &xorder, std::vector<double> &y_std, std::vector<size_t> &candidate_index, size_t index, bool adaptive_cutpoint)
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

    void updateFullSuffStat(std::vector<double> &y_std, std::vector<size_t> &x_info)
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

    double likelihood(double tau, double ntau, double sigma2, double y_sum, bool left_side) const
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
            return 0.5 * (suff_stat_total[2] - Model::suff_stat_model[2]) + 0.5 * std::log((1 / tau) / ((1 / tau) + (suff_stat_total[1] - Model::suff_stat_model[1]))) + 0.5 * tau / (1 + tau * (suff_stat_total[1] - Model::suff_stat_model[1])) * pow(suff_stat_total[0] - Model::suff_stat_model[0], 2) ;// - 0.5 * (suff_stat_total[3] - Model::suff_stat_model[3]);
        }
    }

    double likelihood_no_split(double value, double tau, double ntau, double sigma2) const
    {
        // the likelihood of no-split option is a bit different from others
        // because the sufficient statistics is y_sum here
        // write a separate function, more flexibility

        return 0.5 * (suff_stat_total[2]) + 0.5 * std::log((1 / tau) / ((1 / tau) + (suff_stat_total[1]))) + 0.5 * tau / (1 + tau * (suff_stat_total[1])) * pow(suff_stat_total[0], 2) - 0.5 * suff_stat_total[3];
        ;
    }
    Model *clone() { return new CLTClass(*this); }
};

#endif
