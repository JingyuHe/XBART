
#ifndef model_h
#define model_h


#include "common.h"
#include "utility.h"
#include <memory>
#include "fit_info.h"

using namespace std;

class Model
{

  protected:
    size_t num_classes;
    size_t dim_suffstat;
    size_t dim_suffstat_total;
    std::vector<double> suff_stat_model;
    std::vector<double> suff_stat_total;
    double no_split_penality;

  public:
    Model(size_t num_classes, size_t dim_suff)
    {
        this->num_classes = num_classes;
        this->dim_suffstat = dim_suff;
        Model::suff_stat_model.resize(dim_suff);
    };

    // Abstract functions
    virtual void incrementSuffStat() const { return; };
    virtual void samplePars(bool draw_mu, double y_mean, size_t N_Xorder, double sigma, double tau,
                            std::mt19937 &generator, std::vector<double> &theta_vector, std::vector<double> &y_std, xinfo_sizet &Xorder) { return; };
    virtual void updateResidual(const xinfo &predictions_std, size_t tree_ind, size_t M,
                                std::vector<double> &residual_std) const { return; };
    virtual void calcSuffStat_categorical(std::vector<double> &y, xinfo_sizet &Xorder, size_t &start, size_t &end, const size_t &var) { return; };
    virtual void calcSuffStat_continuous(std::vector<size_t> &xorder, std::vector<double> &y_std, std::vector<size_t> &candidate_index, size_t index, bool adaptive_cutpoint) { return; };
    virtual double likelihood(double tau, double ntau, double sigma2, double y_sum, bool left_side) const { return 0.0; };
    virtual double likelihood_no_split(double value, double tau, double ntau, double sigma2) const { return 0.0; };
    virtual void suff_stat_fill(std::vector<double> &y_std, std::vector<size_t> &xorder) { return; };
    virtual Model *clone() { return nullptr; };

    // Getters and Setters
    // num classes
    size_t getNumClasses() const { return num_classes; };
    void setNumClasses(size_t n_class) { num_classes = n_class; };
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
    void samplePars(bool draw_mu, double y_mean, size_t N_Xorder, double sigma, double tau,
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
    void samplePars(bool draw_mu, double y_mean, size_t N_Xorder, double sigma, double tau,
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


//////////////////////////////////////////////////////////////////////////////////////
//
//
//  Multinomial logistic model 
//
//
//////////////////////////////////////////////////////////////////////////////////////

////[[Rcpp::export]]
double LogitLIL(const vector<double> &suffstats, const double &tau_a, const double &tau_b) {
  
  size_t c = suffstats.size()/2;
  
  //suffstats[0] .. suffstats[c-1]is count of y's in cat 0,...,c-1, i.e. r in proposal
  //suffstats[c] .. suffstats[2c-1] is sum of phi_i*(partial fit j)'s ie s in proposal

  double ret = 0;
  for(size_t j=0; j<c; j++) {
    double r = suffstats[j];
    double s = suffstats[c+j];
    ret += (tau_a + r)*log(tau_b + s) - lgamma(tau_a + r);
  }
  
  return ret;
}

////[[Rcpp::export]]
vector<double> LogitSamplePars(vector<double> &suffstats,  double &tau_a, double &tau_b, std::mt19937 &generator) {
//redefine these to use prior pars from Model class
  int c = suffstats.size()/2;
  vector<double> ret(c);
  for(int j=0; j<c; j++) {
    double r = suffstats[j];
    double s = suffstats[c+j];

    std::gamma_distribution<double> gammadist(tau_a + r, 1);

    ret[j] = gammadist(generator) / (tau_b + s);
  }
  return ret;
}

class LogitClass : public Model
{
  private:
    size_t dim_suffstat_total = 0; // = 2*num_classes;
    //std::vector<double> suff_stat_total;

  public:

    //This is probably unsafe/stupid but a temporary hack #yolo
    FitInfo* fit_info;
    //these should be elements of a class derived from a FitInfo base class for this model
    std::vector<std::vector<double> >* slop;
    std::vector<double>* phi;
    double tau_a = 3.3; //approx 4/sqrt(2) + 0.5
    double tau_b = 2.8;

    LogitClass() : Model(2, 4)
    {
        dim_suffstat_total = 2*num_classes; //num_classes is a member of base Model class
        suff_stat_total.resize(dim_suffstat_total); //suff_stat_total stuff should live in base class
    }

    LogitClass(size_t num_classes) : Model(num_classes, 2*num_classes)
    {
        dim_suffstat_total = 2*num_classes;
        suff_stat_total.resize(dim_suffstat_total);
    }
    //std::vector<double> total_fit; // Keep public to save copies
    std::vector<double> suff_stat_total;

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
    void incSuffStat(std::vector<double> &y_std, size_t ix, std::vector<double> &suffstats) const {
        
        for(size_t j=0; j<num_classes; ++j) {
            if(abs(y_std[ix] - j)<0.1) suffstats[j] += 1; //is it important that y_std be doubles?
            suffstats[num_classes + j] += (*phi)[ix]*(*slop)[ix][j];
        }

        return; 
    };

    // This function call can be much simplified too --  should only require (maybe) theta plus a draw_mu flag?
    void samplePars(bool draw_mu, double y_mean, size_t N_Xorder, double sigma, double tau,
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
            theta_vector[0] = suff_stat_total[0] / (1.0 / tau + suff_stat_total[1]) + sqrt(1.0 / (1.0 / tau + suff_stat_total[1])) * normal_samp(generator); //Rcpp::rnorm(1, 0, 1)[0];//* as_scalar(arma::randn(1,1));
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
    void updateResidual(const xinfo &predictions_std, size_t tree_ind, size_t M, std::vector<double> &residual_std) const
    {
        size_t next_index = tree_ind + 1;
        if (next_index == M)
        {
            next_index = 0;
        }  

        //slop is the partial fit
        for(size_t i=0; i<slop->size(); ++i) {
            for(size_t j=0; j<(*slop)[0].size(); ++j) {
                //output[i] = data_pointers[M][i]->theta_vector[0];
                (*slop)[i][j] *= fit_info->data_pointers[tree_ind][i]->theta_vector[j]/fit_info->data_pointers[next_index][i]->theta_vector[j];
            }
        }
    }

    // Once their function calls are standardized to take a FitInfo we should never have to redefine these
    // in another model class
    void calcSuffStat_categorical(std::vector<double> &y, xinfo_sizet &Xorder, size_t &start, size_t &end, const size_t &var)
    {
        // calculate sufficient statistics for categorical variables

        // compute sum of y[Xorder[start:end, var]]
        for (size_t i = start; i <= end; i++) incSuffStat(y, Xorder[var][i],suff_stat_model);

    }

    void calcSuffStat_continuous(std::vector<size_t> &xorder, std::vector<double> &y_std, 
        std::vector<size_t> &candidate_index, size_t index, bool adaptive_cutpoint)
    {
        // calculate sufficient statistics for continuous variables
        if (adaptive_cutpoint)
        {

            // if use adaptive number of cutpoints, calculated based on vector candidate_index
            for (size_t q = candidate_index[index] + 1; q <= candidate_index[index + 1]; q++)
            {
                incSuffStat(y_std,xorder[q],suff_stat_model);
            }
        }
        else
        {
            incSuffStat(y_std,xorder[index],suff_stat_model);
        }

        return;
    }

    void updateFullSuffStat(std::vector<double> &y_std, std::vector<size_t> &x_info)
    {
        for (size_t i = 0; i < x_info.size(); i++) incSuffStat(y_std, x_info[i],suff_stat_total);
        return;
    }

    double LIL(const std::vector<double> &suffstats) const {
        LogitLIL(suffstats, tau_a, tau_b);
    }

    //this function should call a base LIL() member function that should be redefined in
    double likelihood(double tau, double ntau, double sigma2, double y_sum, bool left_side) const
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

    double likelihood_no_split(double value, double tau, double ntau, double sigma2) const
    {
        // the likelihood of no-split option is a bit different from others
        // because the sufficient statistics is y_sum here
        // write a separate function, more flexibility

        return LIL(suff_stat_total);

        ;
    }
    Model *clone() { return new LogitClass(*this); }
};

#endif
