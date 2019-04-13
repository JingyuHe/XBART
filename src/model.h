
#ifndef model_h
#define model_h

#include "common.h"
#include "utility.h"

using namespace std;

class Model
{

  private:
	size_t num_classes;
	size_t dim_suffstat;
	std::vector<double> suff_stat_model;

  public:
	virtual void incrementSuffStat() const { return; };
	virtual void samplePars(bool draw_mu, double y_mean, size_t N_Xorder, double sigma, double tau,
							std::mt19937 &generator, std::vector<double> &theta_vector) const { return; };

	virtual void updateResidual(const xinfo &predictions_std, size_t tree_ind, size_t M,
								std::vector<double> &residual_std) const { return; };

	virtual size_t getNumClasses() const { return 0; };
	virtual size_t getDimSuffstat() const { return 0; };
	std::vector<double> getSuffstat() const { return std::vector<double>(); };


	virtual void suff_stat_fill(double a) { return; };
	virtual void suff_stat_init() { return; };
	virtual void printSuffstat() const { return; };

	virtual void calcSuffStat_categorical(std::vector<double> &y, xinfo_sizet &Xorder, size_t &start, size_t &end, const size_t &var) { return; };
	virtual void calcSuffStat_continuous(std::vector<size_t> &xorder, std::vector<double> &y_std, std::vector<size_t> &candidate_index, size_t index, bool adaptive_cutpoint) { return; };
	virtual double likelihood(double tau, double ntau, double sigma2, double y_sum, bool left_side) const { return 0.0; };
	virtual double likelihood_no_split(double value, double tau, double ntau, double sigma2) const { return 0.0; };
};

class NormalModel : public Model
{
  private:
	size_t num_classes = 1;
	size_t dim_suffstat = 1;
	std::vector<double> suff_stat_model;

  public:
	void suff_stat_init()
	{
		suff_stat_model.resize(dim_suffstat);
		return;
	}
	void suff_stat_fill(double a)
	{
		// fill the suff_stat_model with a value
		// in function call, a = 0.0 to reset sufficient statistics vector
		
		std::fill(suff_stat_model.begin(), suff_stat_model.end(), a);
		return;
	}
	void incrementSuffStat() const { return; };
	void samplePars(bool draw_mu, double y_mean, size_t N_Xorder, double sigma, double tau,
					std::mt19937 &generator, std::vector<double> &theta_vector) const
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

	size_t getNumClasses() const { return this->num_classes; }
	size_t getDimSuffstat() const { return this->dim_suffstat; }
	std::vector<double> getSuffstat() const { return this->suff_stat_model; };

	void printSuffstat() const
	{
		cout << this->suff_stat_model << endl;
		return;
	};

	void calcSuffStat_categorical(std::vector<double> &y, xinfo_sizet &Xorder, size_t &start, size_t &end, const size_t &var)
	{
		// calculate sufficient statistics for categorical variables

		// compute sum of y[Xorder[start:end, var]]
		size_t loop_count = 0;
		for (size_t i = start; i <= end; i++)
		{
			suff_stat_model[0] += y[Xorder[var][i]];
			loop_count++;
		}
		return;
	}

	void calcSuffStat_continuous(std::vector<size_t> &xorder, std::vector<double> &y_std, std::vector<size_t> &candidate_index, size_t index, bool adaptive_cutpoint)
	{
		// calculate sufficient statistics for continuous variables

		if (adaptive_cutpoint)
		{
			// if use adaptive number of cutpoints, calculated based on vector candidate_index
			for (size_t q = candidate_index[index] + 1; q <= candidate_index[index + 1]; q++)
			{
				suff_stat_model[0] += y_std[xorder[q]];
			}
		}
		else
		{
			// use all data points as candidates
			suff_stat_model[0] += y_std[xorder[index]];
		}
		return;
	}

	double likelihood(double tau, double ntau, double sigma2, double y_sum, bool left_side) const
	{
		// likelihood equation,
		// note the difference of left_side == true / false

		// BE CAREFUL
		// weighting is in function
		// calculate_likelihood_continuous and calculate_likelihood_categorical, tree.cpp
		// see function call there
		// maybe move to model class?

		if (left_side)
		{
			return -0.5 * log(ntau + sigma2) + 0.5 * tau * pow(suff_stat_model[0], 2) / (sigma2 * (ntau + sigma2));
		}
		else
		{
			return -0.5 * log(ntau + sigma2) + 0.5 * tau * pow(y_sum - suff_stat_model[0], 2) / (sigma2 * (ntau + sigma2));
		}
	}

	double likelihood_no_split(double value, double tau, double ntau, double sigma2) const
	{
		// the likelihood of no-split option is a bit different from others
		// because the sufficient statistics is y_sum here
		// write a separate function, more flexibility


		// BE CAREFUL
		// weighting of no split option is in function
		// calculate_likelihood_no_split in tree.cpp
		// maybe move it to model class??
		return -0.5 * log(ntau + sigma2) + 0.5 * tau * pow(value, 2) / (sigma2 * (ntau + sigma2));
	}
};

class CLTClass : public Model
{
  private:
	size_t num_classes = 1;
	size_t dim_suffstat = 3;
	std::vector<double> suff_stat_model;

  public:
  	std::vector<double>  total_fit; // Keep public to save copies
	double sum_ipsi = 0; // sum(1/psi)
	double sum_log_ipsi = 0; //sum(log(1/psi))

	void suff_stat_init()
	{
		suff_stat_model.resize(dim_suffstat);
		return;
	}
	void suff_stat_fill(double a)
	{
		// fill the suff_stat_model with a value
		// in function call, a = 0.0 to reset sufficient statistics vector
		
		std::fill(suff_stat_model.begin(), suff_stat_model.end(), a);
		return;
	}
	void incrementSuffStat() const { return; };
	void samplePars(bool draw_mu, double y_mean, size_t N_Xorder, double sigma, double tau,
					std::mt19937 &generator, std::vector<double> &theta_vector) const
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

	size_t getNumClasses() const { return this->num_classes; }
	size_t getDimSuffstat() const { return this->dim_suffstat; }
	std::vector<double> getSuffstat() const { return this->suff_stat_model; };

	void printSuffstat() const
	{
		cout << this->suff_stat_model << endl;
		return;
	};

	void calcSuffStat_categorical(std::vector<double> &y, xinfo_sizet &Xorder, size_t &start, size_t &end, const size_t &var)
	{
		// calculate sufficient statistics for categorical variables

		// compute sum of y[Xorder[start:end, var]]
		size_t loop_count = 0;
		std::vector <size_t> &xorder_var = Xorder[var];
		for (size_t i = start; i <= end; i++)
		{
			double current_fit_val = total_fit[xorder_var[i]];
			double psi = (current_fit_val +1)*( 1- current_fit_val);
			suff_stat_model[0] += y[xorder_var[i]]/psi;
			suff_stat_model[1] += 1/psi;
			suff_stat_model[2] += std::log(1/psi);
			loop_count++;
		}
		return;
	}

	void calcSuffStat_continuous(std::vector<size_t> &xorder, std::vector<double> &y_std, std::vector<size_t> &candidate_index, size_t index, bool adaptive_cutpoint)
	{
		// calculate sufficient statistics for continuous variables

		if (adaptive_cutpoint)
		{
			// if use adaptive number of cutpoints, calculated based on vector candidate_index
			for (size_t q = candidate_index[index] + 1; q <= candidate_index[index + 1]; q++)
			{
				double current_fit_val = total_fit[xorder[q]];
				double psi = (current_fit_val +1)*( 1- current_fit_val);
				suff_stat_model[0] += y_std[xorder[q]]/psi;
				suff_stat_model[1] += 1/psi;
				suff_stat_model[2] += std::log(1/psi);
			}
		}
		else
		{
			// use all data points as candidates
			double current_fit_val = total_fit[xorder[index]];
			double psi = (current_fit_val +1)*( 1- current_fit_val);
			suff_stat_model[0] += y_std[xorder[index]]/psi;
			suff_stat_model[1] += 1/psi;
			suff_stat_model[2] += std::log(1/psi);
		}
		return;
	}

	void updateFullSuffStat(){
		size_t n = total_fit.size();
		for(size_t i = 0; i < n; i++){
			double current_fit_val = total_fit[i];
			double psi = (current_fit_val +1)*( 1- current_fit_val);
			sum_ipsi += 1/psi;
			sum_log_ipsi += std::log(1/psi);
		}
		return;
	}

	double likelihood(double tau, double ntau, double sigma2, double y_sum, bool left_side) const
	{
		// likelihood equation,
		// note the difference of left_side == true / false

		// BE CAREFUL
		// weighting is in function
		// calculate_likelihood_continuous and calculate_likelihood_categorical, tree.cpp
		// see function call there
		// maybe move to model class?

		if (left_side)
		{
			return suff_stat_model[2] + 0.5 * std::log((1/tau)/((1/tau)+suff_stat_model[1])) + 0.5 * std::log(tau/(1+suff_stat_model[1]))*pow(suff_stat_model[0], 2);
			//return -0.5 * log(ntau + sigma2) + 0.5 * tau * pow(suff_stat_model[0], 2) / (sigma2 * (ntau + sigma2));
		}
		else
		{
			return (sum_log_ipsi - suff_stat_model[2]) + 0.5 * std::log((1/tau)/((1/tau)+ (sum_ipsi- suff_stat_model[1]) )) + 0.5 * std::log(tau/(1+ (sum_ipsi- suff_stat_model[1]) ))* ( y_sum - suff_stat_model[0] );
		}
	}

	double likelihood_no_split(double value, double tau, double ntau, double sigma2) const
	{
		// the likelihood of no-split option is a bit different from others
		// because the sufficient statistics is y_sum here
		// write a separate function, more flexibility


		// BE CAREFUL
		// weighting of no split option is in function
		// calculate_likelihood_no_split in tree.cpp
		// maybe move it to model class??
		return -0.5 * log(ntau + sigma2) + 0.5 * tau * pow(value, 2) / (sigma2 * (ntau + sigma2));
	}
};

#endif