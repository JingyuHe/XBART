
#ifndef model_h
#define model_h

#include "common.h"
#include "utility.h"
#include "beta.h"

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
	virtual double likelihood(double tau, double ntau, double sigma2, double y_sum, bool left_side,size_t N_Xorder) const { return 0.0; };
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

	double likelihood(double tau, double ntau, double sigma2, double y_sum, bool left_side,size_t N_Xorder) const
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


class PoissonClassifcationModel : public Model
{
  private:
	size_t num_classes = 3;
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
		// TODO: Change later so that each class has prior as a private memeber:
		double sum_y = y_mean*N_Xorder;


			// test result should be theta
			// y_mean*N_Xorder = sum_y
			// P
		//std::cout << "sum_y " << sum_y << endl;
		sftrabbit::beta_distribution<double> beta(sigma+sum_y, N_Xorder - sum_y+ tau); // sigma = alpha, tau = beta
		theta_vector[0] = beta(generator);
		//std::cout <<"tv[0] " << theta_vector[0] << endl;
		theta_vector[1] = -std::log(2.0*std::max(theta_vector[0],1.0-theta_vector[0])-1.0)/2.0 ;
		if (theta_vector[0] > 0.5){
			theta_vector[2] = 1.0;
		}else{
			theta_vector[2] = 0.0;
		}
		//std::cout << "tv[1] " << theta_vector[1]<<" tv[2] " << theta_vector[2] << endl;
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

	double likelihood(double tau, double ntau, double sigma2, double y_sum, bool left_side,size_t N_Xorder) const
	{
		// sorry in advance...
		// note the difference of left_side == true / false

		// ntau = n_min

		// BE CAREFUL
		// weighting is in function
		// calculate_likelihood_continuous and calculate_likelihood_categorical, tree.cpp
		// see function call there
		// maybe move to model class?

		// ntau/tau = n_left

		if (left_side)
		{
			return std::lgamma(suff_stat_model[0]+tau) + std::lgamma(ntau/tau-suff_stat_model[0]+std::sqrt(sigma2)) - std::lgamma( tau+ ntau/tau + std::sqrt(sigma2));
		}
		else
		{
			return std::lgamma(y_sum - suff_stat_model[0]+tau)  + std::lgamma(N_Xorder - ntau/tau -(y_sum - suff_stat_model[0])+std::sqrt(sigma2) ) - std::lgamma(tau+N_Xorder -ntau/tau + std::sqrt(sigma2) );
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
		return std::lgamma(suff_stat_model[0]+tau) + std::lgamma(ntau/tau-suff_stat_model[0]+std::sqrt(sigma2)) - std::lgamma( tau+ ntau/tau + std::sqrt(sigma2));
	}

void draw_residual(std::vector<double> &lambs,std::vector<double> &ks,std::vector<double> &lamj, std::vector<double> &kj, std::vector<double> &y_std,std::vector<double> &residual_std,std::mt19937 &gen)
	{
		double p;
		for(size_t i=0; i < residual_std.size();i++)
		{
			double q = 0.5*(1.0 + std::pow(-1.0,1.0-(size_t)ks[i]%2) * std::exp(-2*lambs[i]));
			double qj = 0.5*(1.0 + std::pow(-1.0,1.0-(size_t)kj[i]%2) * std::exp(-2*lamj[i]));
			
			if (y_std[i] < 0.5){
			p = q*qj/(q*qj + (1.0-q)*(1.0-qj));
		}else{
			p = q*(1.0-qj)/(q*(1.0-qj) + (1.0-q)*qj);
		}
			
			std::bernoulli_distribution bern(p);
			
			residual_std[i]= std::abs(y_std[i] - bern(gen));
			
		}
	}


	// void update_partial_fit(std::vector<double> &lambs,std::vector<double> &ks,
	// 	matrix<tree::tree_p> data_points,
	// 	std::vector<tree::tree_p> &next_data_pointers, std::vector<tree::tree_p> &current_data_pointers)
	// 	{

	// 		for(size_t i = 0;i < k.size();i++){
	// 			std::vector<double> thetas_current = current_data_pointers[i]->theta_vector;
	// 			std::vector<double> thetas_next = next_data_pointers[i]->theta_vector;
	// 			lambs[i] = lambs[i]  - thetas_next[1] + thetas_current[1];
	// 			ks[i] = ks[i]  - thetas_next[2] + thetas_current[2];
	// 		}


	// 		return;
	// 	}

};



#endif