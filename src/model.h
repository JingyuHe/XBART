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
	virtual double likelihood(double value, double tau, double ntau, double sigma2) const { return 0; };

	virtual void updateResidual(const xinfo &predictions_std, size_t tree_ind, size_t M,
															std::vector<double> &residual_std) const { return; };

	virtual size_t getNumClasses() const { return 0; };
	virtual size_t getDimSuffstat() const { return 0; };
	std::vector<double> getSuffstat() const {return std::vector<double>();};

	virtual double calcSuffStat_categorical(std::vector<double> &y, xinfo_sizet &Xorder, size_t &start, size_t &end, double &suff_stat, const size_t &var) const { return 0.0; };

	virtual double calcSuffStat_continuous(std::vector<size_t> &xorder, std::vector<double> &y_std, std::vector<size_t> &candidate_index, size_t index, double &suff_stat, bool adaptive_cutpoint) const { return 0.0; };

	virtual std::vector<double> calcSuffStat_continuous_vec(std::vector<size_t> &xorder, std::vector<double> &y_std, std::vector<size_t> &candidate_index, size_t index, std::vector<double> suff_stat, bool adaptive_cutpoint) const {return std::vector<double>(); };

	virtual std::vector<double> calcSuffStat_categorical_vec(std::vector<double> &y, xinfo_sizet &Xorder, size_t &start, size_t &end, std::vector<double> suff_stat, const size_t &var) const {return std::vector<double>();};

	virtual double likelihood_vec(std::vector<double> value, double tau, double ntau, double sigma2) const { return 0; };

	virtual	void suff_stat_fill(double a) {return;};
	virtual void suff_stat_init() {return;};
	virtual void printSuffstat() const {return;};


	virtual void calcSuffStat_categorical_vec_class(std::vector<double> &y, xinfo_sizet &Xorder, size_t &start, size_t &end, const size_t &var)  {return;};
	virtual void calcSuffStat_continuous_vec_class(std::vector<size_t> &xorder, std::vector<double> &y_std, std::vector<size_t> &candidate_index, size_t index, bool adaptive_cutpoint) {return;};
	virtual double likelihood_vec_class(double y_sum, double tau, double ntau, double sigma2, bool left_side) const {return 0.0;};
	virtual double likelihood_vec_test(double tau, double ntau, double sigma2, double y_sum, bool left_side) const {return 0.0;};
};

class NormalModel : public Model
{
private:
	size_t num_classes = 1;
	size_t dim_suffstat = 1;
	std::vector<double> suff_stat_model;

public:
	void suff_stat_init(){
		suff_stat_model.resize(dim_suffstat);
		return;
	}
	void suff_stat_fill(double a){
		// cout << "before " << suff_stat_model << endl;
		std::fill(suff_stat_model.begin(), suff_stat_model.end(), a);
		// cout << "after " << suff_stat_model << endl;
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
	double likelihood(double value, double tau, double ntau, double sigma2) const { return -0.5 * log(ntau + sigma2) + 0.5 * tau * pow(value, 2) / (sigma2 * (ntau + sigma2)); }

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
	std::vector<double> getSuffstat() const {return this->suff_stat_model;};

	void printSuffstat() const{cout << this-> suff_stat_model << endl; return;};

	double calcSuffStat_categorical(std::vector<double> &y, xinfo_sizet &Xorder, size_t &start, size_t &end, double &suff_stat, const size_t &var) const
	{
		// compute sum of y[Xorder[start:end, var]]
		size_t loop_count = 0;
		for (size_t i = start; i <= end; i++)
		{
			suff_stat += y[Xorder[var][i]];
			loop_count++;
			// cout << "Xorder " << Xorder[var][i] << " y value " << y[Xorder[var][i]] << endl;
		}
		return suff_stat;
	}

	double calcSuffStat_continuous(std::vector<size_t> &xorder, std::vector<double> &y_std, std::vector<size_t> &candidate_index, size_t index, double &suff_stat, bool adaptive_cutpoint) const
	{

		if (adaptive_cutpoint)
		{
			// if use adaptive number of cutpoints, calculated based on vector candidate_index
			for (size_t q = candidate_index[index] + 1; q <= candidate_index[index + 1]; q++)
			{
				suff_stat += y_std[xorder[q]];
			}
		}
		else
		{
			// use all data points as candidates
			suff_stat += y_std[xorder[index]];
		}
		return suff_stat;
	}


	std::vector<double> calcSuffStat_continuous_vec(std::vector<size_t> &xorder, std::vector<double> &y_std, std::vector<size_t> &candidate_index, size_t index, std::vector<double> suff_stat, bool adaptive_cutpoint) const
	{

		if (adaptive_cutpoint)
		{
			// if use adaptive number of cutpoints, calculated based on vector candidate_index
			for (size_t q = candidate_index[index] + 1; q <= candidate_index[index + 1]; q++)
			{
				suff_stat[0] += y_std[xorder[q]];
			}
		}
		else
		{
			// use all data points as candidates
			suff_stat[0] += y_std[xorder[index]];
		}
		return suff_stat;
	}

	std::vector<double> calcSuffStat_categorical_vec(std::vector<double> &y, xinfo_sizet &Xorder, size_t &start, size_t &end, std::vector<double> suff_stat, const size_t &var) const
	{
		// compute sum of y[Xorder[start:end, var]]
		size_t loop_count = 0;
		for (size_t i = start; i <= end; i++)
		{
			suff_stat[0] += y[Xorder[var][i]];
			loop_count++;
			// cout << "Xorder " << Xorder[var][i] << " y value " << y[Xorder[var][i]] << endl;
		}
		return suff_stat;
	}

	double likelihood_vec(std::vector<double> value, double tau, double ntau, double sigma2) const 
	{ 
		// cout << "inside " << value << "   " << suff_stat_model << endl;
		// cout << "-----" << endl;
		return -0.5 * log(ntau + sigma2) + 0.5 * tau * pow(value[0], 2) / (sigma2 * (ntau + sigma2)); 
	}


	void calcSuffStat_categorical_vec_class(std::vector<double> &y, xinfo_sizet &Xorder, size_t &start, size_t &end, const size_t &var) 
	{
		// compute sum of y[Xorder[start:end, var]]
		size_t loop_count = 0;
		for (size_t i = start; i <= end; i++)
		{
			suff_stat_model[0] += y[Xorder[var][i]];
			loop_count++;
			// cout << "Xorder " << Xorder[var][i] << " y value " << y[Xorder[var][i]] << endl;
		}
		return;
	}

	void calcSuffStat_continuous_vec_class(std::vector<size_t> &xorder, std::vector<double> &y_std, std::vector<size_t> &candidate_index, size_t index, bool adaptive_cutpoint)
	{

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



	double likelihood_vec_class(double y_sum, double tau, double ntau, double sigma2, bool left_side) const 
	{ 
		if(left_side){
			// cout << "left comparison " << value[0] << "  " << suff_stat_model[0] << endl;
			return -0.5 * log(ntau + sigma2) + 0.5 * tau * pow(suff_stat_model[0], 2) / (sigma2 * (ntau + sigma2));
		}else{
			// cout << "right comparison " << value[0] << " " << suff_stat_model[0] << endl;
			return -0.5 * log(ntau + sigma2) + 0.5 * tau * pow(y_sum - suff_stat_model[0], 2) / (sigma2 * (ntau + sigma2));
		} 
	}

	double likelihood_vec_test(double tau, double ntau, double sigma2, double y_sum, bool left_side) const 
	{ 
		// cout << "inside " << value << "   " << suff_stat_model << endl;
		// cout << "-----" << endl;
		if(left_side){
			// cout << "left comparison " << value[0] << "  " << suff_stat_model[0] << endl;
			return -0.5 * log(ntau + sigma2) + 0.5 * tau * pow(suff_stat_model[0], 2) / (sigma2 * (ntau + sigma2));
		}else{
			// cout << "right comparison " << value[0] << " " << suff_stat_model[0] << endl;
			return -0.5 * log(ntau + sigma2) + 0.5 * tau * pow(y_sum - suff_stat_model[0], 2) / (sigma2 * (ntau + sigma2));
		} 
	}

};

#endif
