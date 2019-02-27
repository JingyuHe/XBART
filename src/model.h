#ifndef model_h
#define model_h

#include "common.h"
#include "utility.h"

class Model
{

  public:
	virtual void incrementSuffStat() const { return; };
	virtual void samplePars(bool draw_mu, double y_mean, size_t N_Xorder, double sigma, double tau,
							std::mt19937 &generator, std::vector<double> &theta_vector) const { return; };
	virtual double likelihood(double value, double tau, double ntau, double sigma2) const { return 0; };

	virtual void updateResidual(const xinfo &predictions_std, size_t tree_ind, size_t M,
								std::vector<double> &residual_std) const { return; };

	virtual size_t getNumClasses() const { return 0; };
};

class NormalModel : public Model
{
  private:
	size_t num_classes = 1;

  public:
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
};

#endif
