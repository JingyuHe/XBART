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
	void calcSuffStat_continuous(xinfo_sizet &Xorder_std, std::vector<double> &y_cumsum, std::vector<double> &y_std,  const size_t &N_Xorder, const size_t &Ncutpoints, size_t var_index, std::vector<size_t> &candidate_index, bool adaptive_cutpoints) const {
		return;}
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


	void calcSuffStat_continuous(xinfo_sizet &Xorder_std, std::vector<double> &y_cumsum, std::vector<double> &y_std,  const size_t &N_Xorder, const size_t &Ncutpoints, size_t var_index, std::vector<size_t> &candidate_index, bool adaptive_cutpoints) const {
		// calculate sufficient statistics for continuous variable

		//var_index : index of X variable working on
		std::vector<size_t> &xorders = Xorder_std[var_index];
		double cumsum = 0.0;

		if(adaptive_cutpoints == false){
			// if use all data points as split point candidates
			for (size_t q = 0; q < N_Xorder; q++)
			{
				cumsum += y_std[xorders[q]];
				y_cumsum[q] = cumsum;
			}
		}else{
			// if use adaptive number of split points
			size_t ind = 0;
			for (size_t q = 0; q < N_Xorder; q++)
			{
				cumsum += y_std[xorders[q]];

				if (q >= candidate_index[ind])
				{
					y_cumsum[ind] = cumsum;
					ind++;

					if (ind >= Ncutpoints)
					{
						// have done cumulative sum, do not care about elements after index of last entry of candidate_index
						break;
					}
				}
			}
		}
		return;
	}

};

#endif
