#ifndef model_h
#define model_h

#include "common.h"

class Model{
private:
    size_t num_classes;
public:
	virtual void getSufficientStatistic() const{return;};
	virtual void samplePars(bool draw_mu, double y_mean, size_t N_Xorder, double sigma, double tau, 
						std::default_random_engine generator,double &theta, double &theta_noise) const {return;};
	virtual double likelihood(double value,double tau,double ntau,double sigma2) const{return 0 ;};

};




class NormalModel: public Model{
private:
    size_t num_classes = 1;
public:
		void getSufficientStatistic() const {return;};
		void samplePars(bool draw_mu, double y_mean, size_t N_Xorder, double sigma, double tau, 
						std::default_random_engine generator,double &theta, double &theta_noise) const {
		std::normal_distribution<double> normal_samp(0.0, 1.0);
    	if (draw_mu == true){
        	theta = y_mean * N_Xorder / pow(sigma, 2) / (1.0 / tau + N_Xorder / pow(sigma, 2)) + sqrt(1.0 / (1.0 / tau + N_Xorder / pow(sigma, 2))) * normal_samp(generator); //Rcpp::rnorm(1, 0, 1)[0];//* as_scalar(arma::randn(1,1));
        	theta_noise = theta;
    	}
    	else {
        	theta = y_mean * N_Xorder / pow(sigma, 2) / (1.0 / tau + N_Xorder / pow(sigma, 2));
        	theta_noise = theta; // identical to theta
		}
		return;
	}
		double likelihood(double value,double tau,double ntau,double sigma2) const { return -0.5 * log(ntau + sigma2)  + 0.5 * tau * pow(value, 2) / (sigma2 * (ntau + sigma2));}
};



#endif
