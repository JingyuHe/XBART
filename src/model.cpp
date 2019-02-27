#include "model.h"

// void NormalModel::samplePars(bool draw_mu, double y_mean, size_t N_Xorder, double sigma, double tau,
// 						std::default_random_engine generator,double &theta, double &theta_noise){
// 	std::normal_distribution<double> normal_samp(0.0, 1.0);

//     if (draw_mu == true)
//     {

//         theta = y_mean * N_Xorder / pow(sigma, 2) / (1.0 / tau + N_Xorder / pow(sigma, 2)) + sqrt(1.0 / (1.0 / tau + N_Xorder / pow(sigma, 2))) * normal_samp(generator); //Rcpp::rnorm(1, 0, 1)[0];//* as_scalar(arma::randn(1,1));
//         theta_noise = theta;
//     }
//     else
//     {

//         theta = y_mean * N_Xorder / pow(sigma, 2) / (1.0 / tau + N_Xorder / pow(sigma, 2));
//         theta_noise = theta; // identical to theta
//     }

// 	return;
// }