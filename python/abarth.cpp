#include <cstddef>
#include <armadillo>
#include <iostream>
#include <vector>
#include "abarth.h"
#include "../src/python_train_forest_std_mtrywithinnode.h"
// #include "helpers.h"


using namespace std;

// Constructors
Abarth::Abarth(AbarthParams params){
	this->params = params;		
}
Abarth::Abarth (size_t M ,size_t L ,size_t N_sweeps ,
        size_t Nmin , size_t Ncutpoints , //CHANGE 
        double alpha , double beta , double tau , //CHANGE!
        size_t burnin, 
        size_t max_depth_num , bool draw_sigma , double kap , 
        double s , bool verbose , bool m_update_sigma, 
        bool draw_mu , bool parallel){
  this->params.M = M; 
  this->params.L = L;
  this->params.N_sweeps = N_sweeps;
  this->params.Nmin = Nmin;
  this->params.Ncutpoints = Ncutpoints;
  this->params.alpha = alpha;
  this->params.beta = beta;
  this->params.tau = tau;
  this->params.burnin = burnin;
  this->params.max_depth_num = max_depth_num;
  this->params.draw_sigma = draw_sigma;
  this->params.kap = kap;
  this->params.s = s;
  this->params.verbose = verbose;
  this->params.m_update_sigma = m_update_sigma;
  this->params.draw_mu = draw_mu;
  this->params.parallel=parallel;
  return;
}

// Destructor
Abarth::~Abarth(){}

// Getter
int Abarth::get_M(){return((int)params.M);} 


// Fitting
double Abarth::fit(int n,double *a){
  this->y_std.reserve(n);
  this->y_std = Abarth::np_to_vec_d(n,a);

  return y_std[n-1];
}

double Abarth::fit_x(int n,int d,double *a){
  xinfo x_std = Abarth::np_to_xinfo(n,d,a);

  return x_std[d-1][n-1];
}

void Abarth::predict(int n,double *a,int size, double *arr){
  this->y_std.reserve(n);
  this->y_std = Abarth::np_to_vec_d(n,a);
  std::copy(y_std.begin(), y_std.end(), arr);
    
  return;
}

void Abarth::__predict_2d(int n,int d,double *a,int size, double *arr){
  xinfo x_std = Abarth::np_to_xinfo(n,d,a);
  Abarth::xinfo_to_np(x_std,arr);
  return;
}

void Abarth::fit_predict(int n,int d,double *a, // Train X 
      int n_y,double *a_y, // Train Y
      int n_test,int d_test,double *a_test, // Test X
      int size, double *arr){ // Result 

        xinfo x_std = Abarth::np_to_xinfo(n,d,a);
        xinfo x_test_std = Abarth::np_to_xinfo(n_test,d_test,a_test);
        this->y_std.reserve(n_y);
        this->y_std = Abarth::np_to_vec_d(n_y,a_y);
        
        // // Fit Model
        // python_train_forest_root_std_mtrywithinnode();
        // params.M,params.L,params.N_sweeps,params.max_depth_num,params.Nmin,
        // params.Ncutpoints,params.alpha,params.beta,params.tau);

        std::copy(y_std.begin(), y_std.end(), arr);


        // return;


      } 

// Private Helper Functions 

// Numpy 1D array to vec_d - std_vector of doubles
vec_d Abarth::np_to_vec_d(int n,double *a){
  vec_d y_std(n,0);
  for (size_t i = 0; i < n; i++){
      y_std[i] = a[i];
     }
  return y_std;
}

// Numpy 2D Array to xinfo- nested std vectors of doubles
xinfo Abarth::np_to_xinfo(int n, int d,double *a){
  // 
  xinfo x_std(d, vector<double> (n, 0));
  // Fill in Values of xinfo from array 
  for(size_t i =0;i<n;i++){
    for(size_t j =0;j<d;j++){
      size_t index = i*d + j;
      x_std[j][i] = a[index];
    }
  }
  return x_std;
}

void Abarth::xinfo_to_np(xinfo x_std,double *arr){
  // Fill in array values from xinfo
  for(size_t i = 0 ,n = (size_t)x_std[0].size();i<n;i++){
    for(size_t j = 0,d = (size_t)x_std.size();j<d;j++){
      size_t index = i*d + j;
      arr[index] = x_std[j][i];
    }
  }
  return;
}




