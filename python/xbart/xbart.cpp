//#ifndef SWIG
#include <cstddef>
#include <iostream>
#include <vector>
//#endif 
#include "xbart.h"
#include <utility.h>
#include <forest.h>



using namespace std;

// Constructors
XBARTcpp::XBARTcpp(XBARTcppParams params){
	this->params = params;		
}
XBARTcpp::XBARTcpp (size_t M ,size_t L ,size_t N_sweeps ,
        size_t Nmin , size_t Ncutpoints , //CHANGE 
        double alpha , double beta , double tau , //CHANGE!
        size_t burnin, 
        size_t mtry , size_t max_depth_num,bool draw_sigma , double kap , 
        double s , bool verbose , bool m_update_sigma, 
        bool draw_mu , bool parallel,int seed){
  this->params.M = M; 
  this->params.L = L;
  this->params.N_sweeps = N_sweeps;
  this->params.Nmin = Nmin;
  this->params.Ncutpoints = Ncutpoints;
  this->params.alpha = alpha;
  this->params.beta = beta;
  this->params.tau = tau;
  this->params.burnin = burnin;
  this->params.mtry = mtry;
  this->params.max_depth_num = max_depth_num;
  this->params.draw_sigma = draw_sigma;
  this->params.kap = kap;
  this->params.s = s;
  this->params.verbose = verbose;
  this->params.m_update_sigma = m_update_sigma;
  this->params.draw_mu = draw_mu;
  this->params.parallel=parallel;
  this->trees = vector<tree>(M);
  this->trees2 = vector< vector<tree>> (N_sweeps);

  // handling seed
  
  if(seed == -1){
    this->seed_flag = false;
    this->seed = 0;
  }else{
    this->seed_flag = true; 
    this->seed = (size_t)seed;
  }

  
  // Create trees3
  for(size_t i = 0; i < N_sweeps;i++){
        this->trees2[i]= vector<tree>(M); 
    }
  return;
}

// Destructor
// XBARTcpp::~XBARTcpp(){}

// Getter
int XBARTcpp::get_M(){return((int)params.M);} 


void XBARTcpp::sort_x(int n,int d,double *a,int size, double *arr){
  xinfo x_std = XBARTcpp::np_to_xinfo(n,d,a);

  xinfo_sizet Xorder_std;
  ini_xinfo_sizet(Xorder_std, n, d);  
      for (size_t j = 0; j < d; j++)
    { 
        std::vector <double> x_temp (n);
        std::copy (x_std[j].begin(), x_std[j].end(), x_temp.begin());
        std::vector<size_t> temp = sort_indexes(x_temp);
        for (size_t i = 0; i < n; i++)
        {
            //cout << temp[i] << endl; 
            Xorder_std[j][i] = temp[i];
        }
    }
  std::copy(Xorder_std[d-1].begin(), Xorder_std[d-1].end(), arr);

}



void XBARTcpp::_fit_predict(int n,int d,double *a, // Train X 
      int n_y,double *a_y, // Train Y
      int n_test,int d_test,double *a_test, // Test X
      int size, double *arr,size_t p_cat){ // Result 

      xinfo x_std = XBARTcpp::np_to_xinfo(n,d,a);
      xinfo x_test_std = XBARTcpp::np_to_xinfo(n_test,d_test,a_test);
      y_std.reserve(n_y);
      y_std = XBARTcpp::np_to_vec_d(n_y,a_y);
                
      // Calculate y_mean
      double y_mean = 0.0;
      for (size_t i = 0; i < n; i++){
          y_mean = y_mean + y_std[i];
        }
      y_mean = y_mean/(double)n;
      this->y_mean = y_mean;
      //   // xorder containers
      xinfo_sizet Xorder_std;
      ini_xinfo_sizet(Xorder_std, n, d);

      // MAKE MORE EFFICIENT! 
      // TODO: Figure out away of working on row major std::vectors
      // Fill in 
      for (size_t j = 0; j < d; j++)
    { 
        std::vector <double> x_temp (n);
        std::copy (x_std[j].begin(), x_std[j].end(), x_temp.begin());
        std::vector<size_t> temp = sort_indexes(x_temp);
        for (size_t i = 0; i < n; i++)
        {
            Xorder_std[j][i] = temp[i];
        }
    }
    // Create new x_std's that are row major
    vec_d x_std_2 = XBARTcpp::xinfo_to_row_major_vec(x_std); // INEFFICIENT - For now to include index sorting
    vec_d x_test_std_2 = XBARTcpp::xinfo_to_row_major_vec(x_test_std); // INEFFICIENT

    // Remove old x_std
    for(int j = 0; j<d;j++){
      x_std[j].clear();
      x_test_std[j].clear();
      x_std[j].shrink_to_fit();
      x_test_std[j].shrink_to_fit();
    }
    x_std.clear();x_test_std.clear();
    x_std.shrink_to_fit();x_test_std.shrink_to_fit();


      // // //max_depth_std container
      xinfo_sizet max_depth_std;

      ini_xinfo_sizet(max_depth_std, this->params.M, this->params.N_sweeps);
      // Fill with max Depth Value
      for(size_t i = 0; i < this->params.M; i++){
        for(size_t j = 0;j < this->params.N_sweeps; j++){
          max_depth_std[j][i] = this->params.max_depth_num;
        }
      }


      // Cpp native objects to return
      ini_xinfo(this->yhats_xinfo, n, this->params.N_sweeps);
      ini_xinfo(this->yhats_test_xinfo, n_test, this->params.N_sweeps);
      ini_xinfo(this->sigma_draw_xinfo, this->params.M, this->params.N_sweeps);
      ini_xinfo(this->split_count_all_tree, d, this->params.M); // initialize at 0

      double *ypointer = &a_y[0];//&y_std[0];
      double *Xpointer = &x_std_2[0];//&x_std[0][0];
      double *Xtestpointer = &x_test_std_2[0];//&x_test_std[0][0];


      fit_std_main_loop_all(Xpointer,y_std,y_mean,Xtestpointer, Xorder_std,
                n,d,n_test,
                this->params.M, this->params.L, this->params.N_sweeps, max_depth_std, 
                this->params.Nmin, this->params.Ncutpoints, this->params.alpha, this->params.beta, 
                this->params.tau, this->params.burnin, this->params.mtry, 
                 this->params.kap , this->params.s, 
                this->params.verbose, 
                this->params.draw_mu, this->params.parallel,
                yhats_xinfo,this->yhats_test_xinfo,sigma_draw_xinfo,split_count_all_tree,
                p_cat,d-p_cat,this->trees2,this->seed_flag, this->seed);



      std::copy(y_std.begin(), y_std.end(), arr);
    } 

void XBARTcpp::_predict(int n,int d,double *a){//,int size, double *arr){

  xinfo x_test_std = XBARTcpp::np_to_xinfo(n,d,a);
  vec_d x_test_std_2 = XBARTcpp::xinfo_to_row_major_vec(x_test_std); // INEFFICIENT

  ini_xinfo(this->yhats_test_xinfo, n, this->params.N_sweeps);

  double *Xtestpointer = &x_test_std_2[0];//&x_test_std[0][0];
  // predict_std(Xtestpointer,n,d,this->params.M,this->params.L,this->params.N_sweeps,
  //       this->yhats_test_xinfo,this->trees,this->y_mean); 
  predict_std(Xtestpointer,n,d,this->params.M,this->params.L,this->params.N_sweeps,
        this->yhats_test_xinfo,this->trees2,this->y_mean); 
}


void XBARTcpp::_fit(int n,int d,double *a, 
      int n_y,double *a_y, size_t p_cat){
  
      xinfo x_std = XBARTcpp::np_to_xinfo(n,d,a);
      y_std.reserve(n_y);
      y_std = XBARTcpp::np_to_vec_d(n_y,a_y);
                
      // Calculate y_mean
      double y_mean = 0.0;
      for (size_t i = 0; i < n; i++){
          y_mean = y_mean + y_std[i];
        }
      y_mean = y_mean/(double)n;
      this->y_mean = y_mean;
      //   // xorder containers
      xinfo_sizet Xorder_std;
      ini_xinfo_sizet(Xorder_std, n, d);

      // MAKE MORE EFFICIENT! 
      // TODO: Figure out away of working on row major std::vectors
      // Fill in 
      for (size_t j = 0; j < d; j++)
    { 
        std::vector <double> x_temp (n);
        std::copy (x_std[j].begin(), x_std[j].end(), x_temp.begin());
        std::vector<size_t> temp = sort_indexes(x_temp);
        for (size_t i = 0; i < n; i++)
        {
            Xorder_std[j][i] = temp[i];
        }
    }
    // Create new x_std's that are row major
    vec_d x_std_2 = XBARTcpp::xinfo_to_row_major_vec(x_std); // INEFFICIENT - For now to include index sorting

    // Remove old x_std
    for(int j = 0; j<d;j++){
      x_std[j].clear();
      x_std[j].shrink_to_fit();
    }
    x_std.clear();
    x_std.shrink_to_fit();


      // // //max_depth_std container
      xinfo_sizet max_depth_std;

      ini_xinfo_sizet(max_depth_std, this->params.M, this->params.N_sweeps);
      // Fill with max Depth Value
      for(size_t i = 0; i < this->params.M; i++){
        for(size_t j = 0;j < this->params.N_sweeps; j++){
          max_depth_std[j][i] = this->params.max_depth_num;
        }
      }


      // Cpp native objects to return
      xinfo yhats_xinfo; // Temp Change
      ini_xinfo(yhats_xinfo, n, this->params.N_sweeps);


      xinfo sigma_draw_xinfo; // Temp Change
      ini_xinfo(sigma_draw_xinfo, this->params.M, this->params.N_sweeps);
      ini_xinfo(this->split_count_all_tree, d, this->params.M); // initialize at 0
      double *ypointer = &a_y[0];//&y_std[0];
      double *Xpointer = &x_std_2[0];//&x_std[0][0];


//size_t mtry, double kap, double s, bool verbose, bool draw_mu, bool parallel, xinfo &yhats_xinfo, xinfo &sigma_draw_xinfo, size_t p_categorical, size_t p_continuous, vector<vector<tree>> &trees, bool set_random_seed, size_t random_seed);
  fit_std(Xpointer,y_std,y_mean, Xorder_std,n,d,
                this->params.M, this->params.L, this->params.N_sweeps, max_depth_std, 
                this->params.Nmin, this->params.Ncutpoints, this->params.alpha, 
                this->params.beta, this->params.tau, this->params.burnin, 
                this->params.mtry,  this->params.kap , 
                this->params.s, this->params.verbose,
                this->params.draw_mu, this->params.parallel,
                yhats_xinfo,sigma_draw_xinfo,p_cat,d-p_cat,this->trees2,
                this->seed_flag, this->seed);
}    

// Getters
void XBARTcpp::get_yhats(int size,double *arr){
  xinfo_to_np(this->yhats_xinfo,arr);
}
void XBARTcpp::get_yhats_test(int size,double *arr){
  xinfo_to_np(this->yhats_test_xinfo,arr);
}
void XBARTcpp::get_sigma_draw(int size,double *arr){
  xinfo_to_np(this->sigma_draw_xinfo,arr);
}
void XBARTcpp::get_importance(int size,double *arr){
  xinfo_to_np(this->split_count_all_tree,arr);
}



// Private Helper Functions 

// Numpy 1D array to vec_d - std_vector of doubles
vec_d XBARTcpp::np_to_vec_d(int n,double *a){
  vec_d y_std(n,0);
  for (size_t i = 0; i < n; i++){
      y_std[i] = a[i];
     }
  return y_std;
}

vec_d XBARTcpp::np_to_row_major_vec(int n, int d,double *a){
  // 
  vec_d x_std(n*d,0);
  // Fill in Values of xinfo from array 
  ;
  for(size_t i =0;i<n;i++){
    for(size_t j =0;j<d;j++){
      size_t index = i*d + j;
      size_t index_std = j*n +i;
      x_std[index_std] = a[index];
    }
  }
  return x_std;
}

vec_d XBARTcpp::xinfo_to_row_major_vec(xinfo x_std){
  size_t n = (size_t)x_std[0].size();
  size_t d = (size_t)x_std.size();
  vec_d x_std_2(n*d,0);
  // Fill in Values of xinfo from array 
  for(size_t i =0;i<n;i++){
    for(size_t j =0;j<d;j++){
      size_t index_std = j*n +i;
      x_std_2[index_std] = x_std[j][i];
    }
  }
  return x_std_2;
}

// Numpy 2D Array to xinfo- nested std vectors of doubles
xinfo XBARTcpp::np_to_xinfo(int n, int d,double *a){
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

void XBARTcpp::xinfo_to_np(xinfo x_std,double *arr){
  // Fill in array values from xinfo
  for(size_t i = 0 ,n = (size_t)x_std[0].size();i<n;i++){
    for(size_t j = 0,d = (size_t)x_std.size();j<d;j++){
      size_t index = i*d + j;
      arr[index] = x_std[j][i];
    }
  }
  return;
}

void XBARTcpp::test_random_generator(){
  std::default_random_engine generator;
  std::normal_distribution<double> normal_samp(0.0,0.0);

  cout << "Random Normal " << normal_samp(generator) <<endl;
}


