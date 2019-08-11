#include <cstddef>
#include <iostream>
#include <vector>
#include "xbart.h"
#include <utility.h>
#include <forest.h>



//using namespace std;

// Constructors
XBARTcpp::XBARTcpp(XBARTcppParams params){
	this->params = params;		
}
XBARTcpp::XBARTcpp (size_t M ,size_t N_sweeps ,
        size_t Nmin , size_t Ncutpoints , //CHANGE 
        double alpha , double beta , double tau , //CHANGE!
        size_t burnin, 
        size_t mtry , size_t max_depth_num, double kap , 
        double s , bool verbose, bool parallel,int seed,size_t model_num,double no_split_penality,bool sample_weights_flag){
  this->params.M = M; 
  this->params.N_sweeps = N_sweeps;
  this->params.Nmin = Nmin;
  this->params.Ncutpoints = Ncutpoints;
  this->params.alpha = alpha;
  this->params.beta = beta;
  this->params.tau = tau;
  this->params.burnin = burnin;
  this->params.mtry = mtry;
  this->params.max_depth_num = max_depth_num;
  this->params.kap = kap;
  this->params.s = s;
  this->params.verbose = verbose;
  this->params.parallel=parallel;
  this->trees =  vector< vector<tree>> (N_sweeps);
  this->model_num =  model_num;
  this->no_split_penality =  no_split_penality;
  this->params.sample_weights_flag =  sample_weights_flag;

  // handling seed
  
  if(seed == -1){
    this->seed_flag = false;
    this->seed = 0;
  }else{
    this->seed_flag = true; 
    this->seed = (size_t)seed;
  }

  
  // Create trees
  for(size_t i = 0; i < N_sweeps;i++){
        this->trees[i]=  vector<tree>(M); 
    }


  // Initialize model
  //if(this->model_num == 0){ // NORMAL
    //define model
    // this->model = new NormalModel(this->params.kap, this->params.s, this->params.tau, this->params.alpha, this->params.beta);
    // this->model->setNoSplitPenality(no_split_penality);
    
  //}

    return;
}

XBARTcpp::XBARTcpp(std::string json_string){
  //std::vector<std::vector<tree>> temp_trees;
  from_json_to_forest(json_string,  this->trees,this->y_mean);  
  this->params.N_sweeps = this->trees.size();
  this->params.M = this->trees[0].size();
}


std::string XBARTcpp::_to_json(void){
  json j = get_forest_json(this->trees,this->y_mean);
  return j.dump();
}

// Getter
int XBARTcpp::get_M(){return((int)params.M);} 

void XBARTcpp::_predict(int n,int d,double *a){//,int size, double *arr){
  
  // Convert *a to col_major std::vector
  vec_d x_test_std_flat(n*d);
  XBARTcpp::np_to_col_major_vec(n,d,a,x_test_std_flat);

  // Initialize result
  ini_matrix(this->yhats_test_xinfo, n, this->params.N_sweeps);
  for(size_t i = 0;i<n;i++)for(size_t j = 0;j<this->params.N_sweeps;j++) this->yhats_test_xinfo[j][i]=0;

  // Convert column major vector to pointer
  const double *Xtestpointer = &x_test_std_flat[0];//&x_test_std[0][0];

  // Predict
  NormalModel *model = new NormalModel(); //(this->params.kap, this->params.s, this->params.tau, this->params.alpha, this->params.beta);

  model->predict_std(Xtestpointer,n,d,this->params.M,this->params.N_sweeps,
        this->yhats_test_xinfo,this->trees); 

  delete model;
}


void XBARTcpp::_fit(int n,int d,double *a, 
      int n_y,double *a_y, size_t p_cat){
  
      // Convert row major *a to column major std::vector
      vec_d x_std_flat(n*d);
      XBARTcpp::np_to_col_major_vec(n,d,a,x_std_flat);

      // Convert a_y to std::vector
      vec_d y_std(n_y);
      XBARTcpp::np_to_vec_d(n_y,a_y,y_std);
                
      // Calculate y_mean
      double y_mean = 0.0;
      for (size_t i = 0; i < n; i++){
          y_mean = y_mean + y_std[i];
        }
      y_mean = y_mean/(double)n;
      this->y_mean = y_mean;
      
      // xorder containers
      matrix<size_t> Xorder_std;
      ini_xinfo_sizet(Xorder_std, n, d);
      XBARTcpp::compute_Xorder(n, d,x_std_flat,Xorder_std);
    
      //max_depth_std container
      matrix<size_t> max_depth_std;
      ini_xinfo_sizet(max_depth_std, this->params.M, this->params.N_sweeps);

      // Fill with max Depth Value
      for(size_t i = 0; i < this->params.M; i++){
        for(size_t j = 0;j < this->params.N_sweeps; j++){
          max_depth_std[j][i] = this->params.max_depth_num;
        }
      }


      // Cpp native objects to return
      matrix<double>  yhats_xinfo; // Temp Change
      ini_xinfo(yhats_xinfo, n, this->params.N_sweeps);

      // Temp Change
      ini_xinfo(this->sigma_draw_xinfo, this->params.M, this->params.N_sweeps);
      this->mtry_weight_current_tree.resize(d);
      //ini_xinfo(this->split_count_all_tree, d, this->params.M); // initialize at 0
      double *ypointer = &a_y[0];
      double *Xpointer = &x_std_flat[0];

  if(this->model_num == 0){ // NORMAL
  
    // define model
    NormalModel *model = new NormalModel(this->params.kap, this->params.s, this->params.tau, this->params.alpha, this->params.beta);
    model->setNoSplitPenality(no_split_penality);

    // // //State settings
    std::vector<double> initial_theta(1, y_mean / (double)this->params.M);
    std::unique_ptr<State> state(new NormalState(Xpointer, Xorder_std, n, d, this->params.M, p_cat, d-p_cat, 
    this->seed_flag, this->seed, this->params.Nmin, this->params.Ncutpoints, this->params.parallel, this->params.mtry, 
    Xpointer, this->params.N_sweeps, this->params.sample_weights_flag, &y_std, 1.0, 
    this->params.max_depth_num, y_mean, this->params.burnin, model->dim_residual));

    // initialize X_struct
    std::unique_ptr<X_struct> x_struct(new X_struct(Xpointer, &y_std, n, Xorder_std, p_cat, d-p_cat, &initial_theta, this->params.M));

    mcmc_loop(Xorder_std, this->params.verbose, sigma_draw_xinfo, this->trees, this->no_split_penality, 
    state, model,x_struct);

    this->mtry_weight_current_tree = state->mtry_weight_current_tree;

    delete model;
    state.reset();
    x_struct.reset();

  }else if(this->model_num == 1){ // CLT
  //     mcmc_loop_clt(Xpointer,y_std,y_mean, Xorder_std,n,d,
  //               this->params.M,  this->params.N_sweeps, max_depth_std, 
  //               this->params.Nmin, this->params.Ncutpoints, this->params.alpha, 
  //               this->params.beta, this->params.tau, this->params.burnin, 
  //               this->params.mtry,  this->params.kap , 
  //               this->params.s, this->params.verbose,
  //               this->params.parallel,
  //               yhats_xinfo,this->sigma_draw_xinfo,this->mtry_weight_current_tree,p_cat,d-p_cat,this->trees,
  //               this->seed_flag, this->seed, this->no_split_penality, this->params.sample_weights_flag);
  }else if(this->model_num == 2){ // Probit
    // define model
    ProbitClass *model = new ProbitClass(this->params.kap, this->params.s, this->params.tau, 
                                          this->params.alpha, this->params.beta,y_std);
    model->setNoSplitPenality(no_split_penality);

    // // //State settings
    std::vector<double> initial_theta(1, y_mean / (double)this->params.M);
    std::unique_ptr<State> state(new NormalState(Xpointer, Xorder_std, n, d, this->params.M, p_cat, d-p_cat, 
    this->seed_flag, this->seed, this->params.Nmin, this->params.Ncutpoints, this->params.parallel, this->params.mtry, 
    Xpointer, this->params.N_sweeps, this->params.sample_weights_flag, &y_std, 1.0, 
    this->params.max_depth_num, y_mean, this->params.burnin, model->dim_residual));

    // initialize X_struct
    std::unique_ptr<X_struct> x_struct(new X_struct(Xpointer, &y_std, n, Xorder_std, p_cat, d-p_cat, &initial_theta, this->params.M));

    mcmc_loop_probit(Xorder_std, this->params.verbose, sigma_draw_xinfo, this->trees, this->no_split_penality, 
    state, model,x_struct);

    this->mtry_weight_current_tree = state->mtry_weight_current_tree;

    delete model;
    state.reset();
    x_struct.reset();
  }
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
void XBARTcpp::_get_importance(int size,double *arr){
  for(size_t i =0; i < size ; i++){
    arr[i] = this->mtry_weight_current_tree[i];
  }
}



// Private Helper Functions 

// Numpy 1D array to vec_d - std_vector of doubles
void XBARTcpp::np_to_vec_d(int n,double *a,vec_d &y_std){
  for (size_t i = 0; i < n; i++){
      y_std[i] = a[i];
     }
}

void XBARTcpp::np_to_col_major_vec(int n, int d,double *a,vec_d &x_std){
  for(size_t i =0;i<n;i++){
    for(size_t j =0;j<d;j++){
      size_t index = i*d + j;
      size_t index_std = j*n +i;
      x_std[index_std] = a[index];
    }
  }

}

void XBARTcpp::xinfo_to_np(matrix<double>  x_std,double *arr){
  // Fill in array values from xinfo
  for(size_t i = 0 ,n = (size_t)x_std[0].size();i<n;i++){
    for(size_t j = 0,d = (size_t)x_std.size();j<d;j++){
      size_t index = i*d + j;
      arr[index] = x_std[j][i];
    }
  }
  return;
}

void XBARTcpp::compute_Xorder(size_t n, size_t d,const vec_d &x_std_flat,matrix<size_t>  & Xorder_std){
        // Create Xorder
      std::vector<size_t> temp;
      std::vector<size_t> *xorder_std;
      for (size_t j = 0; j < d; j++)
      { 
        size_t column_start_index = j*n;
        std::vector<double>::const_iterator first = x_std_flat.begin() + column_start_index;
        std::vector<double>::const_iterator last = x_std_flat.begin() + column_start_index + n;
        std::vector<double> colVec(first, last);

        temp = sort_indexes(colVec);
    
        xorder_std = &Xorder_std[j];
        for(size_t i = 0; i<n; i++) (*xorder_std)[i] = temp[i];

      }
}



