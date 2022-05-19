#include <cstddef>
#include <iostream>
#include <vector>
#include "xbart.h"
#include <utility.h>
#include <forest.h>



using namespace std;

// Constructors
XBARTcpp::XBARTcpp(XBARTcppParams params){
	this->params = params;		
}

XBARTcpp::XBARTcpp(std::string json_string){
  //std::vector<std::vector<tree>> temp_trees;
  from_json_to_forest(json_string,  this->trees,this->y_mean);  
  this->params.num_sweeps = this->trees.size();
  this->params.num_trees = this->trees[0].size();
}


XBARTcpp::XBARTcpp(size_t num_trees, size_t num_sweeps, size_t max_depth,
			 size_t Nmin, size_t Ncutpoints,		//CHANGE
			 double alpha, double beta, double tau, //CHANGE!
			 size_t burnin, size_t mtry,
			 double kap, double s, double tau_kap, double tau_s, 
			 bool verbose, bool sampling_tau, bool parallel, size_t nthread,
			 int seed, double no_split_penality, bool sample_weights){
	this->params.num_trees = num_trees; 
	this->params.num_sweeps = num_sweeps;
	this->params.max_depth = max_depth;
	this->params.Nmin = Nmin;
	this->params.Ncutpoints = Ncutpoints;
	this->params.alpha = alpha;
	this->params.beta = beta;
	this->params.tau = tau;
	this->params.burnin = burnin;
	this->params.mtry = mtry;
	this->params.kap = kap;
	this->params.s = s;
	this->params.tau_kap = tau_kap;
	this->params.tau_s = tau_s;
	this->params.verbose = verbose;
	this->params.sampling_tau = sampling_tau;
	this->params.parallel = parallel;
	this->params.nthread = nthread;
	this->trees = vector< vector<tree>> (num_sweeps);
	this->no_split_penality =  no_split_penality;
	this->params.sample_weights =  sample_weights;

	// handling seed

	if(seed != 0){
	this->seed_flag = true; 
	this->seed = (size_t)seed;
	}else{
	this->seed_flag = false;
	this->seed = 0;
	}


	// Create trees
	for(size_t i = 0; i < num_sweeps;i++){
		this->trees[i]=  vector<tree>(num_trees); 
	}


	//   Initialize model
	// NORMAL
	// define model
	this->model = new NormalModel(kap, s, tau, alpha, beta, sampling_tau, tau_kap, tau_s);
	this->model->setNoSplitPenality(no_split_penality);

	//}

    return;
}



std::string XBARTcpp::_to_json(void){
  json j = get_forest_json(this->trees,this->y_mean);
  return j.dump();
}

// Private Helper Functions 

// Numpy 1D array to vec_d - std_vector of doubles
void XBARTcpp::np_to_vec_d(int n, double *a, vec_d &y_std){
	for (size_t i = 0; i < n; i++){
		y_std[i] = a[i];
	}
}

void XBARTcpp::np_to_col_major_vec(int n, int d, double *a, vec_d &x_std){
	for(size_t i = 0; i < n; i++){
		for(size_t j = 0; j < d; j++){
			size_t index = i * d + j;
			size_t index_std = j * n +i;
			x_std[index_std] = a[index];
		}
	}
}

void XBARTcpp::xinfo_to_np(matrix<double> x_std, double *arr){
	// Fill in array values from xinfo
	for(size_t i = 0 , n = (size_t) x_std[0].size(); i < n; i++){
		for(size_t j = 0, d = (size_t)x_std.size(); j < d; j++){
			size_t index = i * d + j;
			arr[index] = x_std[j][i];
		}
	}
	return;
}

void XBARTcpp::vec_d_to_np(vec_d &y_std, double *arr){
	// Fill in array values from vec_d
	for(size_t i = 0 , n = (size_t) y_std.size(); i < n; i++){
		arr[i] = y_std[i];
	}
	return;
}

void XBARTcpp::compute_Xorder(size_t n, size_t d, const vec_d &x_std_flat, matrix<size_t> &Xorder_std){
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

// Getters
void XBARTcpp::get_yhats(int size,double *arr){
  	xinfo_to_np(this->yhats_xinfo,arr);
}

void XBARTcpp::get_yhats_test(int size,double *arr){
  	xinfo_to_np(this->yhats_test_xinfo,arr);
}

void XBARTcpp::get_yhats_test_multinomial(int size,double *arr){
  	for(size_t i=0; i < size; i++){
		arr[i]=this->yhats_test_multinomial[i];
	}
}

void XBARTcpp::get_sigma_draw(int size, double *arr){
  	xinfo_to_np(this->sigma_draw_xinfo,arr);
}

void XBARTcpp::get_residuals(int size, double *arr){
  	vec_d_to_np(this->resid, arr);
}

void XBARTcpp::_get_importance(int size,double *arr){
  	for(size_t i = 0; i < size ; i++){
    	arr[i] = this->mtry_weight_current_tree[i];
  	}
}

void XBARTcpp::_predict(int n, int p, double *a){//,int size, double *arr){
  
	// Convert *a to col_major std::vector
	vec_d x_test_std_flat(n * p);
	XBARTcpp::np_to_col_major_vec(n, p, a, x_test_std_flat);

	// Initialize result
	ini_matrix(this->yhats_test_xinfo, n, this->params.num_sweeps);
	for(size_t i = 0; i < n; i++) {
		for(size_t j = 0; j <this->params.num_sweeps; j++) {
			this->yhats_test_xinfo[j][i]=0;
		}
	}

	// Convert column major vector to pointer
	const double *Xtestpointer = &x_test_std_flat[0];//&x_test_std[0][0];


	// Predict
	NormalModel *model = new NormalModel(); //(this->params.kap, this->params.s, this->params.tau, this->params.alpha, this->params.beta);

	model->predict_std(Xtestpointer, n, p, this->params.num_trees,this->params.num_sweeps,
		this->yhats_test_xinfo, this->trees);  // *trees2

	delete model;
}

void XBARTcpp::_predict_gp( int n, int d, double *a, int n_y, double *a_y, int n_t, int d_t, double *a_t, size_t p_cat, double theta, double tau){

  
	// Convert row major *a to column major std::vector
	vec_d x_std_flat(n * d);
	XBARTcpp::np_to_col_major_vec(n, d, a, x_std_flat);

	// Convert row major *a_t to column major std::vector
	vec_d xtest_std_flat(n_t * d_t);
	XBARTcpp::np_to_col_major_vec(n_t, d_t, a_t, xtest_std_flat);

	// Convert a_y to std::vector
	vec_d y_std(n);
	XBARTcpp::np_to_vec_d(n, a_y, y_std);
			
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
	XBARTcpp::compute_Xorder(n, d, x_std_flat, Xorder_std);

	// xtestorder containers
	matrix<size_t> Xtestorder_std;
	ini_xinfo_sizet(Xtestorder_std, n_t, d_t);
	XBARTcpp::compute_Xorder(n_t, d_t, xtest_std_flat, Xtestorder_std);

	// //max_depth_std container
	// matrix<size_t> max_depth_std;
	// ini_xinfo_sizet(max_depth_std, this->params.num_trees, this->params.num_sweeps);

	// // Fill with max Depth Value
	// for(size_t i = 0; i < this->params.num_trees; i++){
	// 	for(size_t j = 0;j < this->params.num_sweeps; j++){
	// 		max_depth_std[j][i] = this->params.max_depth;
	// 	}
	// }

	// // Cpp native objects to return
	// matrix<double>  yhats_xinfo; // Temp Change
	// ini_xinfo(yhats_xinfo, n_t, this->params.num_sweeps);

	// Temp Change
	// ini_xinfo(this->sigma_draw_xinfo, this->params.num_trees, this->params.num_sweeps);
	// this->mtry_weight_current_tree.resize(p);
	//ini_xinfo(this->split_count_all_tree, d, this->params.M); // initialize at 0
	
	double *ypointer = &a_y[0];
	double *Xpointer = &x_std_flat[0];
	double *Xtestpointer = &xtest_std_flat[0];

	// Initialize result
	ini_matrix(this->yhats_test_xinfo, n_t, this->params.num_sweeps);
	for(size_t i = 0; i < n_t; i++) {
		for(size_t j = 0; j <this->params.num_sweeps; j++) {
			this->yhats_test_xinfo[j][i]=0;
		}
	}

	std::vector<double> sigma_std(this->params.num_sweeps);
	for(size_t i = 0; i < this->params.num_sweeps; i++){
		sigma_std[i] = this->sigma_draw_xinfo[i][this->params.num_trees - 1];
	}

	// initialize X_struct
    std::vector<double> initial_theta(1, y_mean / (double)this->params.num_trees);
    std::unique_ptr<gp_struct> x_struct(new gp_struct(Xpointer, &y_std, n, Xorder_std, p_cat, d-p_cat, &initial_theta, sigma_std, this->params.num_trees));
	std::unique_ptr<gp_struct> xtest_struct(new gp_struct(Xtestpointer, &y_std, n_t, Xtestorder_std, p_cat, d-p_cat, &initial_theta, sigma_std, this->params.num_trees));
	x_struct->n_y = n;
    xtest_struct->n_y = n_t;

	std::vector<bool> active_var(d);
    std::fill(active_var.begin(), active_var.end(), false);

	// get residuals
    matrix<std::vector<double>> residuals;
    ini_matrix(residuals, this->params.num_trees, this->params.num_sweeps);
    for (size_t i = 0; i < this->params.num_sweeps; i++){
        for (size_t j = 0; j < this->params.num_trees; j++){
            residuals[i][j].resize(n);
            for (size_t k = 0; k < n; k++){
                residuals[i][j][k] =  this->resid[k + i * n + j * this->params.num_sweeps * n];
            }
        }
    }
    x_struct->set_resid(residuals);


	// mcmc loop
    for (size_t sweeps = 0; sweeps < this->params.num_sweeps; sweeps++)
    {
        for (size_t tree_ind = 0; tree_ind < this->params.num_trees; tree_ind++)
        {
            // cout << "sweeps = " << sweeps << ", tree_ind = " << tree_ind << endl;
            (this->trees)[sweeps][tree_ind].gp_predict_from_root(Xorder_std, x_struct, x_struct->X_counts, x_struct->X_num_unique, 
            Xtestorder_std, xtest_struct, xtest_struct->X_counts, xtest_struct->X_num_unique, 
            this->yhats_test_xinfo, active_var, p_cat, sweeps, tree_ind, theta, tau);

        }
    }

    x_struct.reset();
    xtest_struct.reset();
}



// void XBARTcpp::_predict_multinomial(int n, int d, double *a){//,int size, double *arr){
  
//   // Convert *a to col_major std::vector
//   vec_d x_test_std_flat(n*d);
//   XBARTcpp::np_to_col_major_vec(n,d,a,x_test_std_flat);

//   // Initialize result
//   ini_matrix(this->yhats_test_xinfo, n, this->params.N_sweeps);
//   for(size_t i = 0;i<n;i++)for(size_t j = 0;j<this->params.N_sweeps;j++) this->yhats_test_xinfo[j][i]=0;

//   // Convert column major vector to pointer
//   const double *Xtestpointer = &x_test_std_flat[0];//&x_test_std[0][0];
//   this->yhats_test_multinomial = vector<double>(this->params.N_sweeps * n * this->num_classes);
  
//   // define model
//   double tau_a = 1/this->params.tau + 0.5;
//   double tau_b = 1/this->params.tau;
//   std::vector<double> dummy_phi(n);
//   std::vector<size_t> dummy_y(n);
//   for(size_t i=0; i<n; ++i){
//     dummy_phi[i] = 1;
//     dummy_y[i] = 1;
//   }
//   LogitModel *model = new LogitModel(this->num_classes, tau_a, tau_b, this->params.alpha, 
//                                      this->params.beta, &dummy_y, &dummy_phi);

//   // Predict
//   model->predict_std(Xtestpointer,n,d,this->params.M,this->params.N_sweeps,
//         this->yhats_test_xinfo,this->trees,this->yhats_test_multinomial); 

//   delete model;
// }


void XBARTcpp::_fit(int n, int p, double *a, int n_y, double *a_y, size_t p_cat){
  
	// Convert row major *a to column major std::vector
	vec_d x_std_flat(n * p);
	XBARTcpp::np_to_col_major_vec(n, p, a, x_std_flat);

	// Convert a_y to std::vector
	vec_d y_std(n_y);
	XBARTcpp::np_to_vec_d(n_y, a_y, y_std);
			
	// Calculate y_mean
	double y_mean = 0.0;
	for (size_t i = 0; i < n; i++){
		y_mean = y_mean + y_std[i];
	}
	y_mean = y_mean/(double)n;
	this->y_mean = y_mean;
      
	// xorder containers
	matrix<size_t> Xorder_std;
	ini_xinfo_sizet(Xorder_std, n, p);
	XBARTcpp::compute_Xorder(n, p, x_std_flat, Xorder_std);

	//max_depth_std container
	matrix<size_t> max_depth_std;
	ini_xinfo_sizet(max_depth_std, this->params.num_trees, this->params.num_sweeps);

	// Fill with max Depth Value
	for(size_t i = 0; i < this->params.num_trees; i++){
		for(size_t j = 0;j < this->params.num_sweeps; j++){
			max_depth_std[j][i] = this->params.max_depth;
		}
	}

	// Cpp native objects to return
	matrix<double>  yhats_xinfo; // Temp Change
	ini_xinfo(yhats_xinfo, n, this->params.num_sweeps);

	// Temp Change
	ini_xinfo(this->sigma_draw_xinfo, this->params.num_trees, this->params.num_sweeps);
	this->mtry_weight_current_tree.resize(p);
	//ini_xinfo(this->split_count_all_tree, d, this->params.M); // initialize at 0
	double *ypointer = &a_y[0];
	double *Xpointer = &x_std_flat[0];

	// NORMAL
  
    // define model
    // NormalModel *model = new NormalModel(this->params.kap, this->params.s, this->params.tau, this->params.alpha, this->params.beta, 
	// 	this->params.sampling_tau, this->params.tau_kap, this->params.tau_s);
    // model->setNoSplitPenality(this->no_split_penality);

    // // //State settings
    std::vector<double> initial_theta(1, y_mean / (double)this->params.num_trees);
    std::unique_ptr<State> state(new NormalState(Xpointer, Xorder_std, n, p, this->params.num_trees, 
		p_cat, p - p_cat, this->seed_flag, this->seed, this->params.Nmin, this->params.Ncutpoints, 
		this->params.mtry, Xpointer, this->params.num_sweeps, this->params.sample_weights, 
		&y_std, 1.0, this->params.max_depth, y_mean, this->params.burnin, this->model->dim_residual, this->params.nthread, this->params.parallel)); //last input is nthread, need update


    // initialize X_struct
    std::unique_ptr<X_struct> x_struct(new X_struct(Xpointer, &y_std, n, Xorder_std, p_cat, p-p_cat, &initial_theta, this->params.num_trees));

	this->resid.resize(n * this->params.num_sweeps * this->params.num_trees);

    mcmc_loop(Xorder_std, this->params.verbose, sigma_draw_xinfo, this->trees, this->no_split_penality, state, this->model, x_struct, this->resid);

    this->mtry_weight_current_tree = state->mtry_weight_current_tree;

    // delete model;
    state.reset();
    x_struct.reset();

}
//  code for multinomial, set a separate fit funciton later.

//   }else if(this->model_num == 1){ // Multinomial
    
//     // define model
//   double tau_a = 1/this->params.tau + 0.5;
//   double tau_b = 1/this->params.tau;
//   std::vector<double> phi(n);
//   for(size_t i=0; i<n; ++i){
//     phi[i] = 1;
//   }
//   std::vector<size_t> y_size_t(n);
//   for(size_t i=0; i<n; ++i) y_size_t[i] = (size_t)y_std[i];
//   LogitModel *model = new LogitModel(this->num_classes, tau_a, tau_b, this->params.alpha, 
//                                      this->params.beta, &y_size_t, &phi);
//   model->setNoSplitPenality(no_split_penality);
  
//   //data
//   std::vector<double> initial_theta(this->num_classes, 1); 
//   std::unique_ptr<State> state(new State(Xpointer, Xorder_std, n, d, this->params.M, p_cat, d-p_cat, 
//   this->seed_flag, this->seed, this->params.Nmin, this->params.Ncutpoints, this->params.parallel, this->params.mtry, 
//   Xpointer, this->params.N_sweeps, this->params.sample_weights, &y_std, 1.0, 
//   this->params.max_depth_num, y_mean, this->params.burnin, model->dim_residual));
//   std::unique_ptr<X_struct> x_struct(new X_struct(Xpointer, &y_std, n, Xorder_std, p_cat, d-p_cat, &initial_theta, this->params.M));

//   std::vector< std::vector<double> > phi_samples;
//   ini_matrix(phi_samples, n, this->params.N_sweeps * this->params.M);

//   // fit
//   mcmc_loop_multinomial(Xorder_std,this->params.verbose, this->trees, this->no_split_penality, 
//                         state, model, x_struct, phi_samples);
//   this->mtry_weight_current_tree = state->mtry_weight_current_tree;

//   delete model;
//   state.reset();
//   x_struct.reset();

//   }else if(this->model_num == 2){ // Probit
//     // define model
//     ProbitClass *model = new ProbitClass(this->params.kap, this->params.s, this->params.tau, 
//                                           this->params.alpha, this->params.beta,y_std);
//     model->setNoSplitPenality(no_split_penality);

//     // // //State settings
//     std::vector<double> initial_theta(1, y_mean / (double)this->params.M);
//     std::unique_ptr<State> state(new NormalState(Xpointer, Xorder_std, n, d, this->params.M, p_cat, d-p_cat, 
//     this->seed_flag, this->seed, this->params.Nmin, this->params.Ncutpoints, this->params.parallel, this->params.mtry, 
//     Xpointer, this->params.N_sweeps, this->params.sample_weights, &y_std, 1.0, 
//     this->params.max_depth_num, y_mean, this->params.burnin, model->dim_residual));

//     // initialize X_struct
//     std::unique_ptr<X_struct> x_struct(new X_struct(Xpointer, &y_std, n, Xorder_std, p_cat, d-p_cat, &initial_theta, this->params.M));

//     mcmc_loop_probit(Xorder_std, this->params.verbose, sigma_draw_xinfo, this->trees, this->no_split_penality, 
//     state, model,x_struct);

//     this->mtry_weight_current_tree = state->mtry_weight_current_tree;

//     delete model;
//     state.reset();
//     x_struct.reset();
//   }
// }    
