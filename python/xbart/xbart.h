//#ifndef SWIG
#include <iostream>
#include <vector>
//#endif
#include <fit_std_main_loop.h>
#include <json_io.h>

// #include <cstddef>
// #include <armadillo>
// #include "../src/python_train_forest_std_mtrywithinnode.h"
//double vector
typedef std::vector<double> vec_d; 

//vector of vectors, will be split rules    
typedef std::vector<vec_d> xinfo;      



struct XBARTcppParams{
			size_t M;size_t N_sweeps; size_t Nmin; size_t Ncutpoints;
			size_t burnin; size_t mtry;size_t max_depth_num;
			double alpha;double beta;double tau;double kap;double s;
			bool verbose; bool draw_mu;bool parallel;int seed;bool sample_weights_flag;
};

class XBARTcpp{
	private:
		XBARTcppParams params;
		vector<vector<tree>> trees; 
		double y_mean;
		size_t n_train; size_t n_test; size_t d;
		xinfo yhats_xinfo; xinfo yhats_test_xinfo; xinfo sigma_draw_xinfo; 
		vec_d mtry_weight_current_tree;
		//xinfo split_count_all_tree;
		
		// helper functions
		void np_to_vec_d(int n,double *a,vec_d &y_std);
		void np_to_col_major_vec(int n, int d,double *a,vec_d &x_std);
		void xinfo_to_np(xinfo x_std,double *arr);
		void compute_Xorder(size_t n, size_t d, const vec_d &x_std_flat,xinfo_sizet & Xorder_std);
		size_t seed; bool seed_flag;
		size_t model_num; // 0 : normal, 1 : clt
		double no_split_penality;

		//= forest(10);
	
		// void params_to_struct;
	public:
		// Constructors 
		XBARTcpp (XBARTcppParams params);
		XBARTcpp (size_t M,size_t N_sweeps ,
				size_t Nmin , size_t Ncutpoints , //CHANGE 
				double alpha , double beta , double tau , //CHANGE!
				size_t burnin, size_t mtry ,
				size_t max_depth_num , double kap , 
				double s , bool verbose , 
				bool draw_mu , bool parallel,int seed,size_t model_num,double no_split_penality,bool sample_weights_flag);

		XBARTcpp(std::string json_string);

		std::string _to_json(void);
	
		void _fit(int n,int d,double *a, // Train X 
      		int n_y,double *a_y, size_t p_cat);
		void _predict(int n, int d, double *a);//,int size, double *arr);



		// Getters
		int get_M (void);
		int get_N_sweeps(void){return((int)params.N_sweeps);};
		int get_burnin(void){return((int)params.burnin);};
		void get_yhats(int size, double *arr);
		void get_yhats_test(int size, double *arr);
		void get_sigma_draw(int size, double *arr);
		void _get_importance(int size, double *arr);
};
