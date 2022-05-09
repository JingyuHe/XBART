#include <iostream>
#include <vector>
#include <mcmc_loop.h>
#include <json_io.h>
#include <model.h>

struct XBARTcppParams
{
	size_t M;
	size_t N_sweeps;
	size_t Nmin;
	size_t Ncutpoints;
	size_t burnin;
	size_t mtry;
	size_t max_depth_num;
	double alpha;
	double beta;
	double tau;
	double kap;
	double s;
	bool verbose;
	bool parallel;
	int seed;
	bool sample_weights_flag;
};

class XBARTcpp
{
private:
	XBARTcppParams params;
	vector<vector<tree>> trees;
	double y_mean;
	size_t n_train;
	size_t n_test;
	size_t d;
	matrix<double> yhats_xinfo;
	matrix<double>  yhats_test_xinfo;
	matrix<double>  sigma_draw_xinfo;
	vec_d mtry_weight_current_tree;

	// multinomial
	vec_d yhats_test_multinomial;
	size_t num_classes;
	//xinfo split_count_all_tree;

	// helper functions
	void np_to_vec_d(int n, double *a, vec_d &y_std);
	void np_to_col_major_vec(int n, int d, double *a, vec_d &x_std);
	void xinfo_to_np(matrix<double>  x_std, double *arr);
	void compute_Xorder(size_t n, size_t d, const vec_d &x_std_flat, matrix<size_t> &Xorder_std);
	size_t seed;
	bool seed_flag;
	size_t model_num; // 0 : normal, 1 : multinomial; 2 : probit
	double no_split_penality;

public:
	// Constructors
	XBARTcpp(XBARTcppParams params);
	XBARTcpp(size_t M, size_t N_sweeps,
			 size_t Nmin, size_t Ncutpoints,		//CHANGE
			 double alpha, double beta, double tau, //CHANGE!
			 size_t burnin, size_t mtry,
			 size_t max_depth_num, double kap,
			 double s, bool verbose,
			 bool parallel, int seed, size_t model_num, double no_split_penality, bool sample_weights_flag, size_t num_classes);

	XBARTcpp(std::string json_string);

	std::string _to_json(void);

	void _fit(int n, int d, double *a, // Train X
			  int n_y, double *a_y, size_t p_cat);
	void _predict(int n, int d, double *a); //,int size, double *arr);
	void _predict_multinomial(int n, int d, double *a); //,int size, double *arr);

	// Getters
	int get_M(void);
	int get_N_sweeps(void) { return ((int)params.N_sweeps); };
	int get_burnin(void) { return ((int)params.burnin); };
	void get_yhats(int size, double *arr);
	void get_yhats_test(int size, double *arr);
	void get_yhats_test_multinomial(int size,double *arr);
	void get_sigma_draw(int size, double *arr);
	void _get_importance(int size, double *arr);
};
