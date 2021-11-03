#include <iostream>
#include <vector>
#include <mcmc_loop.h>
#include <json_io.h>
#include <model.h>

struct XBARTcppParams
{
	size_t num_trees;
	size_t num_sweeps;
	size_t max_depth;
	size_t Nmin;
	size_t Ncutpoints;
	size_t burnin;
	size_t mtry;
	double alpha;
	double beta;
	double tau;
	double kap;
	double s;
	double tau_kap;
	double tau_s;
	bool verbose;
	bool sampling_tau;
	bool parallel;
	size_t nthread;
	int seed;
	bool sample_weights_flag;
};

class XBARTcpp
{
public:
	XBARTcppParams params;
	vector<vector<tree>> trees;
	double y_mean;
	size_t n_train;
	size_t n_test;
	size_t p;
	matrix<double> yhats_xinfo;
	matrix<double>  yhats_test_xinfo;
	matrix<double>  sigma_draw_xinfo;
	vec_d mtry_weight_current_tree;

	NormalModel *model;

	// multinomial
	vec_d yhats_test_multinomial;
	size_t num_classes;
	//xinfo split_count_all_tree;

	std::vector<double> resid;

	// Constructors
	XBARTcpp(XBARTcppParams params);
	XBARTcpp(std::string json_string);
	XBARTcpp(size_t num_trees, size_t num_sweeps, size_t max_depth,
			 size_t Nmin, size_t Ncutpoints,		//CHANGE
			 double alpha, double beta, double tau, //CHANGE!
			 size_t burnin, size_t mtry,
			 double kap, double s, double tau_kap, double tau_s, 
			 bool verbose, bool sampling_tau, bool parallel, size_t nthread,
			 bool set_random_seed, int seed, double no_split_penality, bool sample_weights_flag);

	void _fit(int n, int d, double *a, int n_y, double *a_y, size_t p_cat);
	void _predict(int n, int d, double *a); //,int size, double *arr);
	void _gp_predict(int n, int p, double *a, double *a_y, int n_t, double *a_t, size_t p_cat);
  

	// helper functions
	void np_to_vec_d(int n, double *a, vec_d &y_std);
	void np_to_col_major_vec(int n, int d, double *a, vec_d &x_std);
	void xinfo_to_np(matrix<double>  x_std, double *arr);
	void compute_Xorder(size_t n, size_t d, const vec_d &x_std_flat, matrix<size_t> &Xorder_std);
	size_t seed;
	bool seed_flag;
	double no_split_penality;

	std::string _to_json(void);

	// Getters
	int get_num_trees(void) { return ((int)params.num_trees); };;
	int get_num_sweeps(void) { return ((int)params.num_sweeps); };
	int get_burnin(void) { return ((int)params.burnin); };
	void get_yhats(int size, double *arr);
	void get_yhats_test(int size, double *arr);
	void get_yhats_test_multinomial(int size,double *arr);
	void get_sigma_draw(int size, double *arr);
	void get_residuals(double *arr);
	void _get_importance(int size, double *arr);
};
