#include <iostream>
#include <vector>
#include <cstddef>
#include <armadillo>
#include "../src/python_train_forest_std_mtrywithinnode.h"
//double vector
typedef std::vector<double> vec_d; 

//vector of vectors, will be split rules    
typedef std::vector<vec_d> xinfo;      



struct AbarthParams{
			size_t M;
			size_t L;size_t N_sweeps; size_t Nmin; size_t Ncutpoints;
			size_t burnin; size_t mtry;size_t max_depth_num;
			double alpha;double beta;double tau;double kap;double s;
			bool draw_sigma;bool verbose; bool m_update_sigma;
			bool draw_mu;bool parallel;
};

class Abarth{
	private:
		AbarthParams params;
		vec_d y_std;

		// helper functions
		vec_d np_to_vec_d(int n,double *a);
		xinfo np_to_xinfo(int n, int d,double *a);
		void xinfo_to_np(xinfo x_std,double *arr);

	
		// void params_to_struct;
	public:
		// Constructors 
		Abarth (AbarthParams params);
		Abarth (size_t M ,size_t L ,size_t N_sweeps ,
				size_t Nmin , size_t Ncutpoints , //CHANGE 
				double alpha , double beta , double tau , //CHANGE!
				size_t burnin, 
				size_t max_depth_num , bool draw_sigma , double kap , 
				double s , bool verbose , bool m_update_sigma, 
				bool draw_mu , bool parallel);

		// Destructor
		~Abarth();

		// Public Functions 
		int get_M (void);
		double fit(int n, double *a);
		double fit_x(int n, int d, double *a);
		void predict(int n,double *a,int size, double *arr);
		void __predict_2d(int n,int d,double *a,int size, double *arr);

		void fit_predict(int n,int d,double *a, // Train X 
			int n_y,double *a_y, // Train Y
			int n_test,int d_test,double *a_test, // Test X
			int size, double *arr); // Result 

};
