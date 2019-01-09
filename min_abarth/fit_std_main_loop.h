#ifndef SWIG
#include <ctime>
#include <chrono>
#endif
#include "tree.h"
#include "treefuns.h"
#include "forest.h"


void fit_std_main_loop(const double *Xpointer,std::vector<double> &y_std,
					double& y_mean,const double *Xtestpointer, xinfo_sizet &Xorder_std,
                size_t N,size_t p,size_t N_test,
                size_t M, size_t L, size_t N_sweeps, xinfo_sizet &max_depth_std, 
                size_t Nmin, size_t Ncutpoints, double alpha, double beta, 
                double tau, size_t burnin, size_t mtry, 
                bool draw_sigma , double kap , double s, 
                bool verbose, bool m_update_sigma, 
                bool draw_mu, bool parallel,
                xinfo &yhats_xinfo,xinfo &yhats_test_xinfo,xinfo &sigma_draw_xinfo);


void fit_std_main_loop_all(const double *Xpointer,std::vector<double> &y_std,double& y_mean,const double *Xtestpointer, xinfo_sizet &Xorder_std,
                size_t N,size_t p,size_t N_test,
                size_t M, size_t L, size_t N_sweeps, xinfo_sizet &max_depth_std, 
                size_t Nmin, size_t Ncutpoints, double alpha, double beta, 
                double tau, size_t burnin, size_t mtry, 
                bool draw_sigma , double kap , double s, 
                bool verbose, bool m_update_sigma, 
                bool draw_mu, bool parallel,
                xinfo &yhats_xinfo,xinfo &yhats_test_xinfo,xinfo &sigma_draw_xinfo,size_t p_categorical,size_t p_continuous);