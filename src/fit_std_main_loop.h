#include <ctime>
#include "tree.h"
#include "forest.h"
#include <chrono>
#include "model.h"


void fit_std_main_loop_all(const double *Xpointer, std::vector<double> &y_std, double &y_mean, const double *Xtestpointer, xinfo_sizet &Xorder_std, size_t N, size_t p, size_t N_test, size_t num_trees, size_t num_sweeps, xinfo_sizet &max_depth_std, size_t n_min, size_t Ncutpoints, double alpha, double beta, double tau, size_t burnin, size_t mtry, double kap, double s, bool verbose, bool draw_mu, bool parallel, xinfo &yhats_xinfo, xinfo &yhats_test_xinfo, xinfo &sigma_draw_xinfo, xinfo &split_count_all_tree, size_t p_categorical, size_t p_continuous, vector<vector<tree>> &trees, bool set_random_seed, size_t random_seed);

void fit_std(const double *Xpointer, std::vector<double> &y_std, double y_mean, xinfo_sizet &Xorder_std, size_t N, size_t p, size_t num_trees, size_t num_sweeps, xinfo_sizet &max_depth_std, size_t n_min, size_t Ncutpoints, double alpha, double beta, double tau, size_t burnin, size_t mtry, double kap, double s, bool verbose, bool draw_mu, bool parallel, xinfo &yhats_xinfo, xinfo &sigma_draw_xinfo, size_t p_categorical, size_t p_continuous, vector<vector<tree>> &trees, bool set_random_seed, size_t random_seed);

void predict_std(const double *Xtestpointer, size_t N_test, size_t p, size_t num_trees, size_t num_sweeps, xinfo &yhats_test_xinfo, vector<vector<tree>> &trees, double y_mean);
