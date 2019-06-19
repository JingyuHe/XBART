#include <ctime>
#include "tree.h"
#include "forest.h"
#include <chrono>
#include "model.h"
#include "node_data.h"
#include "state.h"
#include "cdf.h"

void fit_std(std::vector<double> &y_std, double y_mean, xinfo_sizet &Xorder_std, xinfo_sizet &max_depth_std, size_t burnin, bool verbose, xinfo &yhats_xinfo, xinfo &sigma_draw_xinfo, vec_d &mtry_weight_current_tree, vector<vector<tree>> &trees, double no_split_penality, std::unique_ptr<State> &state, NormalModel *model);

void predict_std(const double *Xtestpointer, size_t N_test, size_t p, size_t num_trees, size_t num_sweeps, xinfo &yhats_test_xinfo, vector<vector<tree>> &trees, double y_mean);

void predict_std_multinomial(const double *Xtestpointer, size_t N_test, size_t p, size_t num_trees, size_t num_sweeps, xinfo &yhats_test_xinfo, vector<vector<tree>> &trees, double y_mean);

void fit_std_clt(std::vector<double> &y_std, double y_mean, xinfo_sizet &Xorder_std, xinfo_sizet &max_depth_std, size_t burnin, bool verbose, xinfo &yhats_xinfo, xinfo &sigma_draw_xinfo, vec_d &mtry_weight_current_tree, vector<vector<tree>> &trees, double no_split_penality, std::unique_ptr<State> &state, CLTClass *model);

void fit_std_multinomial(std::vector<double> &y_std, double y_mean, xinfo_sizet &Xorder_std, xinfo_sizet &max_depth_std, size_t burnin, bool verbose, xinfo &yhats_xinfo, xinfo &sigma_draw_xinfo, vec_d &mtry_weight_current_tree, vector<vector<tree>> &trees, double no_split_penality, std::unique_ptr<State> &state, LogitClass *model);

void fit_std_probit(std::vector<double> &y_std, double y_mean, xinfo_sizet &Xorder_std, xinfo_sizet &max_depth_std, size_t burnin, bool verbose, xinfo &yhats_xinfo, xinfo &sigma_draw_xinfo, vec_d &mtry_weight_current_tree, vector<vector<tree>> &trees, double no_split_penality, std::unique_ptr<State> &state, NormalModel *model);

void fit_std_MH(std::vector<double> &y_std, double y_mean, xinfo_sizet &Xorder_std, xinfo_sizet &max_depth_std, size_t burnin, bool verbose, xinfo &yhats_xinfo, xinfo &sigma_draw_xinfo, vec_d &mtry_weight_current_tree, vector<vector<tree>> &trees, double no_split_penality, std::unique_ptr<State> &state, NormalModel *model, std::vector<double>& accept_count, std::vector<double>& MH_vector, std::vector<double>& P_ratio, std::vector<double>& Q_ratio, std::vector<double>& prior_ratio);

