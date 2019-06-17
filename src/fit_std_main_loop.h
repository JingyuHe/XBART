#include <ctime>
#include "tree.h"
#include "forest.h"
#include <chrono>
#include "model.h"
#include "fit_info.h"
#include "cdf.h"
#include "node_data.h"

void fit_std(std::vector<double> &y_std, double y_mean, xinfo_sizet &Xorder_std, xinfo_sizet &max_depth_std, size_t burnin, bool verbose, xinfo &yhats_xinfo, xinfo &sigma_draw_xinfo, vec_d &mtry_weight_current_tree, vector<vector<tree>> &trees, double no_split_penality, Prior &prior, std::unique_ptr<FitInfo> &fit_info, Model *model, NodeData *node_data);

void predict_std(const double *Xtestpointer, size_t N_test, size_t p, size_t num_trees, size_t num_sweeps, xinfo &yhats_test_xinfo, vector<vector<tree>> &trees, double y_mean);

void predict_std_multinomial(const double *Xtestpointer, size_t N_test, size_t p, size_t num_trees, size_t num_sweeps, xinfo &yhats_test_xinfo, vector<vector<tree>> &trees, double y_mean);

void fit_std_clt(std::vector<double> &y_std, double y_mean, xinfo_sizet &Xorder_std, xinfo_sizet &max_depth_std, size_t burnin, bool verbose, xinfo &yhats_xinfo, xinfo &sigma_draw_xinfo, vec_d &mtry_weight_current_tree, vector<vector<tree>> &trees, double no_split_penality, Prior &prior, std::unique_ptr<FitInfo> &fit_info, CLTClass *model);

void fit_std_multinomial(std::vector<double> &y_std, double y_mean, xinfo_sizet &Xorder_std, xinfo_sizet &max_depth_std, size_t burnin, bool verbose, xinfo &yhats_xinfo, xinfo &sigma_draw_xinfo, vec_d &mtry_weight_current_tree, vector<vector<tree>> &trees, double no_split_penality, Prior &prior, std::unique_ptr<FitInfo> &fit_info, LogitClass *model);

void fit_std_probit(std::vector<double> &y_std, double y_mean, xinfo_sizet &Xorder_std, xinfo_sizet &max_depth_std, size_t burnin, bool verbose, xinfo &yhats_xinfo, xinfo &sigma_draw_xinfo, vec_d &mtry_weight_current_tree, vector<vector<tree>> &trees, double no_split_penality, Prior &prior, std::unique_ptr<FitInfo> &fit_info, Model *model);

void fit_std_MH(std::vector<double> &y_std, double y_mean, xinfo_sizet &Xorder_std, xinfo_sizet &max_depth_std, size_t burnin, bool verbose, xinfo &yhats_xinfo, xinfo &sigma_draw_xinfo, vec_d &mtry_weight_current_tree, vector<vector<tree>> &trees, double no_split_penality, Prior &prior, std::unique_ptr<FitInfo> &fit_info, Model *model, std::vector<double>& accept_count, std::vector<double>& MH_vector, std::vector<double>& P_ratio, std::vector<double>& Q_ratio, std::vector<double>& prior_ratio);

