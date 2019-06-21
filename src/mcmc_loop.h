#include <ctime>
#include "tree.h"
#include "forest.h"
#include <chrono>
#include "model.h"
#include "node_data.h"
#include "state.h"
#include "cdf.h"
#include "X_struct.h"

void mcmc_loop(xinfo_sizet &Xorder_std, xinfo_sizet &max_depth_std, size_t burnin, bool verbose, xinfo &yhats_xinfo, xinfo &sigma_draw_xinfo, vector<vector<tree>> &trees, double no_split_penality, std::unique_ptr<State> &state, NormalModel *model, std::unique_ptr<X_struct> &x_struct);

void predict_std(const double *Xtestpointer, size_t N_test, size_t p, size_t num_trees, size_t num_sweeps, xinfo &yhats_test_xinfo, vector<vector<tree>> &trees);

void predict_std_multinomial(const double *Xtestpointer, size_t N_test, size_t p, size_t num_trees, size_t num_sweeps, xinfo &yhats_test_xinfo, vector<vector<tree>> &trees);

void mcmc_loop_clt(xinfo_sizet &Xorder_std, xinfo_sizet &max_depth_std, size_t burnin, bool verbose, xinfo &yhats_xinfo, xinfo &sigma_draw_xinfo, vector<vector<tree>> &trees, double no_split_penality, std::unique_ptr<State> &state, CLTClass *model, std::unique_ptr<X_struct> &x_struct);

void mcmc_loop_multinomial(xinfo_sizet &Xorder_std, xinfo_sizet &max_depth_std, size_t burnin, bool verbose, xinfo &yhats_xinfo, xinfo &sigma_draw_xinfo, vector<vector<tree>> &trees, double no_split_penality, std::unique_ptr<State> &state, LogitClass *model, std::unique_ptr<X_struct> &x_struct);

void mcmc_loop_probit(xinfo_sizet &Xorder_std, xinfo_sizet &max_depth_std, size_t burnin, bool verbose, xinfo &yhats_xinfo, xinfo &sigma_draw_xinfo, vector<vector<tree>> &trees, double no_split_penality, std::unique_ptr<State> &state, NormalModel *model, std::unique_ptr<X_struct> &x_struct);

void mcmc_loop_MH(xinfo_sizet &Xorder_std, xinfo_sizet &max_depth_std, size_t burnin, bool verbose, xinfo &yhats_xinfo, xinfo &sigma_draw_xinfo, vector<vector<tree>> &trees, double no_split_penality, std::unique_ptr<State> &state, NormalModel *model, std::unique_ptr<X_struct> &x_struct, std::vector<double> &accept_count, std::vector<double> &MH_vector, std::vector<double> &P_ratio, std::vector<double> &Q_ratio, std::vector<double> &prior_ratio);
