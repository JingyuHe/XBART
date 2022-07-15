#include <ctime>
#include "tree.h"
#include <chrono>
#include "model.h"
#include "state.h"
#include "cdf.h"
#include "X_struct.h"

//////////////////////////////////////////////////////////////////////////////////////
// main function of the Bayesian backfitting algorithm
//////////////////////////////////////////////////////////////////////////////////////

// normal regression model
void mcmc_loop(matrix<size_t> &Xorder_std, bool verbose, matrix<double> &sigma_draw_xinfo, vector<vector<tree>> &trees, double no_split_penalty, std::unique_ptr<State> &state, NormalModel *model, std::unique_ptr<X_struct> &x_struct, std::vector<double> &resid);

// classification, all classes share the same tree structure
void mcmc_loop_multinomial(matrix<size_t> &Xorder_std, bool verbose, vector<vector<tree>> &trees, double no_split_penality, std::unique_ptr<State> &state, LogitModel *model, std::unique_ptr<X_struct> &x_struct, std::vector<std::vector<double>> &weight_samples, std::vector<double> &lambda_samples, std::vector<std::vector<double>> &tau_samples);

// classification, each class has its own tree structure
void mcmc_loop_multinomial_sample_per_tree(matrix<size_t> &Xorder_std, bool verbose, vector<vector<vector<tree>>> &trees, double no_split_penality, std::unique_ptr<State> &state, LogitModelSeparateTrees *model, std::unique_ptr<X_struct> &x_struct, std::vector<std::vector<double>> &weight_samples);

// XBCF for continuous treatment
void mcmc_loop_XBCF_continuous(matrix<size_t> &Xorder_std, bool verbose, matrix<double> &sigma_draw_xinfo, vector<vector<tree>> &trees_ps, vector<vector<tree>> &trees_trt, double no_split_penalty, std::unique_ptr<State> &state, NormalLinearModel *model, std::unique_ptr<X_struct> &x_struct_ps, std::unique_ptr<X_struct> &x_struct_trt);
