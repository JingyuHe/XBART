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
void mcmc_loop(matrix<size_t> &Xorder_std, bool verbose, matrix<double> &sigma_draw_xinfo, vector<vector<tree>> &trees, double no_split_penalty, State &state, NormalModel *model, X_struct &x_struct, std::vector<double> &resid);

// classification, all classes share the same tree structure
void mcmc_loop_multinomial(matrix<size_t> &Xorder_std, bool verbose, vector<vector<tree>> &trees, double no_split_penalty, State &state, LogitModel *model, X_struct &x_struct,
std::vector<std::vector<double>> &weight_samples, std::vector<double> &lambda_samples, std::vector<std::vector<double>> &phi_samples, std::vector<std::vector<double>> &logloss,
std::vector<std::vector<double>> &tree_size);

// classification, each class has its own tree structure
void mcmc_loop_multinomial_sample_per_tree(matrix<size_t> &Xorder_std, bool verbose, vector<vector<vector<tree>>> &trees, double no_split_penalty, State &state,
LogitModelSeparateTrees *model, X_struct &x_struct, std::vector<std::vector<double>> &weight_samples, std::vector<std::vector<double>> &tau_samples, std::vector<std::vector<double>> &logloss,
std::vector<std::vector<double>> &tree_size);

// XBCF for continuous treatment
void mcmc_loop_xbcf_continuous(matrix<size_t> &Xorder_std_con, matrix<size_t> &Xorder_std_mod, bool verbose, matrix<double> &sigma_draw_xinfo, vector<vector<tree>> &trees_con, vector<vector<tree>> &trees_mod, double no_split_penalty, State &state, XBCFContinuousModel *model, X_struct &x_struct_con, X_struct &x_struct_mod);

// XBCF for discrete treatment
void mcmc_loop_xbcf_discrete(matrix<size_t> &Xorder_std_con, matrix<size_t> &Xorder_std_mod, bool verbose, matrix<double> &sigma0_draw_xinfo, matrix<double> &sigma1_draw_xinfo, matrix<double> &a_xinfo, matrix<double> &b_xinfo, vector<vector<tree>> &trees_con, vector<vector<tree>> &trees_mod, double no_split_penalty, State &state, XBCFDiscreteModel *model, X_struct &x_struct_con, X_struct &x_struct_mod);
