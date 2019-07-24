#include <ctime>
#include "tree.h"
#include "forest.h"
#include <chrono>
#include "model.h"
#include "node_data.h"
#include "state.h"
#include "cdf.h"
#include "X_struct.h"
//#include "MH.h"

void mcmc_loop(matrix<size_t> &Xorder_std, bool verbose, matrix<double> &sigma_draw_xinfo, vector<vector<tree>> &trees, double no_split_penality, std::unique_ptr<State> &state, NormalModel *model, std::unique_ptr<X_struct> &x_struct);

// void predict_std_multinomial(const double *Xtestpointer, size_t N_test, size_t p, size_t num_trees, size_t num_sweeps, matrix<double> &yhats_test_xinfo, vector<vector<tree>> &trees);

void mcmc_loop_clt(matrix<size_t> &Xorder_std, bool verbose, matrix<double> &sigma_draw_xinfo, vector<vector<tree>> &trees, double no_split_penality, std::unique_ptr<State> &state, CLTClass *model, std::unique_ptr<X_struct> &x_struct);

void mcmc_loop_multinomial(matrix<size_t> &Xorder_std, bool verbose, matrix<double> &sigma_draw_xinfo, vector<vector<tree>> &trees, double no_split_penality, std::unique_ptr<State> &state, LogitClass *model, std::unique_ptr<X_struct> &x_struct);

void mcmc_loop_probit(matrix<size_t> &Xorder_std, bool verbose, matrix<double> &sigma_draw_xinfo, vector<vector<tree>> &trees, double no_split_penality, std::unique_ptr<State> &state, ProbitClass *model, std::unique_ptr<X_struct> &x_struct);

//void mcmc_loop_MH(matrix<size_t> &Xorder_std, bool verbose, matrix<double> &sigma_draw_xinfo, vector<vector<tree>> &trees, double no_split_penality, std::unique_ptr<State> &state, NormalModel *model, std::unique_ptr<X_struct> &x_struct, std::vector<double> &accept_count, std::vector<double> &MH_vector, std::vector<double> &P_ratio, std::vector<double> &Q_ratio, std::vector<double> &prior_ratio);
