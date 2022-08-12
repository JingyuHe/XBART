#include <ctime>
#include "tree.h"
#include <chrono>
#include "model.h"
#include "xbcf_model.h"
#include "node_data.h"
#include "state.h"
#include "cdf.h"
#include "X_struct.h"
//#include "MH.h"

void mcmc_loop_xbcf(matrix<size_t> &Xorder_std, matrix<size_t> &Xorder_tau_std,
                    const double *X_std, const double *X_tau_std,
                    bool verbose,
                    matrix<double> &sigma0_draw_xinfo,
                    matrix<double> &sigma1_draw_xinfo,
                    matrix<double> &b_xinfo,
                    matrix<double> &a_xinfo,
                    // matrix<double> &b0_draw_xinfo,
                    // matrix<double> &b1_draw_xinfo,
                    // matrix<double> &total_fit,
                    vector<vector<tree>> &trees_ps,
                    vector<vector<tree>> &trees_trt,
                    double no_split_penality,
                    std::unique_ptr<State> &state,
                    //std::unique_ptr<State> &state_trt,
                    xbcfModel *model_ps,
                    xbcfModel *model_trt,
                    std::unique_ptr<X_struct> &x_struct_ps,
                    std::unique_ptr<X_struct> &x_struct_trt,
                    bool a_scaling,
                    bool b_scaling);