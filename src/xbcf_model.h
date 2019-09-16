
#ifndef xbcfmodel_h
#define xbcfmodel_h

#include "model.h"

using namespace std;

class tree;

//////////////////////////////////////////////////////////////////////////////////////
//
//
//  XBCF Model ||| a copy of Normal model atm
//
//
//////////////////////////////////////////////////////////////////////////////////////

class xbcfModel : public Model
{
public:
  size_t dim_suffstat = 4;

  // model prior
  // prior on sigma
  double kap;
  double s;
  // prior on leaf parameter
  double tau;

  xbcfModel(double kap, double s, double tau, double alpha, double beta) : Model(1, 4)
  {
    this->kap = kap;
    this->s = s;
    this->tau = tau;
    this->alpha = alpha;
    this->beta = beta;
    this->dim_residual = 1;
  }

  xbcfModel() : Model(1, 4) {}

  Model *clone() { return new xbcfModel(*this); }

  void incSuffStat(matrix<double> &residual_std, size_t index_next_obs, std::vector<double> &suffstats);

  void samplePars(std::unique_ptr<State> &state, std::vector<double> &suff_stat, std::vector<double> &theta_vector, double &prob_leaf);

  void update_state(std::unique_ptr<State> &state, size_t tree_ind, std::unique_ptr<X_struct> &x_struct);

  void initialize_root_suffstat(std::unique_ptr<State> &state, std::vector<double> &suff_stat);

  void updateNodeSuffStat(std::vector<double> &suff_stat, matrix<double> &residual_std, matrix<size_t> &Xorder_std, size_t &split_var, size_t row_ind);

  void calculateOtherSideSuffStat(std::vector<double> &parent_suff_stat, std::vector<double> &lchild_suff_stat, std::vector<double> &rchild_suff_stat, size_t &N_parent, size_t &N_left, size_t &N_right, bool &compute_left_side);

  void state_sweep(size_t tree_ind, size_t M, matrix<double> &residual_std, std::unique_ptr<X_struct> &x_struct) const;

  double likelihood(std::vector<double> &temp_suff_stat, std::vector<double> &suff_stat_all, size_t N_left, bool left_side, bool no_split, std::unique_ptr<State> &state) const;

  // double likelihood_no_split(std::vector<double> &suff_stat, std::unique_ptr<State> &state) const;

  void ini_residual_std(std::unique_ptr<State> &state);

  void transfer_residual_std(std::unique_ptr<State> &state_ps, std::unique_ptr<State> &state_trt);

  void predict_std(const double *Xtestpointer, size_t N_test, size_t p, size_t num_trees, size_t num_sweeps, matrix<double> &yhats_test_xinfo, vector<vector<tree>> &trees);
};

#endif