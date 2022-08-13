
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

  void incSuffStat(State &state, size_t index_next_obs, std::vector<double> &suffstats);

  void samplePars(State &state, std::vector<double> &suff_stat, std::vector<double> &theta_vector, double &prob_leaf);

  void draw_sigma(State &state, size_t ind);

  void initialize_root_suffstat(State &state, std::vector<double> &suff_stat);

  void updateNodeSuffStat(std::vector<double> &suff_stat, State &state, matrix<size_t> &Xorder_std, size_t &split_var, size_t row_ind);

  void calculateOtherSideSuffStat(std::vector<double> &parent_suff_stat, std::vector<double> &lchild_suff_stat, std::vector<double> &rchild_suff_stat, size_t &N_parent, size_t &N_left, size_t &N_right, bool &compute_left_side);

  void state_sweep(size_t tree_ind, std::vector<double> &fit, X_struct &x_struct) const;

  void update_xinfo(matrix<double> &yhats_xinfo, size_t sweep_num, size_t num_trees, size_t N, X_struct &x_struct);

  double likelihood(std::vector<double> &temp_suff_stat, std::vector<double> &suff_stat_all, size_t N_left, bool left_side, bool no_split, State &state) const;

  void predict_std(const double *Xtestpointer, size_t N_test, size_t p, size_t num_sweeps, matrix<double> &yhats_test_xinfo, vector<vector<tree>> &trees);

  void update_a_value(State &state);

  void update_b_values(State &state);

  void subtract_old_tree_fit(size_t tree_ind, std::vector<double> &fit, X_struct &x_struct);

  void set_state_status(State &state, size_t value, const double *X, matrix<size_t> &Xorder);
};

#endif