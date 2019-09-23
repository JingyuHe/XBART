#include "tree.h"
#include "xbcf_model.h"

//////////////////////////////////////////////////////////////////////////////////////
//
//
//  XBCF Model ||| a copy of Normal Model atm
//
//
//////////////////////////////////////////////////////////////////////////////////////

// called in tree.cpp
// called from calcSuffStat_categorical, calcSuffStat_continuous
// adds residual to suff stats for normalmodel (updates the suff stats by adding new values to the old ones)
void xbcfModel::incSuffStat(matrix<double> &residual_std, size_t index_next_obs, std::vector<double> &suffstats)
{
  // I have to pass matrix<double> &residual_std, size_t index_next_obs
  // which allows more flexibility for multidimensional residual_std

  suffstats[0] += residual_std[0][index_next_obs];
  return;
}

// called in tree.cpp
// called from GFR
// samples theta for leafs (does it depend on the task or it's really just a matter of prior choice?)
void xbcfModel::samplePars(std::unique_ptr<State> &state, std::vector<double> &suff_stat, std::vector<double> &theta_vector, double &prob_leaf)
{
  std::normal_distribution<double> normal_samp(0.0, 1.0);

  // test result should be theta
  theta_vector[0] = suff_stat[0] / pow(state->sigma, 2) / (1.0 / tau + suff_stat[2] / pow(state->sigma, 2)) + sqrt(1.0 / (1.0 / tau + suff_stat[2] / pow(state->sigma, 2))) * normal_samp(state->gen); //Rcpp::rnorm(1, 0, 1)[0];//* as_scalar(arma::randn(1,1));

  // also update probability of leaf parameters
  prob_leaf = normal_density(theta_vector[0], suff_stat[0] / pow(state->sigma, 2) / (1.0 / tau + suff_stat[2] / pow(state->sigma, 2)), 1.0 / (1.0 / tau + suff_stat[2] / pow(state->sigma, 2)), true);

  return;
}

// called in xbcf_mcmc_loop.cpp
// called from xbcf_mcmc_loop
// updates sigma and sigma2 (is it for normalmodel only?)
void xbcfModel::update_state(std::unique_ptr<State> &state, size_t tree_ind, std::unique_ptr<X_struct> &x_struct)
{
  // Draw Sigma
  // state->residual_std_full = state->residual_std - state->predictions_std[tree_ind];

  // residual_std is only 1 dimensional for regression model

  std::vector<double> full_residual_trt(state->n_trt);               // residual for the treated group
  std::vector<double> full_residual_ctrl(state->n_y - state->n_trt); // residual for the control group

  for (size_t i = 0; i < state->n_trt - 1; i++)
  {
    full_residual_trt[i] = state->residual_std[0][i] - (*(x_struct->data_pointers[tree_ind][i]))[0];
  }

  for (size_t i = state->n_trt; i < state->residual_std[0].size(); i++)
  {
    full_residual_ctrl[i] = state->residual_std[0][i] - (*(x_struct->data_pointers[tree_ind][i]))[0];
  }

  // compute sigma1 for the treated group
  std::gamma_distribution<double> gamma_samp1((state->n_trt + kap) / 2.0, 2.0 / (sum_squared(full_residual_trt) + s));
  double sigma1 = 1.0 / sqrt(gamma_samp1(state->gen));

  // compute sigma0 for the control group
  std::gamma_distribution<double> gamma_samp0((state->n_y - state->n_trt + kap) / 2.0, 2.0 / (sum_squared(full_residual_ctrl) + s));
  double sigma0 = 1.0 / sqrt(gamma_samp0(state->gen));

  //update sigma vector for the state
  state->update_sigma(sigma0, sigma1);
  return;
}

// called in xbcf_mcmc_loop.cpp
// called from xbcf_mcmc_loop
// initializes root suffstats
void xbcfModel::initialize_root_suffstat(std::unique_ptr<State> &state, std::vector<double> &suff_stat)
{
  // sum of y
  suff_stat[0] = sum_vec(state->residual_std[0]);
  // sum of y squared
  suff_stat[1] = sum_squared(state->residual_std[0]);
  // number of observations in the node
  suff_stat[2] = state->n_y;
  return;
}

// called in tree.cpp
// called from split_xorder_std_continuous, split_xorder_std_categorical
// updates node suffstats for the split (or after the split? no, doesn't seem so)
// it is executed after suffstats for the node has been initialized by suff_stats_ini [defined in tree.h]
void xbcfModel::updateNodeSuffStat(std::vector<double> &suff_stat, matrix<double> &residual_std, matrix<size_t> &Xorder_std, size_t &split_var, size_t row_ind)
{
  suff_stat[0] += residual_std[0][Xorder_std[split_var][row_ind]];
  suff_stat[1] += pow(residual_std[0][Xorder_std[split_var][row_ind]], 2);
  suff_stat[2] += 1;
  return;
}

// called in tree.cpp
// called from split_xorder_std_continuous, split_xorder_std_categorical
// updates the other side node's side suffstats for the split
// probably should be completely changed for xbcf (no, it's just vector subtraction)
void xbcfModel::calculateOtherSideSuffStat(std::vector<double> &parent_suff_stat, std::vector<double> &lchild_suff_stat, std::vector<double> &rchild_suff_stat, size_t &N_parent, size_t &N_left, size_t &N_right, bool &compute_left_side)
{

  // in function split_xorder_std_categorical, for efficiency, the function only calculates suff stat of ONE child
  // this function calculate the other side based on parent and the other child

  if (compute_left_side)
  {
    rchild_suff_stat = parent_suff_stat - lchild_suff_stat;
  }
  else
  {
    lchild_suff_stat = parent_suff_stat - rchild_suff_stat;
  }
  return;
}

// called in xbcf_mcmc_loop.cpp
// called from xbcf_mcmc_loop
// updates partial residual for the next tree to fit
void xbcfModel::state_sweep(size_t tree_ind, size_t M, matrix<double> &residual_std, std::unique_ptr<X_struct> &x_struct) const
{
  size_t next_index = tree_ind + 1;
  if (next_index == M)
  {
    next_index = 0;
  }

  ////////////////////////////////////////////////////////
  // Be care of line 151 in train_all.cpp, initial_theta
  ////////////////////////////////////////////////////////

  for (size_t i = 0; i < residual_std[0].size(); i++)
  {
    residual_std[0][i] = residual_std[0][i] - (*(x_struct->data_pointers[tree_ind][i]))[0] + (*(x_struct->data_pointers[next_index][i]))[0];
  }
  return;
}

// called in tree.cpp
// called from GFR, calculate_loglikelihood_continuous, calculate_loglikelihood_categorical, calculate_loglikelihood_nosplit
// computes likelihood of a split
double xbcfModel::likelihood(std::vector<double> &temp_suff_stat, std::vector<double> &suff_stat_all, size_t N_left, bool left_side, bool no_split, std::unique_ptr<State> &state) const
{
  // likelihood equation,
  // note the difference of left_side == true / false
  // node_suff_stat is mean of y, sum of square of y, saved in tree class
  // double y_sum = (double)suff_stat_all[2] * suff_stat_all[0];
  double y_sum = suff_stat_all[0];
  double sigma2 = state->sigma2;
  double ntau;
  double suff_one_side;

  /////////////////////////////////////////////////////////////////////////
  //
  //  I know combining likelihood and likelihood_no_split looks nicer
  //  but this is a very fundamental function, executed many times
  //  the extra if(no_split) statement and value assignment make the code about 5% slower!!
  //
  /////////////////////////////////////////////////////////////////////////

  if (no_split)
  {
    ntau = suff_stat_all[2] * tau;
    suff_one_side = y_sum;
  }
  else
  {
    if (left_side)
    {
      ntau = (N_left + 1) * tau;
      suff_one_side = temp_suff_stat[0];
    }
    else
    {
      ntau = (suff_stat_all[2] - N_left - 1) * tau;
      suff_one_side = y_sum - temp_suff_stat[0];
    }
  }

  return 0.5 * log(sigma2) - 0.5 * log(ntau + sigma2) + 0.5 * tau * pow(suff_one_side, 2) / (sigma2 * (ntau + sigma2));
}

// double xbcfModel::likelihood_no_split(std::vector<double> &suff_stat, std::unique_ptr<State> &state) const
// {
//     // the likelihood of no-split option is a bit different from others
//     // because the sufficient statistics is y_sum here
//     // write a separate function, more flexibility
//     double ntau = suff_stat[2] * tau;
//     // double sigma2 = pow(state->sigma, 2);
//     double sigma2 = state->sigma2;
//     double value = suff_stat[2] * suff_stat[0]; // sum of y

//     return 0.5 * log(sigma2) - 0.5 * log(ntau + sigma2) + 0.5 * tau * pow(value, 2) / (sigma2 * (ntau + sigma2));
// }

// called in xbcf_mcmc_loop.cpp
// called from xbcf_mcmc_loop
// initializes the residual for 0th tree
void xbcfModel::ini_residual_std(std::unique_ptr<State> &state)
{
  double value = state->ini_var_yhat * ((double)state->num_trees - 1.0) / (double)state->num_trees;
  for (size_t i = 0; i < state->residual_std[0].size(); i++)
  {
    state->residual_std[0][i] = (*state->y_std)[i] - value;
  }
  return;
}

// NEW
// called in xbcf_mcmc_loop.cpp
// called from xbcf_mcmc_loop
// passes over the residual from the prognostic term forest to the treatment term forest
void xbcfModel::transfer_residual_std(std::unique_ptr<State> &state_ps, std::unique_ptr<State> &state_trt)
{
  for (size_t i = 0; i < state_trt->residual_std[0].size(); i++)
  {
    state_trt->residual_std[0][i] = state_ps->residual_std[0][i];
  }

  return;
}

void xbcfModel::predict_std(const double *Xtestpointer, size_t N_test, size_t p, size_t num_trees, size_t num_sweeps, matrix<double> &yhats_test_xinfo, vector<vector<tree>> &trees)
{
  matrix<double> output;

  // row : dimension of theta, column : number of trees
  ini_matrix(output, this->dim_theta, trees.size());

  for (size_t sweeps = 0; sweeps < num_sweeps; sweeps++)
  {
    for (size_t data_ind = 0; data_ind < N_test; data_ind++)
    {
      getThetaForObs_Outsample(output, trees[sweeps], data_ind, Xtestpointer, N_test, p);

      // take sum of predictions of each tree, as final prediction
      for (size_t i = 0; i < trees.size(); i++)
      {
        yhats_test_xinfo[sweeps][data_ind] += output[i][0];
      }
    }
  }
  return;
}