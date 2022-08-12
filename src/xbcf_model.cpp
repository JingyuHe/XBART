#include "tree.h"
#include "xbcf_model.h"

//////////////////////////////////////////////////////////////////////////////////////
//
//
//  XBCF Model
//
//
//////////////////////////////////////////////////////////////////////////////////////

// adds residual to suff stats for normalmodel (updates the suff stats by adding new values to the old ones)
// called from calcSuffStat_categorical, calcSuffStat_continuous in tree.cpp
void xbcfModel::incSuffStat(std::unique_ptr<State> &state, size_t index_next_obs, std::vector<double> &suffstats)
{
  if (state->fl == 0)
  {
    if (state->z[index_next_obs] == 1)
    {
      // old: suffstats[1] += state->residual_std[0][index_next_obs];
      suffstats[1] += ((*state->y_std)[index_next_obs] - state->a * state->mu_fit[index_next_obs] - state->b_vec[1] * state->tau_fit[index_next_obs]) / state->a;
      suffstats[3] += 1;
    }
    else
    {
      suffstats[0] += ((*state->y_std)[index_next_obs] - state->a * state->mu_fit[index_next_obs] - state->b_vec[0] * state->tau_fit[index_next_obs]) / state->a;
      suffstats[2] += 1;
    }
  }
  else
  {
    if (state->z[index_next_obs] == 1)
    {
      // old: suffstats[1] += state->residual_std[0][index_next_obs];
      suffstats[1] += ((*state->y_std)[index_next_obs] - state->a * state->mu_fit[index_next_obs] - state->b_vec[1] * state->tau_fit[index_next_obs]) / state->b_vec[1];
      suffstats[3] += 1;
    }
    else
    {
      suffstats[0] += ((*state->y_std)[index_next_obs] - state->a * state->mu_fit[index_next_obs] - state->b_vec[0] * state->tau_fit[index_next_obs]) / state->b_vec[0];
      suffstats[2] += 1;
    }
  }

  return;
}

// samples leaf parameter
// called from GFR in tree.cpp
void xbcfModel::samplePars(std::unique_ptr<State> &state, std::vector<double> &suff_stat, std::vector<double> &theta_vector, double &prob_leaf)
{
  std::normal_distribution<double> normal_samp(0.0, 1.0);
  double s0 = 0;
  double s1 = 0;

  if (state->fl == 0)
  {
    s0 = state->sigma_vec[0] / fabs(state->a);
    s1 = state->sigma_vec[1] / fabs(state->a);
  }
  else
  {
    s0 = state->sigma_vec[0] / fabs(state->b_vec[0]);
    s1 = state->sigma_vec[1] / fabs(state->b_vec[1]);
  }
  // step 1 (control group)
  double denominator0 = 1.0 / tau + suff_stat[2] / pow(s0, 2);
  double m0 = (suff_stat[0] / pow(s0, 2)) / denominator0;
  double v0 = 1.0 / denominator0;

  // step 2 (treatment group)
  double denominator1 = (1.0 / v0 + suff_stat[3] / pow(s1, 2));
  double m1 = (1.0 / v0) * m0 / denominator1 + suff_stat[1] / pow(s1, 2) / denominator1;
  double v1 = 1.0 / denominator1;

  // sample leaf parameter
  theta_vector[0] = m1 + sqrt(v1) * normal_samp(state->gen);

  // also update probability of leaf parameters
  prob_leaf = 1.0;
  return;
}

// updates sigmas (new)
// called from mcmc_loop_xbcf in xbcf_mcmc_loop.cpp
void xbcfModel::draw_sigma(std::unique_ptr<State> &state, size_t ind)
{
  // computing both sigmas here due to structural complexity of splitting them
  std::gamma_distribution<double> gamma_samp1((state->n_trt + kap) / 2.0, 2.0 / (sum_squared(state->full_residual_trt) + s));
  std::gamma_distribution<double> gamma_samp0((state->n_y - state->n_trt + kap) / 2.0, 2.0 / (sum_squared(state->full_residual_ctrl) + s));

  // then we choose only one of them based on the group we are updating for
  double sigma;
  if(ind == 0) {
    sigma = 1.0 / sqrt(gamma_samp0(state->gen));
  } else {
    sigma = 1.0 / sqrt(gamma_samp1(state->gen));
  }

  // update the corresponding value in the state object
  state->update_sigma(sigma, ind);
  return;
}

// initializes root suffstats
// called from mcmc_loop_xbcf in xbcf_mcmc_loop.cpp
void xbcfModel::initialize_root_suffstat(std::unique_ptr<State> &state, std::vector<double> &suff_stat)
{
  suff_stat.resize(4);

  std::fill(suff_stat.begin(), suff_stat.end(), 0.0);

  for (size_t i = 0; i < state->n_y; i++)
  {
    incSuffStat(state, i, suff_stat);
  }

  return;
}

// updates node suffstats for the split
// called from split_xorder_std_continuous, split_xorder_std_categorical in tree.cpp
// it is executed after suffstats for the node has been initialized by suff_stats_ini [defined in tree.h]
void xbcfModel::updateNodeSuffStat(std::vector<double> &suff_stat, std::unique_ptr<State> &state, matrix<size_t> &Xorder_std, size_t &split_var, size_t row_ind)
{

  incSuffStat(state, Xorder_std[split_var][row_ind], suff_stat);

  return;
}

// updates the other side node's side suffstats for the split
// called from split_xorder_std_continuous, split_xorder_std_categorical in tree.cpp
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

// updates partial residual for the next tree to fit
// called from mcmc_loop_xbcf in xbcf_mcmc_loop.cpp
void xbcfModel::state_sweep(size_t tree_ind, std::vector<double> &fit, std::unique_ptr<X_struct> &x_struct) const
{
  for (size_t i = 0; i < fit.size(); i++)
  {
    fit[i] += (*(x_struct->data_pointers[tree_ind][i]))[0];
  }
  return;
}

// computes likelihood of a split
// called from GFR, calculate_loglikelihood_continuous, calculate_loglikelihood_categorical, calculate_loglikelihood_nosplit in tree.cpp
double xbcfModel::likelihood(std::vector<double> &temp_suff_stat, std::vector<double> &suff_stat_all, size_t N_left, bool left_side, bool no_split, std::unique_ptr<State> &state) const
{
  // helper variables
  double s0 = 0;
  double s1 = 0;
  double denominator;   // the denominator (1 + tau * precision_squared) is the same for both terms
  double s_psi_squared; // (residual * precision_squared)^2

  if (state->fl == 0)
  {
    s0 = state->sigma_vec[0] / fabs(state->a);
    s1 = state->sigma_vec[1] / fabs(state->a);
  }
  else
  {
    s0 = state->sigma_vec[0] / fabs(state->b_vec[0]);
    s1 = state->sigma_vec[1] / fabs(state->b_vec[1]);
  }

  if (no_split)
  {
    denominator = 1 + (suff_stat_all[2] / pow(s0, 2) + suff_stat_all[3] / pow(s1, 2)) * tau;
    s_psi_squared = suff_stat_all[0] / pow(s0, 2) + suff_stat_all[1] / pow(s1, 2);
  }
  else
  {
    if (left_side)
    {
      denominator = 1 + (temp_suff_stat[2] / pow(s0, 2) + temp_suff_stat[3] / pow(s1, 2)) * tau;
      s_psi_squared = temp_suff_stat[0] / pow(s0, 2) + temp_suff_stat[1] / pow(s1, 2);
    }
    else
    {
      denominator = 1 + ((suff_stat_all[2] - temp_suff_stat[2]) / pow(s0, 2) + (suff_stat_all[3] - temp_suff_stat[3]) / pow(s1, 2)) * tau;
      s_psi_squared = (suff_stat_all[0] - temp_suff_stat[0]) / pow(s0, 2) + (suff_stat_all[1] - temp_suff_stat[1]) / pow(s1, 2);
    }
  }

  return 0.5 * log(1 / denominator) + 0.5 * pow(s_psi_squared, 2) * tau / denominator;
}

// makes a prediction for treatment effect on the given Xtestpointer data
void xbcfModel::predict_std(const double *Xtestpointer, size_t N_test, size_t p, size_t num_sweeps, matrix<double> &yhats_test_xinfo, vector<vector<tree>> &trees)
{
  matrix<double> output;

  // row : dimension of theta, column : number of trees
  ini_matrix(output, this->dim_theta, trees[0].size());

  for (size_t sweeps = 0; sweeps < num_sweeps; sweeps++)
  {
    for (size_t data_ind = 0; data_ind < N_test; data_ind++)
    {
      getThetaForObs_Outsample(output, trees[sweeps], data_ind, Xtestpointer, N_test, p);

      // take sum of predictions of each tree, as final prediction
      for (size_t i = 0; i < trees[0].size(); i++)
      {
        yhats_test_xinfo[sweeps][data_ind] += output[i][0];
      }
    }
  }
  return;
}

// updates parameter a
// called from mcmc_loop_xbcf in xbcf_mcmc_loop.cpp
void xbcfModel::update_a_value(std::unique_ptr<State> &state)
{
  std::normal_distribution<double> normal_samp(0.0, 1.0);

  double mu2sum_ctrl = 0;
  double mu2sum_trt = 0;
  double muressum_ctrl = 0;
  double muressum_trt = 0;

  // compute the residual y-b*tau(x) using state's objects y_std, tau_fit and b_vec
  for (size_t i = 0; i < state->n_y; i++)
  {
    if (state->z[i] == 1)
    {
      state->residual[i] = (*state->y_std)[i] - state->tau_fit[i] * state->b_vec[1];
    }
    else
    {
      state->residual[i] = (*state->y_std)[i] - state->tau_fit[i] * state->b_vec[0];
    }
  }

  for (size_t i = 0; i < state->n_y; i++)
  {
    if (state->z[i] == 1)
    {
      mu2sum_trt += state->mu_fit[i] * state->mu_fit[i];
      muressum_trt += state->mu_fit[i] * state->residual[i];
    }
    else
    {
      mu2sum_ctrl += state->mu_fit[i] * state->mu_fit[i];
      muressum_ctrl += state->mu_fit[i] * state->residual[i];
    }
  }

  // update parameters
  double v0 = 1 / (1.0 + mu2sum_ctrl / pow(state->sigma_vec[0], 2));
  double m0 = v0 * (muressum_ctrl) / pow(state->sigma_vec[0], 2);

  double v1 = 1 / (1.0 / v0 + mu2sum_trt / pow(state->sigma_vec[1], 2));
  double m1 = v1 * (m0 / v0 + (muressum_trt) / pow(state->sigma_vec[1], 2));

  // sample a
  state->a = m1 + sqrt(v1) * normal_samp(state->gen);

  return;
}

// updates parameters b0, b1
// called from mcmc_loop_xbcf in xbcf_mcmc_loop.cpp
void xbcfModel::update_b_values(std::unique_ptr<State> &state)
{
  std::normal_distribution<double> normal_samp(0.0, 1.0);

  double tau2sum_ctrl = 0;
  double tau2sum_trt = 0;
  double tauressum_ctrl = 0;
  double tauressum_trt = 0;

  // compute the residual y-a*mu(x) using state's objects y_std, mu_fit and a
  for (size_t i = 0; i < state->n_y; i++)
  {
    state->residual[i] = (*state->y_std)[i] - state->a * state->mu_fit[i];
  }

  for (size_t i = 0; i < state->n_y; i++)
  {
    if (state->z[i] == 1)
    {
      tau2sum_trt += state->tau_fit[i] * state->tau_fit[i];
      tauressum_trt += state->tau_fit[i] * state->residual[i];
    }
    else
    {
      tau2sum_ctrl += state->tau_fit[i] * state->tau_fit[i];
      tauressum_ctrl += state->tau_fit[i] * state->residual[i];
    }
  }

  // update parameters
  double v0 = 1 / (2 + tau2sum_ctrl / pow(state->sigma_vec[0], 2));
  double v1 = 1 / (2 + tau2sum_trt / pow(state->sigma_vec[1], 2));

  double m0 = v0 * (tauressum_ctrl) / pow(state->sigma_vec[0], 2);
  double m1 = v1 * (tauressum_trt) / pow(state->sigma_vec[1], 2);

  // sample b0, b1
  double b0 = m0 + sqrt(v0) * normal_samp(state->gen);
  double b1 = m1 + sqrt(v1) * normal_samp(state->gen);

  state->b_vec[1] = b1;
  state->b_vec[0] = b0;

  return;
}

// subtracts old tree contribution from the fit
// called from mcmc_loop_xbcf in xbcf_mcmc_loop.cpp
void xbcfModel::subtract_old_tree_fit(size_t tree_ind, std::vector<double> &fit, std::unique_ptr<X_struct> &x_struct)
{
  for (size_t i = 0; i < fit.size(); i++)
  {
    fit[i] -= (*(x_struct->data_pointers[tree_ind][i]))[0];
  }
  return;
}

// sets unique term parameters in the state object depending on the term being updated
// called from mcmc_loop_xbcf in xbcf_mcmc_loop.cpp
void xbcfModel::set_state_status(std::unique_ptr<State> &state, size_t value, const double *X, matrix<size_t> &Xorder)
{
  state->fl = value; // value can only be 0 or 1 (to alternate between arms)
  state->iniSplitStorage(state->fl);
  state->adjustMtry(state->fl);
  state->X_std = X;
  state->Xorder_std = Xorder;
  if(value == 0)
  {
    state->p = state->p_pr;
    state->p_categorical = state->p_categorical_pr;
    state->p_continuous = state->p_continuous_pr;
  } else {
    state->p = state->p_trt;
    state->p_categorical = state->p_categorical_trt;
    state->p_continuous = state->p_continuous_trt;
  }

}