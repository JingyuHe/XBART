#include "xbcf_mcmc_loop.h"

// BCF main loop
// input includes information about two sets of trees (one for prognostic term, the other for treatment term)
// thus there are two of each tree-object, model-object, state-object, x_struct-object

void mcmc_loop_xbcf(matrix<size_t> &Xorder_std, bool verbose,
                    matrix<double> &sigma0_draw_xinfo,
                    matrix<double> &sigma1_draw_xinfo,
                    matrix<double> &b_xinfo,
                    matrix<double> &b0_draw_xinfo,
                    matrix<double> &b1_draw_xinfo,
                    matrix<double> &total_fit,
                    vector<vector<tree>> &trees_ps,
                    vector<vector<tree>> &trees_trt,
                    double no_split_penality,
                    std::unique_ptr<State> &state_ps,
                    std::unique_ptr<State> &state_trt,
                    xbcfModel *model_ps,
                    xbcfModel *model_trt,
                    std::unique_ptr<X_struct> &x_struct_ps,
                    std::unique_ptr<X_struct> &x_struct_trt,
                    bool b_scaling)
{

  if (state_ps->parallel)
    thread_pool.start();

  // Residual for 0th tree in Prognostic term forest
  model_ps->ini_residual_std(state_ps);
  //model_trt->ini_residual_std(state_ps);

  // initialize vectors needed for b update (initialization verified)
  std::vector<double> b_residual(state_ps->n_y, 0);
  std::vector<double> tau_fit(state_ps->n_y, 0);
  //std::vector<double> total_fit(state_ps->n_y, 0);

  for (size_t sweeps = 0; sweeps < state_ps->num_sweeps; sweeps++)
  {
    //cout << "sweep: " << sweeps << endl;
    if (verbose == true)
    {
      COUT << "--------------------------------" << endl;
      COUT << "number of sweeps " << sweeps << endl;
      COUT << "--------------------------------" << endl;
    }

    // Prognostic term loop
    for (size_t tree_ind = 0; tree_ind < state_ps->num_trees; tree_ind++)
    {
      // Draw Sigma
      model_ps->update_state_ps(state_ps, tree_ind, x_struct_ps); // state_ps->sigma and state_ps->sigma2 are updated via state->update_sigma

      if (state_ps->use_all && (sweeps > state_ps->burnin) && (state_ps->mtry != state_ps->p))
      {
        state_ps->use_all = false;
      }

      // clear counts of splits for one tree
      std::fill(state_ps->split_count_current_tree.begin(), state_ps->split_count_current_tree.end(), 0.0);

      // subtract old tree for sampling case
      if (state_ps->sample_weights_flag)
      {
        state_ps->mtry_weight_current_tree = state_ps->mtry_weight_current_tree - state_ps->split_count_all_tree[tree_ind];
      }

      model_ps->initialize_root_suffstat(state_ps, trees_ps[sweeps][tree_ind].suff_stat);

      trees_ps[sweeps][tree_ind].grow_from_root(state_ps, Xorder_std, x_struct_ps->X_counts, x_struct_ps->X_num_unique, model_ps, x_struct_ps, sweeps, tree_ind, true, false, true);

      state_ps->update_split_counts(tree_ind);

      // update partial residual for the next tree to fit
      model_ps->state_sweep(tree_ind, state_ps->num_trees, state_ps->residual_std, x_struct_ps);
    }

    // store fitted values in tauhats_xinfo
    // model_ps->update_xinfo(muhats_xinfo, sweeps, state_ps->num_trees, state_ps->n_y, x_struct_ps);

    // compute full residual y - mu(x) needed for updating scaling parameters b0, b1
    if (b_scaling)
    {
      model_trt->compute_residual_b(state_ps, x_struct_ps, b_residual);
    }
    /*
    if (sweeps != 0) // for the very first sweep, b0 = -0.5, b1 = 0.5 (update will give b0 = b1 = 0, since taufit is 0)
    {
    // compute b0, b1 based on residual and current tau fit
    model_trt->update_b_values(state_trt, b_residual, tau_fit);

    // update b_std vector which is a state attribute used for state update
    // had to do it outside of the trestment term loop since we need b to compute residual
    //model_trt->update_b_vector(state_trt);
    }*/
    // update the residual to pass it over to the treatment term loop
    // NEW: pass over sigmas from state_ps to state_trt
    model_trt->compute_residual_trt(state_ps, state_trt, x_struct_ps, x_struct_trt);

    // Treatment term loop
    for (size_t tree_ind = 0; tree_ind < state_trt->num_trees; tree_ind++)
    {
      //cout << state_trt->sigma_vec[0] << " <- s0 | s1 -> " << state_trt->sigma_vec[1] << endl;
      // Draw Sigma
      model_trt->update_state_trt(state_trt, tree_ind, x_struct_trt);

      // store sigma draws
      sigma0_draw_xinfo[sweeps][tree_ind] = state_trt->sigma_vec[0]; // storing sigmas
      sigma1_draw_xinfo[sweeps][tree_ind] = state_trt->sigma_vec[1]; // storing sigmas

      if (state_trt->use_all && (sweeps > state_trt->burnin) && (state_trt->mtry != state_trt->p))
      {
        state_trt->use_all = false;
      }

      // clear counts of splits for one tree
      std::fill(state_trt->split_count_current_tree.begin(), state_trt->split_count_current_tree.end(), 0.0);

      // subtract old tree for sampling case
      if (state_trt->sample_weights_flag)
      {
        state_trt->mtry_weight_current_tree = state_trt->mtry_weight_current_tree - state_trt->split_count_all_tree[tree_ind];
      }

      model_trt->initialize_root_suffstat(state_trt, trees_trt[sweeps][tree_ind].suff_stat);

      // adjust tau fit by subtracting the old value of the tree
      //model_trt->adjust_tau_fit(tau_fit, tree_ind, x_struct_trt);

      trees_trt[sweeps][tree_ind].grow_from_root(state_trt, Xorder_std, x_struct_trt->X_counts, x_struct_trt->X_num_unique, model_trt, x_struct_trt, sweeps, tree_ind, true, false, true);

      state_trt->update_split_counts(tree_ind);

      if (b_scaling)
      {
        // compute the current tau fit, needed for b update
        model_trt->compute_tau_fit(tau_fit, tree_ind, x_struct_trt);
        //model_trt->compute_total_fit(total_fit[sweeps], tree_ind, x_struct_trt, state_trt->b_vec);

        // update partial residual for the next tree to fit
        model_trt->adjust_residual_trt(tree_ind, state_trt->num_trees, state_trt->residual_std, x_struct_trt, state_trt->b_std);

        //update b parameters
        // compute new b values based on y - mu(x) residual, current tau fit and current sigma
        model_trt->update_b_values(state_trt, b_residual, tau_fit);
        // update b_std vector which is a state attribute used for state update
        //model_trt->update_b_vector(state_trt);

        // complete state sweep
        model_trt->state_sweep_trt(tree_ind, state_trt->num_trees, state_trt->residual_std, x_struct_trt, state_trt->b_std);

        // subtract the value of the old fit for the next tree we will be growing
        if (tree_ind + 1 == state_trt->num_trees)
        {
          model_trt->adjust_tau_fit(tau_fit, 0, x_struct_trt);
        }
        else
        {
          model_trt->adjust_tau_fit(tau_fit, tree_ind + 1, x_struct_trt);
        }
      }
      else
      {
        model_trt->state_sweep(tree_ind, state_trt->num_trees, state_trt->residual_std, x_struct_trt);
      }
    }

    // store draws for b0 and b1
    b_xinfo[0][sweeps] = state_trt->b_vec[0];
    b_xinfo[1][sweeps] = state_trt->b_vec[1];

    // compute average values of b0, b1
    //model_trt->compute_average_b_value(state_trt, sweeps, b0_draw_xinfo, b1_draw_xinfo, b_avg);

    // update b_std vector with computed average b0 and b1
    //model_trt->update_b_vector(state_trt);

    // update the residual to pass it over back to the prognostic term loop
    model_trt->compute_residual_ps(state_trt, state_ps, x_struct_trt, x_struct_ps);
    //std::fill(tau_fit.begin(), tau_fit.end(), 0.0);
    // store fitted values in tauhats_xinfo
    // model_trt->update_xinfo(tauhats_xinfo, sweeps, state_trt->num_trees, state_trt->n_y, x_struct_trt);
  }

  thread_pool.stop();
  return;
}