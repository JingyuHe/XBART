#include "xbcf_mcmc_loop.h"

// BCF main loop
// input includes information about two sets of trees (one for prognostic term, the other for treatment term)
// thus there are two of each tree-object, model-object, state-object, x_struct-object

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
                    bool b_scaling)
{

  if (state->parallel)
    thread_pool.start();

  for (size_t sweeps = 0; sweeps < state->num_sweeps; sweeps++)
  {
    //cout << "sweep: " << sweeps << endl;
    if (verbose == true)
    {
      COUT << "--------------------------------" << endl;
      COUT << "number of sweeps " << sweeps << endl;
      COUT << "--------------------------------" << endl;
    }

    model_ps->set_flag(state->fl, 0); // set this flag to 0 so that likelihood function can recognize the mu-loop
    state->iniSplitStorage(state->fl);
    state->adjustMtry(state->fl);
    state->X_std = X_std;
    state->p = state->p_pr;
    state->p_categorical = state->p_categorical_pr;
    state->p_continuous = state->p_continuous_pr;
    ////////////// Prognostic term loop
    for (size_t tree_ind = 0; tree_ind < state->num_trees_vec[0]; tree_ind++)
    {
      model_ps->update_state(state, tree_ind, x_struct_ps); // Draw Sigma -- the residual needed for the update is computed inside of the function

      if (state->use_all && (sweeps > state->burnin) && (state->mtry_pr != state->p_pr))
      {
        state->use_all = false;
      }

      std::fill(state->split_count_current_tree.begin(), state->split_count_current_tree.end(), 0.0); // clear counts of splits for one tree

      if (state->sample_weights_flag)
      {
        state->mtry_weight_current_tree = state->mtry_weight_current_tree - state->split_count_all_tree_pr[tree_ind]; // subtract old tree for sampling case
      }

      model_ps->subtract_old_tree_fit(tree_ind, state->mu_fit, x_struct_ps); // for GFR we will need partial mu_fit -- thus take out the old fitted values

      model_ps->initialize_root_suffstat(state, trees_ps[sweeps][tree_ind].suff_stat); // initialize suff stat using partial fit
      //GFR
      trees_ps[sweeps][tree_ind].grow_from_root(state, Xorder_std, x_struct_ps->X_counts, x_struct_ps->X_num_unique, model_ps, x_struct_ps, sweeps, tree_ind, true, false, true);

      model_ps->state_sweep(tree_ind, state->mu_fit, x_struct_ps); // update total mu_fit by adding just fitted values

      state->update_split_counts(tree_ind, 0); // update split counts for mu
      if (sweeps != 0)
      {
        if (a_scaling) // in case b_scaling on, we update b0 and b1
        {
          model_ps->update_a_value(state);
        }
        if (b_scaling) // in case b_scaling on, we update b0 and b1
        {
          model_trt->update_b_values(state);
        }
      }
    }

    model_ps->set_flag(state->fl, 1); // set this flag to 1 so that likelihood function can recognize the tau-loop

    ////////////// Treatment term loop
    state->iniSplitStorage(state->fl);
    state->adjustMtry(state->fl);
    state->X_std = X_tau_std;
    state->p = state->p_trt;
    state->p_categorical = state->p_categorical_trt;
    state->p_continuous = state->p_continuous_trt;
    for (size_t tree_ind = 0; tree_ind < state->num_trees_vec[1]; tree_ind++)
    {
      // Draw Sigma
      model_trt->update_state(state, tree_ind, x_struct_trt);

      // store sigma draws
      sigma0_draw_xinfo[sweeps][tree_ind] = state->sigma_vec[0]; // storing sigmas
      sigma1_draw_xinfo[sweeps][tree_ind] = state->sigma_vec[1]; // storing sigmas

      if (state->use_all && (sweeps > state->burnin) && (state->mtry_trt != state->p_trt))
      {
        state->use_all = false;
      }

      std::fill(state->split_count_current_tree.begin(), state->split_count_current_tree.end(), 0.0); // clear counts of splits for one tree

      if (state->sample_weights_flag)
      {
        state->mtry_weight_current_tree = state->mtry_weight_current_tree - state->split_count_all_tree_trt[tree_ind]; // subtract old tree for sampling case
      }

      model_trt->subtract_old_tree_fit(tree_ind, state->tau_fit, x_struct_trt); // for GFR we will need partial tau_fit -- thus take out the old fitted values

      model_trt->initialize_root_suffstat(state, trees_trt[sweeps][tree_ind].suff_stat); // initialize suff stat using partial fit
      // GFR
      trees_trt[sweeps][tree_ind].grow_from_root(state, Xorder_tau_std, x_struct_trt->X_counts, x_struct_trt->X_num_unique, model_trt, x_struct_trt, sweeps, tree_ind, true, false, true);

      model_trt->state_sweep(tree_ind, state->tau_fit, x_struct_trt); // update total tau_fit by adding just fitted values

      state->update_split_counts(tree_ind, 1); // update split counts for tau
      if (sweeps != 0)
      {
        if (a_scaling) // in case b_scaling on, we update b0 and b1
        {
          model_ps->update_a_value(state);
        }
        if (b_scaling) // in case b_scaling on, we update b0 and b1
        {
          model_trt->update_b_values(state);
        }
      }
    }


    // store draws for b0, b1 and a, although they are updated per tree, we save results per forest (sweep)
    b_xinfo[0][sweeps] = state->b_vec[0];
    b_xinfo[1][sweeps] = state->b_vec[1];
    a_xinfo[0][sweeps] = state->a;
  }

  // print out all residuals, total fits, etc

  // for(size_t kk = 0; kk < 50; kk ++ ){
  //   cout << kk << "  " << state->mu_fit[kk] << "  " << state->tau_fit[kk] << endl;
  // }
 

  tree::tree_p bn;

  cout << "number of trees " << state->num_trees_vec[0] << endl;

  std::vector<double> fit_ps(50);
  std::vector<double> fit_trt(50);

  for(size_t data_ind = 0; data_ind < 50; data_ind ++ ){
    for(size_t tree_ind = 0; tree_ind < state->num_trees_vec[0]; tree_ind ++ ){
      fit_ps[data_ind] += (*(x_struct_ps->data_pointers[tree_ind][data_ind]))[0];
    }
  }

  for(size_t data_ind = 0; data_ind < 50; data_ind ++ ){
    for(size_t tree_ind = 0; tree_ind < state->num_trees_vec[1]; tree_ind ++ ){
      fit_trt[data_ind] += (*(x_struct_trt->data_pointers[tree_ind][data_ind]))[0];
    }
  }


  std::vector<double> fit_ps2(50);
  std::vector<double> fit_trt2(50);


  for (size_t i = 0; i < 50; i++)
  {
    for(size_t tree_ind = 0; tree_ind < state->num_trees_vec[0]; tree_ind ++ )
    {
        // loop over observations
        // tree search
        bn = trees_ps[state->num_sweeps - 1][tree_ind].search_bottom_std(X_std, i, state->p, state->n_y);
        fit_ps2[i] += bn->theta_vector[0];
    }

    for(size_t tree_ind = 0; tree_ind < state->num_trees_vec[1]; tree_ind ++ )
    {
        // loop over observations
        // tree search
        bn = trees_trt[state->num_sweeps - 1][tree_ind].search_bottom_std(X_tau_std, i, state->p, state->n_y);
        fit_trt2[i] += bn->theta_vector[0];
    }
  }

  cout << "-------------------------" << endl;
  cout << "mu fit, data pointer  |  mu, tree search  |  tau fit data pointer  |  tau tree search " << endl;
  for(size_t kk = 0; kk < 50; kk ++ ){
    cout << kk << "  " << fit_ps[kk] << "  " << fit_trt[kk] << "  " << fit_ps2[kk] <<  "  " << fit_trt2[kk] << endl;
  }



  cout << "++++++++++++++++++++++++++" << endl;
  cout << "print mu trees " << endl;
  for(size_t i = 0; i < state->num_trees_vec[0]; i ++){
    cout << trees_ps[state->num_sweeps - 1][i] << endl;
  }
  
  thread_pool.stop();
  return;
}