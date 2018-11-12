#include "fit_std_main_loop.h"


void fit_std_main_loop(const double *Xpointer,std::vector<double> &y_std,double& y_mean,const double *Xtestpointer, xinfo_sizet &Xorder_std,
    size_t N,size_t p,size_t N_test,
    size_t M, size_t L, size_t N_sweeps, xinfo_sizet &max_depth_std, 
    size_t Nmin, size_t Ncutpoints, double alpha, double beta, 
    double tau, size_t burnin, size_t mtry, 
    bool draw_sigma , double kap , double s, 
    bool verbose, bool m_update_sigma, 
    bool draw_mu, bool parallel,
    xinfo &yhats_xinfo,xinfo &yhats_test_xinfo,xinfo &sigma_draw_xinfo)
{
    // Set Random Generator
    std::default_random_engine(generator);

    std::vector<std::vector<double>> predictions_std;
    ini_xinfo(predictions_std, N, M);

    xinfo predictions_test_std;
    ini_xinfo(predictions_test_std, N_test, M);

    std::vector<double> yhat_std(N);
    row_sum(predictions_std, yhat_std);
    std::vector<double> yhat_test_std(N_test);
    row_sum(predictions_test_std, yhat_test_std);

    xinfo sigma_draw_std;
    ini_xinfo(sigma_draw_std, M, N_sweeps);


    xinfo split_count_all_tree;  // initialize at 0
    ini_xinfo(split_count_all_tree, p, M);

    // split_count_all_tree = split_count_all_tree + 1; // initialize at 1
    std::vector<double> split_count_current_tree(p, 1.0);
    std::vector<double> mtry_weight_current_tree(p, 1.0);

    // Rcpp::NumericVector split_count_current_tree(p, 1);
    // Rcpp::NumericVector mtry_weight_current_tree(p, 1);

    double sigma;
    // double tau;
    forest trees(M);
    std::vector<double> prob(2, 0.5);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> d(prob.begin(), prob.end());
    // // sample one index of split point
    size_t prune;

    // current residual
    std::vector<double> residual_std(N);

    // std::vector<double> split_var_count(p);
    // std::fill(split_var_count.begin(), split_var_count.end(), 1);
    // Rcpp::NumericVector split_var_count(p, 1);


    

    // double *split_var_count_pointer = &split_var_count[0];


    // in the burnin samples, use all variables
    std::vector<size_t> subset_vars(p);
    std::iota(subset_vars.begin() + 1, subset_vars.end(), 1);
    
    // cout << subset_vars << endl;

    double run_time = 0.0;

    // save tree objects to strings
    // std::stringstream treess;
    // treess.precision(10);
    // treess << L << " " << M << " " << p << endl;

    // L, number of samples
    // M, number of trees

    bool use_all = true;

    for (size_t mc = 0; mc < L; mc++)
    {

        // initialize predcitions and predictions_test
        for (size_t ii = 0; ii < M; ii++)
        {
            std::fill(predictions_std[ii].begin(), predictions_std[ii].end(), y_mean / (double)M);
            std::fill(predictions_test_std[ii].begin(), predictions_test_std[ii].end(), y_mean / (double)M);
        }

        row_sum(predictions_std, yhat_std);
        row_sum(predictions_test_std, yhat_test_std);

        residual_std = y_std - yhat_std;

        for (size_t sweeps = 0; sweeps < N_sweeps; sweeps++)
        {

            if (verbose == true)
            {
                cout << "--------------------------------" << endl;
                cout << "number of sweeps " << sweeps << endl;
                cout << "--------------------------------" << endl;
            }

            for (size_t tree_ind = 0; tree_ind < M; tree_ind++)
            {

                // if update sigma based on residual of all m trees
                if (m_update_sigma == true)
                {

                    std::gamma_distribution<double> gamma_samp((N + kap) / 2.0, 2.0 / (sum_squared(residual_std) + s));

                    sigma = 1.0 / sqrt(gamma_samp(generator));

                    sigma_draw_xinfo[sweeps][tree_ind] = sigma;
                }

                // save sigma
                sigma_draw_xinfo[sweeps][tree_ind] = sigma;

                // add prediction of current tree back to residual
                // then it's m - 1 trees residual

                residual_std = residual_std + predictions_std[tree_ind];

                // do the samething for residual_theta_noise, residual of m - 1 trees

                yhat_std = yhat_std - predictions_std[tree_ind];

                yhat_test_std = yhat_test_std - predictions_test_std[tree_ind];

                if (use_all && (sweeps > burnin) && (mtry != p))
                {
                    // subset_vars = Rcpp::as<std::vector<size_t>>(sample(var_index_candidate, mtry, false, split_var_count));
                    use_all = false;
                }

                // cout << "variables used " << subset_vars << endl;
                // cout << "------------------" << endl;

                // clear counts of splits for one tree
                std::fill(split_count_current_tree.begin(), split_count_current_tree.end(), 0.0);

                for(int i=0;i<p;i++){
                    mtry_weight_current_tree[i] = mtry_weight_current_tree[i] - split_count_all_tree[tree_ind][i];
                }
                // mtry_weight_current_tree = mtry_weight_current_tree - split_count_all_tree(Rcpp::_, tree_ind);

                // cout << "before " << mtry_weight_current_tree << endl;

                trees.t[tree_ind].grow_tree_adaptive_abarth_train(sum_vec(residual_std) / (double)N, 0, max_depth_std[sweeps][tree_ind], Nmin, Ncutpoints, tau, sigma, alpha, beta, draw_sigma, draw_mu, parallel, residual_std, Xorder_std, Xpointer, mtry, run_time, use_all, mtry_weight_current_tree, split_count_current_tree);

                mtry_weight_current_tree = mtry_weight_current_tree + split_count_current_tree;

                // cout << "after " << mtry_weight_current_tree << endl; 


                for(int i = 0;i<p;i++){
                    split_count_all_tree[tree_ind][i] = split_count_current_tree[i];
                }
                // split_count_all_tree(Rcpp::_, tree_ind) = split_count_current_tree; 


                if (verbose == true)
                {
                    cout << "tree " << tree_ind << " size is " << trees.t[tree_ind].treesize() << endl;
                }

                // update prediction of current tree
                fit_new_std(trees.t[tree_ind], Xpointer, N, p, predictions_std[tree_ind]);

                // update prediction of current tree, test set
                fit_new_std(trees.t[tree_ind], Xtestpointer, N_test, p, predictions_test_std[tree_ind]);

                // update sigma based on residual of m - 1 trees, residual_theta_noise
                if (m_update_sigma == false)
                {

                    std::gamma_distribution<double> gamma_samp((N + kap) / 2.0, 2.0 / (sum_squared(residual_std) + s));

                    sigma = 1.0 / sqrt(gamma_samp(generator));

                    sigma_draw_xinfo[sweeps][tree_ind] = sigma;
                }

                // update residual, now it's residual of m trees

                residual_std = residual_std - predictions_std[tree_ind];

                yhat_std = yhat_std + predictions_std[tree_ind];
                yhat_test_std = yhat_test_std + predictions_test_std[tree_ind];

                // treess << trees.t[tree_ind];
            }

            // save predictions to output matrix
            yhats_xinfo[sweeps] = yhat_std;
            yhats_test_xinfo[sweeps] = yhat_test_std;

        }
    }

}
