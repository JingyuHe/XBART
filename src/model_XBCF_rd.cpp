#include "tree.h"
#include "model.h"
#include <cfenv>

//////////////////////////////////////////////////////////////////////////////////////
//
//
//  binary treatment XBCF for regression discontinuity design extrpolation
//
//
//////////////////////////////////////////////////////////////////////////////////////

void XBCFrdModel::predict_std(matrix<size_t> &Xorder_std, rd_struct &x_struct, std::vector<size_t> &X_counts, std::vector<size_t> &X_num_unique,
                            matrix<size_t> &Xtestorder_std, rd_struct &xtest_struct, std::vector<size_t> &Xtest_counts, std::vector<size_t> &Xtest_num_unique,
                            const double *Xtestpointer_con, const double *Xtestpointer_mod,
                            size_t N_test, size_t p_con, size_t p_mod, size_t num_trees_con, size_t num_trees_mod, size_t num_sweeps,
                            matrix<double> &prognostic_xinfo, matrix<double> &treatment_xinfo,
                            vector<vector<tree>> &trees_con, vector<vector<tree>> &trees_mod,
                            const double &theta, const double &tau)
{
    // predict the output as a matrix
    matrix<double> output_mod;

    // row : dimension of theta, column : number of trees
    ini_matrix(output_mod, this->dim_theta, trees_mod[0].size());

    matrix<double> output_con;
    ini_matrix(output_con, this->dim_theta, trees_con[0].size());

    for (size_t sweeps = 0; sweeps < num_sweeps; sweeps++)
    {
        for (size_t data_ind = 0; data_ind < N_test; data_ind++)
        {
            // getThetaForObs_Outsample(output_mod, trees_mod[sweeps], data_ind, Xtestpointer_mod, N_test, p_mod);

            getThetaForObs_Outsample(output_con, trees_con[sweeps], data_ind, Xtestpointer_con, N_test, p_con);

            // take sum of predictions of each tree, as final prediction
            // for (size_t i = 0; i < trees_mod[0].size(); i++)
            // {
            //     treatment_xinfo[sweeps][data_ind] += output_mod[i][0];
            // }

            for (size_t i = 0; i < trees_con[0].size(); i++)
            {
                prognostic_xinfo[sweeps][data_ind] += output_con[i][0];
            }

        }

        // get local ate
        std::vector<double> local_ate(num_trees_mod, 0.0);
        const double *run_var_x_pointer = x_struct.X_std + x_struct.n_y * (x_struct.p_continuous - 1);
        double run_var_value;
        size_t count_local = 0;
        for (size_t data_ind = 0; data_ind < x_struct.n_y; data_ind ++){
            run_var_value = *(run_var_x_pointer + data_ind);
            if ( (run_var_value <= x_struct.cutoff + x_struct.Owidth) & (run_var_value >= x_struct.cutoff - x_struct.Owidth) ){
                count_local += 1;
                getThetaForObs_Outsample(output_mod, trees_mod[sweeps], data_ind, x_struct.X_std, x_struct.n_y, p_mod);

                for (size_t tree_ind = 0; tree_ind < num_trees_mod; tree_ind++){
                    local_ate[tree_ind] += output_mod[tree_ind][0];
                }
            }
        }

        for (size_t tree_ind = 0; tree_ind < num_trees_mod; tree_ind++)
        {
            // cout << "sweeps " << sweeps << " tree " << tree_ind << " ate " << local_ate[tree_ind] / count_local << endl;
            std::vector<bool> active_var(Xorder_std.size(), false);
             trees_mod[sweeps][tree_ind].rd_predict_from_root(Xorder_std, x_struct, X_counts, X_num_unique, Xtestorder_std, xtest_struct, Xtest_counts, Xtest_num_unique,
                              treatment_xinfo, active_var, sweeps, tree_ind, theta, tau, local_ate[tree_ind] / count_local);
            // TODO: local_ate should be obtained on the tree level.
        }


    }
    return;
}
