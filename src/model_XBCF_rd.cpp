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

void XBCFrdModel::predict_std(matrix<double> &Ztestpointer, const double *Xtestpointer_con, const double *Xtestpointer_mod, size_t N_test, size_t p_con, size_t p_mod, size_t num_trees_con, size_t num_trees_mod, size_t num_sweeps, matrix<double> &yhats_test_xinfo, matrix<double> &prognostic_xinfo, matrix<double> &treatment_xinfo, vector<vector<tree>> &trees_con, vector<vector<tree>> &trees_mod, std::vector<double> &local_ate)
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
            getThetaForObs_Outsample(output_mod, trees_mod[sweeps], data_ind, Xtestpointer_mod, N_test, p_mod);

            getThetaForObs_Outsample(output_con, trees_con[sweeps], data_ind, Xtestpointer_con, N_test, p_con);

            // take sum of predictions of each tree, as final prediction
            for (size_t i = 0; i < trees_mod[0].size(); i++)
            {
                treatment_xinfo[sweeps][data_ind] += output_mod[i][0];
            }

            for (size_t i = 0; i < trees_con[0].size(); i++)
            {
                prognostic_xinfo[sweeps][data_ind] += output_con[i][0];
            }

            if (Ztestpointer[0][data_ind] == 1)
            {
                // yhats_test_xinfo[sweeps][data_ind] = (state.a) * prognostic_xinfo[sweeps][data_ind] + (state.b_vec[1]) * treatment_xinfo[sweeps][data_ind];
            }
            else
            {
                // yhats_test_xinfo[sweeps][data_ind] = (state.a) * prognostic_xinfo[sweeps][data_ind] + (state.b_vec[0]) * treatment_xinfo[sweeps][data_ind];
            }
            yhats_test_xinfo[sweeps][data_ind] = prognostic_xinfo[sweeps][data_ind] + treatment_xinfo[sweeps][data_ind];
        }
    }
    return;
}
