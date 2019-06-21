#ifndef GUARD_X_struct_h
#define GUARD_X_struct_h
#include "common.h"
#include "utility.h"

struct X_struct
{
public:
    // Vector pointers
    matrix<std::vector<double> *> data_pointers;
    // copy of data_pointers object, for MH update
    matrix<std::vector<double> *> data_pointers_copy;


    std::vector<double> X_values;
    std::vector<size_t> X_counts;
    std::vector<size_t> variable_ind;
    std::vector<size_t> X_num_unique;
    const double *X_std;    // pointer to original data
    const std::vector<double> *y_std; // pointer to y data
    size_t n_y; // number of total data points in root node

    X_struct(const double *X_std, const std::vector<double> *y_std, size_t n_y, std::vector< std::vector<size_t> > &Xorder_std, size_t p_categorical, size_t p_continuous, std::vector<double> *initial_theta, size_t num_trees){

        this->variable_ind = std::vector<size_t>(p_categorical + 1);
        this->X_num_unique = std::vector<size_t>(p_categorical);

        init_tree_pointers(initial_theta, n_y, num_trees);

        unique_value_count2(X_std, Xorder_std, X_values, X_counts, variable_ind, n_y, X_num_unique, p_categorical, p_continuous);

        this->X_std = X_std;
        this->y_std = y_std;
        this->n_y = n_y;
        this->data_pointers_copy = this->data_pointers;
    }



    void create_backup_data_pointers()
    {
        // create a backup copy of data_pointers
        // used in MH adjustment
        data_pointers_copy = data_pointers;
        return;
    }

    void restore_data_pointers(size_t tree_ind)
    {
        // restore pointers of one tree from data_pointers_copy
        // used in MH adjustment
        data_pointers[tree_ind] = data_pointers_copy[tree_ind];
        return;
    }

    void init_tree_pointers(std::vector<double> *initial_theta, size_t N, size_t num_trees)
    {
        ini_matrix(data_pointers, N, num_trees);
        for (size_t i = 0; i < num_trees; i++)
        {
            std::vector<std::vector<double> *> &pointer_vec = data_pointers[i];
            for (size_t j = 0; j < N; j++)
            {
                pointer_vec[j] = initial_theta;
            }
        }
    }
};

#endif