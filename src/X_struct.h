#ifndef GUARD_X_struct_h
#define GUARD_X_struct_h
#include "common.h"
#include "utility.h"

struct X_struct
{
public:
    std::vector<double> X_values;
    std::vector<size_t> X_counts;
    std::vector<size_t> variable_ind;
    std::vector<size_t> X_num_unique;
    const double *X_std;    // pointer to original data
    const std::vector<double> *y_std; // pointer to y data
    size_t n_y; // number of total data points in root node

    X_struct(const double *X_std, const std::vector<double> *y_std, size_t n_y, std::vector< std::vector<size_t> > &Xorder_std, size_t p_categorical, size_t p_continuous){
        
        this->variable_ind = std::vector<size_t>(p_categorical + 1);
        this->X_num_unique = std::vector<size_t>(p_categorical);

        unique_value_count2(X_std, Xorder_std, X_values, X_counts, variable_ind, n_y, X_num_unique, p_categorical, p_continuous);

        this->X_std = X_std;
        this->y_std = y_std;
        this->n_y = n_y;
    }
};

#endif