#ifndef GUARD_node_data_h
#define GUARD_node_data_h

#include <ctime>
#include "common.h"
#include "utility.h"
#include <chrono>
#include "fit_info.h"

class NodeData
{
    // strcuct of data at each specific node, including Xorder for continuous variable and X_counts for categorical variable
public:
    // continuous variables
    xinfo_sizet Xorder_std;
    size_t N_Xorder; // number of data observations in current node

    // categorical variables
    std::vector<size_t> X_counts;
    std::vector<size_t> X_num_unique;

    // constructor
    NodeData(size_t N_Xorder, size_t p, size_t p_categorical)
    {
        this->X_num_unique = std::vector<size_t>(p_categorical);

        this->N_Xorder = N_Xorder;
        return;
    }

    NodeData(size_t N_Xorder, size_t p, size_t p_categorical, size_t X_counts_size, size_t X_num_unique_size)
    {
        this->X_num_unique = std::vector<size_t>(p_categorical);
        this->X_counts = std::vector<size_t>(X_counts_size);
        this->X_num_unique = std::vector<size_t>(X_num_unique_size);
        this->N_Xorder = N_Xorder;
        return;
    }

    void unique_value_count(std::unique_ptr<FitInfo> &fit_info);
    
};

#endif