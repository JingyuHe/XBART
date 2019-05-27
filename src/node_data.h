#ifndef GUARD_node_data_h
#define GUARD_node_data_h

#include <ctime>
#include "common.h"
#include "utility.h"
#include <chrono>

struct NodeData
{
    // strcuct of data at each specific node, including Xorder for continuous variable and X_counts for categorical variable
    public:
        // continuous variables
        xinfo_sizet Xorder_std;
                size_t N_Xorder;

        // categorical variables
        std::vector<size_t> X_counts;
        std::vector<size_t> X_num_unique;

        // response variable
        double y_mean;
}

#endif