#ifndef GUARD_node_data_h
#define GUARD_node_data_h

#include <ctime>
#include "common.h"
#include "utility.h"
#include <chrono>

struct NodeData
{
public:
    double sigma;
    size_t N_Xorder; // number of data observations in the current node

    NodeData(){
        sigma = 1.0;
        N_Xorder = 1;
        return;
    }

    NodeData(double sigma, size_t N_Xorder){
        this->sigma = sigma;
        this->N_Xorder = N_Xorder;
        return;
    }

    void update_value(double sigma, size_t N_Xorder){
        this->sigma = sigma;
        this->N_Xorder = N_Xorder;
        return;
    }

    void update_sigma(double sigma){
        this->sigma = sigma;
        return;
    }

    void update_N_Xorder(size_t N_Xorder){
        this->N_Xorder = N_Xorder;
        return;
    }
};

// struct NodeData
// {
//     // strcuct of data at each specific node, including Xorder for continuous variable and X_counts for categorical variable
// public:
//     // continuous variables
//     xinfo_sizet Xorder_std;
//     size_t N_Xorder;

//     // categorical variables
//     std::vector<size_t> X_counts;
//     std::vector<size_t> X_num_unique;

//     NodeData(size_t N_Xorder, size_t p, size_t N_X_counts, size_t p_categorical)
//     {
//         // initialize everything, prespecify size of X_counts and X_num_unique
//         ini_xinfo_sizet(Xorder_std, N_Xorder, p);

//         X_counts = std::vector<size_t>(N_X_counts);

//         X_num_unique = std::vector<size_t>(p_categorical);

//         return;
//     }

//     NodeData(size_t N_Xorder, size_t p, const double *Xpointer, State *state)
//     {
//         // initialize Xorder_std with prespecify size
//         // count X_counts and X_num_unique from data
//         // Use for initialize NodeData for ROOT node
//         ini_xinfo_sizet(Xorder_std, N_Xorder, p);

//         this->ini_categorical_var(Xpointer, state);

//         return;
//     }

//     void ini_categorical_var(const double *Xpointer, State *state)
//     {
//         size_t total_points = 0;
//         size_t N = Xorder_std[0].size();
//         size_t p = Xorder_std.size();
//         double current_value = 0.0;
//         size_t count_unique = 0;
//         size_t N_unique;
//         state->variable_ind[0] = 0;

//         for (size_t i = state->p_continuous; i < p; i++)
//         {
//             // only loop over categorical variables
//             // suppose p = (p_continuous, p_categorical)
//             // index starts from p_continuous
//             this->X_counts.push_back(1);
//             current_value = *(Xpointer + i * N + this->Xorder_std[i][0]);
//             state->X_values.push_back(current_value);
//             count_unique = 1;

//             for (size_t j = 1; j < N; j++)
//             {
//                 if (*(Xpointer + i * N + this->Xorder_std[i][j]) == current_value)
//                 {
//                     this->X_counts[total_points]++;
//                 }
//                 else
//                 {
//                     current_value = *(Xpointer + i * N + this->Xorder_std[i][j]);
//                     state->X_values.push_back(current_value);
//                     this->X_counts.push_back(1);
//                     count_unique++;
//                     total_points++;
//                 }
//             }
//             state->variable_ind[i + 1 - state->p_continuous] = count_unique + state->variable_ind[i - state->p_continuous];
//             this->X_num_unique[i - state->p_continuous] = count_unique;
//             total_points++;
//         }
//         return;
//     }
// };

#endif
