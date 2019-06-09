#ifndef GUARD_fit_info_h
#define GUARD_fit_info_h

#include <ctime>
#include "common.h"
#include "utility.h"
#include <chrono>

struct FitInfo
{
  public:
    // Categorical
    bool categorical_variables = false;
    std::vector<double> X_values;
    std::vector<size_t> X_counts;
    std::vector<size_t> variable_ind;
    size_t total_points;
    std::vector<size_t> X_num_unique;

    // Result containers
    xinfo predictions_std;
    std::vector<double> yhat_std;
    std::vector<double> residual_std;
    std::vector<double> residual_std_full;

    // Random
    std::vector<double> prob;
    std::random_device rd;
    std::mt19937 gen;
    std::discrete_distribution<> d;

    // Splits
    xinfo split_count_all_tree;
    std::vector<double> split_count_current_tree;
    std::vector<double> mtry_weight_current_tree;

    // mtry
    bool use_all = true;

    // Vector pointers
    matrix<std::vector<double>*> data_pointers;
    void init_tree_pointers(std::vector<double>* initial_theta, size_t N, size_t num_trees)
    {
        ini_matrix(data_pointers, N, num_trees);
        for (size_t i = 0; i < num_trees; i++)
        {
            std::vector<std::vector<double>*> &pointer_vec = data_pointers[i];
            for (size_t j = 0; j < N; j++)
            {
                pointer_vec[j] = initial_theta;
            }
        }
    }
    FitInfo(const double *Xpointer, xinfo_sizet &Xorder_std, size_t N, size_t p,
            size_t num_trees, size_t p_categorical, size_t p_continuous,
            bool set_random_seed, size_t random_seed, std::vector<double>* initial_theta)
    {

        // Handle Categorical
        if (p_categorical > 0)
        {
            this->categorical_variables = true;
        }
        this->variable_ind = std::vector<size_t>(p_categorical + 1);
        this->X_num_unique = std::vector<size_t>(p_categorical);
        unique_value_count2(Xpointer, Xorder_std, this->X_values, this->X_counts,
                            this->variable_ind, this->total_points, this->X_num_unique, p_categorical, p_continuous);

        // // Init containers
        ini_xinfo(this->predictions_std, N, num_trees);

        yhat_std = std::vector<double>(N);
        row_sum(this->predictions_std, this->yhat_std);
        this->residual_std = std::vector<double>(N);

        // Random
        this->prob = std::vector<double>(2, 0.5);
        this->gen = std::mt19937(rd());
        if (set_random_seed)
        {
            gen.seed(random_seed);
        }
        this->d = std::discrete_distribution<>(prob.begin(), prob.end());

        // Splits
        ini_xinfo(this->split_count_all_tree, p, num_trees);
        this->split_count_current_tree = std::vector<double>(p, 0);
        this->mtry_weight_current_tree = std::vector<double>(p, 0.1);

        init_tree_pointers(initial_theta, N, num_trees);
    }
};

#endif