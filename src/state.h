#ifndef GUARD_fit_info_h
#define GUARD_fit_info_h

#include <ctime>
#include "common.h"
#include "utility.h"
#include <chrono>

class State
{
public:
    // residual size
    size_t dim_residual;

    // vectors (slop?)
    matrix<double> residual_std;

    // Random
    std::vector<double> prob;
    std::random_device rd;
    std::mt19937 gen;
    std::discrete_distribution<> d;

    // Splits
    matrix<double> split_count_all_tree;
    matrix<double> split_count_all_tree_pr;  // TODO: move to xbcfState
    matrix<double> split_count_all_tree_trt; // TODO: move to xbcfState
    std::vector<double> split_count_current_tree;
    std::vector<double> mtry_weight_current_tree;

    // mtry
    bool use_all = true;

    // fitinfo
    size_t n_min;
    size_t n_cutpoints;
    bool parallel;
    size_t p_categorical;
    size_t p_continuous;
    size_t p; // total number of variables = p_categorical + p_continuous
    size_t mtry;
    size_t n_y;                       // number of total data points in root node
    const double *X_std;              // pointer to original data
    const std::vector<double> *y_std; // pointer to y data
    std::vector<double> b_std;        // the scaled treatment vector            TODO: move to xbcfState
    std::vector<size_t> z;            // the scaled treatment vector            TODO: move to xbcfState
    size_t n_trt;                     // the number of treated individuals      TODO: check if it's used anywhere after restructuring
    std::vector<double> mu_fit;       // total mu_fit                           TODO: move to xbcfState
    std::vector<double> tau_fit;      // total tau_fit                          TODO: move to xbcfState
    std::vector<double> b_vec;        // scaling parameters for tau (b0,b1)     TODO: move to xbcfState
    std::vector<double> sigma_vec;    // residual standard deviations           TODO: move to xbcfState
    double a;                         // scaling parameter for mu               TODO: move to xbcfState
    size_t p_categorical_pr;          // TODO: move to xbcfState
    size_t p_continuous_pr;           // TODO: move to xbcfState
    size_t p_categorical_trt;         // TODO: move to xbcfState
    size_t p_continuous_trt;          // TODO: move to xbcfState
    size_t p_pr;                      // total number of variables for mu          TODO: move to xbcfState
    size_t p_trt;                     // total number of variables for tau          TODO: move to xbcfState
    size_t mtry_pr;                   // TODO: move to xbcfState
    size_t mtry_trt;                  // TODO: move to xbcfState

    size_t max_depth;
    size_t num_trees;
    std::vector<size_t> num_trees_vec;
    size_t num_sweeps;
    size_t burnin;
    bool sample_weights_flag;
    double ini_var_yhat;
    size_t fl; // flag for likelihood function to alternate between mu loop and tau loop calculations  TODO: move to xbcfState

    matrix<size_t> Xorder_std;

    // residual standard deviation
    double sigma;
    double sigma2; // sigma squared

    //std::vector<double> precision_squared;

    void update_sigma(double sigma)
    {
        this->sigma = sigma;
        this->sigma2 = pow(sigma, 2);
        return;
    }

    // sigma update for xbcfModel       TODO: move to xbcfClass
    void update_sigma(double sigma0, double sigma1)
    {
        this->sigma_vec[0] = sigma0; // sigma for the control group
        this->sigma_vec[1] = sigma1; // sigma for the treatment group

        return;
    }

    // sigma update for xbcfModel       TODO: move to xbcfClass
    void update_bscales(double b0, double b1)
    {
        this->b_vec[0] = b0; // sigma for the control group
        this->b_vec[1] = b1; // sigma for the treatment group

        return;
    }

    // update precision squared vector (also scales included) based on recently updated sigmas
    /*    void update_precision_squared(double sigma0, double sigma1)
    {

        // update for all individuals in the treatment group
        for (size_t i = 0; i < this->n_trt - 1; i++)
        {
            this->precision_squared[i] = pow(this->b_std[i], 2) / pow(sigma1, 2);
        }

        // update for all individuals in the control group
        for (size_t i = this->n_trt; i < this->n_y - 1; i++)
        {
            this->precision_squared[i] = pow(this->b_std[i], 2) / pow(sigma0, 2);
        }
        return;
    }
*/
    State(const double *Xpointer, matrix<size_t> &Xorder_std, size_t N, size_t p, size_t num_trees, size_t p_categorical, size_t p_continuous, bool set_random_seed, size_t random_seed, size_t n_min, size_t n_cutpoints, bool parallel, size_t mtry, const double *X_std, size_t num_sweeps, bool sample_weights_flag, std::vector<double> *y_std, double sigma, size_t max_depth, double ini_var_yhat, size_t burnin, size_t dim_residual)
    {

        // Init containers
        // initialize predictions_std at given value / number of trees
        // ini_xinfo(this->predictions_std, N, num_trees, ini_var_yhat / (double)num_trees);

        // initialize yhat at given value

        // this->residual_std = std::vector<double>(N);
        // this->residual_std_full = std::vector<double>(N);

        // Warning! ini_matrix(matrix, N, p).
        ini_matrix(this->residual_std, N, dim_residual);

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
        this->mtry_weight_current_tree = std::vector<double>(p, 0);
        this->sigma = sigma;
        
        this->n_min = n_min;
        this->n_cutpoints = n_cutpoints;
        this->parallel = parallel;
        this->p_categorical = p_categorical;
        this->p_continuous = p_continuous;
        this->mtry = mtry;
        this->X_std = X_std;
        this->p = p_categorical + p_continuous;
        this->n_y = N;
        this->num_trees = num_trees;
        this->num_sweeps = num_sweeps;
        this->sample_weights_flag = sample_weights_flag;
        this->y_std = y_std;
        this->max_depth = max_depth;
        this->burnin = burnin;
        this->ini_var_yhat = ini_var_yhat;
        this->Xorder_std = Xorder_std;

        return;
    }

    //  TODO: update the constructor / get rid of it (if all new vars can be moved to xbcfState constructor)
    State(const double *Xpointer, matrix<size_t> &Xorder_std, size_t N, size_t p_pr, size_t p_trt, std::vector<size_t> num_trees_vec, size_t p_categorical_pr, size_t p_categorical_trt, size_t p_continuous_pr, size_t p_continuous_trt, bool set_random_seed, size_t random_seed, size_t n_min, size_t n_cutpoints, bool parallel, size_t mtry_pr, size_t mtry_trt, const double *X_std, size_t num_sweeps, bool sample_weights_flag, std::vector<double> *y_std, std::vector<double> b_std, std::vector<size_t> z, std::vector<double> sigma_vec, std::vector<double> b_vec, size_t max_depth, double ini_var_yhat, size_t burnin, size_t dim_residual)
    {

        // Init containers
        // initialize predictions_std at given value / number of trees
        // ini_xinfo(this->predictions_std, N, num_trees, ini_var_yhat / (double)num_trees);

        // initialize yhat at given value

        // this->residual_std = std::vector<double>(N);
        // this->residual_std_full = std::vector<double>(N);

        // Warning! ini_matrix(matrix, N, p).
        ini_matrix(this->residual_std, N, dim_residual);

        // Random
        this->prob = std::vector<double>(2, 0.5);
        this->gen = std::mt19937(rd());
        if (set_random_seed)
        {
            gen.seed(random_seed);
        }
        this->d = std::discrete_distribution<>(prob.begin(), prob.end());

        // Splits
        ini_xinfo(this->split_count_all_tree_pr, p_pr, num_trees_vec[0]);
        ini_xinfo(this->split_count_all_tree_trt, p_trt, num_trees_vec[1]);

        this->n_min = n_min;
        this->n_cutpoints = n_cutpoints;
        this->parallel = parallel;
        this->p_categorical_pr = p_categorical_pr;
        this->p_continuous_pr = p_continuous_pr;
        this->p_categorical_trt = p_categorical_trt;
        this->p_continuous_trt = p_continuous_trt;
        this->mtry_pr = mtry_pr;
        this->mtry_trt = mtry_trt;
        this->X_std = X_std;
        this->p_pr = p_categorical_pr + p_continuous_pr;
        this->p_trt = p_categorical_trt + p_continuous_trt;
        this->n_y = N;
        this->num_trees_vec = num_trees_vec; // stays the same even for vector
        this->num_sweeps = num_sweeps;
        this->sample_weights_flag = sample_weights_flag;
        this->y_std = y_std;
        this->max_depth = max_depth;
        this->burnin = burnin;
        this->ini_var_yhat = ini_var_yhat;
        this->Xorder_std = Xorder_std;

        return;
    }

    void update_split_counts(size_t tree_ind)
    {
        mtry_weight_current_tree = mtry_weight_current_tree + split_count_current_tree;
        split_count_all_tree[tree_ind] = split_count_current_tree;
    }

    void update_split_counts(size_t tree_ind, size_t flag)
    {
        mtry_weight_current_tree = mtry_weight_current_tree + split_count_current_tree;
        if (flag == 0)
        {
            split_count_all_tree_pr[tree_ind] = split_count_current_tree;
        }
        else
        {
            split_count_all_tree_trt[tree_ind] = split_count_current_tree;
        }
        return;
    }

    void iniSplitStorage(size_t flag)
    {
        if (flag == 0)
        {
            this->split_count_current_tree = std::vector<double>(this->p_pr, 0);
            this->mtry_weight_current_tree = std::vector<double>(this->p_pr, 0);
        }
        else if (flag == 1)
        {
            this->split_count_current_tree = std::vector<double>(this->p_trt, 0);
            this->mtry_weight_current_tree = std::vector<double>(this->p_trt, 0);
        }
    }

    void adjustMtry(size_t flag)
    {
        if (flag == 0)
        {
            this->mtry = this->mtry_pr;
        }
        else if (flag == 1)
        {
            this->mtry = this->mtry_trt;
        }
    }
};

class NormalState : public State
{
public:
    NormalState(const double *Xpointer, matrix<size_t> &Xorder_std, size_t N, size_t p, size_t num_trees, size_t p_categorical, size_t p_continuous, bool set_random_seed, size_t random_seed, size_t n_min, size_t n_cutpoints, bool parallel, size_t mtry, const double *X_std, size_t num_sweeps, bool sample_weights_flag, std::vector<double> *y_std, double sigma, size_t max_depth, double ini_var_yhat, size_t burnin, size_t dim_residual) : State(Xpointer, Xorder_std, N, p, num_trees, p_categorical, p_continuous, set_random_seed, random_seed, n_min, n_cutpoints, parallel, mtry, X_std, num_sweeps, sample_weights_flag, y_std, sigma, max_depth, ini_var_yhat, burnin, dim_residual)
    {
        this->sigma = sigma;
        this->sigma2 = pow(sigma, 2);
    }


};

class xbcfState : public State
{
public:
    xbcfState(const double *Xpointer, matrix<size_t> &Xorder_std, size_t N, size_t n_trt, size_t p, size_t p_tau, std::vector<size_t> num_trees_vec, size_t p_categorical_pr, size_t p_categorical_trt, size_t p_continuous_pr, size_t p_continuous_trt, bool set_random_seed, size_t random_seed, size_t n_min, size_t n_cutpoints, bool parallel, size_t mtry_pr, size_t mtry_trt, const double *X_std, size_t num_sweeps, bool sample_weights_flag, std::vector<double> *y_std, std::vector<double> b_std, std::vector<size_t> z, std::vector<double> sigma_vec, std::vector<double> b_vec, size_t max_depth, double ini_var_yhat, size_t burnin, size_t dim_residual) : State(Xpointer, Xorder_std, N, p, p_tau, num_trees_vec, p_categorical_pr, p_categorical_trt, p_continuous_pr, p_continuous_trt, set_random_seed, random_seed, n_min, n_cutpoints, parallel, mtry_pr, mtry_trt, X_std, num_sweeps, sample_weights_flag, y_std, b_std, z, sigma_vec, b_vec, max_depth, ini_var_yhat, burnin, dim_residual)
    {
        this->sigma_vec = sigma_vec;
        this->b_vec = b_vec;
        this->n_trt = n_trt;
        this->num_trees_vec = num_trees_vec;
        this->b_std = b_std;
        this->z = z;
        this->a = 1; // initialize a at 1 for now

        this->mu_fit = std::vector<double>(N, 0);
        this->tau_fit = std::vector<double>(N, 0);
    }
};

#endif