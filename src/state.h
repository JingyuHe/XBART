//////////////////////////////////////////////////////////////////////////////////////
// class to carry all intermediate data vectors, parameters across all functions
//////////////////////////////////////////////////////////////////////////////////////

#ifndef GUARD_fit_info_h
#define GUARD_fit_info_h

#include <ctime>
#include "common.h"
#include "utility.h"
#include <chrono>

class State
{
public:
    size_t dim_residual;          // residual size
    matrix<double> *residual_std; // a matrix to save all residuals
    matrix<size_t> *Xorder_std;

    // random number generators
    std::vector<double> prob;
    std::random_device rd;
    std::mt19937 gen;
    std::discrete_distribution<> d;

    // Splits
    matrix<double> *split_count_all_tree;
    std::vector<double> *split_count_all;
    std::vector<double> *split_count_current_tree;
    std::vector<double> *mtry_weight_current_tree;

    // for XBCF
    matrix<double> *split_count_all_tree_con;
    std::vector<double> *split_count_all_con;
    std::vector<double> *mtry_weight_current_tree_con;

    matrix<double> *split_count_all_tree_mod;
    std::vector<double> *split_count_all_mod;
    std::vector<double> *mtry_weight_current_tree_mod;

    // mtry
    bool use_all = true;
    bool parallel = true;

    // fitinfo
    size_t n_min;
    size_t n_cutpoints;
    size_t p_categorical;
    size_t p_continuous;
    size_t p; // total number of variables = p_categorical + p_continuous
    size_t mtry;
    size_t n_y;                 // number of total data points in root node
    const double *X_std;        // pointer to original data
    std::vector<double> *y_std; // pointer to y data
    size_t max_depth;
    size_t num_trees;
    size_t num_sweeps;
    size_t burnin;
    bool sample_weights;
    double ini_var_yhat;

    // residual standard deviation
    double sigma;
    double sigma2; // sigma squared

    // paralization
    size_t nthread;

    // Logit Model
    // lambdas
    std::vector<std::vector<std::vector<double>>> *lambdas;
    std::vector<std::vector<std::vector<double>>> *lambdas_separate;

    // for continuous treatment XBCF
    matrix<double> *Z_std;
    std::vector<double> *tau_fit;
    std::vector<double> *mu_fit;
    bool treatment_flag;
    matrix<size_t> *Xorder_std_con;
    matrix<size_t> *Xorder_std_mod;
    size_t p_con;
    size_t p_mod;
    size_t p_categorical_con;
    size_t p_categorical_mod;
    size_t p_continuous_con;
    size_t p_continuous_mod;
    size_t mtry_con;
    size_t mtry_mod;
    size_t num_trees_con;
    size_t num_trees_mod;

    // for discrete treatment XBCF
    size_t fl; // flag for likelihood function to alternate between mu loop and tau loop calculations  TODO: move to xbcfState
    // total residual vector
    std::vector<double> residual;
    // residual for treated group, length n_trt
    std::vector<double> full_residual_trt; //(state.n_trt);               // residual for the treated group
    // residual for control group, length n_y - n_trt
    std::vector<double> full_residual_ctrl;
    matrix<double> split_count_all_tree_pr;  // TODO: move to xbcfState
    matrix<double> split_count_all_tree_trt; // TODO: move to xbcfState
    std::vector<double> b_std;        // the scaled treatment vector            TODO: move to xbcfState
    std::vector<size_t> z;            // the scaled treatment vector            TODO: move to xbcfState
    size_t n_trt;                     // the number of treated individuals      TODO: check if it's used anywhere after restructuring
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
    matrix<size_t> *Xorder_std_pr;
    matrix<size_t> *Xorder_std_trt;
    bool a_scaling;
    bool b_scaling;
    size_t N_trt;
    size_t N_ctrl;

    void update_residuals()
    {
        size_t index_trt = 0;  // index of the current observation in the treatment group
        size_t index_ctrl = 0; // index of the current observation in the control group

        for (size_t i = 0; i < this->n_y; i++)
        {
            if (this->z[i] == 1)
            {
                this->full_residual_trt[index_trt] = (*(this->y_std))[i] - (this->a) * (*(this->mu_fit))[i] - (this->b_vec)[1] * (*(this->tau_fit))[i];
                index_trt++;
            }
            else
            {
                this->full_residual_ctrl[index_ctrl] = (*this->y_std)[i] - (this->a) * (*this->mu_fit)[i] - (this->b_vec)[0] * (*this->tau_fit)[i];
                index_ctrl++;
            }
        }
    }

    void update_sigma(double sigma)
    {
        this->sigma = sigma;
        this->sigma2 = pow(sigma, 2);
        return;
    }

    // sigma update for xbcfModel       TODO: move to xbcfClass
    void update_sigma(double sigma, size_t ind)
    {
        this->sigma_vec[ind] = sigma; // sigma for the "ind" group
        return;
    }

    // sigma update for xbcfModel       TODO: move to xbcfClass
    void update_bscales(double b0, double b1)
    {
        this->b_vec[0] = b0; // sigma for the control group
        this->b_vec[1] = b1; // sigma for the treatment group

        return;
    }
    State(const double *Xpointer, matrix<size_t> &Xorder_std, size_t N, size_t p, size_t num_trees, size_t p_categorical, size_t p_continuous, bool set_random_seed, size_t random_seed, size_t n_min, size_t n_cutpoints, size_t mtry, const double *X_std, size_t num_sweeps, bool sample_weights, std::vector<double> *y_std, double sigma, size_t max_depth, double ini_var_yhat, size_t burnin, size_t dim_residual, size_t nthread)
    {

        // Init containers
        // initialize predictions_std at given value / number of trees
        this->residual_std = new matrix<double>();
        ini_matrix((*this->residual_std), N, dim_residual);

        // Random
        this->prob = std::vector<double>(2, 0.5);
        this->gen = std::mt19937(rd());
        if (set_random_seed)
        {
            gen.seed(random_seed);
        }
        this->d = std::discrete_distribution<>(prob.begin(), prob.end());

        // Splits
        this->split_count_all_tree = new matrix<double>();
        ini_xinfo((*this->split_count_all_tree), p, num_trees);
        this->split_count_current_tree = new std::vector<double>(p, 0);
        this->mtry_weight_current_tree = new std::vector<double>(p, 0);
        this->split_count_all = new std::vector<double>(p, 0);
        this->sigma = sigma;
        this->n_min = n_min;
        this->n_cutpoints = n_cutpoints;
        this->p_categorical = p_categorical;
        this->p_continuous = p_continuous;
        this->mtry = mtry;
        this->X_std = X_std;
        this->p = p_categorical + p_continuous;
        this->n_y = N;
        this->num_trees = num_trees;
        this->num_sweeps = num_sweeps;
        this->sample_weights = sample_weights;
        this->y_std = y_std;
        this->max_depth = max_depth;
        this->burnin = burnin;
        this->ini_var_yhat = ini_var_yhat;
        this->Xorder_std = &Xorder_std;
        this->nthread = nthread;

        this->split_count_all_tree_con = NULL;
        this->split_count_all_tree_mod = NULL;
        this->split_count_all_con = NULL;
        this->split_count_all_mod = NULL;
        this->mtry_weight_current_tree_con = NULL;
        this->mtry_weight_current_tree_mod = NULL;
        return;
    }

    //  TODO: update the constructor / get rid of it (if all new vars can be moved to xbcfState constructor)
    State(const double *Xpointer, matrix<size_t> &Xorder_std_pr, matrix<size_t> &Xorder_std_trt, size_t N, size_t p_pr, size_t p_trt, size_t num_trees_con, size_t num_trees_mod, size_t p_categorical_pr, size_t p_categorical_trt, size_t p_continuous_pr, size_t p_continuous_trt, bool set_random_seed, size_t random_seed, size_t n_min, size_t n_cutpoints, bool parallel, size_t mtry_pr, size_t mtry_trt, const double *X_std, size_t num_sweeps, bool sample_weights, std::vector<double> *y_std, std::vector<double> b_std, std::vector<size_t> z, std::vector<double> sigma_vec, std::vector<double> b_vec, size_t max_depth, double ini_var_yhat, size_t burnin, size_t dim_residual)
    {

        // Init containers
        // initialize predictions_std at given value / number of trees
        // ini_xinfo(this->predictions_std, N, num_trees, ini_var_yhat / (double)num_trees);

        // initialize yhat at given value

        // this->residual_std = std::vector<double>(N);
        // this->residual_std_full = std::vector<double>(N);

        // Warning! ini_matrix(matrix, N, p).
        this->residual_std = new matrix<double>();
        ini_matrix((*this->residual_std), N, dim_residual);

        // Random
        this->prob = std::vector<double>(2, 0.5);
        this->gen = std::mt19937(rd());
        if (set_random_seed)
        {
            gen.seed(random_seed);
        }
        this->d = std::discrete_distribution<>(prob.begin(), prob.end());

        // Splits
        ini_xinfo(this->split_count_all_tree_pr, p_pr, num_trees_con);
        ini_xinfo(this->split_count_all_tree_trt, p_trt, num_trees_mod);

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
        this->num_trees_con = num_trees_con;
        this->num_trees_mod = num_trees_mod;
        this->num_sweeps = num_sweeps;
        this->sample_weights = sample_weights;
        this->y_std = y_std;
        this->max_depth = max_depth;
        this->burnin = burnin;
        this->ini_var_yhat = ini_var_yhat;
        this->Xorder_std_pr = &Xorder_std_pr;
        this->Xorder_std_trt = &Xorder_std_trt;

        // those are for XBCF, initialize at a length 1 vector
        this->residual = std::vector<double>(1, 0);
        this->full_residual_ctrl = std::vector<double>(1, 0);
        this->full_residual_trt = std::vector<double>(1, 0);
        return;
    }

    void update_split_counts(size_t tree_ind)
    {
        (*mtry_weight_current_tree) = (*mtry_weight_current_tree) + (*split_count_current_tree);
        (*split_count_all_tree)[tree_ind] = (*split_count_current_tree);
    }

    void update_split_counts(size_t tree_ind, size_t flag)
    {
        (*mtry_weight_current_tree) = (*mtry_weight_current_tree) + (*split_count_current_tree);
        if (flag == 0)
        {
            (split_count_all_tree_pr)[tree_ind] = (*split_count_current_tree);
        }
        else
        {
            (split_count_all_tree_trt)[tree_ind] = (*split_count_current_tree);
        }
        return;
    }

    void iniSplitStorage(size_t flag)
    {
        if (flag == 0)
        {
            this->split_count_current_tree = new std::vector<double>(this->p_pr, 0);
            this->mtry_weight_current_tree = new std::vector<double>(this->p_pr, 0);
        }
        else if (flag == 1)
        {
            this->split_count_current_tree = new std::vector<double>(this->p_trt, 0);
            this->mtry_weight_current_tree = new std::vector<double>(this->p_trt, 0);
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
    NormalState(const double *Xpointer, matrix<size_t> &Xorder_std, size_t N, size_t p, size_t num_trees, size_t p_categorical, size_t p_continuous, bool set_random_seed, size_t random_seed, size_t n_min, size_t n_cutpoints, size_t mtry, const double *X_std, size_t num_sweeps, bool sample_weights, std::vector<double> *y_std, double sigma, size_t max_depth, double ini_var_yhat, size_t burnin, size_t dim_residual, size_t nthread, bool parallel) : State(Xpointer, Xorder_std, N, p, num_trees, p_categorical, p_continuous, set_random_seed, random_seed, n_min, n_cutpoints, mtry, X_std, num_sweeps, sample_weights, y_std, sigma, max_depth, ini_var_yhat, burnin, dim_residual, nthread)
    {
        this->sigma = sigma;
        this->sigma2 = pow(sigma, 2);
        this->parallel = parallel;
    }
};

class LogitState : public State
{

    void ini_lambda(std::vector<std::vector<std::vector<double>>> &lambdas, size_t num_trees, size_t dim_residual)
    {
        // each tree has different number of theta vectors, each is of the size dim_residual (num classes)
        lambdas.resize(num_trees);
    }

    void ini_lambda_separate(std::vector<std::vector<std::vector<double>>> &lambdas, size_t num_trees, size_t dim_residual)
    {
        // each tree have (num classes) of lambda vectors
        lambdas.resize(num_trees);
        for (size_t i = 0; i < num_trees; i++)
        {
            lambdas[i].resize(dim_residual);
        }
    }

public:
    LogitState(const double *Xpointer, matrix<size_t> &Xorder_std, size_t N, size_t p, size_t num_trees, size_t p_categorical, size_t p_continuous, bool set_random_seed, size_t random_seed, size_t n_min, size_t n_cutpoints, size_t mtry, const double *X_std, size_t num_sweeps, bool sample_weights, std::vector<double> *y_std, double sigma, size_t max_depth, double ini_var_yhat, size_t burnin, size_t dim_residual, size_t nthread) : State(Xpointer, Xorder_std, N, p, num_trees, p_categorical, p_continuous, set_random_seed, random_seed, n_min, n_cutpoints, mtry, X_std, num_sweeps, sample_weights, y_std, sigma, max_depth, ini_var_yhat, burnin, dim_residual, nthread)
    {
        this->lambdas = new std::vector<std::vector<std::vector<double>>>();
        this->lambdas_separate = new std::vector<std::vector<std::vector<double>>>();
        ini_lambda((*this->lambdas), num_trees, dim_residual);
        ini_lambda_separate((*this->lambdas_separate), num_trees, dim_residual);
    }
};

class XBCFcontinuousState : public State
{
public:
    XBCFcontinuousState(matrix<double> *Z_std, const double *Xpointer_con, const double *Xpointer_mod, matrix<size_t> &Xorder_std_con, matrix<size_t> &Xorder_std_mod, size_t N, size_t p_con, size_t p_mod, size_t num_trees_con, size_t num_trees_mod, size_t p_categorical_con, size_t p_categorical_mod, size_t p_continuous_con, size_t p_continuous_mod, bool set_random_seed, size_t random_seed, size_t n_min, size_t n_cutpoints, size_t mtry_con, size_t mtry_mod, const double *X_std, size_t num_sweeps, bool sample_weights, std::vector<double> *y_std, double sigma, size_t max_depth, double ini_var_yhat, size_t burnin, size_t dim_residual, size_t nthread, bool parallel) : State(Xpointer_con, Xorder_std_con, N, p_con, num_trees_con, p_categorical_con, p_continuous_con, set_random_seed, random_seed, n_min, n_cutpoints, mtry_con, Xpointer_con, num_sweeps, sample_weights, y_std, sigma, max_depth, ini_var_yhat, burnin, dim_residual, nthread)
    {
        this->split_count_all_tree_con = new matrix<double>();
        this->split_count_all_tree_mod = new matrix<double>();
        ini_xinfo((*this->split_count_all_tree_con), p_con, num_trees_con);
        ini_xinfo((*this->split_count_all_tree_mod), p_mod, num_trees_mod);
        this->split_count_all_con = new std::vector<double>(p_con, 0);
        this->mtry_weight_current_tree_con = new std::vector<double>(p_con, 0);
        this->split_count_all_mod = new std::vector<double>(p_mod, 0);
        this->mtry_weight_current_tree_mod = new std::vector<double>(p_mod, 0);
        this->Z_std = Z_std;
        this->sigma = sigma;
        this->sigma2 = pow(sigma, 2);
        this->parallel = parallel;
        this->tau_fit = (new std::vector<double>(N, 0));
        this->mu_fit = (new std::vector<double>(N, 0));
        this->Xorder_std_con = &Xorder_std_con;
        this->Xorder_std_mod = &Xorder_std_mod;
        this->p_con = p_con;
        this->p_mod = p_mod;
        this->p_categorical_con = p_categorical_con;
        this->p_categorical_mod = p_categorical_mod;
        this->p_continuous_con = p_continuous_con;
        this->p_continuous_mod = p_continuous_mod;
        this->mtry_con = mtry_con;
        this->mtry_mod = mtry_mod;
        this->num_trees_con = num_trees_con;
        this->num_trees_mod = num_trees_mod;
    }
};

class XBCFdiscreteState : public State
{
public:
    XBCFdiscreteState(matrix<double> *Z_std, const double *Xpointer_con, const double *Xpointer_mod, matrix<size_t> &Xorder_std_con, matrix<size_t> &Xorder_std_mod, size_t N, size_t p_con, size_t p_mod, size_t num_trees_con, size_t num_trees_mod, size_t p_categorical_con, size_t p_categorical_mod, size_t p_continuous_con, size_t p_continuous_mod, bool set_random_seed, size_t random_seed, size_t n_min, size_t n_cutpoints, size_t mtry_con, size_t mtry_mod, const double *X_std, size_t num_sweeps, bool sample_weights, std::vector<double> *y_std, double sigma, size_t max_depth, double ini_var_yhat, size_t burnin, size_t dim_residual, size_t nthread, bool parallel, bool a_scaling, bool b_scaling, size_t N_trt, size_t N_ctrl) : State(Xpointer_con, Xorder_std_con, N, p_con, num_trees_con, p_categorical_con, p_continuous_con, set_random_seed, random_seed, n_min, n_cutpoints, mtry_con, Xpointer_con, num_sweeps, sample_weights, y_std, sigma, max_depth, ini_var_yhat, burnin, dim_residual, nthread)
    {
        this->split_count_all_tree_con = new matrix<double>();
        this->split_count_all_tree_mod = new matrix<double>();
        ini_xinfo((*this->split_count_all_tree_con), p_con, num_trees_con);
        ini_xinfo((*this->split_count_all_tree_mod), p_mod, num_trees_mod);
        this->split_count_all_con = new std::vector<double>(p_con, 0);
        this->mtry_weight_current_tree_con = new std::vector<double>(p_con, 0);
        this->split_count_all_mod = new std::vector<double>(p_mod, 0);
        this->mtry_weight_current_tree_mod = new std::vector<double>(p_mod, 0);
        this->Z_std = Z_std;
        this->sigma = sigma;
        this->sigma2 = pow(sigma, 2);
        this->parallel = parallel;
        this->tau_fit = (new std::vector<double>(N, 0));
        this->mu_fit = (new std::vector<double>(N, 0));
        this->Xorder_std_con = &Xorder_std_con;
        this->Xorder_std_mod = &Xorder_std_mod;
        this->p_con = p_con;
        this->p_mod = p_mod;
        this->p_categorical_con = p_categorical_con;
        this->p_categorical_mod = p_categorical_mod;
        this->p_continuous_con = p_continuous_con;
        this->p_continuous_mod = p_continuous_mod;
        this->mtry_con = mtry_con;
        this->mtry_mod = mtry_mod;
        this->num_trees_con = num_trees_con;
        this->num_trees_mod = num_trees_mod;
        this->a_scaling = a_scaling;
        this->b_scaling = b_scaling;
        this->N_trt = N_trt;
        this->N_ctrl = N_ctrl;
    }
};

class xbcfState : public State
{
public:
    xbcfState(const double *Xpointer, matrix<size_t> &Xorder_std_pr, matrix<size_t> &Xorder_std_trt, size_t N, size_t n_trt, size_t p, size_t p_tau, size_t num_trees_con, size_t num_trees_mod, size_t p_categorical_pr, size_t p_categorical_trt, size_t p_continuous_pr, size_t p_continuous_trt, bool set_random_seed, size_t random_seed, size_t n_min, size_t n_cutpoints, bool parallel, size_t mtry_pr, size_t mtry_trt, const double *X_std, size_t num_sweeps, bool sample_weights, std::vector<double> *y_std, std::vector<double> b_std, std::vector<size_t> z, std::vector<double> sigma_vec, std::vector<double> b_vec, size_t max_depth, double ini_var_yhat, size_t burnin, size_t dim_residual) : State(Xpointer, Xorder_std_pr, Xorder_std_trt, N, p, p_tau, num_trees_con, num_trees_mod, p_categorical_pr, p_categorical_trt, p_continuous_pr, p_continuous_trt, set_random_seed, random_seed, n_min, n_cutpoints, parallel, mtry_pr, mtry_trt, X_std, num_sweeps, sample_weights, y_std, b_std, z, sigma_vec, b_vec, max_depth, ini_var_yhat, burnin, dim_residual)
    {
        this->sigma_vec = sigma_vec;
        this->b_vec = b_vec;
        this->n_trt = n_trt;
        this->num_trees_con = num_trees_con;
        this->num_trees_mod = num_trees_mod;
        this->b_std = b_std;
        this->z = z;
        this->a = 1; // initialize a at 1 for now

        this->mu_fit = new std::vector<double>(N, 0);
        this->tau_fit = new std::vector<double>(N, 0);

        // those are for XBCF, initialize at a length 1 vector
        this->residual = std::vector<double>(N, 0);
        this->full_residual_ctrl = std::vector<double>(N - n_trt, 0);
        this->full_residual_trt = std::vector<double>(n_trt, 0);
    }
};

#endif
