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

//    std::vector<double> *var_fit;    // for heteroskedastic XBART
//
    // residual standard deviation
    double sigma;
    double sigma2; // sigma squared

    // for heteroskedastic case
    // std::vector<double> sigma_vec; // residual standard deviation

    // paralization
    size_t nthread;

    // Logit Model
    // lambdas
    std::vector<std::vector<std::vector<double>>> *lambdas;
    std::vector<std::vector<std::vector<double>>> *lambdas_separate;
    size_t weight_exponent;
    double logloss_last_sweep;

// for heteroskedastic XBART
    matrix<double> *split_count_all_tree_m;
    std::vector<double> *split_count_all_m;
    std::vector<double> *mtry_weight_current_tree_m;
    matrix<double> *split_count_all_tree_v;
    std::vector<double> *split_count_all_v;
    std::vector<double> *mtry_weight_current_tree_v;

    bool forest_flag; // 0 = mean forest; 1 = variance forest
    std::vector<double> *mean_res; // temporary storage for mean model residual
    std::vector<double> *precision;
    std::vector<double> *res_x_precision;
    size_t n_min_m;
    size_t n_min_v;
    size_t n_cutpoints_m;
    size_t n_cutpoints_v;
    size_t max_depth_m;
    size_t max_depth_v;
    size_t num_trees_m;
    size_t num_trees_v;
    size_t mtry_v;

    // for continuous treatment XBCF
    matrix<double> *split_count_all_tree_con;
    std::vector<double> *split_count_all_con;
    std::vector<double> *mtry_weight_current_tree_con;
    matrix<double> *split_count_all_tree_mod;
    std::vector<double> *split_count_all_mod;
    std::vector<double> *mtry_weight_current_tree_mod;
    const double *X_std_con; // pointer to original data
    const double *X_std_mod; // pointer to original data

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

    // extra variables for binary treatment XBCF
    std::vector<double> b_vec;     // scaling parameters for tau (b0,b1)     TODO: move to xbcfState
    // a is also used for logit model
    double a;                      // scaling parameter for mu               TODO: move to xbcfState
    std::vector<double> sigma_vec; // residual standard deviations           TODO: move to xbcfState
    bool a_scaling;
    bool b_scaling;
    size_t N_trt;
    size_t N_ctrl;


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

        this->split_count_all_tree_m = NULL;
        this->split_count_all_tree_v = NULL;
        this->split_count_all_m = NULL;
        this->split_count_all_v = NULL;
        this->mtry_weight_current_tree_m = NULL;
        this->mtry_weight_current_tree_v = NULL;
        return;
    }

    void update_split_counts(size_t tree_ind)
    {
        (*mtry_weight_current_tree) = (*mtry_weight_current_tree) + (*split_count_current_tree);
        (*split_count_all_tree)[tree_ind] = (*split_count_current_tree);
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
//        this->var_fit = (new std::vector<double>(N, 0)); //initialize fit at log(1)=0 for all observations
    }
};

class LogitState : public State
{

    void ini_lambda(std::vector<std::vector<std::vector<double>>> &lambdas, size_t num_trees, size_t dim_residual)
    {
        // TODO: the two lambda structure can be merged, change this to lambda_separate structure
        // each tree has different number of theta vectors, each is of the size dim_residual (num classes)
        lambdas.resize(num_trees);
        for (size_t i = 0; i < num_trees; i++)
        {
            lambdas[i].resize(1);
            lambdas[i][0].resize(dim_residual);
            std::fill(lambdas[i][0].begin(), lambdas[i][0].end(), 1.0);
        }
    }

    void ini_lambda_separate(std::vector<std::vector<std::vector<double>>> &lambdas, size_t num_trees, size_t dim_residual)
    {
        // each tree have (num classes) of lambda vectors
        lambdas.resize(num_trees);
        for (size_t i = 0; i < num_trees; i++)
        {
            lambdas[i].resize(dim_residual);
            for(size_t j = 0; j < dim_residual; j++){
                lambdas[i][j].push_back(1);
            }
        }
    }

public:
    LogitState(const double *Xpointer, matrix<size_t> &Xorder_std, size_t N, size_t p, size_t num_trees, size_t p_categorical, size_t p_continuous, bool set_random_seed, size_t random_seed, size_t n_min, size_t n_cutpoints, size_t mtry, const double *X_std, size_t num_sweeps, bool sample_weights, std::vector<double> *y_std, double sigma, size_t max_depth, double ini_var_yhat, size_t burnin, size_t dim_residual, size_t nthread, double a, size_t weight_exponent) : State(Xpointer, Xorder_std, N, p, num_trees, p_categorical, p_continuous, set_random_seed, random_seed, n_min, n_cutpoints, mtry, X_std, num_sweeps, sample_weights, y_std, sigma, max_depth, ini_var_yhat, burnin, dim_residual, nthread)
    {
        this->a = a;
        this->weight_exponent = weight_exponent;
        this->lambdas = new std::vector<std::vector<std::vector<double>>>();
        this->lambdas_separate = new std::vector<std::vector<std::vector<double>>>();
        ini_lambda((*this->lambdas), num_trees, dim_residual);
        ini_lambda_separate((*this->lambdas_separate), num_trees, dim_residual);
    }
};

class XBCFcontinuousState : public State
{
public:
    XBCFcontinuousState(matrix<double> *Z_std, const double *Xpointer_con, const double *Xpointer_mod, matrix<size_t> &Xorder_std_con, matrix<size_t> &Xorder_std_mod, size_t N, size_t p_con, size_t p_mod, size_t num_trees_con, size_t num_trees_mod, size_t p_categorical_con, size_t p_categorical_mod, size_t p_continuous_con, size_t p_continuous_mod, bool set_random_seed, size_t random_seed, size_t n_min, size_t n_cutpoints, size_t mtry_con, size_t mtry_mod, size_t num_sweeps, bool sample_weights, std::vector<double> *y_std, double sigma, size_t max_depth, double ini_var_yhat, size_t burnin, size_t dim_residual, size_t nthread, bool parallel) : State(Xpointer_con, Xorder_std_con, N, p_con, num_trees_con, p_categorical_con, p_continuous_con, set_random_seed, random_seed, n_min, n_cutpoints, mtry_con, Xpointer_con, num_sweeps, sample_weights, y_std, sigma, max_depth, ini_var_yhat, burnin, dim_residual, nthread)
    {
        this->X_std_con = Xpointer_con;
        this->X_std_mod = Xpointer_mod;
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
    XBCFdiscreteState(matrix<double> *Z_std, const double *Xpointer_con, const double *Xpointer_mod, matrix<size_t> &Xorder_std_con, matrix<size_t> &Xorder_std_mod, size_t N, size_t p_con, size_t p_mod, size_t num_trees_con, size_t num_trees_mod, size_t p_categorical_con, size_t p_categorical_mod, size_t p_continuous_con, size_t p_continuous_mod, bool set_random_seed, size_t random_seed, size_t n_min, size_t n_cutpoints, size_t mtry_con, size_t mtry_mod, size_t num_sweeps, bool sample_weights, std::vector<double> *y_std, double sigma, size_t max_depth, double ini_var_yhat, size_t burnin, size_t dim_residual, size_t nthread, bool parallel, bool a_scaling, bool b_scaling, size_t N_trt, size_t N_ctrl) : State(Xpointer_con, Xorder_std_con, N, p_con, num_trees_con, p_categorical_con, p_continuous_con, set_random_seed, random_seed, n_min, n_cutpoints, mtry_con, Xpointer_con, num_sweeps, sample_weights, y_std, sigma, max_depth, ini_var_yhat, burnin, dim_residual, nthread)
    {
        this->X_std_con = Xpointer_con;
        this->X_std_mod = Xpointer_mod;
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
        this->a = 1.0;
        this->b_vec.resize(2);
        this->b_vec[0] = -0.5;
        this->b_vec[1] = 0.5;
        this->sigma_vec.resize(2);
        this->sigma_vec[0] = 1;
        this->sigma_vec[1] = 1;
    }
};

class hskState : public State
{
    // for hskState we have residual_std of size 3xN
    // residual[0] is the original residual we try to learn
    // residual[1] is the precision vector
    // residual[2] is the element-wise product of residual[0] and residual[1]

    private:

        void ini_sigma(std::vector<double> &sigma, std::vector<double> &input)
        {
            sigma.resize(input.size());
            for (size_t i = 0; i < input.size(); i++)
            {
                sigma[i] = input[i];
            }
        }


    public:

        hskState(const double *Xpointer, matrix<size_t> &Xorder_std, size_t N, size_t p, size_t num_trees, size_t p_categorical, size_t p_continuous, bool set_random_seed, size_t random_seed, size_t n_min, size_t n_cutpoints, size_t mtry, const double *X_std, size_t num_sweeps, bool sample_weights, std::vector<double> *y_std, double sigma, size_t max_depth, double ini_var_yhat, size_t burnin, size_t dim_residual, size_t nthread, bool parallel, std::vector<double> &sigma_vec) : State(Xpointer, Xorder_std, N, p, num_trees, p_categorical, p_continuous, set_random_seed, random_seed, n_min, n_cutpoints, mtry, X_std, num_sweeps, sample_weights, y_std, sigma, max_depth, ini_var_yhat, burnin, dim_residual, nthread)
        {
            //COUT << "hsk " <<ini_var_yhat << endl;
            ini_sigma(this->sigma_vec, sigma_vec);
            this->parallel = parallel;
            this->precision = (new std::vector<double>(N, 1));
            this->res_x_precision = (new std::vector<double>(N, 0));
        }


};

class HeteroskedasticState : public State
{
    private:

        void ini_sigma(std::vector<double> &sigma, std::vector<double> &input)
        {
            sigma.resize(input.size());
            for (size_t i = 0; i < input.size(); i++)
            {
                sigma[i] = input[i];
            }
        }
    public:
        HeteroskedasticState(const double *Xpointer, matrix<size_t> &Xorder_std, size_t N, size_t p, size_t num_trees_m, size_t num_trees_v,
                            size_t p_categorical, size_t p_continuous, bool set_random_seed, size_t random_seed, size_t n_min_m, size_t n_min_v,
                            size_t n_cutpoints_m, size_t n_cutpoints_v, size_t mtry, const double *X_std, size_t num_sweeps,
                            bool sample_weights, std::vector<double> *y_std, double sigma, size_t max_depth_m, size_t max_depth_v,
                            double ini_var_yhat, size_t burnin, size_t dim_residual, size_t nthread, bool parallel, std::vector<double> &sigma_vec) :
                    State(Xpointer, Xorder_std, N, p, num_trees_m, p_categorical, p_continuous, set_random_seed, random_seed,
                        n_min_m, n_cutpoints_m, mtry, Xpointer, num_sweeps, sample_weights, y_std, sigma, max_depth_m,
                        ini_var_yhat, burnin, dim_residual, nthread)
        {
            this->split_count_all_tree_m = new matrix<double>();
            this->split_count_all_tree_v = new matrix<double>();
            ini_xinfo((*this->split_count_all_tree_m), p, num_trees_m);
            ini_xinfo((*this->split_count_all_tree_v), p, num_trees_v);
            this->split_count_all_m = new std::vector<double>(p, 0);
            this->mtry_weight_current_tree_m = new std::vector<double>(p, 0);
            this->split_count_all_v = new std::vector<double>(p, 0);
            this->mtry_weight_current_tree_v = new std::vector<double>(p, 0);
            ini_sigma(this->sigma_vec, sigma_vec);
            this->sigma = sigma;
            this->parallel = parallel;
            this->mean_res = (new std::vector<double>(N, 0));
            this->precision = (new std::vector<double>(N, 1));
            this->res_x_precision = (new std::vector<double>(N, 0));
            this->mtry = mtry;
            this->num_trees_m = num_trees_m;
            this->num_trees_v = num_trees_v;
            this->n_cutpoints_m = n_cutpoints_m;
            this->n_cutpoints_v = n_cutpoints_v;
            this->n_min_m = n_min_m;
            this->n_min_v = n_min_v;
            this->max_depth_m = max_depth_m;
            this->max_depth_v = max_depth_v;
        }
};

class XBCFdiscreteHeteroskedasticState : public State
{
    private:

        void ini_sigma(std::vector<double> &sigma, std::vector<double> &input)
        {
            sigma.resize(input.size());
            for (size_t i = 0; i < input.size(); i++)
            {
                sigma[i] = input[i];
            }
        }
    public:
        XBCFdiscreteHeteroskedasticState(matrix<double> *Z_std,
                                         const double *Xpointer_con,
                                         const double *Xpointer_mod,
                                         matrix<size_t> &Xorder_std_con,
                                         matrix<size_t> &Xorder_std_mod,
                                         size_t N, size_t p_con, size_t p_mod,
                                         size_t num_trees_con, size_t num_trees_mod,
                                         size_t num_trees_v,
                                         size_t p_categorical_con, size_t p_categorical_mod,
                                         size_t p_continuous_con, size_t p_continuous_mod,
                                         bool set_random_seed, size_t random_seed,
                                         size_t n_min, size_t n_min_v,
                                         size_t n_cutpoints, size_t n_cutpoints_v,
                                         size_t mtry_con, size_t mtry_mod, size_t mtry_v,
                                         size_t num_sweeps, bool sample_weights,
                                         std::vector<double> *y_std, double sigma,
                                         size_t max_depth, size_t max_depth_v,
                                         double ini_var_yhat,
                                         size_t burnin, size_t dim_residual,
                                         size_t nthread, bool parallel,
                                         bool a_scaling, bool b_scaling,
                                         size_t N_trt, size_t N_ctrl, std::vector<double> &sigma_vec) :
                State(Xpointer_con, Xorder_std_con,
                      N, p_con, num_trees_con, p_categorical_con, p_continuous_con,
                      set_random_seed, random_seed,
                      n_min, n_cutpoints,
                      mtry_con, Xpointer_con,
                      num_sweeps, sample_weights,
                      y_std, sigma, max_depth,
                      ini_var_yhat, burnin, dim_residual, nthread)
        {
            this->X_std_con = Xpointer_con;
            this->X_std_mod = Xpointer_mod;
            this->split_count_all_tree_con = new matrix<double>();
            this->split_count_all_tree_mod = new matrix<double>();
            this->split_count_all_tree_v = new matrix<double>();
            ini_xinfo((*this->split_count_all_tree_con), p_con, num_trees_con);
            ini_xinfo((*this->split_count_all_tree_mod), p_mod, num_trees_mod);
            ini_xinfo((*this->split_count_all_tree_v), p_con, num_trees_v);
            this->split_count_all_con = new std::vector<double>(p_con, 0);
            this->mtry_weight_current_tree_con = new std::vector<double>(p_con, 0);
            this->split_count_all_mod = new std::vector<double>(p_mod, 0);
            this->mtry_weight_current_tree_mod = new std::vector<double>(p_mod, 0);
            this->split_count_all_v = new std::vector<double>(p_con, 0);
            this->mtry_weight_current_tree_v = new std::vector<double>(p_con, 0);
            this->Z_std = Z_std;
            this->sigma = sigma;
            this->sigma2 = pow(sigma, 2);
            this->parallel = parallel;
            this->tau_fit = (new std::vector<double>(N, 0));
            this->mu_fit = (new std::vector<double>(N, 0));
            this->precision = (new std::vector<double>(N, 1));
            this->res_x_precision = (new std::vector<double>(N, 0));
            this->mean_res = (new std::vector<double>(N, 0));
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
            this->mtry_v = mtry_v;
            this->num_trees_con = num_trees_con;
            this->num_trees_mod = num_trees_mod;
            this->a_scaling = a_scaling;
            this->b_scaling = b_scaling;
            this->N_trt = N_trt;
            this->N_ctrl = N_ctrl;
            this->a = 1.0;
            this->b_vec.resize(2);
            this->b_vec[0] = -0.5;
            this->b_vec[1] = 0.5;
            ini_sigma(this->sigma_vec,sigma_vec);
            this->num_trees_m = num_trees_con;
            this->num_trees_v = num_trees_v;
            this->n_cutpoints_m = n_cutpoints;
            this->n_cutpoints_v = n_cutpoints_v;
            this->n_min_m = n_min;
            this->n_min_v = n_min_v;
            this->max_depth_m = max_depth;
            this->max_depth_v = max_depth_v;

        }
};


#endif