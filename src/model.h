
#ifndef model_h
#define model_h

#include "common.h"
#include "utility.h"
#include <memory>
#include "state.h"
#include "X_struct.h"
#include "cdf.h"

using namespace std;

class tree;

class Model
{

public:
    size_t dim_theta;

    size_t dim_suffstat;

    size_t dim_residual;

    size_t class_operating;

    /////////////////////////////////////
    //
    //  suff_stat_model and suff_stat_total
    //  are useless for NormalModel now
    //  They are still here because CLT class depends on them
    //  Delelte them later
    //
    /////////////////////////////////////
    std::vector<double> suff_stat_model;

    std::vector<double> suff_stat_total;

    double no_split_penality;

    // tree prior
    double alpha;

    double beta;

    Model(size_t dim_theta, size_t dim_suff)
    {
        this->dim_theta = dim_theta;
        this->dim_suffstat = dim_suff;
    };

    // Abstract functions
    virtual void incSuffStat(std::unique_ptr<State> &state, size_t index_next_obs, std::vector<double> &suffstats) { return; };

    virtual void samplePars(std::unique_ptr<State> &state, std::vector<double> &suff_stat, std::vector<double> &theta_vector, double &prob_leaf) { return; };

    virtual void update_state(std::unique_ptr<State> &state, size_t tree_ind, std::unique_ptr<X_struct> &x_struct) { return; };

    virtual void initialize_root_suffstat(std::unique_ptr<State> &state, std::vector<double> &suff_stat) { return; };

    virtual void updateNodeSuffStat(std::unique_ptr<State> &state, std::vector<double> &suff_stat, matrix<size_t> &Xorder_std, size_t &split_var, size_t row_ind) { return; };

    virtual void calculateOtherSideSuffStat(std::vector<double> &parent_suff_stat, std::vector<double> &lchild_suff_stat, std::vector<double> &rchild_suff_stat, size_t &N_parent, size_t &N_left, size_t &N_right, bool &compute_left_side) { return; };

    virtual void state_sweep(size_t tree_ind, size_t M, matrix<double> &residual_std, std::unique_ptr<X_struct> &x_struct) const { return; };

    virtual double likelihood(std::vector<double> &temp_suff_stat, std::vector<double> &suff_stat_all, size_t N_left, bool left_side, bool no_split, std::unique_ptr<State> &state) const { return 0.0; };

    // virtual double likelihood_no_split(std::vector<double> &suff_stat, std::unique_ptr<State> &state) const { return 0.0; };

    virtual void ini_residual_std(std::unique_ptr<State> &state) { return; };

    // virtual double predictFromTheta(const std::vector<double> &theta_vector) const { return 0.0; };

    virtual void predict_std(const double *Xtestpointer, size_t N_test, size_t p, size_t num_trees, size_t num_sweeps, matrix<double> &yhats_test_xinfo, vector<vector<tree>> &trees) { return; };

    virtual Model *clone() { return nullptr; };

    // Getters and Setters
    // num classes
    size_t getNumClasses() const { return dim_theta; };

    void setNumClasses(size_t n_class) { dim_theta = n_class; };

    // dim suff stat
    size_t getDimSuffstat() const { return dim_suffstat; };

    void setDimSuffStat(size_t dim_suff) { dim_suffstat = dim_suff; };

    // penality
    double getNoSplitPenality()
    {
        return no_split_penality;
        ;
    };
    void setNoSplitPenality(double pen) { this->no_split_penality = pen; };

    virtual size_t get_class_operating() { return class_operating; };

    virtual void set_class_operating(size_t i)
    {
        class_operating = i;
        return;
    };
};

class NormalModel : public Model
{
public:
    size_t dim_suffstat = 3;

    // model prior
    // prior on sigma
    double kap;
    double s;
    double tau_kap;
    double tau_s;
    // prior on leaf parameter
    double tau; // might be updated if sampling tau
    double tau_prior;

    double tau_mean; // copy of the original value
    bool sampling_tau;

    NormalModel(double kap, double s, double tau, double alpha, double beta, bool sampling_tau, double tau_kap, double tau_s) : Model(1, 3)
    {
        this->kap = kap;
        this->s = s;
        this->tau_kap = tau_kap;
        this->tau_s = tau_s;
        this->tau_prior = tau;
        this->tau = tau;
        this->tau_mean = tau;
        this->alpha = alpha;
        this->beta = beta;
        this->dim_residual = 1;
        this->class_operating = 0;
        this->sampling_tau = sampling_tau;
    }

    NormalModel(double kap, double s, double tau, double alpha, double beta) : Model(1, 3)
    {
        this->kap = kap;
        this->s = s;
        this->tau = tau;
        this->tau_mean = tau;
        this->alpha = alpha;
        this->beta = beta;
        this->dim_residual = 1;
        this->class_operating = 0;
        this->sampling_tau = true;
    }

    NormalModel() : Model(1, 3) {}

    Model *clone() { return new NormalModel(*this); }

    void incSuffStat(std::unique_ptr<State> &state, size_t index_next_obs, std::vector<double> &suffstats);

    void samplePars(std::unique_ptr<State> &state, std::vector<double> &suff_stat, std::vector<double> &theta_vector, double &prob_leaf);

    void update_state(std::unique_ptr<State> &state, size_t tree_ind, std::unique_ptr<X_struct> &x_struct);

    void update_tau(std::unique_ptr<State> &state, size_t tree_ind, size_t sweeps, vector<vector<tree>> &trees);

    void update_tau_per_forest(std::unique_ptr<State> &state, size_t sweeps, vector<vector<tree>> &trees);

    void initialize_root_suffstat(std::unique_ptr<State> &state, std::vector<double> &suff_stat);

    void updateNodeSuffStat(std::unique_ptr<State> &state, std::vector<double> &suff_stat, matrix<size_t> &Xorder_std, size_t &split_var, size_t row_ind);

    void calculateOtherSideSuffStat(std::vector<double> &parent_suff_stat, std::vector<double> &lchild_suff_stat, std::vector<double> &rchild_suff_stat, size_t &N_parent, size_t &N_left, size_t &N_right, bool &compute_left_side);

    void state_sweep(size_t tree_ind, size_t M, matrix<double> &residual_std, std::unique_ptr<X_struct> &x_struct) const;

    double likelihood(std::vector<double> &temp_suff_stat, std::vector<double> &suff_stat_all, size_t N_left, bool left_side, bool no_split, std::unique_ptr<State> &state) const;

    // double likelihood_no_split(std::vector<double> &suff_stat, std::unique_ptr<State> &state) const;

    void ini_residual_std(std::unique_ptr<State> &state);

    void predict_std(const double *Xtestpointer, size_t N_test, size_t p, size_t num_trees, size_t num_sweeps, matrix<double> &yhats_test_xinfo, vector<vector<tree>> &trees);

    void predict_whole_std(const double *Xtestpointer, size_t N_test, size_t p, size_t num_trees, size_t num_sweeps, std::vector<double> &output_vec, vector<vector<tree>> &trees);
};

//////////////////////////////////////////////////////////////////////////////////////
//
//
//  Multinomial logistic model
//
//
//////////////////////////////////////////////////////////////////////////////////////

class LogitModel : public Model
{
private:
    double LogitLIL(const vector<double> &suffstats) const
    {

        size_t c = dim_residual;

        double ret = 0;

        for (size_t j = 0; j < c; j++)
        {
            ret += -(tau_a + suffstats[j]) * log(tau_b + suffstats[c + j]) + lgamma(tau_a + suffstats[j]); // - lgamma(suffstats[j] +1);
        }
        return ret;
    }

public:
    // prior on leaf parameter
    double tau_a, tau_b; // leaf parameter is ~ G(tau_a, tau_b). tau_a = 1/tau + 1/2, tau_b = 1/tau -> f(x)\sim N(0,tau) approx

    // Should these pointers live in model subclass or state subclass?
    std::vector<size_t> *y_size_t; // a y vector indicating response categories in 0,1,2,...,c-1
    std::vector<double> *phi;

    bool update_weight, update_tau; // option to update tau_a
    double weight, logloss;         // pseudo replicates of observations
    double hmult, heps;             // weight ~ Gamma(n, hmult * entropy + heps);

    LogitModel(size_t num_classes, double tau_a, double tau_b, double alpha, double beta, std::vector<size_t> *y_size_t, std::vector<double> *phi, double weight, bool update_weight, bool update_tau, double hmult, double heps) : Model(num_classes, 2 * num_classes)
    {
        this->y_size_t = y_size_t;
        this->phi = phi;
        this->tau_a = tau_a;
        this->tau_b = tau_b;
        this->alpha = alpha;
        this->beta = beta;
        // what should this be?
        this->dim_residual = num_classes;

        this->update_weight = update_weight;
        this->update_tau = update_tau;
        this->weight = weight;
        this->hmult = hmult;
        this->heps = heps;
        this->logloss = 0;
    }

    LogitModel() : Model(2, 4) {}

    Model *clone() { return new LogitModel(*this); }

    void incSuffStat(std::unique_ptr<State> &state, size_t index_next_obs, std::vector<double> &suffstats);

    void samplePars(std::unique_ptr<State> &state, std::vector<double> &suff_stat, std::vector<double> &theta_vector, double &prob_leaf);

    void update_state(std::unique_ptr<State> &state, size_t tree_ind, std::unique_ptr<X_struct> &x_struct);

    void initialize_root_suffstat(std::unique_ptr<State> &state, std::vector<double> &suff_stat);

    void updateNodeSuffStat(std::unique_ptr<State> &state, std::vector<double> &suff_stat, matrix<size_t> &Xorder_std, size_t &split_var, size_t row_ind);

    void calculateOtherSideSuffStat(std::vector<double> &parent_suff_stat, std::vector<double> &lchild_suff_stat, std::vector<double> &rchild_suff_stat, size_t &N_parent, size_t &N_left, size_t &N_right, bool &compute_left_side);

    void state_sweep(size_t tree_ind, size_t M, matrix<double> &residual_std, std::unique_ptr<X_struct> &x_struct) const;

    double likelihood(std::vector<double> &temp_suff_stat, std::vector<double> &suff_stat_all, size_t N_left, bool left_side, bool no_split, std::unique_ptr<State> &state) const;

    void ini_residual_std(std::unique_ptr<State> &state);

    using Model::predict_std;
    void predict_std(const double *Xtestpointer, size_t N_test, size_t p, size_t num_trees, size_t num_sweeps, matrix<double> &yhats_test_xinfo, vector<vector<tree>> &trees, std::vector<double> &output_vec);

    void predict_std_standalone(const double *Xtestpointer, size_t N_test, size_t p, size_t num_trees, size_t num_sweeps, matrix<double> &yhats_test_xinfo, vector<vector<tree>> &trees, std::vector<double> &output_vec, std::vector<size_t> &iteration);
};

class LogitModelSeparateTrees : public LogitModel
{
private:
    double LogitLIL(const vector<double> &suffstats) const
    {
        double ret = -(tau_a + suffstats[class_operating]) * log(tau_b + suffstats[dim_residual + class_operating]) + lgamma(tau_a + suffstats[class_operating]);

        return ret;
    }

    void ini_class_count(std::vector<double> &class_count, double &pseudo_norm, const double num_classes)
    {
        class_count.resize(num_classes);
        std::fill(class_count.begin(), class_count.end(), 0.0);
        for (size_t i = 0; i < (*y_size_t).size(); i++)
        {
            class_count[(*y_size_t)[i]] += 1.0;
        }
        for (size_t i = 0; i < num_classes; i++)
        {
            class_count[i] = class_count[i] / (*y_size_t).size();
        }
        pseudo_norm = 0.0;
        for (size_t k = 0; k < class_count.size(); k++)
        {
            // pseudo_norm += lgamma(class_count[k] + 1);
            pseudo_norm = class_count[k] * (*y_size_t).size() * log(class_count[k]);
        }
    }

public:
    LogitModelSeparateTrees(size_t num_classes, double tau_a, double tau_b, double alpha, double beta, std::vector<size_t> *y_size_t, std::vector<double> *phi, double weight, bool update_weight, bool update_tau) : LogitModel(num_classes, tau_a, tau_b, alpha, beta, y_size_t, phi, weight, update_weight, update_tau, 1, 0.1) {}

    LogitModelSeparateTrees() : LogitModel() {}

    Model *clone() { return new LogitModelSeparateTrees(*this); }

    void samplePars(std::unique_ptr<State> &state, std::vector<double> &suff_stat, std::vector<double> &theta_vector, double &prob_leaf);

    void update_state(std::unique_ptr<State> &state, size_t tree_ind, std::unique_ptr<X_struct> &x_struct);

    void state_sweep(size_t tree_ind, size_t M, matrix<double> &residual_std, std::unique_ptr<X_struct> &x_struct) const;

    double likelihood(std::vector<double> &temp_suff_stat, std::vector<double> &suff_stat_all, size_t N_left, bool left_side, bool no_split, std::unique_ptr<State> &state) const;

    void predict_std(const double *Xtestpointer, size_t N_test, size_t p, size_t num_trees, size_t num_sweeps, matrix<double> &yhats_test_xinfo, vector<vector<vector<tree>>> &trees, std::vector<double> &output_vec);

    void predict_std_standalone(const double *Xtestpointer, size_t N_test, size_t p, size_t num_trees, size_t num_sweeps, matrix<double> &yhats_test_xinfo, vector<vector<vector<tree>>> &trees, std::vector<double> &output_vec, std::vector<size_t> &iteration, double weight);
};

//////////////////////////////////////////////////////////////////////////////////////
//
//
//  Normal Linear Model for continuous treatment XBCF
//
//
//////////////////////////////////////////////////////////////////////////////////////

class NormalLinearModel : public Model
{
public:
    size_t dim_suffstat = 4;

    // model prior
    // prior on sigma
    double kap;
    double s;
    double tau_kap;
    double tau_s;
    // prior on leaf parameter
    double tau;      // might be updated if sampling tau
    double tau_mean; // copy of the original value
    bool sampling_tau;

    NormalLinearModel(double kap, double s, double tau, double alpha, double beta, bool sampling_tau, double tau_kap, double tau_s) : Model(1, 4)
    {
        this->kap = kap;
        this->s = s;
        this->tau_kap = tau_kap;
        this->tau_s = tau_s;
        this->tau = tau;
        this->tau_mean = tau;
        this->alpha = alpha;
        this->beta = beta;
        this->dim_residual = 1;
        this->class_operating = 0;
        this->sampling_tau = sampling_tau;
    }

    NormalLinearModel(double kap, double s, double tau, double alpha, double beta) : Model(1, 4)
    {
        this->kap = kap;
        this->s = s;
        this->tau = tau;
        this->tau_mean = tau;
        this->alpha = alpha;
        this->beta = beta;
        this->dim_residual = 1;
        this->class_operating = 0;
        this->sampling_tau = true;
    }

    NormalLinearModel() : Model(1, 4) {}

    Model *clone() { return new NormalLinearModel(*this); }

    void incSuffStat(std::unique_ptr<State> &state, size_t index_next_obs, std::vector<double> &suffstats);

    void samplePars(std::unique_ptr<State> &state, std::vector<double> &suff_stat, std::vector<double> &theta_vector, double &prob_leaf);

    void update_state(std::unique_ptr<State> &state, size_t tree_ind, std::unique_ptr<X_struct> &x_struct);

    void update_tau(std::unique_ptr<State> &state, size_t tree_ind, size_t sweeps, vector<vector<tree>> &trees);

    void update_tau_per_forest(std::unique_ptr<State> &state, size_t sweeps, vector<vector<tree>> &trees);

    void initialize_root_suffstat(std::unique_ptr<State> &state, std::vector<double> &suff_stat);

    void updateNodeSuffStat(std::unique_ptr<State> &state, std::vector<double> &suff_stat, matrix<size_t> &Xorder_std, size_t &split_var, size_t row_ind);

    void calculateOtherSideSuffStat(std::vector<double> &parent_suff_stat, std::vector<double> &lchild_suff_stat, std::vector<double> &rchild_suff_stat, size_t &N_parent, size_t &N_left, size_t &N_right, bool &compute_left_side);

    // void state_sweep(std::unique_ptr<State> &state, size_t tree_ind, size_t M, std::unique_ptr<X_struct> &x_struct) const;

    double likelihood(std::vector<double> &temp_suff_stat, std::vector<double> &suff_stat_all, size_t N_left, bool left_side, bool no_split, std::unique_ptr<State> &state) const;

    void ini_tau_mu_fit(std::unique_ptr<State> &state);

    void ini_residual_std(std::unique_ptr<State> &state);

    void ini_tau_mu_fit(std::unique_ptr<State> &state);

    void predict_std(matrix<double> &Ztestpointer, const double *Xtestpointer, size_t N_test, size_t p, size_t num_trees, size_t num_sweeps, matrix<double> &yhats_test_xinfo, vector<vector<tree>> &trees_ps, vector<vector<tree>> &trees_trt);

    void set_treatmentflag(std::unique_ptr<State> &state, bool value);

    void subtract_old_tree_fit(size_t tree_ind, std::unique_ptr<State> &state, std::unique_ptr<X_struct> &x_struct);

    void add_new_tree_fit(size_t tree_ind, std::unique_ptr<State> &state, std::unique_ptr<X_struct> &x_struct);

    void update_partial_residuals(size_t tree_ind, std::unique_ptr<State> &state, std::unique_ptr<X_struct> &x_struct);

};

#endif
