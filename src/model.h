
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

    double no_split_penality;

    // tree prior
    double alpha;

    double beta;

    Model(size_t dim_theta, size_t dim_suff)
    {
        this->dim_theta = dim_theta;
        this->dim_suffstat = dim_suff;
    };

    virtual ~Model() = default;

    // Abstract functions
    virtual void incSuffStat(State &state, size_t index_next_obs, std::vector<double> &suffstats) { return; };

    virtual void samplePars(State &state, std::vector<double> &suff_stat, std::vector<double> &theta_vector, double &prob_leaf) { return; };

    virtual void update_state(State &state, size_t tree_ind, X_struct &x_struct) { return; };

    virtual void initialize_root_suffstat(State &state, std::vector<double> &suff_stat) { return; };

    virtual void updateNodeSuffStat(State &state, std::vector<double> &suff_stat, matrix<size_t> &Xorder_std, size_t &split_var, size_t row_ind) { return; };

    virtual void calculateOtherSideSuffStat(std::vector<double> &parent_suff_stat, std::vector<double> &lchild_suff_stat, std::vector<double> &rchild_suff_stat, size_t &N_parent, size_t &N_left, size_t &N_right, bool &compute_left_side) { return; };

    virtual void state_sweep(size_t tree_ind, size_t M, matrix<double> &residual_std, X_struct &x_struct) const { return; };

    virtual double likelihood(std::vector<double> &temp_suff_stat, std::vector<double> &suff_stat_all, size_t N_left, bool left_side, bool no_split, State &state) const { return 0.0; };

    // virtual double likelihood_no_split(std::vector<double> &suff_stat, State&state) const { return 0.0; };

    virtual void ini_residual_std(State &state) { return; };

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

    void incSuffStat(State &state, size_t index_next_obs, std::vector<double> &suffstats);

    void samplePars(State &state, std::vector<double> &suff_stat, std::vector<double> &theta_vector, double &prob_leaf);

    void update_state(State &state, size_t tree_ind, X_struct &x_struct);

    void update_tau(State &state, size_t tree_ind, size_t sweeps, vector<vector<tree>> &trees);

    void update_tau_per_forest(State &state, size_t sweeps, vector<vector<tree>> &trees);

    void initialize_root_suffstat(State &state, std::vector<double> &suff_stat);

    void updateNodeSuffStat(State &state, std::vector<double> &suff_stat, matrix<size_t> &Xorder_std, size_t &split_var, size_t row_ind);

    void calculateOtherSideSuffStat(std::vector<double> &parent_suff_stat, std::vector<double> &lchild_suff_stat, std::vector<double> &rchild_suff_stat, size_t &N_parent, size_t &N_left, size_t &N_right, bool &compute_left_side);

    void state_sweep(size_t tree_ind, size_t M, matrix<double> &residual_std, X_struct &x_struct) const;

    double likelihood(std::vector<double> &temp_suff_stat, std::vector<double> &suff_stat_all, size_t N_left, bool left_side, bool no_split, State &state) const;

    // double likelihood_no_split(std::vector<double> &suff_stat, State&state) const;

    void ini_residual_std(State &state);

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

        // size_t c = dim_residual;

        double ret = 0;
        // double z1, z2, n, sy, minz;
        double logz1, logz2, n, sy, numrt, logminz;

        for (size_t j = 0; j < dim_residual; j++)
        {
            // ret += -(tau_a + suffstats[j]) * log(tau_b + suffstats[c + j]) + lgamma(tau_a + suffstats[j]); // - lgamma(suffstats[j] +1);
            n = suffstats[j];
            sy = suffstats[dim_residual + j];

            logz1 = loggignorm(-c + n, 2*d, 2*sy);
            logz2 = loggignorm(c + n, 0, 2*(d + sy));

            logminz = logz1 < logz2 ? logz1 : logz2;
            if (logz1 - logminz > 100) {
                numrt = logz1; // approximate log(exp(x) + 1) = x
            } else if (logz2 - logminz > 100)
            { 
                numrt = logz2;
            } else {
                numrt = log(exp(logz1 - logminz) + exp(logz2 - logminz)) + logminz;
            }
            ret += numrt - log(2) - logz3;
            // ret += log((z1 + z2) / 2 / z3);
            // cout <<" n = " << n << " sy = " << sy << " logz1 = " << logz1 << " logz2 = " << logz2 << " numrt = " << numrt << " ret = " << ret << endl;
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
    double weight, logloss, accuracy;         // pseudo replicates of observations
    double hmult, heps;             // weight ~ Gamma(n, hmult * entropy + heps);
    std::vector<double> acc_gp; // track accuracy per group

    double c, d, z3, logz3; // param for mixture prior, c = m / tau_a^2 + 0.5; d = m / tau_a^2; m = num_trees = tau_b

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
        this->accuracy = 0;
        this->acc_gp.resize(dim_residual);

        this->c = tau_b / pow(tau_a, 2) + 0.5;
        this->d = tau_b / pow(tau_a, 2);
        this->z3 = exp(lgamma(this->c) - this->c * log(this->d));
        this->logz3 = lgamma(this->c) - this->c * log(this->d);
        cout << "c = " << c << " d = " << d << " z3 = " << z3 << " logz3 " << logz3 << endl;
    }

    LogitModel() : Model(2, 4) {}

    Model *clone() { return new LogitModel(*this); }

    void incSuffStat(State &state, size_t index_next_obs, std::vector<double> &suffstats);

    void samplePars(State &state, std::vector<double> &suff_stat, std::vector<double> &theta_vector, double &prob_leaf);

    void update_state(State &state, size_t tree_ind, X_struct &x_struct, double &mean_lambda, std::vector<double>& var_lambda, size_t &count_lambda);
    
    void copy_initialization(State &state, X_struct &x_struct, vector<vector<tree>> &trees, size_t sweeps, size_t tree_ind, matrix<size_t> &Xorder_std);

    void initialize_root_suffstat(State &state, std::vector<double> &suff_stat);

    void updateNodeSuffStat(State &state, std::vector<double> &suff_stat, matrix<size_t> &Xorder_std, size_t &split_var, size_t row_ind);

    void calculateOtherSideSuffStat(std::vector<double> &parent_suff_stat, std::vector<double> &lchild_suff_stat, std::vector<double> &rchild_suff_stat, size_t &N_parent, size_t &N_left, size_t &N_right, bool &compute_left_side);

    void state_sweep(size_t tree_ind, size_t M, matrix<double> &residual_std, X_struct &x_struct) const;

    double likelihood(std::vector<double> &temp_suff_stat, std::vector<double> &suff_stat_all, size_t N_left, bool left_side, bool no_split, State &state) const;

    void ini_residual_std(State &state);

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

    void samplePars(State &state, std::vector<double> &suff_stat, std::vector<double> &theta_vector, double &prob_leaf);

    void update_state(State &state, size_t tree_ind, X_struct &x_struct, double &mean_lambda, std::vector<double>& var_lambda, size_t &count_lambda);

    void state_sweep(size_t tree_ind, size_t M, matrix<double> &residual_std, X_struct &x_struct) const;

    double likelihood(std::vector<double> &temp_suff_stat, std::vector<double> &suff_stat_all, size_t N_left, bool left_side, bool no_split, State &state) const;

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

class XBCFContinuousModel : public Model
{
public:
    size_t dim_suffstat = 4;

    // model prior
    // prior on sigma
    double kap;
    double s;
    double tau_con_kap;
    double tau_con_s;
    double tau_mod_kap;
    double tau_mod_s;
    // prior on leaf parameter
    double tau_con; // might be updated if sampling tau
    double tau_mod;
    double tau_con_mean; // copy of the original value
    double tau_mod_mean;

    double alpha_con;
    double alpha_mod;
    double beta_con;
    double beta_mod;
    bool sampling_tau;

    XBCFContinuousModel(double kap, double s, double tau_con, double tau_mod, double alpha_con, double beta_con, double alpha_mod, double beta_mod, bool sampling_tau, double tau_con_kap, double tau_con_s, double tau_mod_kap, double tau_mod_s) : Model(1, 4)
    {
        this->kap = kap;
        this->s = s;
        this->tau_con_kap = tau_con_kap;
        this->tau_con_s = tau_con_s;
        this->tau_mod_kap = tau_mod_kap;
        this->tau_mod_s = tau_mod_s;
        this->tau_con = tau_con;
        this->tau_mod = tau_mod;
        this->tau_con_mean = tau_con;
        this->tau_mod_mean = tau_mod;
        this->alpha_con = alpha_con;
        this->alpha_mod = alpha_mod;
        this->beta_con = beta_con;
        this->beta_mod = beta_mod;
        this->alpha = alpha_con;
        this->beta = beta_con;
        this->dim_residual = 1;
        this->class_operating = 0;
        this->sampling_tau = sampling_tau;
    }

    XBCFContinuousModel() : Model(1, 4) {}

    Model *clone() { return new XBCFContinuousModel(*this); }

    void incSuffStat(State &state, size_t index_next_obs, std::vector<double> &suffstats);

    void samplePars(State &state, std::vector<double> &suff_stat, std::vector<double> &theta_vector, double &prob_leaf);

    void update_state(State &state, size_t tree_ind, X_struct &x_struct);

    void update_tau(State &state, size_t tree_ind, size_t sweeps, vector<vector<tree>> &trees);

    void update_tau_per_forest(State &state, size_t sweeps, vector<vector<tree>> &trees);

    void initialize_root_suffstat(State &state, std::vector<double> &suff_stat);

    void updateNodeSuffStat(State &state, std::vector<double> &suff_stat, matrix<size_t> &Xorder_std, size_t &split_var, size_t row_ind);

    void calculateOtherSideSuffStat(std::vector<double> &parent_suff_stat, std::vector<double> &lchild_suff_stat, std::vector<double> &rchild_suff_stat, size_t &N_parent, size_t &N_left, size_t &N_right, bool &compute_left_side);

    // void state_sweep(State&state, size_t tree_ind, size_t M, X_struct &x_struct) const;

    double likelihood(std::vector<double> &temp_suff_stat, std::vector<double> &suff_stat_all, size_t N_left, bool left_side, bool no_split, State &state) const;

    void ini_tau_mu_fit(State &state);

    void ini_residual_std(State &state);

    void predict_std(matrix<double> &Ztestpointer, const double *Xtestpointer_con, const double *Xtestpointer_mod, size_t N_test, size_t p_con, size_t p_mod, size_t num_trees_con, size_t num_trees_mod, size_t num_sweeps, matrix<double> &yhats_test_xinfo, matrix<double> &prognostic_xinfo, matrix<double> &treatment_xinfo, vector<vector<tree>> &trees_con, vector<vector<tree>> &trees_mod);

    void set_treatmentflag(State &state, bool value);

    void subtract_old_tree_fit(size_t tree_ind, State &state, X_struct &x_struct);

    void add_new_tree_fit(size_t tree_ind, State &state, X_struct &x_struct);

    void update_partial_residuals(size_t tree_ind, State &state, X_struct &x_struct);

    void update_split_counts(State &state, size_t tree_ind);
};

#endif
