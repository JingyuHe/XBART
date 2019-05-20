
#ifndef model_h
#define model_h

#include "common.h"
#include "utility.h"
#include <memory>
//#include "fit_info.h"

using namespace std;

class Model
{

protected:
    size_t num_classes;
    size_t dim_suffstat;
    size_t dim_suffstat_total;
    std::vector<double> suff_stat_model;
    std::vector<double> suff_stat_total;
    double no_split_penality;

public:
    Model(size_t num_classes, size_t dim_suff)
    {
        this->num_classes = num_classes;
        this->dim_suffstat = dim_suff;
        Model::suff_stat_model.resize(dim_suff);
    };

    // Abstract functions
    virtual void incrementSuffStat() const { return; };
    virtual void samplePars(bool draw_mu, double y_mean, size_t N_Xorder, double sigma, double tau,
                            std::mt19937 &generator, std::vector<double> &theta_vector, std::vector<double> &y_std, xinfo_sizet &Xorder) { return; };
    virtual void updateResidual(const xinfo &predictions_std, size_t tree_ind, size_t M,
                                std::vector<double> &residual_std) const { return; };
    virtual void calcSuffStat_categorical(std::vector<double> &y, xinfo_sizet &Xorder, size_t &start, size_t &end, const size_t &var) { return; };
    virtual void calcSuffStat_continuous(std::vector<size_t> &xorder, std::vector<double> &y_std, std::vector<size_t> &candidate_index, size_t index, bool adaptive_cutpoint) { return; };
    virtual double likelihood(double tau, double ntau, double sigma2, double y_sum, bool left_side) const { return 0.0; };
    virtual double likelihood_no_split(double value, double tau, double ntau, double sigma2) const { return 0.0; };
    virtual void suff_stat_fill(std::vector<double> &y_std, std::vector<size_t> &xorder) { return; };
    virtual Model *clone() { return nullptr; };

    // Getters and Setters
    // num classes
    size_t getNumClasses() const { return num_classes; };
    void setNumClasses(size_t n_class) { num_classes = n_class; };
    // dim suff stat
    size_t getDimSuffstat() const { return dim_suffstat; };
    void setDimSuffStat(size_t dim_suff) { dim_suffstat = dim_suff; };
    // suff stat
    std::vector<double> getSuffstat() const { return this->suff_stat_model; };
    //penality
    double getNoSplitPenality()
    {
        return no_split_penality;
        ;
    };
    void setNoSplitPenality(double pen) { this->no_split_penality = pen; };

    // Other Pre-Defined functions
    void suff_stat_fill_zero()
    {
        std::fill(Model::suff_stat_model.begin(), Model::suff_stat_model.end(), 0);
        return;
    };
    void printSuffstat() const
    {
        cout << this->suff_stat_model << endl;
        return;
    };
};

class NormalModel : public Model
{
private:
    size_t dim_suffstat_total = 1;
    std::vector<double> suff_stat_total;

public:
    NormalModel() : Model(1, 1)
    {
    }

    void suff_stat_fill(std::vector<double> &y_std, std::vector<size_t> &xorder);

    void incrementSuffStat() const;

    void samplePars(bool draw_mu, double y_mean, size_t N_Xorder, double sigma, double tau,
                    std::mt19937 &generator, std::vector<double> &theta_vector, std::vector<double> &y_std, xinfo_sizet &Xorder);

    void updateResidual(const xinfo &predictions_std, size_t tree_ind, size_t M, std::vector<double> &residual_std) const;

    void calcSuffStat_categorical(std::vector<double> &y, xinfo_sizet &Xorder, size_t &start, size_t &end, const size_t &var);

    void calcSuffStat_continuous(std::vector<size_t> &xorder, std::vector<double> &y_std, std::vector<size_t> &candidate_index, size_t index, bool adaptive_cutpoint);

    double likelihood(double tau, double ntau, double sigma2, double y_sum, bool left_side) const;

    double likelihood_no_split(double value, double tau, double ntau, double sigma2) const;

    Model *clone() { return new NormalModel(*this); }
};

class CLTClass : public Model
{
private:
    size_t dim_suffstat_total = 4;
    //std::vector<double> suff_stat_total;

public:
    CLTClass() : Model(1, 4)
    {
        suff_stat_total.resize(dim_suffstat_total);
    }
    std::vector<double> total_fit; // Keep public to save copies
    std::vector<double> suff_stat_total;

    void suff_stat_fill(std::vector<double> &y_std, std::vector<size_t> &xorder);

    void incrementSuffStat() const;

    void samplePars(bool draw_mu, double y_mean, size_t N_Xorder, double sigma, double tau,
                    std::mt19937 &generator, std::vector<double> &theta_vector, std::vector<double> &y_std, xinfo_sizet &Xorder);

    void updateResidual(const xinfo &predictions_std, size_t tree_ind, size_t M, std::vector<double> &residual_std) const;

    void calcSuffStat_categorical(std::vector<double> &y, xinfo_sizet &Xorder, size_t &start, size_t &end, const size_t &var);

    void calcSuffStat_continuous(std::vector<size_t> &xorder, std::vector<double> &y_std, std::vector<size_t> &candidate_index, size_t index, bool adaptive_cutpoint);

    void updateFullSuffStat(std::vector<double> &y_std, std::vector<size_t> &x_info);

    double likelihood(double tau, double ntau, double sigma2, double y_sum, bool left_side) const;

    double likelihood_no_split(double value, double tau, double ntau, double sigma2) const;

    Model *clone() { return new CLTClass(*this); }
};

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
    matrix<std::vector<double> *> data_pointers;
    void init_tree_pointers(std::vector<double> *initial_theta, size_t N, size_t num_trees);

    FitInfo(const double *Xpointer, xinfo_sizet &Xorder_std, size_t N, size_t p,
            size_t num_trees, size_t p_categorical, size_t p_continuous,
            bool set_random_seed, size_t random_seed, std::vector<double> *initial_theta);
};

//////////////////////////////////////////////////////////////////////////////////////
//
//
//  Multinomial logistic model
//
//
//////////////////////////////////////////////////////////////////////////////////////

double LogitLIL(const vector<double> &suffstats, const double &tau_a, const double &tau_b);

vector<double> LogitSamplePars(vector<double> &suffstats, double &tau_a, double &tau_b, std::mt19937 &generator);

class LogitClass : public Model
{
private:
    size_t dim_suffstat_total = 0; // = 2*num_classes;
                                   //std::vector<double> suff_stat_total;

public:
    //This is probably unsafe/stupid but a temporary hack #yolo
    FitInfo *fit_info;
    //these should be elements of a class derived from a FitInfo base class for this model
    std::vector<std::vector<double>> *slop;
    std::vector<double> *phi;
    double tau_a = 3.3; //approx 4/sqrt(2) + 0.5
    double tau_b = 2.8;

    LogitClass() : Model(2, 4)
    {
        dim_suffstat_total = 2 * num_classes;       //num_classes is a member of base Model class
        suff_stat_total.resize(dim_suffstat_total); //suff_stat_total stuff should live in base class
    }

    LogitClass(size_t num_classes) : Model(num_classes, 2 * num_classes)
    {
        dim_suffstat_total = 2 * num_classes;
        suff_stat_total.resize(dim_suffstat_total);
    }
    //std::vector<double> total_fit; // Keep public to save copies
    std::vector<double> suff_stat_total;

    // Initialize the sufficient stat vector to the sufficient stat for the first obs
    // when sorting by xorder
    //no longer necessary
    /*
    void suff_stat_fill(std::vector<double> &y_std, std::vector<size_t> &xorder);
    */

    // this function should ultimately take a FitInfo and DataInfo
    void incSuffStat(std::vector<double> &y_std, size_t ix, std::vector<double> &suffstats) const;

    // This function call can be much simplified too --  should only require (maybe) theta plus a draw_mu flag?
    void samplePars(bool draw_mu, double y_mean, size_t N_Xorder, double sigma, double tau,
                    std::mt19937 &generator, std::vector<double> &theta_vector, std::vector<double> &y_std,
                    xinfo_sizet &Xorder);

    /*

    void updateResidual(const xinfo &predictions_std, size_t tree_ind, size_t M, std::vector<double> &residual_std) const; 
    */

    //    void updateResidualNew(size_t tree_ind, size_t M, std::unique_ptr<FitInfo> fit_info, std::vector<std::vector<double> > &slop) {
    void updateResidual(const xinfo &predictions_std, size_t tree_ind, size_t M, std::vector<double> &residual_std) const;

    // Once their function calls are standardized to take a FitInfo we should never have to redefine these
    // in another model class
    void calcSuffStat_categorical(std::vector<double> &y, xinfo_sizet &Xorder, size_t &start, size_t &end, const size_t &var);

    void calcSuffStat_continuous(std::vector<size_t> &xorder, std::vector<double> &y_std,
                                 std::vector<size_t> &candidate_index, size_t index, bool adaptive_cutpoint);

    void updateFullSuffStat(std::vector<double> &y_std, std::vector<size_t> &x_info);

    double LIL(const std::vector<double> &suffstats) const;

    //this function should call a base LIL() member function that should be redefined in
    double likelihood(double tau, double ntau, double sigma2, double y_sum, bool left_side) const;

    double likelihood_no_split(double value, double tau, double ntau, double sigma2) const;

    Model *clone() { return new LogitClass(*this); }
};

#endif
