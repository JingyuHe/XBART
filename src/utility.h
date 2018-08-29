#ifndef GUARD_utility_h
#define GUARD_utility_h

#include "common.h"
#include <algorithm>
#include <functional>


// copy NumericMatrix to STD matrix
xinfo copy_xinfo(Rcpp::NumericMatrix& X);

// copy IntegerMatrix to STD matrix
xinfo_sizet copy_xinfo_sizet(Rcpp::IntegerMatrix& X);

// // initialize STD matrix
void ini_xinfo(xinfo& X, size_t N, size_t p);

// // initialize STD integer matrix
void ini_xinfo_sizet(xinfo_sizet& X, size_t N, size_t p);


void row_sum(xinfo& X, std::vector<double>& output);

void col_sum(xinfo& X, std::vector<double>& output);

void vec_sum(std::vector<double>& vector, double& sum);

double sum_squared(std::vector<double>& v);

double sum_vec(std::vector<double>& v);

void seq_gen(size_t start, size_t end, size_t length_out, arma::uvec& vec);

void seq_gen_std(size_t start, size_t end, size_t length_out, std::vector<size_t>& vec);


void calculate_y_cumsum_std(const double * y, const size_t N_y, double y_sum, std::vector<size_t>& ind, std::vector<double>& y_cumsum, std::vector<double>& y_cumsum_inv);

void compute_partial_sum_adaptive(std::vector<double>& y_std, std::vector<size_t>& candidate_index, std::vector<double>& y_cumsum, xinfo_sizet& Xorder_std, const size_t& var);

void compute_partial_sum_adaptive_newXorder(std::vector<double>& y_std, std::vector<size_t>& candidate_index, std::vector<double>& y_cumsum, xinfo_sizet& Xorder_std, const size_t& var, xinfo_sizet& Xorder_next_index, std::vector<size_t>& Xorder_firstline, size_t N_Xorder, std::vector<double>& possible_cutpoints, size_t N_y, const double* X_std);



double subnode_mean(const std::vector<double>& y, xinfo_sizet& Xorder, const size_t& split_var);

double subnode_mean_newXorder(const std::vector<double>& y, const xinfo_sizet& Xorder_full, const xinfo_sizet& Xorder_next_index, const size_t& split_var, const std::vector<size_t>& Xorder_firstline, const size_t& N_Xorder);




struct likelihood_fullset_std : public Worker {
    // input variables, pass by reference
    const std::vector<double>& y_std;
    const xinfo_sizet& Xorder_std;
    const size_t& N_Xorder;
    const std::vector<size_t> subset_vars;
    const double& tau;
    const double& Ntau;
    const double& sigma2;
    std::vector<double>& loglike;

    // constructor
    likelihood_fullset_std(const std::vector<double>& y_std, const xinfo_sizet& Xorder_std, const size_t& N_Xorder, const std::vector<size_t> subset_vars, const double& tau, const  double& Ntau, const double& sigma2, std::vector<double>& loglike) : y_std(y_std), Xorder_std(Xorder_std), N_Xorder(N_Xorder), subset_vars(subset_vars), tau(tau), Ntau(Ntau), sigma2(sigma2), loglike(loglike){}

    void operator()(std::size_t begin, std::size_t end){
        std::vector<double> y_cumsum(N_Xorder);
        std::vector<double> y_cumsum_inv(N_Xorder);
        std::vector<double> Y_sort(N_Xorder);
        double* ypointer;
        size_t var_ind;
        double n1tau;
        double n2tau;
        double y_sum;
        for(size_t i = begin; i < end; i ++ ){
            var_ind = subset_vars[i];
            for(size_t q = 0; q < N_Xorder; q ++ ){
                Y_sort[q] = y_std[Xorder_std[var_ind][q]];
            }

            ypointer = &Y_sort[0];

            std::partial_sum(Y_sort.begin(), Y_sort.end(), y_cumsum.begin());

            y_sum = y_cumsum[y_cumsum.size() - 1];

            for(size_t k = 0; k < N_Xorder; k ++ ){
                y_cumsum_inv[k] = y_sum - y_cumsum[k];
            }

            for(size_t j = 0; j < N_Xorder - 1; j ++ ){
                n1tau = (j + 1) * tau;
                n2tau = Ntau - n1tau;
                loglike[(N_Xorder-1) * var_ind + j] = - 0.5 * log(n1tau + sigma2) - 0.5 * log(n2tau + sigma2) + 0.5 * tau * pow(y_cumsum[j], 2) / (sigma2 * (n1tau + sigma2)) + 0.5 * tau * pow(y_cumsum_inv[j], 2) / (sigma2 * (n2tau + sigma2));

            }

        }
        return;
    }

};



struct likelihood_subset_std : public Worker{
    // input variables, pass by reference
    const std::vector<double>& y_std;
    const xinfo_sizet& Xorder_std;
    const size_t& N_Xorder;
    const size_t& Ncutpoints;
    const std::vector<size_t> subset_vars;
    const double& tau;
    const double& sigma2;
    std::vector<size_t>& candidate_index;
    std::vector<double>& loglike;

    // constructor
    likelihood_subset_std(const std::vector<double>& y_std, const xinfo_sizet& Xorder_std, const size_t& N_Xorder, size_t& Ncutpoints, const std::vector<size_t> subset_vars, const double& tau, const double& sigma2, std::vector<size_t>& candidate_index, std::vector<double>& loglike) : y_std(y_std), Xorder_std(Xorder_std), N_Xorder(N_Xorder), Ncutpoints(Ncutpoints), subset_vars(subset_vars), tau(tau), sigma2(sigma2), candidate_index(candidate_index), loglike(loglike){}


    void operator()(std::size_t begin, std::size_t end){
        std::vector<double> Y_sort(N_Xorder);
        double* ypointer;
        double n1tau;
        double n2tau;
        double Ntau = N_Xorder * tau;
        size_t var_ind;
        double y_sum = 0.0;
        std::vector<double> y_cumsum(Ncutpoints);
        std::vector<double> y_cumsum_inv(Ncutpoints);

        bool firstrun = true;   // flag of the first loop


        // for(size_t i = 0; i < p; i ++ ){
        for(size_t i = begin; i < end; i ++ ){

            var_ind = subset_vars[i];

            for(size_t q = 0;  q < N_Xorder; q++ ){
                Y_sort[q] = y_std[Xorder_std[var_ind][q]];
            }
            ypointer = &Y_sort[0];

            if(firstrun){
                y_sum = sum_vec(Y_sort);
                firstrun = false;
            }

            calculate_y_cumsum_std(ypointer, Y_sort.size(), y_sum, candidate_index, y_cumsum, y_cumsum_inv);

            for(size_t j = 0; j < Ncutpoints; j ++ ){
                // loop over all possible cutpoints
                n1tau = (candidate_index[j] + 1) * tau; // number of points on left side (x <= cutpoint)
                n2tau = Ntau - n1tau; // number of points on right side (x > cutpoint)
                loglike[(Ncutpoints) * var_ind + j] = - 0.5 * log(n1tau + sigma2) - 0.5 * log(n2tau + sigma2) + 0.5 * tau * pow(y_cumsum[j], 2) / (sigma2 * (n1tau + sigma2)) + 0.5 * tau * pow(y_cumsum_inv[j], 2) / (sigma2 * (n2tau + sigma2));

            }
        }

        return;

    }

};



// overload plus for std vectors
template <typename T>
std::vector<T> operator+(const std::vector<T>& a, const std::vector<T>& b)
{
    assert(a.size() == b.size());

    std::vector<T> result;
    result.reserve(a.size());

    std::transform(a.begin(), a.end(), b.begin(),
                   std::back_inserter(result), std::plus<T>());
    return result;
}



// overload minus for std vectors
template <typename T>
std::vector<T> operator-(const std::vector<T>& a, const std::vector<T>& b)
{
    assert(a.size() == b.size());

    std::vector<T> result;
    result.reserve(a.size());

    std::transform(a.begin(), a.end(), b.begin(),
                   std::back_inserter(result), std::minus<T>());
    return result;
}




// sort std vectors from small to large numbers, return indexes
std::vector<size_t> sort_indexes(const Rcpp::NumericVector &v);



// overload print out for std vectors
template <typename T>
std::ostream& operator<< (std::ostream& out, const std::vector<T>& v) {
  if ( !v.empty() ) {
    out << '[';
    std::copy (v.begin(), v.end(), std::ostream_iterator<T>(out, ", "));
    out << "\b\b]";
  }
  return out;
}

double sq_diff_arma_std(arma::vec vec1, std::vector<double> vec2);
double sq_vec_diff(std::vector<double>& v1, std::vector<double>& v2);



void recover_Xorder(xinfo_sizet& Xorder, std::vector<size_t>& Xorder_firstline, xinfo_sizet& Xorder_next_index, xinfo_sizet& Xorder_new);

void create_y_sort(std::vector<double>& Y_sort, const std::vector<double>& y_std, const xinfo_sizet& Xorder, const xinfo_sizet& Xorder_next_index, const std::vector<size_t>& Xorder_firstline, const size_t& var);

void create_y_sort_2(std::vector<double>& Y_sort, std::vector<double>& possible_cutpoints, const double* X_std, const std::vector<double>& y_std, const xinfo_sizet& Xorder, const xinfo_sizet& Xorder_next_index, const std::vector<size_t>& Xorder_firstline, const size_t& var, const size_t& N_y);
void create_y_sort_3(std::vector<double>& Y_sort, std::vector<double>& possible_cutpoints, const double* X_std, const std::vector<double>& y_std, const xinfo_sizet& Xorder, const xinfo_sizet& Xorder_next_index, const std::vector<size_t>& Xorder_firstline, const size_t& var, const size_t& N_y, std::vector<size_t>& candidate_index);


void compute_partial_sum(std::vector<double>& Y, xinfo_sizet& Xorder, const size_t& var, std::vector<double>& y_cumsum);



void compute_partial_sum_newXorder(const std::vector<double>& y_std, const xinfo_sizet& Xorder, const xinfo_sizet& Xorder_next_index, const std::vector<size_t>& Xorder_firstline, const size_t& var, const size_t N_y, std::vector<double>& y_cumsum, std::vector<double>& possible_cutpoints, const double* X_std);

#endif
