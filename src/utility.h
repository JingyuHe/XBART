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


void row_sum(xinfo& X, std::vector<double> output);

void col_sum(xinfo& X, std::vector<double> output);

void vec_sum(std::vector<double> vector, double& sum);

double sum_squared(std::vector<double> v);

double sum_vec(std::vector<double>& v);

void seq_gen(size_t start, size_t end, size_t length_out, arma::uvec& vec);

void seq_gen_std(size_t start, size_t end, size_t length_out, std::vector<size_t>& vec);

void calculate_y_cumsum(arma::vec& y, double y_sum, arma::uvec& ind, arma::vec& y_cumsum, arma::vec& y_cumsum_inv);

void calculate_y_cumsum_std(const double * y, const size_t N_y, double y_sum, std::vector<size_t>& ind, std::vector<double>& y_cumsum, std::vector<double>& y_cumsum_inv);

double subnode_mean(const std::vector<double>& y, xinfo_sizet& Xorder, const size_t& split_var);

struct likelihood_evaluation_subset : public Worker {
    // input variables, pass by reference
    const arma::vec& y;
    const arma::umat& Xorder;
    arma::uvec& candidate_index;
    arma::vec& loglike;
    const double& sigma2;
    const double& tau;
    const double& y_sum;
    const size_t& Ncutpoints;
    const size_t& N;
    const arma::vec& n1tau;
    const arma::vec& n2tau;
    
    // constructor
    likelihood_evaluation_subset(const arma::vec& y, const arma::umat& Xorder, arma::uvec& candidate_index, arma::vec&loglike, const double& sigma2, const double& tau, const double& y_sum, const size_t& Ncutpoints, const size_t& N, const arma::vec& n1tau, const arma::vec& n2tau) : y(y), Xorder(Xorder), candidate_index(candidate_index), loglike(loglike), sigma2(sigma2), tau(tau), y_sum(y_sum), Ncutpoints(Ncutpoints), N(N), n1tau(n1tau), n2tau(n2tau){}

    // fucntion call operator that work for specified index range
    void operator()(std::size_t begin, std::size_t end){
        arma::vec y_cumsum(Ncutpoints);
        arma::vec y_cumsum_inv(Ncutpoints);
        arma::vec y_sort(N);
        for(size_t i = begin; i < end; i ++ ){

            y_sort = y(Xorder.col(i));
            calculate_y_cumsum(y_sort, y_sum, candidate_index, y_cumsum, y_cumsum_inv);
            loglike(arma::span(i * Ncutpoints, i * Ncutpoints + Ncutpoints - 1)) = - 0.5 * log(n1tau + sigma2) - 0.5 * log(n2tau + sigma2) + 0.5 * tau * pow(y_cumsum, 2) / (sigma2 * (n1tau + sigma2)) + 0.5 * tau * pow(y_cumsum_inv, 2)/(sigma2 * (n2tau + sigma2));
                        // cout << "    ----  ---   " << endl;
        }
        return;
    }
};




struct likelihood_evaluation_fullset : public Worker {
    // input variables, pass by reference
    const arma::vec& y;
    const arma::umat& Xorder;
    arma::vec& loglike;
    const double& sigma2;
    const double& tau;
    const size_t& N;
    const arma::vec& n1tau;
    const arma::vec& n2tau;
    
    // constructor
    likelihood_evaluation_fullset(const arma::vec& y, const arma::umat& Xorder, arma::vec&loglike, const double& sigma2, const double& tau, const size_t& N, const arma::vec& n1tau, const arma::vec& n2tau) : y(y), Xorder(Xorder), loglike(loglike), sigma2(sigma2), tau(tau), N(N), n1tau(n1tau), n2tau(n2tau){}

    // fucntion call operator that work for specified index range
    void operator()(std::size_t begin, std::size_t end){
        arma::vec y_cumsum(y.n_elem);
        arma::vec y_cumsum_inv(y.n_elem);
        double y_sum = 0.0;
        for(size_t i = begin; i < end; i++){ // loop over variables 
            y_cumsum = arma::cumsum(y.rows(Xorder.col(i)));
            y_sum = y_cumsum(y_cumsum.n_elem - 1);
            y_cumsum_inv = y_sum - y_cumsum;  // redundant copy!
            loglike(arma::span(i * (N - 1), i * (N - 1) + N - 2)) = - 0.5 * log(n1tau + sigma2) - 0.5 * log(n2tau + sigma2) + 0.5 * tau * pow(y_cumsum(arma::span(0, N - 2)), 2) / (sigma2 * (n1tau + sigma2)) + 0.5 * tau * pow(y_cumsum_inv(arma::span(0, N - 2)), 2)/(sigma2 * (n2tau + sigma2));   
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
template <typename T>
std::vector<size_t> sort_indexes(const std::vector<T> &v) {

  // initialize original index locations
  std::vector<size_t> idx(v.size());
  iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  sort(idx.begin(), idx.end(),
       [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

  return idx;
}



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

#endif