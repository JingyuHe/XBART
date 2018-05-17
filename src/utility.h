#ifndef GUARD_utility_h
#define GUARD_utility_h

#include "common.h"


// copy NumericMatrix to STD matrix
xinfo copy_xinfo(Rcpp::NumericMatrix& X);

// copy IntegerMatrix to STD matrix
xinfo_sizet copy_xinfo_sizet(Rcpp::IntegerMatrix& X);

// // initialize STD matrix
xinfo ini_xinfo(size_t N, size_t p);

// // initialize STD integer matrix
xinfo_sizet ini_xinfo_sizet(size_t N, size_t p);

std::vector<double> row_sum(xinfo& X);

std::vector<double> col_sum(xinfo& X);

double sum_squared(std::vector<double> v);

double sum_vec(std::vector<double>& v);

void seq_gen(size_t start, size_t end, size_t length_out, arma::uvec& vec);

void calculate_y_cumsum(arma::vec& y, double y_sum, arma::uvec& ind, arma::vec& y_cumsum, arma::vec& y_cumsum_inv);


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
        arma::vec y_cumsum(N);
        arma::vec y_cumsum_inv(N);
        arma::vec y_sort(N);
        double y_sum;
        for(size_t i = begin; i < end; i++){ // loop over variables 
            y_cumsum = arma::cumsum(y(Xorder.col(i)));
            y_sum = y_cumsum(y_cumsum.n_elem - 1);
            y_cumsum_inv = y_sum - y_cumsum;  // redundant copy!
            loglike(arma::span(i * (N - 1), i * (N - 1) + N - 2)) = - 0.5 * log(n1tau + sigma2) - 0.5 * log(n2tau + sigma2) + 0.5 * tau * pow(y_cumsum(arma::span(0, N - 2)), 2) / (sigma2 * (n1tau + sigma2)) + 0.5 * tau * pow(y_cumsum_inv(arma::span(0, N - 2)), 2)/(sigma2 * (n2tau + sigma2));   
        }
        return;
    }
};





struct likelihood_evaluation_adaptive : public Worker {
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
    likelihood_evaluation_adaptive(const arma::vec& y, const arma::umat& Xorder, arma::vec&loglike, const double& sigma2, const double& tau, const size_t& N, const arma::vec& n1tau, const arma::vec& n2tau) : y(y), Xorder(Xorder), loglike(loglike), sigma2(sigma2), tau(tau), N(N), n1tau(n1tau), n2tau(n2tau){}

    // fucntion call operator that work for specified index range
    void operator()(std::size_t begin, std::size_t end){
        arma::vec y_cumsum(N);
        arma::vec y_cumsum_inv(N);
        arma::vec y_sort(N);
        double y_sum;
        for(size_t i = begin; i < end; i++){ // loop over variables 
            y_cumsum = arma::cumsum(y(Xorder.col(i)));
            y_sum = y_cumsum(y_cumsum.n_elem - 1);
            y_cumsum_inv = y_sum - y_cumsum;  // redundant copy!
            loglike(arma::span(i * (N - 1), i * (N - 1) + N - 2)) = - 0.5 * log(n1tau + sigma2) - 0.5 * log(n2tau + sigma2) + 0.5 * tau * pow(y_cumsum(arma::span(0, N - 2)), 2) / (sigma2 * (n1tau + sigma2)) + 0.5 * tau * pow(y_cumsum_inv(arma::span(0, N - 2)), 2)/(sigma2 * (n2tau + sigma2));   
        }
        return;
    }
};




#endif