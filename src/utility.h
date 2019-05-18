#ifndef GUARD_utility_h
#define GUARD_utility_h

#include "common.h"

#include "thread_pool.h"
extern ThreadPool thread_pool;

#ifndef SWIG
#include <algorithm>
#include <functional>
#include <iterator>
#define _USE_MATH_DEFINES
#include <math.h>
#include <numeric>
#endif

template <typename T>
void ini_matrix(matrix<T> &matrix, size_t N, size_t p)
{
    matrix.resize(p);

    for (size_t i = 0; i < p; i++)
    {
        matrix[i].resize(N);
    }
    return;
}

// // initialize STD matrix
void ini_xinfo(xinfo &X, size_t N, size_t p);

// // initialize STD integer matrix
void ini_xinfo_sizet(xinfo_sizet &X, size_t N, size_t p);

void row_sum(xinfo &X, std::vector<double> &output);

void col_sum(xinfo &X, std::vector<double> &output);

void vec_sum(std::vector<double> &vector, double &sum);

void vec_sum_sizet(std::vector<size_t> &vector, size_t &sum);

double sum_squared(std::vector<double> &v);

double sum_vec(std::vector<double> &v);

void seq_gen_std(size_t start, size_t end, size_t length_out, std::vector<size_t> &vec);
void seq_gen_std_categorical(size_t start, size_t end, size_t length_out, std::vector<size_t> &vec, std::vector<size_t> output);

void seq_gen_std2(size_t start, size_t end, size_t length_out, std::vector<size_t> &vec);

void calculate_y_cumsum_std(const double *y, const size_t N_y, double y_sum, std::vector<size_t> &ind, std::vector<double> &y_cumsum, std::vector<double> &y_cumsum_inv);

void compute_partial_sum_adaptive(std::vector<double> &y_std, std::vector<size_t> &candidate_index, std::vector<double> &y_cumsum, xinfo_sizet &Xorder_std, const size_t &var);

void compute_partial_sum_adaptive_newXorder(std::vector<double> &y_std, std::vector<size_t> &candidate_index, std::vector<double> &y_cumsum, xinfo_sizet &Xorder_std, const size_t &var, xinfo_sizet &Xorder_next_index, std::vector<size_t> &Xorder_firstline, size_t N_Xorder, std::vector<double> &possible_cutpoints, size_t N_y, const double *X_std);

double subnode_mean(const std::vector<double> &y, xinfo_sizet &Xorder, const size_t &split_var);

double subnode_mean_newXorder(const std::vector<double> &y, const xinfo_sizet &Xorder_full, const xinfo_sizet &Xorder_next_index, const size_t &split_var, const std::vector<size_t> &Xorder_firstline, const size_t &N_Xorder);

// overload plus for std vectors
template <typename T>
std::vector<T> operator+(const std::vector<T> &a, const std::vector<T> &b)
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
std::vector<T> operator-(const std::vector<T> &a, const std::vector<T> &b)
{
    assert(a.size() == b.size());

    std::vector<T> result;
    result.reserve(a.size());

    std::transform(a.begin(), a.end(), b.begin(),
                   std::back_inserter(result), std::minus<T>());
    return result;
}

template <typename T>
std::vector<T> operator/(const std::vector<T> &a, const T &b)
{
    std::vector<T> result;
    for (size_t i = 0; i < a.size(); i++)
    {
        result[i] = a[i] / b;
    }
    return result;
}

template <typename T>
std::vector<T> operator+(const std::vector<T> &a, const T &b)
{
    std::vector<T> result;
    for (size_t i = 0; i < a.size(); i++)
    {
        result[i] = a[i] + b;
    }
    return result;
}

// overload print out for std vectors
template <typename T>
std::ostream &operator<<(std::ostream &out, const std::vector<T> &v)
{
    if (!v.empty())
    {
        out << '[';
        std::copy(v.begin(), v.end(), std::ostream_iterator<T>(out, ", "));
        out << "\b\b]";
    }
    return out;
}


template <typename T>
std::vector<size_t> sort_indexes(const std::vector<T> &v)
{

    // initialize original index locations
    std::vector<size_t> idx(v.size());
    iota(idx.begin(), idx.end(), 0);

    // sort indexes based on comparing values in v
    sort(idx.begin(), idx.end(),
         [&v](size_t i1, size_t i2) {
             // Compare index values by their respective v values
             return v[i1] < v[i2];
         });

    return idx;
}

// double sq_diff_arma_std(arma::vec vec1, std::vector<double> vec2);
double sq_vec_diff(std::vector<double> &v1, std::vector<double> &v2);
double sq_vec_diff_sizet(std::vector<size_t> &v1, std::vector<size_t> &v2);

void recover_Xorder(xinfo_sizet &Xorder, std::vector<size_t> &Xorder_firstline, xinfo_sizet &Xorder_next_index, xinfo_sizet &Xorder_new);

void create_y_sort(std::vector<double> &Y_sort, const std::vector<double> &y_std, const xinfo_sizet &Xorder, const xinfo_sizet &Xorder_next_index, const std::vector<size_t> &Xorder_firstline, const size_t &var);

void create_y_sort_2(std::vector<double> &Y_sort, std::vector<double> &possible_cutpoints, const double *X_std, const std::vector<double> &y_std, const xinfo_sizet &Xorder, const xinfo_sizet &Xorder_next_index, const std::vector<size_t> &Xorder_firstline, const size_t &var, const size_t &N_y);
void create_y_sort_3(std::vector<double> &Y_sort, std::vector<double> &possible_cutpoints, const double *X_std, const std::vector<double> &y_std, const xinfo_sizet &Xorder, const xinfo_sizet &Xorder_next_index, const std::vector<size_t> &Xorder_firstline, const size_t &var, const size_t &N_y, std::vector<size_t> &candidate_index);

void compute_partial_sum(std::vector<double> &Y, xinfo_sizet &Xorder, const size_t &var, std::vector<double> &y_cumsum);

void compute_partial_sum_newXorder(const std::vector<double> &y_std, const xinfo_sizet &Xorder, const xinfo_sizet &Xorder_next_index, const std::vector<size_t> &Xorder_firstline, const size_t &var, const size_t N_y, std::vector<double> &y_cumsum, std::vector<double> &possible_cutpoints, const double *X_std);

void partial_sum_y(std::vector<double> &y, xinfo_sizet &Xorder, size_t &start, size_t &end, double &y_sum, const size_t &var);

void unique_value_count2(const double *Xpointer, xinfo_sizet &Xorder_std, //std::vector<size_t> &X_values,
                         std::vector<double> &X_values, std::vector<size_t> &X_counts, std::vector<size_t> &variable_ind, size_t &total_points, std::vector<size_t> &X_num_unique, size_t &p_categorical, size_t &p_continuous);

#endif
