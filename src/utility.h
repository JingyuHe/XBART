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
void ini_xinfo(matrix<double> &X, size_t N, size_t p);

void ini_xinfo(matrix<double> &X, size_t N, size_t p, double var);

// // initialize STD integer matrix
void ini_xinfo_sizet(matrix<size_t> &X, size_t N, size_t p);

void row_sum(matrix<double> &X, std::vector<double> &output);

void col_sum(matrix<double> &X, std::vector<double> &output);

void vec_sum(std::vector<double> &vector, double &sum);

void vec_sum_sizet(std::vector<size_t> &vector, size_t &sum);

double sum_squared(std::vector<double> &v);

double sum_vec(std::vector<double> &v);

void seq_gen_std(size_t start, size_t end, size_t length_out, std::vector<size_t> &vec);

void seq_gen_std2(size_t start, size_t end, size_t length_out, std::vector<size_t> &vec);

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

double sq_vec_diff(std::vector<double> &v1, std::vector<double> &v2);

double sq_vec_diff_sizet(std::vector<size_t> &v1, std::vector<size_t> &v2);

void unique_value_count2(const double *Xpointer, matrix<size_t> &Xorder_std, std::vector<double> &X_values, std::vector<size_t> &X_counts, std::vector<size_t> &variable_ind, size_t &total_points, std::vector<size_t> &X_num_unique, size_t &p_categorical, size_t &p_continuous);

double normal_density(double y, double mean, double var, bool take_log);

bool is_non_zero(size_t x);

size_t count_non_zero(std::vector<double> &vec);

#endif
