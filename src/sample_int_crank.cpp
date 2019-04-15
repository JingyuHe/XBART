#include "sample_int_crank.h"

void check_args(int n, int size, const std::vector<double> &prob)
{
    if (n < size)
    {
        throw std::range_error("cannot take a sample larger than the population");
    }

    if (prob.size() != (size_t) n)
    {
        throw std::range_error("incorrect number of probabilities");
        return;
    }
}

struct Comp
{
    Comp(const std::vector<double> &v) : _v(v) {}
    // Inverted comparison!
    bool operator()(int a, int b) { return _v[a] > _v[b]; }
    const std::vector<double> &_v;
};

template <class T>
T _divide_by_rexp(T t) { return t / M_E; }

template <class T>
T _add_one(T t) { return t + 1; }

struct CComp
{
    CComp(const std::vector<double> &v) : _v(v) {}
    // Inverted comparison!
    bool operator()(int a, int b) { return _v[a] > _v[b]; }
    const std::vector<double> &_v;
};

struct UniqueNumber
{
    int current;
    UniqueNumber(int start = 0) { current = start; }
    int operator()() { return current++; }
};

std::vector<double> sample_int_ccrank(int n, int size, std::vector<double> prob)
{
    check_args(n, size, prob);

    // We need the last "size" elements of
    // U ^ (1 / prob) ~ log(U) / prob
    //                ~ -Exp(1) / prob
    //                ~ prob / Exp(1)
    // Here, ~ means "doesn't change order statistics".
    std::vector<double> rnd(n + 1);
    //std::vector<double> rnd = std::vector<double>(n );

    // Already shift by one, rnd[0] is uninitialized (and never accessed)
    std::transform(prob.begin(), prob.end(), rnd.begin() + 1, &_divide_by_rexp<double>);
    //std::transform(prob.begin(), prob.end(), rnd.begin() , &_divide_by_rexp<double>);

    // Find the indexes of the first "size" elements under inverted
    // comparison.  Here, vx is zero-based.
    std::vector<double> vx = std::vector<double>(n);
    std::generate(vx.begin(), vx.end(), UniqueNumber(1));
    std::partial_sort(vx.begin(), vx.begin() + size, vx.end(), CComp(rnd));

    // Initialize with the first "size" elements of vx[1:size], they are already
    // 1-based.
    return vx;
}

template <class T>
T _rexp_divide_by(T t) { return M_E / t; }

template <class T>
T find_min_item(T begin, T end)
{
    T T_w = begin;
    for (T iT_w = T_w + 1; iT_w != end; ++iT_w)
    {
        if (*iT_w < *T_w)
            T_w = iT_w;
    }

    return T_w;
}

struct Indirection
{
    Indirection(const std::vector<size_t> &v) : _v(v) {}
    // Inverted comparison!
    int operator()(int a) { return _v[a]; }
    const std::vector<size_t> &_v;
};