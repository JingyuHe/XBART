
#include "utility.h"

using namespace Rcpp;

#include <queue>

void check_args2(int n, int size, const std::vector<double> &prob)
{
    if (n < size)
    {
        Rcpp::stop("cannot take a sample larger than the population");
    }

    if (prob.size() != n)
    {
        Rcpp::stop("incorrect number of probabilities");
    }
}

struct Comp2
{
    Comp2(const std::vector<double> &v) : _v(v) {}
    // Inverted comparison!
    bool operator()(int a, int b) { return _v[a] > _v[b]; }
    const std::vector<double> &_v;
};

template <class T>
T _divide_by_rexp(T t) { return t / M_E; }

template <class T>
T _add_one(T t) { return t + 1; }

// [[Rcpp::export(sample_int_crank)]]
std::vector<size_t> sample_int_crank2(int n, int size, std::vector<double> prob)
{
    check_args2(n, size, prob);

    // We need the last "size" elements of
    // U ^ (1 / prob) ~ log(U) / prob
    //                ~ -Exp(1) / prob
    //                ~ prob / Exp(1)
    // Here, ~ means "doesn't change order statistics".
    std::vector<double> rnd = prob / M_E;

    // Find the indexes of the first "size" elements under inverted
    // comparison.  Here, vx is zero-based.
    // IntegerVector vx = seq(0, n - 1);

    std::vector<size_t> vx(n);
    std::iota(vx.begin() + 1, vx.end(), 1);

    std::partial_sort(vx.begin(), vx.begin() + size, vx.end(), Comp2(rnd));

    // add one to each element
    std::transform(std::begin(vx), std::end(vx), std::begin(vx), [](size_t x) { return x + 1.0; });

    // Initialize with elements vx[1:size], applying transform "+ 1" --
    // we return one-based values.
    return vx;
}

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

// [[Rcpp::export(sample_int_ccrank)]]
std::vector<double> sample_int_ccrank2(int n, int size, std::vector<double> prob)
{
    check_args2(n, size, prob);

    // We need the last "size" elements of
    // U ^ (1 / prob) ~ log(U) / prob
    //                ~ -Exp(1) / prob
    //                ~ prob / Exp(1)
    // Here, ~ means "doesn't change order statistics".
    std::vector<double> rnd = std::vector<double>(n + 1);

    // Already shift by one, rnd[0] is uninitialized (and never accessed)
    std::transform(prob.begin(), prob.end(), rnd.begin() + 1, &_divide_by_rexp<double>);

    // Find the indexes of the first "size" elements under inverted
    // comparison.  Here, vx is zero-based.
    std::vector<double> vx = std::vector<double>(n);
    std::generate(vx.begin(), vx.end(), UniqueNumber(1));
    std::partial_sort(vx.begin(), vx.begin() + size, vx.end(), CComp(rnd));

    // Initialize with the first "size" elements of vx[1:size], they are already
    // 1-based.

    vx = vx + 1.0;

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

// [[Rcpp::export(sample_int_expj)]]
std::vector<size_t> sample_int_expj2(int n, int size, std::vector<double> prob)
{
    check_args2(n, size, prob);

    // Corner case
    if (size == 0)
    {
        std::vector<size_t> output;
        return output;
    }

    // Step 1: The first m items of V are inserted into R
    // Step 2: For each item v_i ∈ R: Calculate a key k_i = u_i^(1/w),
    // where u_i = random(0, 1)
    // (Modification: Calculate and store -log k_i = e_i / w where e_i = exp(1),
    //  reservoir is a priority queue that pops the *maximum* elements)
    std::priority_queue<std::pair<double, int>> R;

    for (std::vector<double>::iterator iprob = prob.begin();
         iprob != prob.begin() + size; ++iprob)
    {
        double k_i = _rexp_divide_by<double>(*iprob);
        R.push(std::make_pair(k_i, iprob - prob.begin() + 1));
    }

    // Step 4: Repeat Steps 5–10 until the population is exhausted
    {
        // Step 3: The threshold T_w is the minimum key of R
        // (Modification: This is now the logarithm)
        // Step 10: The new threshold T w is the new minimum key of R
        const std::pair<double, int> &T_w = R.top();

        // Incrementing iprob is part of Step 7
        for (std::vector<double>::iterator iprob = prob.begin() + size; iprob != prob.end(); ++iprob)
        {

            // Step 5: Let r = random(0, 1) and X_w = log(r) / log(T_w)
            // (Modification: Use e = -exp(1) instead of log(r))
            double X_w = Rf_rexp(1.0) / T_w.first;

            // Step 6: From the current item v_c skip items until item v_i, such that:
            double w = 0.0;

            // Step 7: w_c + w_{c+1} + ··· + w_{i−1} < X_w <= w_c + w_{c+1} + ··· + w_{i−1} + w_i
            for (; iprob != prob.end(); ++iprob)
            {
                w += *iprob;
                if (X_w <= w)
                    break;
            }

            // Step 7: No such item, terminate
            if (iprob == prob.end())
                break;

            // Step 9: Let t_w = T_w^{w_i}, r_2 = random(t_w, 1) and v_i’s key: k_i = (r_2)^{1/w_i}
            // (Mod: Let t_w = log(T_w) * {w_i}, e_2 = log(random(e^{t_w}, 1)) and v_i’s key: k_i = -e_2 / w_i)
            double t_w = -T_w.first * *iprob;

            // draw random uniform distribution here
            double e_2 = std::log(Rf_runif(std::exp(t_w), 1.0));
            double k_i = -e_2 / *iprob;

            // Step 8: The item in R with the minimum key is replaced by item v_i
            R.pop();
            R.push(std::make_pair(k_i, iprob - prob.begin() + 1));
        }
    }
    std::vector<size_t> ret(size);

    for (std::vector<size_t>::iterator iret = ret.end(); iret != ret.begin();)
    {
        --iret;

        if (R.empty())
        {
            stop("Reservoir empty before all elements have been filled");
        }

        *iret = R.top().second;
        R.pop();
    }

    if (!R.empty())
    {
        stop("Reservoir not empty after all elements have been filled");
    }

    return ret;
}

template <class T>
T _runif_to_one_by(T t) { return std::pow(Rf_runif(0.0, 1.0), 1.0 / t); }

// [[Rcpp::export]]
std::vector<size_t> sample_int_expjs2(int n, int size, std::vector<double> prob)
{
    check_args2(n, size, prob);

    // Corner case
    if (size == 0)
    {
        std::vector<size_t> output;
        return output;
    }

    // Step 1: The first m items of V are inserted into R
    // generate a sequence from 1 to size
    // std::vector<size_t> vx = seq(1, size);
    std::vector<size_t> vx(size);

    // if start from 1, vx.begin()
    // if start from 0, vx.begin() + 1
    std::iota(vx.begin(), vx.end(), 1);

    // Step 2: For each item v_i ∈ R: Calculate a key k_i = u_i^(1/w),
    // where u_i = random(0, 1)
    // std::vector<double> R = NumericVector(prob.begin(), prob.begin() + size, &_runif_to_one_by<double>);

    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(0.0,1.0);

    std::vector<double> R(size);
    for(size_t i = 0; i < size; i ++ ){
        R[i] = std::pow(distribution(generator), 1.0 / prob[i]);
    }

    // Step 3: The threshold T_w is the minimum key of R
    std::vector<double>::iterator T_w = find_min_item(R.begin(), R.end());

    // Step 4: Repeat Steps 5–10 until the population is exhausted
    {
        // Incrementing iprob is part of Step 7
        for (std::vector<double>::iterator iprob = prob.begin() + size; iprob != prob.end(); ++iprob)
        {

            // Step 5: Let r = random(0, 1) and X_w = log(r) / log(T_w)
            double X_w = std::log(Rf_runif(0.0, 1.0)) / std::log(*T_w);

            if (X_w < 0)
                Rcpp::stop("X_w < 0");

            // Step 6: From the current item v_c skip items until item v_i, such that:
            double w = 0.0;

            // Step 7: w_c + w_{c+1} + ··· + w_{i−1} < X_w <= w_c + w_{c+1} + ··· + w_{i−1} + w_i
            for (; iprob != prob.end(); ++iprob)
            {
                w += *iprob;
                if (X_w <= w)
                    break;
            }

            // Step 7: No such item, terminate
            if (iprob == prob.end())
                break;

            // Step 9: Let t_w = T_w^{w_i}, r_2 = random(t_w, 1) and v_i’s key: k_i = (r_2)^{1/w_i}
            // (Mod: Let t_w = log(T_w) * {w_i}, e_2 = log(random(e^{t_w}, 1)) and v_i’s key: k_i = e_2 / w_i)
            double t_w = std::pow(*T_w, *iprob);
            if (t_w < 0.0)
                Rcpp::stop("t_w < 0");
            if (t_w > 1.0)
                Rcpp::stop("t_w > 1");
            double r_2 = Rf_runif(t_w, 1.0);
            double k_i = std::pow(r_2, 1.0 / *iprob);

            // Step 8: The item in R with the minimum key is replaced by item v_i
            vx[(size_t)(T_w - R.begin())] = iprob - prob.begin() + 1;
            *T_w = k_i;

            // Step 10: The new threshold T w is the new minimum key of R
            T_w = find_min_item(R.begin(), R.end());
        }
    }

    // Create an array of indices in vx
    std::vector<double> vvx = std::vector<double>(size);
    std::generate(vvx.begin(), vvx.end(), UniqueNumber(0));

    // Sort it according to the key values in the reservoir
    std::sort(vvx.begin(), vvx.end(), Comp2(R));

    // Map to indices in the input array
    std::vector<size_t> output(vvx.size());

    for(size_t i = 0; i < vx.size(); i ++ ){
        output[i] = vvx[vx[i]];
    }

    // IntegerVector(vvx.begin(), vvx.end(), Indirection(vx));
    return output;
}