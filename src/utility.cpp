#include "utility.h"

// ThreadPool thread_pool;
//
void ini_xinfo(matrix<double> &X, size_t N, size_t p)
{
    // matrix<double> X;
    X.resize(p);

    for (size_t i = 0; i < p; i++)
    {
        X[i].resize(N);
    }

    // return std::move(X);
    return;
}

void ini_xinfo(matrix<double> &X, size_t N, size_t p, double var)
{
    // matrix<double> X;
    X.resize(p);

    for (size_t i = 0; i < p; i++)
    {
        X[i].resize(N, var);
    }

    // return std::move(X);
    return;
}

void ini_xinfo_sizet(matrix<size_t> &X, size_t N, size_t p)
{
    // matrix<size_t> X;
    X.resize(p);

    for (size_t i = 0; i < p; i++)
    {
        X[i].resize(N);
    }

    // return std::move(X);
    return;
}

void row_sum(matrix<double> &X, std::vector<double> &output)
{
    size_t p = X.size();
    size_t N = X[0].size();
    // std::vector<double> output(N);
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < p; j++)
        {
            // COUT << X[j][i] << endl;
            output[i] = output[i] + X[j][i];
        }
    }
    return;
}

void col_sum(matrix<double> &X, std::vector<double> &output)
{
    size_t p = X.size();
    size_t N = X[0].size();
    // std::vector<double> output(p);
    for (size_t i = 0; i < p; i++)
    {
        for (size_t j = 0; j < N; j++)
        {
            output[i] = output[i] + X[i][j];
        }
    }
    return;
}

double sum_squared(std::vector<double> &v)
{
    size_t N = v.size();
    double output = 0.0;
    for (size_t i = 0; i < N; i++)
    {
        output = output + pow(v[i], 2);
    }
    return output;
}

double sum_vec(std::vector<double> &v)
{
    size_t N = v.size();
    double output = 0;
    for (size_t i = 0; i < N; i++)
    {
        output = output + v[i];
    }
    return output;
}

void seq_gen_std(size_t start, size_t end, size_t length_out, std::vector<size_t> &vec)
{
    // generate a sequence of integers, save in std vector container
    double incr = (double)(end - start) / (double)length_out;

    for (size_t i = 0; i < length_out; i++)
    {
        vec[i] = (size_t)incr * i + start;
    }

    return;
}

void seq_gen_std2(size_t start, size_t end, size_t length_out, std::vector<size_t> &vec)
{
    // generate a sequence of integers, save in std vector container
    // different from seq_gen_std
    // always put the first element 0, actual vector output have length length_out + 1!
    double incr = (double)(end - start) / (double)length_out;

    vec[0] = 0;

    for (size_t i = 1; i < length_out + 1; i++)
    {
        vec[i] = (size_t)incr * (i - 1) + start;
    }

    return;
}

void vec_sum(std::vector<double> &vector, double &sum)
{
    sum = 0.0;
    for (size_t i = 0; i < vector.size(); i++)
    {
        sum = sum + vector[i];
    }
    return;
}

void vec_sum_sizet(std::vector<size_t> &vector, size_t &sum)
{
    sum = 0;
    for (size_t i = 0; i < vector.size(); i++)
    {
        sum = sum + vector[i];
    }
    return;
}

double sq_vec_diff(std::vector<double> &v1, std::vector<double> &v2)
{
    assert(v1.size() == v2.size());
    size_t N = v1.size();
    double output = 0.0;
    for (size_t i = 0; i < N; i++)
    {
        output = output + pow(v1[i] - v2[i], 2);
    }
    return output;
}

double sq_vec_diff_sizet(std::vector<size_t> &v1, std::vector<size_t> &v2)
{
    assert(v1.size() == v2.size());
    size_t N = v1.size();
    double output = 0.0;
    for (size_t i = 0; i < N; i++)
    {
        output = output + pow(v1[i] - v2[i], 2);
    }
    return output;
}

void unique_value_count2(const double *Xpointer, matrix<size_t> &Xorder_std, // std::vector<size_t> &X_values,
                         std::vector<double> &X_values, std::vector<size_t> &X_counts, std::vector<size_t> &variable_ind, size_t &total_points, std::vector<size_t> &X_num_unique, size_t &p_categorical, size_t &p_continuous)
{
    size_t N = Xorder_std[0].size();
    size_t p = Xorder_std.size();
    double current_value = 0.0;
    size_t count_unique = 0;
    // size_t N_unique;
    variable_ind[0] = 0;

    total_points = 0;
    for (size_t i = p_continuous; i < p; i++)
    {
        // only loop over categorical variables
        // suppose p = (p_continuous, p_categorical)
        // index starts from p_continuous
        X_counts.push_back(1);
        current_value = *(Xpointer + i * N + Xorder_std[i][0]);
        X_values.push_back(current_value);
        count_unique = 1;

        for (size_t j = 1; j < N; j++)
        {
            if (*(Xpointer + i * N + Xorder_std[i][j]) == current_value)
            {
                X_counts[total_points]++;
            }
            else
            {
                current_value = *(Xpointer + i * N + Xorder_std[i][j]);
                X_values.push_back(current_value);
                X_counts.push_back(1);
                count_unique++;
                total_points++;
            }
        }
        variable_ind[i + 1 - p_continuous] = count_unique + variable_ind[i - p_continuous];
        X_num_unique[i - p_continuous] = count_unique;
        total_points++;
    }

    // std::cout << "total_points " << total_points << std::endl;

    return;
}

void get_X_range(const double *Xpointer, std::vector<std::vector<size_t>> &Xorder_std, std::vector<std::vector<double>> &X_range, size_t &n_y)
{
    size_t N = Xorder_std[0].size();
    size_t p = Xorder_std.size();
    ini_matrix(X_range, 2, p);

    // get 95% quantile to avoid outliers
    double alpha = 0.05;
    size_t low_idx = (size_t)floor(N * alpha / 2);
    size_t up_idx = (size_t)floor(N * (1 - alpha / 2));
    // cout << "N = " << N << ", low = " << low_idx << ", up_idx = " << up_idx<< endl;
    for (size_t i = 0; i < p; i++)
    {
        // X_range[i][0] = *(Xpointer + i * n_y + Xorder_std[i][0]);
        // X_range[i][1] = *(Xpointer + i * n_y + Xorder_std[i][N-1]);
        X_range[i][0] = *(Xpointer + i * n_y + Xorder_std[i][low_idx]);
        X_range[i][1] = *(Xpointer + i * n_y + Xorder_std[i][up_idx]);
    }

    // std::cout << "total_points " << total_points << std::endl;

    return;
}

double normal_density(double y, double mean, double var, bool take_log)
{
    // density of normal distribution
    double output = 0.0;

    output = -0.5 * log(2.0 * 3.14159265359 * var) - pow(y - mean, 2) / 2.0 / var;
    if (!take_log)
    {
        output = exp(output);
    }
    return output;
}

bool is_non_zero(size_t x) { return (x > 0); }

size_t count_non_zero(std::vector<double> &vec)
{
    size_t output = 0;
    for (size_t i = 0; i < vec.size(); i++)
    {
        if (vec[i] != 0)
        {
            output++;
        }
    }
    return output;
}

double wrap(double x)
{
    return (x - std::floor(x));
}

void multinomial_distribution(const size_t size, std::vector<double> &prob, std::vector<double> &draws, std::mt19937 &gen)
{
    std::discrete_distribution<> d(prob.begin(), prob.end());
    draws.resize(prob.size());
    std::fill(draws.begin(), draws.end(), 0);

    for (size_t i = 0; i < size; i++)
    {
        draws[d(gen)] += 1;
    }
    return;
}

void dirichlet_distribution(std::vector<double> &prob, std::vector<double> &alpha, std::mt19937 &gen)
{
    size_t p = alpha.size();
    prob.resize(p);

    for (size_t i = 0; i < p; i++)
    {
        std::gamma_distribution<double> temp_dist(alpha[i], 1.0);
        prob[i] = temp_dist(gen);
    }
    // normalize
    double weight_sum = accumulate(prob.begin(), prob.end(), 0.0);
    for (size_t i = 0; i < p; i++)
    {
        prob[i] = prob[i] / weight_sum;
    }
    return;
}

void get_rel_covariance(mat &cov, mat &X, std::vector<double> X_range, double theta, double tau)
{
    double temp;
    for (size_t i = 0; i < X.n_rows; i++)
    {
        for (size_t j = i; j < X.n_rows; j++)
        {
            // (tau*exp(-sum(theta * abs(x - y) / range)))
            temp = 0;
            for (size_t k = 0; k < X.n_cols; k++)
            {
                temp += pow(X(i, k) - X(j, k), 2) / pow(X_range[k], 2) / 2;
                // temp += std::abs(X(i,k) - X(j, k)) / X_range[k];
            }
            // cout << "distance = " << temp << endl;
            cov(i, j) = tau * exp(-theta * temp);
            cov(j, i) = cov(i, j);
        }
    }
    return;
}