#include "utility.h"

ThreadPool thread_pool;

void ini_xinfo(xinfo &X, size_t N, size_t p)
{
    // xinfo X;
    X.resize(p);

    for (size_t i = 0; i < p; i++)
    {
        X[i].resize(N);
    }

    // return std::move(X);
    return;
}

void ini_xinfo_sizet(xinfo_sizet &X, size_t N, size_t p)
{
    // xinfo_sizet X;
    X.resize(p);

    for (size_t i = 0; i < p; i++)
    {
        X[i].resize(N);
    }

    // return std::move(X);
    return;
}

double subnode_mean(const std::vector<double> &y, xinfo_sizet &Xorder, const size_t &split_var)
{
    // calculate mean of y falls into the same subnode
    double output = 0.0;
    size_t N_Xorder = Xorder[split_var].size();
    for (size_t i = 0; i < N_Xorder; i++)
    {
        output = output + y[Xorder[split_var][i]];
    }
    output = output / N_Xorder;
    return output;
}

double subnode_mean_newXorder(const std::vector<double> &y, const xinfo_sizet &Xorder_full, const xinfo_sizet &Xorder_next_index, const size_t &split_var, const std::vector<size_t> &Xorder_firstline, const size_t &N_Xorder)
{
    // calculate mean of y falls into the same subnode
    double output = 0.0;
    // size_t N_Xorder = Xorder[split_var].size();

    size_t current_index = Xorder_firstline[split_var];
    for (size_t i = 0; i < N_Xorder; i++)
    {
        output = output + y[Xorder_full[split_var][current_index]];
        current_index = Xorder_next_index[split_var][current_index];
    }
    output = output / N_Xorder;
    return output;
}

void row_sum(xinfo &X, std::vector<double> &output)
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

void col_sum(xinfo &X, std::vector<double> &output)
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

void seq_gen_std_categorical(size_t start, size_t end, size_t length_out, std::vector<size_t> &vec, std::vector<size_t> output)
{
    // take a subset of vec, write to output, with fixed length
    double incr = (double)(end - start) / (double)length_out;
    size_t temp;
    for (size_t i = 0; i < length_out; i++)
    {
        temp = (size_t)incr * i + start;
        output[i] = vec[temp];
    }
    return;
}

void calculate_y_cumsum_std(const double *y, const size_t N_y, double y_sum, std::vector<size_t> &ind, std::vector<double> &y_cumsum, std::vector<double> &y_cumsum_inv)
{
    // compute cumulative sum of chunks for y, separate by ind vector
    // N is length of y (total)
    // y_cumsum_chunk should be lenght M + 1
    size_t M = y_cumsum.size();
    size_t ind_ind = 0;
    std::vector<double> y_cumsum_chunk(M + 1);

    y_cumsum_chunk[0] = 0; // initialize

    for (size_t i = 0; i < N_y; i++)
    {
        if (i <= ind[ind_ind])
        {
            y_cumsum_chunk[ind_ind] = y_cumsum_chunk[ind_ind] + y[i];
        }
        else
        {
            if (ind_ind < M - 1)
            {
                ind_ind = ind_ind + 1;
            }
            y_cumsum_chunk[ind_ind] = 0;
            y_cumsum_chunk[ind_ind] = y_cumsum_chunk[ind_ind] + y[i];
        }
    }

    y_cumsum[0] = y_cumsum_chunk[0];
    y_cumsum_inv[0] = y_sum - y_cumsum[0];
    for (size_t i = 1; i < M; i++)
    {
        y_cumsum[i] = y_cumsum[i - 1] + y_cumsum_chunk[i];
        y_cumsum_inv[i] = y_sum - y_cumsum[i];
    }

    return;
}

void compute_partial_sum_adaptive(std::vector<double> &y_std, std::vector<size_t> &candidate_index, std::vector<double> &y_cumsum, xinfo_sizet &Xorder_std, const size_t &var)
{
    size_t M = y_cumsum.size();
    size_t N_Xorder = Xorder_std[0].size();
    size_t ind = 0;
    y_cumsum[0] = 0.0;
    // size_t N_Xorder = Xorder_std[0].size();
    for (size_t i = 0; i < N_Xorder; i++)
    {
        if (i <= candidate_index[ind])
        {
            y_cumsum[ind] = y_cumsum[ind] + y_std[Xorder_std[var][i]];
        }
        else
        {
            if (ind < M - 1)
            {
                ind++;
            }
            y_cumsum[ind] = y_cumsum[ind - 1] + y_std[Xorder_std[var][i]];
        }
    }
    return;
}

void compute_partial_sum_adaptive_newXorder(std::vector<double> &y_std, std::vector<size_t> &candidate_index, std::vector<double> &y_cumsum, xinfo_sizet &Xorder_std, const size_t &var, xinfo_sizet &Xorder_next_index, std::vector<size_t> &Xorder_firstline, size_t N_Xorder, std::vector<double> &possible_cutpoints, size_t N_y, const double *X_std)
{
    size_t M = y_cumsum.size();

    size_t current_index = Xorder_firstline[var];

    size_t ind = 0;

    y_cumsum[0] = 0.0;

    for (size_t i = 0; i < N_Xorder; i++)
    {
        if (i <= candidate_index[ind])
        {
            y_cumsum[ind] = y_cumsum[ind] + y_std[Xorder_std[var][current_index]];
        }
        else
        {
            if (ind < M - 1)
            {
                ind++;
            }
            y_cumsum[ind] = y_cumsum[ind - 1] + y_std[Xorder_std[var][current_index]];

            possible_cutpoints[ind] = *(X_std + N_y * var + Xorder_std[var][current_index]);
        }
        current_index = Xorder_next_index[var][current_index];
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

void recover_Xorder(xinfo_sizet &Xorder, std::vector<size_t> &Xorder_firstline, xinfo_sizet &Xorder_next_index, xinfo_sizet &Xorder_new)
{
    size_t p = Xorder.size();
    // size_t n = Xorder[0].size();
    size_t current_index;
    std::vector<size_t> temp;

    for (size_t i = 0; i < p; i++)
    {
        current_index = Xorder_firstline[i];
        temp.clear();
        while (current_index < UINT_MAX)
        {
            temp.push_back(Xorder[i][current_index]);
            current_index = Xorder_next_index[i][current_index];
        }
        Xorder_new[i] = temp;
    }
    return;
}

void create_y_sort(std::vector<double> &Y_sort, const std::vector<double> &y_std, const xinfo_sizet &Xorder, const xinfo_sizet &Xorder_next_index, const std::vector<size_t> &Xorder_firstline, const size_t &var)
{
    // recover sorted Y using Xorder linked list object
    // only consider variable var
    size_t current_index = Xorder_firstline[var];

    size_t i = 0;
    while (current_index < UINT_MAX)
    {
        Y_sort[i] = y_std[Xorder[var][current_index]];
        current_index = Xorder_next_index[var][current_index];
        i++;
    }

    return;
}

void create_y_sort_2(std::vector<double> &Y_sort, std::vector<double> &possible_cutpoints, const double *X_std, const std::vector<double> &y_std, const xinfo_sizet &Xorder, const xinfo_sizet &Xorder_next_index, const std::vector<size_t> &Xorder_firstline, const size_t &var, const size_t &N_y)
{
    // recover sorted Y using Xorder linked list object
    // only consider variable var
    size_t current_index = Xorder_firstline[var];
    size_t i = 0;

    while (current_index < UINT_MAX)
    {

        possible_cutpoints[i] = *(X_std + N_y * var + Xorder[var][current_index]);

        Y_sort[i] = y_std[Xorder[var][current_index]];
        current_index = Xorder_next_index[var][current_index];
        i++;
    }

    return;
}

void compute_partial_sum_newXorder(const std::vector<double> &y_std, const xinfo_sizet &Xorder, const xinfo_sizet &Xorder_next_index, const std::vector<size_t> &Xorder_firstline, const size_t &var, const size_t N_y, std::vector<double> &y_cumsum, std::vector<double> &possible_cutpoints, const double *X_std)
{
    size_t current_index = Xorder_firstline[var];
    size_t i = 0;
    // first element
    y_cumsum[0] = y_std[Xorder[var][current_index]];
    possible_cutpoints[0] = *(X_std + N_y * var + Xorder[var][current_index]);
    current_index = Xorder_next_index[var][current_index];
    i = 1;

    while (current_index < UINT_MAX)
    {
        possible_cutpoints[i] = *(X_std + N_y * var + Xorder[var][current_index]);
        y_cumsum[i] = y_cumsum[i - 1] + y_std[Xorder[var][current_index]];
        current_index = Xorder_next_index[var][current_index];
        i++;
    }
    return;
}

void create_y_sort_3(std::vector<double> &Y_sort, std::vector<double> &possible_cutpoints, const double *X_std, const std::vector<double> &y_std, const xinfo_sizet &Xorder, const xinfo_sizet &Xorder_next_index, const std::vector<size_t> &Xorder_firstline, const size_t &var, const size_t &N_y, std::vector<size_t> &candidate_index)
{
    // recover sorted Y using Xorder linked list object
    // only consider variable var
    size_t current_index = Xorder_firstline[var];

    size_t i = 0; // loop over y vector
    size_t index = 0; // loop over index of possible_cutpoints
    while (current_index < UINT_MAX)
    {
        Y_sort[i] = y_std[Xorder[var][current_index]];

        if (index < candidate_index.size() && i == (candidate_index[index]))
        {
            possible_cutpoints[index] = *(X_std + N_y * var + Xorder[var][current_index]);
            index = index + 1;
        }

        current_index = Xorder_next_index[var][current_index];

        i++;
    }

    return;
}

void compute_partial_sum(std::vector<double> &Y, xinfo_sizet &Xorder, const size_t &var, std::vector<double> &y_cumsum)
{
    // size_t p = Xorder.size();
    size_t N = Xorder[0].size();

    // first element
    y_cumsum[0] = Y[Xorder[var][0]];

    for (size_t q = 1; q < N; q++)
    {
        y_cumsum[q] = y_cumsum[q - 1] + Y[Xorder[var][q]];
    }
    return;
}

void partial_sum_y(std::vector<double> &y, xinfo_sizet &Xorder, size_t &start, size_t &end, double &y_sum, const size_t &var)
{
    // compute sum of y[Xorder[start:end, var]]
    size_t loop_count = 0;
    for (size_t i = start; i <= end; i++)
    {
        y_sum = y_sum + y[Xorder[var][i]];
        loop_count++;
    }

    return;
}

void unique_value_count2(const double *Xpointer, xinfo_sizet &Xorder_std, //std::vector<size_t> &X_values,
                         std::vector<double> &X_values, std::vector<size_t> &X_counts, std::vector<size_t> &variable_ind, size_t &total_points, std::vector<size_t> &X_num_unique, size_t &p_categorical, size_t &p_continuous)
{
    size_t N = Xorder_std[0].size();
    size_t p = Xorder_std.size();
    double current_value = 0.0;
    size_t count_unique = 0;
    size_t N_unique;
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

    return;
}

