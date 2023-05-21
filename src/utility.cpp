#include "utility.h"
#include "cdf.h"
#include <gsl/gsl_sf_bessel.h>
#include <Rcpp.h>

ThreadPool thread_pool;

using namespace Rcpp;

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

void unique_value_count2(const double *Xpointer, matrix<size_t> &Xorder_std, std::vector<double> &X_values, std::vector<size_t> &X_counts, std::vector<size_t> &variable_ind, size_t &total_points, std::vector<size_t> &X_num_unique, std::vector<size_t> &X_num_cutpoints, size_t &p_categorical, size_t &p_continuous)
{
    // count categorical variables, how many cutpoints
    size_t N = Xorder_std[0].size();
    size_t p = Xorder_std.size();
    double current_value = 0.0;
    size_t count_unique = 0;
    variable_ind[0] = 0;
    size_t count_points = 0;

    X_counts.resize(0);
    X_values.resize(0);
    X_num_unique.resize(p_categorical);
    X_num_cutpoints.resize(p_categorical);

    for (size_t i = p_continuous; i < p; i++)
    {
        // only loop over categorical variables
        // suppose X = (X_continuous, X_categorical)
        // p = p_continuous + p_categorical
        // index starts from p_continuous
        X_counts.push_back(1);
        current_value = *(Xpointer + i * N + Xorder_std[i][0]);
        X_values.push_back(current_value);
        count_unique = 1;

        for (size_t j = 1; j < N; j++)
        {
            // loop over all data
            if (*(Xpointer + i * N + Xorder_std[i][j]) == current_value)
            {
                // still the same X value
                X_counts[count_points]++;
            }
            else
            {
                // find one more unique value, move forward
                current_value = *(Xpointer + i * N + Xorder_std[i][j]);
                X_values.push_back(current_value);
                X_counts.push_back(1);
                count_unique++;
                count_points++;
            }
        }
        variable_ind[i + 1 - p_continuous] = count_unique + variable_ind[i - p_continuous];

        // number of unique values for each categorical X
        X_num_unique[i - p_continuous] = count_unique;
        // number of possible cutpoints is 1 less than total number of unique values
        X_num_cutpoints[i - p_continuous] = count_unique - 1;
        count_points++;
    }
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

    for (size_t i = 0; i < p; i++)
    {
        // X_range[i][0] = *(Xpointer + i * n_y + Xorder_std[i][0]);
        // X_range[i][1] = *(Xpointer + i * n_y + Xorder_std[i][N-1]);
        X_range[i][0] = *(Xpointer + i * n_y + Xorder_std[i][low_idx]);
        X_range[i][1] = *(Xpointer + i * n_y + Xorder_std[i][up_idx]);
    }
    return;
}

double normal_density(double y, double mean, double var, bool take_log)
{
    // density of normal distribution
    double output = 0.0;

    output = -0.5 * log(2.0 * M_PI * var) - pow(y - mean, 2) / 2.0 / var;
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
            cov(i, j) = tau * exp(-theta * temp);
            cov(j, i) = cov(i, j);
        }
    }
    return;
}

double sum_vec_yz(std::vector<double> &v, matrix<double> &z)
{
    double output = 0;
    for (size_t i = 0; i < v.size(); i++)
    {
        output = output + v[i] * (z[0][i]);
    }
    return output;
}

double sum_vec_z_squared(matrix<double> &z, size_t n)
{
    double output = 0;
    for (size_t i = 0; i < n; i++)
    {
        output = output + pow(z[0][i], 2);
    }
    return output;
}

double sum_vec_yzsq(std::vector<double> &v, matrix<double> &z)
{
    double output = 0;
    for (size_t i = 0; i < v.size(); i++)
    {
        output = output + v[i] * pow(z[0][i], 2);
    }
    return output;
}

double sum_vec_y_z(std::vector<double> &v, matrix<double> &z)
{
    double output = 0;
    for (size_t i = 0; i < v.size(); i++)
    {
        output = output + v[i] / z[0][i];
    }
    return output;
}

double drawlambdafromR(size_t n, double sy, double c, double d, std::mt19937 &gen)
{
    double logz1 = loggignorm(-c + n, 2 * d, 2 * sy);
    double logz2 = loggignorm(c + n, 0, 2 * (d + sy));
    // cout << "z1 = " << z1 << " z2 = " << z2 << endl;
    // double _pi =  z1 / (z1+z2) = 1 / (1 + z2 / z1) = 1 / (1 + exp(log(z2 / z1))) = 1 / (1 + exp(log(z2) - log(z1)))
    double _pi = 1 / (1 + exp(logz2 - logz1));
    double ret;
    std::uniform_real_distribution<double> udist(0, 1);

    if (udist(gen) < _pi)
    { // draw from gig(-c+r, 2*d, 2*s)
        double eta = -c + n;
        double chi = 2 * d;
        double psi = 2 * sy;
        Function f("rgig");
        NumericVector ret_r = f(1, eta, chi, psi);
        ret = ret_r(0);
        // if (ret > 1000)
        // {
        //     cout << "pi " << _pi << " logz1 " << logz1 << " logz2 " << logz2 << " z1 " << exp(logz1) << " z2" << exp(logz2) << endl;
        //     cout << "gig " << " eta " << eta << " chi " << chi << " psi " << psi << " ret " << ret << endl;
        // }
    }
    else
    {
        // draw from gig(c+r, 0, 2*(d+s)) or equivalently gamma(c+r, d+s)

        double eta = c + n;
        double chi = 0;
        double psi = 2 * (d + sy);
        Function f("rgig");
        NumericVector ret_r = f(1, eta, chi, psi);
        ret = ret_r(0);
        // if (ret > 1000)
        // {
        //     cout << "pi " << _pi << " logz1 " << logz1 << " logz2 " << logz2 << " z1 " << exp(logz1) << " z2" << exp(logz2) << endl;
        //     cout << "gamma " << " eta " << eta << " chi " << chi << " psi " << psi << " ret " << ret << endl;
        // }

        // cout << "draw from gamma" << endl;
        // std::gamma_distribution gammadist(c+n, d+sy);
        // ret = gammadist(gen);
        // if (ret > 1000)
        // {
        //     cout << "pi " << _pi << " logz1 " << logz1 << " logz2 " << logz2 << " z1 " << exp(logz1) << " z2" << exp(logz2) << endl;
        //     cout << "gamma dist " << " c+n " << c+n << " d+sy " << d+sy << " ret " << ret << endl;
        // }
    }
    return ret;
}

double drawnodelambda(size_t n, double sy, double c, double d, std::mt19937 &gen)
{
    /////////////////////////// generalize inversed Gaussian distribution
    // lambda ~ pi*GIG(-c+r, 2d, 2s) + (1-pi)*Gamma(c+r, d+s)
    // pi = Z(-c+r, 2*d, 2*s) / (Z(-c+r, 2d, 2s) + Z(c+r, 0, 2*(d+s)))
    // r = n, s = sy
    // cout << "n = " << n << " sy = " << sy << endl;
    double logz1 = loggignorm(-c + n, 2 * d, 2 * sy);
    double logz2 = loggignorm(c + n, 0, 2 * (d + sy));
    // cout << "z1 = " << z1 << " z2 = " << z2 << endl;
    // double _pi =  z1 / (z1+z2) = 1 / (1 + z2 / z1) = 1 / (1 + exp(log(z2 / z1))) = 1 / (1 + exp(log(z2) - log(z1)))
    double _pi = 1 / (1 + exp(logz2 - logz1));

    std::uniform_real_distribution<double> udist(0, 1);

    if (udist(gen) < _pi)
    { // draw from gig(-c+r, 2*d, 2*s)
        double eta = -c + n;
        double chi = 2 * d;
        double psi = 2 * sy;
        size_t num_try = 0;
        double u, v, x;

        double lambda = abs(eta);
        double beta = sqrt(chi * psi);
        double ret;

        if ((eta < 0) && (abs(psi) < pow(10, -6)))
            psi = 0;

        // Check Domain
        if ((eta > 0) && !((psi > 0) && (chi >= 0)))
        {
            cout << "Out of domain "
                 << "eta = " << eta << " chi = " << chi << " psi = " << psi << endl;
            exit(1);
        }
        else if ((eta == 0) && !((psi > 0) && (chi > 0)))
        {
            cout << "Out of domain "
                 << "eta = " << eta << " chi = " << chi << " psi = " << psi << endl;
            exit(1);
        }
        else if ((eta < 0) && !((psi >= 0) && (chi > 0)))
        {
            cout << "Out of domain "
                 << "eta = " << eta << " chi = " << chi << " psi = " << psi << endl;
            exit(1);
        }

        if ((psi == 0) && (eta < 0) && (chi > 0))
        {
            // cout << "case 1" << endl;
            std::gamma_distribution gammadist(-eta, 2 / chi); // if psi == 0, its a inverse gamma distribution invGamma(-eta, chi/2)
            return 1 / gammadist(gen);
        }
        else if ((chi == 0) && (eta > 0) && (psi > 0))
        {
            // cout << "case 2" << endl;
            std::gamma_distribution gammadist(eta, 2 / psi); // if chi == 0, it's Gamma(eta, psi/2)
            return gammadist(gen);
        }
        else if ((lambda < 1) && (lambda >= 0) && (beta <= sqrt(1 - lambda) * 2 / 3))
        {
            // cout << "case 3" << endl;
            /////////////// Rejection method for non-T-concave part ///////////////////////
            // source: https://core.ac.uk/download/pdf/11008021.pdf
            double k1, k2, k3, A1, A2, A3, A, h;
            double m = beta / ((1 - lambda) + sqrt(pow(1 - lambda, 2) + pow(beta, 2)));
            double x0 = beta / (1 - lambda);
            double xs = x0 > 2 / beta ? x0 : 2 / beta;
            k1 = exp((lambda - 1) * log(m) - beta * (m + 1 / m) / 2); // g(m) = x^(eta-1)*exp(-beta * (m+1/m) / 2)
            A1 = k1 * x0;

            if (x0 < 2 / beta)
            {
                k2 = exp(-beta);
                if (lambda == 0)
                {
                    A2 = k2 * log(2 / pow(beta, 2));
                }
                else
                {
                    A2 = k2 * (pow(2 / beta, lambda) - pow(x0, lambda)) / lambda;
                }
            }
            else
            {
                k2 = 0;
                A2 = 0;
            }

            k3 = pow(xs, lambda - 1);
            A3 = 2 * k3 * exp(-xs * beta / 2) / beta;
            A = A1 + A2 + A3;

            while (num_try < 1000)
            {
                u = udist(gen);
                v = udist(gen) * A;
                if (v <= A1)
                {
                    x = x0 * v / A1;
                    h = k1;
                }
                else if (v <= (A1 + A2))
                {
                    v = v - A1;
                    if (lambda == 0)
                    {
                        x = beta * exp(v * exp(beta));
                    }
                    else
                    {
                        x = pow(pow(x0, lambda) + v * lambda / k2, 1 / lambda);
                        h = k2 * pow(x, lambda - 1);
                    }
                }
                else
                {
                    v = v - A1 - A2;
                    x = -2 / beta * log(exp(-xs * beta / 2) - v * beta / 2 / k3);
                    h = k3 * exp(-x * beta / 2);
                }
                if (u * h <= exp((lambda - 1) * log(x) - beta * (x + 1 / x) / 2))
                { // uh <= g(x, eta, chi , psi)
                    // ret = eta >= 0 ? x : 1/x;
                    // return ret;
                    break;
                }
                else
                {
                    num_try += 1;
                }
            }
            // cout << "Warning: Sampling lambda exceeds 1000 iterations in rejection methhod for non-T-concave part" << endl;
            // cout << "eta = " << eta << "; chi = " << chi << "; psi = " << psi << endl;
            // cout << "c = " << c << "; d = " << d << "; n = " << n << "; sy = " << sy << endl;
            // cout << "k1 = " << k1 << "; k2 = " << k2 << "; k3 = " << k3 << endl;
            // cout << "A1 = " << A1 << "; A2 = " << A2 << "; A3 = " << A3 << "; x = " << x << endl;

            cout << "case 3 eta " << eta << " x " << x << " 1/x " << 1 / x << endl;
            ret = eta >= 0 ? x : 1 / x;
            return ret;
        }
        else if ((lambda <= 1) && (lambda >= 0) && (beta <= 1) && ((beta >= 1 / 2) | (beta >= sqrt(1 - lambda) * 2 / 3)))
        {
            // cout << "case 4" << endl;
            /////////////// Ratio-of-Uniforms without node shift ///////////////////////
            // source: https://core.ac.uk/download/pdf/11008021.pdf
            double m = beta / ((1 - lambda) + sqrt((pow(1 - lambda, 2) + pow(beta, 2))));
            double xp = ((1 + lambda) + sqrt(pow(1 + lambda, 2) + pow(beta, 2))) / beta;
            double vp = sqrt(exp((lambda - 1) * log(m) - beta * (m + 1 / m) / 2)); // sqrt(g(m))
            double up = xp * sqrt(exp((lambda - 1) * log(xp) - beta * (xp + 1 / xp) / 2));

            while (num_try < 1000)
            {
                u = udist(gen) * up;
                v = udist(gen) * vp;
                x = u / v;
                if (pow(v, 2) <= exp((lambda - 1) * log(x) - beta * (x + 1 / x) / 2))
                {
                    return x;
                }
                else
                {
                    num_try += 1;
                }
            }
            // cout << "Warning: Sampling lambda exceeds 1000 iterations in ratio-of-uniforms without mode shift" << endl;
            // cout << "eta = " << eta << "; chi = " << chi << "; psi = " << psi << endl;
            // cout << "c = " << c << "; d = " << d << "; n = " << n << "; sy = " << sy << endl;
            // cout << "m = " << m << "; xp = " << xp << "; vp = " << vp << "; up = " << up << "; x = " << x << endl;
            cout << "case 4 eta " << eta << " x " << x << " 1/x " << 1 / x << endl;
            ret = eta >= 0 ? x : 1 / x;
            return ret;
        }
        else if ((lambda > 1) && (beta > 1))
        {
            /////////////// Ratio-of-Uniforms method ///////////////////////
            double bx, dx, logib, logid, logu1, logu2;
            bx = sqrt(pow(lambda, 2) - 2 * lambda + chi * psi + 1) + lambda - 1 == 0 ? chi / (2 - 2 * lambda) : (sqrt(pow(lambda, 2) - 2 * lambda + chi * psi + 1) + lambda - 1) / psi;
            dx = sqrt(pow(lambda, 2) + 2 * lambda + chi * psi + 1) + lambda + 1 == 0 ? -chi / (2 * lambda + 2) : (sqrt(pow(lambda, 2) + 2 * lambda + chi * psi + 1) + lambda + 1) / psi;
            // ib = sqrt(exp(lgigkernel(bx, eta, chi, psi)));
            // id = dx * sqrt(exp(lgigkernel(dx, eta, chi, psi)));
            logib = 1 / 2 * lgigkernel(bx, lambda, chi, psi);
            logid = log(dx) + 1 / 2 * lgigkernel(dx, lambda, chi, psi);
            // u2 / u1 = exp(log(u2) - log(u1)) = exp(log(u) + log(ib) - log(u) - log(id))
            // log(ib) = log(exp(lgigkernel(bx, -)))^1/2) = 1/2 * lgigkernel(bx, -);

            // if bx or dx is less than 0, likely psi is too closed to zero and caused an rounding error.
            // if ((bx <= 0 | dx <= 0 | id <= 0 | ib <= 0) && (eta < 0)) return 1/gen.gamma(-eta, chi);

            while (num_try < 1000)
            {
                // u1 = udist(gen)*ib;
                // u2 = udist(gen)*id;
                logu1 = log(udist(gen)) + logib;
                logu2 = log(udist(gen)) + logid;
                if (isinf(logu1) | isinf(logu2) | isnan(logu1) | isnan(logu2))
                {
                    cout << "logu1 = " << logu1 << "; logu2 = " << logu2 << endl;
                    cout << "bx = " << bx << "; logib = " << logib << "; dx = " << dx << "; logid = " << logid << endl;
                    cout << "eta = " << eta << "; chi = " << chi << "; psi = " << psi << endl;
                    cout << "c = " << c << "; d = " << d << "; n = " << n << "; sy = " << sy << endl;
                    exit(1);
                }
                if (2 * logu1 <= lgigkernel(exp(logu2 - logu1), lambda, chi, psi))
                {
                    ret = exp(logu2 - logu1);
                    cout << "case 5 iter " << num_try << " eta " << eta << " ret " << ret << endl;
                    cout << "logu1 = " << logu1 << "; logu2 = " << logu2 << endl;
                    cout << "bx = " << bx << "; logib = " << logib << "; dx = " << dx << "; logid = " << logid << endl;
                    cout << "eta = " << eta << "; chi = " << chi << "; psi = " << psi << endl;
                    return ret;
                }
                else
                {
                    num_try += 1;
                }
            }

            // When psi is extremely small and the sampling can not converge, it will eventually cause overflows
            // So we try to consider psi as the case psi == 0
            if (eta < 0)
            {
                // cout << "case 5" << endl;
                std::gamma_distribution gammadist(-eta, 2 / chi); // if psi == 0, its a inverse gamma distribution invGamma(-eta, chi/2)
                ret = 1 / gammadist(gen);
                ret = exp(logu2 - logu1);
                cout << "case 5 invGamma eta " << eta << " ret " << ret << endl;
                return ret;
            }
            else
            {
                // cout << "case 6" << endl;
                // cout << "Warning: Sampling lambda exceeds 1000 iterations." << endl;
                // cout << "logu1 = " << logu1 << "; logu2 = " << logu2 << endl;
                // cout << "logbx = " << bx << "; logib = " << logib << "; dx = " << dx << "; logid = " << logid << endl;
                // cout << "eta = " << eta << "; chi = " << chi << "; psi = " << psi << endl;
                // cout << "c = " << c << "; d = " << d << "; n = " << n << "; sy = " << sy << endl;
                // cout << "u2 / u1 = " << exp(logu2 - logu1) << endl;
                ret = exp(logu2 - logu1);
                cout << "case 6 eta " << eta << " ret " << ret << endl;
                return ret;
            }
        }
        else if ((eta < 0) && (psi < 1))
        {
            std::gamma_distribution gammadist(-eta, 2 / chi); // if psi == 0, its a inverse gamma distribution invGamma(-eta, chi/2)
            ret = 1 / gammadist(gen);
            cout << "Warning psi is too small, treat as inverse gamma distribution, return " << ret << endl;
            return ret;
        }
        else
        {
            cout << "Currently can not sample "
                 << " eta = " << eta << " chi = " << chi << " psi = " << psi << " beta = " << beta << endl;
            exit(1);
        }
    }
    else
    {
        // draw from gig(c+r, 0, 2*(d+s)) or equivalently gamma(c+r, d+s)
        // cout << "draw from gamma" << endl;
        std::gamma_distribution gammadist(c + n, d + sy);
        return gammadist(gen);
    }
}

double gignorm(double eta, double chi, double psi)
{
    // cout << "eta = " << eta << " chi = " << chi << " psi = " << psi << endl;
    double ret;
    if ((eta > 0) && (chi == 0) && (psi > 0))
    {
        ret = exp(lgamma(eta) + eta * log(2 / psi));
    }
    else if ((eta < 0) && (chi > 0) && (psi == 0))
    {
        ret = exp(lgamma(-eta) - eta * log(2 / chi));
    }
    else if ((chi > 0) && (psi > 0))
    {
        double sq = sqrt(chi * psi);
        double bessel_k = boost::math::cyl_bessel_k(eta, sq);
        // cout << "eta = " << eta << " sqrt(chi*psi) = " << sqrt(chi*psi) << " bessel_k = " << bessel_k;
        double lbessel_k;
        if (eta > 0)
        {
            lbessel_k = gsl_sf_bessel_lnKnu(eta, sqrt(chi * psi));
        }
        else
        {
            lbessel_k = log(bessel_k);
        }
        // ret = exp(log(2*bessel_k) - (eta / 2) * log(psi / chi));
        ret = exp(log(2) + lbessel_k - (eta / 2) * log(psi / chi));
        // cout << " lnKnu = " << lbessel_k <<  " exp(lKn) = " << exp(lbessel_k) << " log(ret) = " << log(2) + lbessel_k - (eta / 2) * log(psi / chi) << " ret = " << ret << endl;
    }
    return ret;
}

double loggignorm(double eta, double chi, double psi)
{
    // cout << "eta = " << eta << " chi = " << chi << " psi = " << psi << endl;
    double ret;
    if ((eta > 0) && (chi == 0) && (psi > 0))
    {
        ret = lgamma(eta) + eta * log(2 / psi);
    }
    else if ((eta < 0) && (chi > 0) && (psi == 0))
    {
        ret = (lgamma(-eta) - eta * log(2 / chi));
    }
    else if ((chi > 0) && (psi > 0))
    {
        // cout << "eta = " << eta << " sqrt(chi*psi) = " << sqrt(chi*psi) << " bessel_k = " << bessel_k;
        double sq = sqrt(chi * psi);
        double lbessel_k = eta > 0 ? gsl_sf_bessel_lnKnu(eta, sq) : log(boost::math::cyl_bessel_k(eta, sq));
        // ret = exp(log(2*bessel_k) - (eta / 2) * log(psi / chi));
        ret = (log(2) + lbessel_k - (eta / 2) * log(psi / chi));
        // cout << " lnKnu = " << lbessel_k <<  " exp(lKn) = " << exp(lbessel_k) << " log(ret) = " << log(2) + lbessel_k - (eta / 2) * log(psi / chi) << " ret = " << ret << endl;
    }
    return ret;
}

double lgigkernel(double x, double eta, double chi, double psi)
{
    // return pow(x, eta-1)*exp(-(chi/x + psi*x)/2);
    return (eta - 1) * log(x) - (chi / x + psi * x) / 2;
}

double sample_truncated_normal(std::mt19937 &gen, double mu, double precision, double cutoff, bool greater)
{
    // draw from truncated normal
    // X ~ N(mu, sigma2) * I(X >= cutoff) if greater = true
    // X ~ N(mu, sigma2) * I(X < cutoff) if greater = false

    double u;
    double sigma = sqrt(1.0 / precision);
    double mu_quantile = normCDF((mu - cutoff) / sigma);

    double a = 0;
    double b = 1;

    if (greater)
    {
        a = std::min(mu_quantile, 0.999);
    }
    else
    {
        b = std::max(mu_quantile, 0.001);
    }

    std::uniform_real_distribution<double> unif(a, b);

    u = unif(gen);

    double output = normCDFInv(u) * sigma + mu;

    return output;
}

// double sample_truncated_normal(std::mt19937 &gen, double mu, double precision, double cutoff, bool greater)
// {
//     // draw from truncated normal
//     // X ~ N(mu, sigma2) * I(X >= cutoff) if greater = true
//     // X ~ N(mu, sigma2) * I(X < cutoff) if greater = false

//     double u;
//     double sigma = sqrt(1.0 / precision);
//     double mu_quantile = normCDF((mu - cutoff) / sigma);

//     double a = 0;
//     double b = 1;

//     std::uniform_real_distribution<double> unif(a, b);
//     u = unif(gen);

//     double output = normCDFInv(u + (1 - u) * mu_quantile) * sigma + mu;

//     return output;
// }
