#include "utility_rcpp.h"

////////////////////////////////////////////////////////////////////////
//                                                                    //
//                                                                    //
//  Full function, support both continuous and categorical variables  //
//                                                                    //
//                                                                    //
////////////////////////////////////////////////////////////////////////

void rcpp_to_std2(arma::mat &y, arma::mat &X, arma::mat &Xtest, std::vector<double> &y_std, double &y_mean, Rcpp::NumericMatrix &X_std, Rcpp::NumericMatrix &Xtest_std, matrix<size_t> &Xorder_std)
{
    // The goal of this function is to convert RCPP object to std objects

    // TODO: Refactor code so for loops are self contained functions
    // TODO: Why RCPP and not std?
    // TODO: inefficient Need Replacement?

    size_t N = X.n_rows;
    size_t p = X.n_cols;
    size_t N_test = Xtest.n_rows;

    // Create y_std
    for (size_t i = 0; i < N; i++)
    {
        y_std[i] = y(i, 0);
        y_mean = y_mean + y_std[i];
    }
    y_mean = y_mean / (double)N;

    // X_std
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < p; j++)
        {
            X_std(i, j) = X(i, j);
        }
    }

    // X_std_test
    for (size_t i = 0; i < N_test; i++)
    {
        for (size_t j = 0; j < p; j++)
        {
            Xtest_std(i, j) = Xtest(i, j);
        }
    }

    // Create Xorder
    // Order
    umat Xorder(X.n_rows, X.n_cols);
    for (size_t i = 0; i < X.n_cols; i++)
    {
        Xorder.col(i) = sort_index(X.col(i));
    }
    // Create
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < p; j++)
        {
            Xorder_std[j][i] = Xorder(i, j);
        }
    }

    return;
}

void rcpp_to_std2(arma::mat &y, arma::mat &X, std::vector<double> &y_std, double &y_mean, Rcpp::NumericMatrix &X_std, matrix<size_t> &Xorder_std)
{
    // The goal of this function is to convert RCPP object to std objects

    // TODO: Refactor code so for loops are self contained functions
    // TODO: Why RCPP and not std?
    // TODO: inefficient Need Replacement?

    size_t N = X.n_rows;
    size_t p = X.n_cols;

    // Create y_std
    for (size_t i = 0; i < N; i++)
    {
        y_std[i] = y(i, 0);
        y_mean = y_mean + y_std[i];
    }
    y_mean = y_mean / (double)N;

    // X_std
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < p; j++)
        {
            X_std(i, j) = X(i, j);
        }
    }

    // Create Xorder
    // Order
    umat Xorder(X.n_rows, X.n_cols);
    for (size_t i = 0; i < X.n_cols; i++)
    {
        Xorder.col(i) = sort_index(X.col(i));
    }
    // Create
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < p; j++)
        {
            Xorder_std[j][i] = Xorder(i, j);
        }
    }

    return;
}

void rcpp_to_std2(arma::mat &X, Rcpp::NumericMatrix &X_std, matrix<size_t> &Xorder_std)
{
    // The goal of this function is to convert RCPP object to std objects

    // TODO: Refactor code so for loops are self contained functions
    // TODO: Why RCPP and not std?
    // TODO: inefficient Need Replacement?

    size_t N = X.n_rows;
    size_t p = X.n_cols;

    // X_std
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < p; j++)
        {
            X_std(i, j) = X(i, j);
        }
    }

    // Create Xorder
    // Order
    umat Xorder(X.n_rows, X.n_cols);
    for (size_t i = 0; i < X.n_cols; i++)
    {
        Xorder.col(i) = sort_index(X.col(i));
    }
    // Create
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < p; j++)
        {
            Xorder_std[j][i] = Xorder(i, j);
        }
    }

    return;
}

void Matrix_to_NumericMatrix(matrix<double> &a, Rcpp::NumericMatrix &b)
{
    // copy from a to b

    size_t a_cols = a.size();
    size_t a_rows = a[0].size();

    for (size_t i = 0; i < a_rows; i++)
    {
        for (size_t j = 0; j < a_cols; j++)
        {
            b(i, j) = a[j][i];
        }
    }
    return;
}

void rcpp_to_std2(arma::mat &y, arma::mat &Z, arma::mat &X, arma::mat &Ztest, arma::mat &Xtest, std::vector<double> &y_std, double &y_mean, matrix<double> &Z_std, Rcpp::NumericMatrix &X_std, matrix<double> &Ztest_std, Rcpp::NumericMatrix &Xtest_std, matrix<size_t> &Xorder_std)
{
    // The goal of this function is to convert RCPP object to std objects
    // TODO: Refactor code so for loops are self contained functions
    // TODO: Why RCPP and not std?
    // TODO: inefficient Need Replacement?
    size_t N = X.n_rows;
    size_t p = X.n_cols;
    size_t N_test = Xtest.n_rows;
    size_t p_z = Z.n_cols;
    // Create y_std
    for (size_t i = 0; i < N; i++)
    {
        y_std[i] = y(i, 0);
        y_mean = y_mean + y_std[i];
    }
    y_mean = y_mean / (double)N;
    // Z_std
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < p_z; j++)
        {
            Z_std[j][i] = Z(i, j);
        }
    }
    // X_std
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < p; j++)
        {
            X_std(i, j) = X(i, j);
        }
    }
    // X_std_test
    for (size_t i = 0; i < N_test; i++)
    {
        for (size_t j = 0; j < p; j++)
        {
            Xtest_std(i, j) = Xtest(i, j);
        }
    }
    // Z_std_test
    for (size_t i = 0; i < N_test; i++)
    {
        for (size_t j = 0; j < p_z; j++)
        {
            Ztest_std[j][i] = Ztest(i, j);
        }
    }
    // Create Xorder
    // Order
    arma::umat Xorder(X.n_rows, X.n_cols);
    // #pragma omp parallel for schedule(dynamic, 1) shared(X, Xorder)
    for (size_t i = 0; i < X.n_cols; i++)
    {
        Xorder.col(i) = arma::sort_index(X.col(i));
    }
    // Create
    // #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < p; j++)
        {
            Xorder_std[j][i] = Xorder(i, j);
        }
    }
    return;
}

void rcpp_to_std2(arma::mat &y, arma::mat &Z, arma::mat &X, arma::mat &X_ps, arma::mat &X_trt, arma::mat &Ztest, arma::mat &Xtest, arma::mat &Xtest_ps, arma::mat &Xtest_trt, std::vector<double> &y_std, double &y_mean, matrix<double> &Z_std, Rcpp::NumericMatrix &X_std, Rcpp::NumericMatrix &X_std_ps, Rcpp::NumericMatrix &X_std_trt, matrix<double> &Ztest_std, Rcpp::NumericMatrix &Xtest_std, Rcpp::NumericMatrix &Xtest_std_ps, Rcpp::NumericMatrix &Xtest_std_trt, matrix<size_t> &Xorder_std, matrix<size_t> &Xorder_std_ps, matrix<size_t> &Xorder_std_trt)
{
    // The goal of this function is to convert RCPP object to std objects
    // TODO: Refactor code so for loops are self contained functions
    // TODO: Why RCPP and not std?
    // TODO: inefficient Need Replacement?
    size_t N = X.n_rows;
    size_t p = X.n_cols;
    size_t p_ps = X_ps.n_cols;
    size_t p_trt = X_trt.n_cols;
    size_t N_test = Xtest.n_rows;
    size_t p_z = Z.n_cols;
    // Create y_std
    for (size_t i = 0; i < N; i++)
    {
        y_std[i] = y(i, 0);
        y_mean = y_mean + y_std[i];
    }
    y_mean = y_mean / (double)N;
    // Z_std
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < p_z; j++)
        {
            Z_std[j][i] = Z(i, j);
        }
    }
    // X_std
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < p; j++)
        {
            X_std(i, j) = X(i, j);
        }
    }
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < p_ps; j++)
        {
            X_std_ps(i, j) = X_ps(i, j);
        }
    }
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < p_trt; j++)
        {
            X_std_trt(i, j) = X_trt(i, j);
        }
    }

    // X_std_test
    for (size_t i = 0; i < N_test; i++)
    {
        for (size_t j = 0; j < p; j++)
        {
            Xtest_std(i, j) = Xtest(i, j);
        }
    }
    for (size_t i = 0; i < N_test; i++)
    {
        for (size_t j = 0; j < p_ps; j++)
        {
            Xtest_std_ps(i, j) = Xtest_ps(i, j);
        }
    }
    for (size_t i = 0; i < N_test; i++)
    {
        for (size_t j = 0; j < p_trt; j++)
        {
            Xtest_std_trt(i, j) = Xtest_trt(i, j);
        }
    }
    // Z_std_test
    for (size_t i = 0; i < N_test; i++)
    {
        for (size_t j = 0; j < p_z; j++)
        {
            Ztest_std[j][i] = Ztest(i, j);
        }
    }
    // Create Xorder
    // Order
    arma::umat Xorder(X.n_rows, X.n_cols);
    // #pragma omp parallel for schedule(dynamic, 1) shared(X, Xorder)
    for (size_t i = 0; i < X.n_cols; i++)
    {
        Xorder.col(i) = arma::sort_index(X.col(i));
    }
    // Create
    // #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < p; j++)
        {
            Xorder_std[j][i] = Xorder(i, j);
        }
    }

    // Create Xorder
    // Order
    arma::umat Xorder_ps(X_ps.n_rows, X_ps.n_cols);
    // #pragma omp parallel for schedule(dynamic, 1) shared(X, Xorder)
    for (size_t i = 0; i < X_ps.n_cols; i++)
    {
        Xorder_ps.col(i) = arma::sort_index(X_ps.col(i));
    }
    // Create
    // #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < p_ps; j++)
        {
            Xorder_std_ps[j][i] = Xorder_ps(i, j);
        }
    }

    // Create Xorder
    // Order
    arma::umat Xorder_trt(X_trt.n_rows, X_trt.n_cols);
    // #pragma omp parallel for schedule(dynamic, 1) shared(X, Xorder)
    for (size_t i = 0; i < X_trt.n_cols; i++)
    {
        Xorder_trt.col(i) = arma::sort_index(X_trt.col(i));
    }
    // Create
    // #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < p_trt; j++)
        {
            Xorder_std_trt[j][i] = Xorder_trt(i, j);
        }
    }
    return;
}