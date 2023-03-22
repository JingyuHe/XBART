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

void NumericMatrix_to_Matrix(Rcpp::NumericMatrix &a, matrix<double> &b)
{
    // copy from a to b

    size_t a_cols = a.ncol();
    size_t a_rows = a.nrow();
    ini_matrix(b, a_rows, a_cols);

    for (size_t i = 0; i < a_rows; i++)
    {
        for (size_t j = 0; j < a_cols; j++)
        {
            b[j][i] = a(i, j);
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

void rcpp_to_std2(arma::mat &y, arma::mat &Z, arma::mat &X_con, arma::mat &X_mod, std::vector<double> &y_std, double &y_mean, matrix<double> &Z_std, Rcpp::NumericMatrix &X_std_con, Rcpp::NumericMatrix &X_std_mod, matrix<size_t> &Xorder_std_con, matrix<size_t> &Xorder_std_mod)
{
    // The goal of this function is to convert RCPP object to std objects
    // TODO: Refactor code so for loops are self contained functions
    // TODO: Why RCPP and not std?
    // TODO: inefficient Need Replacement?
    size_t N = X_con.n_rows;
    size_t p_con = X_con.n_cols;
    size_t p_mod = X_mod.n_cols;
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
        for (size_t j = 0; j < p_con; j++)
        {
            X_std_con(i, j) = X_con(i, j);
        }
    }
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < p_mod; j++)
        {
            X_std_mod(i, j) = X_mod(i, j);
        }
    }
    // Create Xorder
    // Order
    arma::umat Xorder_con(X_con.n_rows, X_con.n_cols);
    // #pragma omp parallel for schedule(dynamic, 1) shared(X, Xorder)
    for (size_t i = 0; i < X_con.n_cols; i++)
    {
        Xorder_con.col(i) = arma::sort_index(X_con.col(i));
    }
    // Create
    // #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < p_con; j++)
        {
            Xorder_std_con[j][i] = Xorder_con(i, j);
        }
    }

    // Create Xorder
    // Order
    arma::umat Xorder_mod(X_mod.n_rows, X_mod.n_cols);
    // #pragma omp parallel for schedule(dynamic, 1) shared(X, Xorder)
    for (size_t i = 0; i < X_mod.n_cols; i++)
    {
        Xorder_mod.col(i) = arma::sort_index(X_mod.col(i));
    }
    // Create
    // #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < p_mod; j++)
        {
            Xorder_std_mod[j][i] = Xorder_mod(i, j);
        }
    }
    return;
}

void tree_to_string(vector<vector<tree>> &trees, Rcpp::StringVector &output_tree, size_t num_sweeps, size_t num_trees, size_t p)
{
    std::stringstream treess;
    for (size_t i = 0; i < num_sweeps; i++)
    {
        treess.precision(10);

        treess.str(std::string());
        treess << num_trees << " " << p << endl;

        for (size_t t = 0; t < num_trees; t++)
        {
            treess << (trees)[i][t];
        }

        output_tree(i) = treess.str();
    }
    return;
}
