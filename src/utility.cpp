#include "utility.h"

xinfo copy_xinfo(Rcpp::NumericMatrix& X){
    size_t n_row = X.nrow();
    size_t n_col = X.ncol();

    xinfo Xinfo;

    //stacked by column
    Xinfo.resize(n_col);

    //copy to std matrix 
    for(size_t i = 0; i < n_col; i ++){
        Xinfo[i].resize(n_row);
        for(size_t j = 0; j < n_row; j++){
            Xinfo[i][j] = X(j, i);
        }
    }

    // std::move to avoid copying
    return std::move(Xinfo);
}


xinfo_sizet copy_xinfo_sizet(Rcpp::IntegerMatrix& X){
    size_t n_row = X.nrow();
    size_t n_col = X.ncol();

    xinfo_sizet Xinfo;

    //stacked by column
    Xinfo.resize(n_col);

    //copy to std matrix 
    for(size_t i = 0; i < n_col; i ++){
        Xinfo[i].resize(n_row);
        for(size_t j = 0; j < n_row; j++){
            Xinfo[i][j] = X(j, i);
        }
    }
    // std::move to avoid copying
    return std::move(Xinfo);
}


xinfo ini_xinfo(size_t N, size_t p){
    xinfo X;
    X.resize(p);

    for(size_t i = 0; i < p; i ++){
        X[i].resize(N);
    }

    return std::move(X);
}

xinfo_sizet ini_xinfo_sizet(size_t N, size_t p){
    xinfo_sizet X;
    X.resize(p);

    for(size_t i = 0; i < p; i ++){
        X[i].resize(N);
    }

    return std::move(X);
}


std::vector<double> row_sum(xinfo& X){
    size_t p = X.size();
    size_t N = X[0].size();
    std::vector<double> output(N);
    for(size_t i = 0; i < N; i ++){
        for(size_t j = 0; j < p; j ++ ){
            output[i] = output[i] + X[j][i];
        }
    }
    return output;
}


std::vector<double> col_sum(xinfo& X){
    size_t p = X.size();
    size_t N = X[0].size();
    std::vector<double> output(p);
    for(size_t i = 0; i < p; i ++){
        for(size_t j = 0; j < N; j ++){
            output[i] = output[i] + X[i][j];
        }
    }
    return output;
}
