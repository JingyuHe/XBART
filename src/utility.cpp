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


xinfo ini_xinfo(size_t p, size_t N){
    xinfo X;
    X.resize(p);

    for(size_t i = 0; i < p; i ++){
        X[i].resize(N);
    }

    return std::move(X);
}

xinfo_sizet ini_xinfo_sizet(size_t p, size_t N){
    xinfo_sizet X;
    X.resize(p);

    for(size_t i = 0; i < p; i ++){
        X[i].resize(N);
    }

    return std::move(X);
}