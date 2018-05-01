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


double sum_squared(std::vector<double> v){
    size_t N = v.size();
    double output = 0.0;
    for(size_t i = 0; i < N; i ++ ){
        output = output + pow(v[i], 2);
    }
    return output;
}


double sum_vec(std::vector<double>& v){
    size_t N = v.size();
    double output = 0;
    for(size_t i = 0; i < N; i ++){
        output = output + v[i];
    }
    return output;
}


void seq_gen(size_t start, size_t end, size_t length_out, arma::uvec& vec){
    // generate a sequence of INTEGERS
    double incr = (double) (end - start) / (double) length_out;

    for(size_t i = 0; i < length_out; i ++ ){
        vec[i] = (size_t) incr * i + start;
    }

    return;
}


void calculate_y_cumsum(arma::vec& y, double y_sum, arma::uvec& ind, arma::vec& y_cumsum, arma::vec& y_cumsum_inv){
    // compute cumulative sum of chunks for y, separate by ind vector
    // y_cumsum_chunk should be lenght M + 1
    size_t N = y.n_elem;
    size_t M = y_cumsum.n_elem;
    assert(y_cumsum.n_elem == y_cumsum_inv.n_elem);
    size_t ind_ind = 0;
    arma::vec y_cumsum_chunk(M + 1);

    y_cumsum_chunk[0] = 0; // initialize  

    for(size_t i = 0; i < N; i ++ ){
        if(i <= ind[ind_ind]){
            y_cumsum_chunk[ind_ind] = y_cumsum_chunk[ind_ind] + y[i];
        }else{
            if(ind_ind < M){
                ind_ind = ind_ind + 1;
            }
            y_cumsum_chunk[ind_ind] = 0;
            y_cumsum_chunk[ind_ind] = y_cumsum_chunk[ind_ind] + y[i];
        }
    }

    y_cumsum[0] = y_cumsum_chunk[0];
    for(size_t i = 1; i < M; i ++ ){
        y_cumsum[i] = y_cumsum[i - 1] + y_cumsum_chunk[i];
        y_cumsum_inv[i] = y_sum - y_cumsum[i];
    }
    
    return;
}
