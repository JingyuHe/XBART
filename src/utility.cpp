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


void ini_xinfo(xinfo& X, size_t N, size_t p){
    // xinfo X;
    X.resize(p);

    for(size_t i = 0; i < p; i ++){
        X[i].resize(N);
    }

    // return std::move(X);
    return;
}

void ini_xinfo_sizet(xinfo_sizet& X, size_t N, size_t p){
    // xinfo_sizet X;
    X.resize(p);

    for(size_t i = 0; i < p; i ++){
        X[i].resize(N);
    }

    // return std::move(X);
    return;
}


double subnode_mean(const std::vector<double>& y, xinfo_sizet& Xorder, const size_t& split_var){
    // calculate mean of y falls into the same subnode
    double output = 0.0;
    size_t N_Xorder = Xorder[split_var].size();
    for(size_t i = 0; i < N_Xorder; i ++ ){
        output = output + y[Xorder[split_var][i]];
    }
    output = output / N_Xorder;
    return output;
}




double subnode_mean_newXorder(const std::vector<double>& y, const xinfo_sizet& Xorder_full, const xinfo_sizet& Xorder_next_index, const size_t& split_var, const std::vector<size_t>& Xorder_firstline, const size_t& N_Xorder){
    // calculate mean of y falls into the same subnode
    double output = 0.0;
    // size_t N_Xorder = Xorder[split_var].size();

    size_t current_index = Xorder_firstline[split_var];
    for(size_t i = 0; i < N_Xorder; i ++ ){
        output = output + y[Xorder_full[split_var][current_index]];
        current_index = Xorder_next_index[split_var][current_index];
    }
    output = output / N_Xorder;
    return output;
}





void row_sum(xinfo& X, std::vector<double>& output){
    size_t p = X.size();
    size_t N = X[0].size();
    // std::vector<double> output(N);
    for(size_t i = 0; i < N; i ++){
        for(size_t j = 0; j < p; j ++ ){
            // cout << X[j][i] << endl;
            output[i] = output[i] + X[j][i];
        }
    }
    return;
}


void col_sum(xinfo& X, std::vector<double>& output){
    size_t p = X.size();
    size_t N = X[0].size();
    // std::vector<double> output(p);
    for(size_t i = 0; i < p; i ++){
        for(size_t j = 0; j < N; j ++){
            output[i] = output[i] + X[i][j];
        }
    }
    return;
}


double sum_squared(std::vector<double>& v){
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


void seq_gen_std(size_t start, size_t end, size_t length_out, std::vector<size_t>& vec){
    // generate a sequence of integers, save in std vector container
    double incr = (double)(end - start) / (double)length_out;

    for(size_t i = 0; i < length_out; i ++ ){
        vec[i] = (size_t) incr * i + start;
    }

    return;
}



void calculate_y_cumsum_std(const double * y, const size_t N_y, double y_sum, std::vector<size_t>& ind, std::vector<double>& y_cumsum, std::vector<double>& y_cumsum_inv){
    // compute cumulative sum of chunks for y, separate by ind vector
    // N is length of y (total)
    // y_cumsum_chunk should be lenght M + 1
    size_t M = y_cumsum.size();
    size_t ind_ind = 0;
    std::vector<double> y_cumsum_chunk(M + 1);

    y_cumsum_chunk[0] = 0; // initialize

    for(size_t i = 0; i < N_y; i ++ ){
        if(i <= ind[ind_ind]){
            y_cumsum_chunk[ind_ind] = y_cumsum_chunk[ind_ind] + y[i];
        }else{
              if(ind_ind < M - 1){
                 ind_ind = ind_ind + 1;
             }
            y_cumsum_chunk[ind_ind] = 0;
            y_cumsum_chunk[ind_ind] = y_cumsum_chunk[ind_ind] + y[i];
        }
    }

    y_cumsum[0] = y_cumsum_chunk[0];
    y_cumsum_inv[0] = y_sum - y_cumsum[0];
    for(size_t i = 1; i < M; i ++ ){
        y_cumsum[i] = y_cumsum[i - 1] + y_cumsum_chunk[i];
        y_cumsum_inv[i] = y_sum - y_cumsum[i];
    }

    return;
}


void compute_partial_sum_adaptive(std::vector<double>& y_std, std::vector<size_t>& candidate_index, std::vector<double>& y_cumsum, xinfo_sizet& Xorder_std, const size_t& var){
    size_t M = y_cumsum.size();
    size_t N_Xorder = Xorder_std[0].size();
    size_t ind = 0;
    y_cumsum[0] = 0.0;
    // size_t N_Xorder = Xorder_std[0].size();
    for(size_t i = 0; i < N_Xorder; i ++ ){
        if(i <= candidate_index[ind]){
            y_cumsum[ind] = y_cumsum[ind] + y_std[Xorder_std[var][i]];
        }else{
            if(ind < M - 1){
                ind ++;
            }
            y_cumsum[ind] = y_cumsum[ind - 1] + y_std[Xorder_std[var][i]];
        }
    }
    return;
}


void compute_partial_sum_adaptive_newXorder(std::vector<double>& y_std, std::vector<size_t>& candidate_index, std::vector<double>& y_cumsum, xinfo_sizet& Xorder_std, const size_t& var, xinfo_sizet& Xorder_next_index, std::vector<size_t>& Xorder_firstline, size_t N_Xorder, std::vector<double>& possible_cutpoints, size_t N_y, const double* X_std){
    size_t M = y_cumsum.size();
    
    size_t current_index = Xorder_firstline[var];

    size_t ind = 0;

    y_cumsum[0] = 0.0;

    for(size_t i = 0; i < N_Xorder; i ++ ){
        if(i <= candidate_index[ind]){
            y_cumsum[ind] = y_cumsum[ind] + y_std[Xorder_std[var][current_index]];
        }else{
            if(ind < M - 1){
                ind ++ ;
            }
            y_cumsum[ind] = y_cumsum[ind - 1] + y_std[Xorder_std[var][current_index]];

            possible_cutpoints[ind] = *(X_std + N_y * var + Xorder_std[var][current_index]);
        }
        current_index = Xorder_next_index[var][current_index];

    }



    // size_t current_index = Xorder_firstline[var];

    // size_t i = 0;
    // // cout << "Xorder_next" << Xorder_next_index[var] << endl;

    // // cout << "xoder" << Xorder[var] << endl;

    // // cout << "size of Y_sort " << Y_sort.size() << "  ";
    // size_t index = 0;
    // while(current_index < UINT_MAX){
    //     // cout << "  " << i ;
    //     // cout << "  current_index " << current_index;       
    //     // cout << " Xorder " << Xorder[var][current_index];
    //     Y_sort[i] = y_std[Xorder[var][current_index]];
    //     current_index = Xorder_next_index[var][current_index];

    //     if(index < candidate_index.size() && i == (candidate_index[index] - 1)){
    //         possible_cutpoints[index] = *(X_std + N_y * var + Xorder[var][current_index]);
    //         index = index + 1;
    //     }

    //     i ++;
    // }



    return;
}


void vec_sum(std::vector<double>& vector, double& sum){
    sum = 0.0;
    for(size_t i = 0; i < vector.size(); i ++ ){
        sum = sum + vector[i];
    }
    return;
}


double sq_diff_arma_std(arma::vec vec1, std::vector<double> vec2){
    // compute squared difference between an armadillo vector and a std vector
    // for debug use
    assert(vec1.n_elem == vec2.size());
    size_t N = vec1.n_elem;
    double output = 0.0;
    for(size_t i = 0; i < N; i ++ ){
        output = output + pow(arma::as_scalar(vec1(i)) - vec2[i], 2);
    }
    return output;
}



double sq_vec_diff(std::vector<double>& v1, std::vector<double>& v2){
    assert(v1.size() == v2.size());
    size_t N = v1.size();
    double output = 0.0;
    for(size_t i =0; i < N; i ++ ){
        output = output + pow(v1[i] - v2[i], 2);
    }
    return output;
}


std::vector<size_t> sort_indexes(const Rcpp::NumericVector &v) {

  // initialize original index locations
  std::vector<size_t> idx(v.size());
  iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  sort(idx.begin(), idx.end(),
       [&v](size_t i1, size_t i2) {return v(i1) < v(i2);});

  return idx;
}



void recover_Xorder(xinfo_sizet& Xorder, std::vector<size_t>& Xorder_firstline, xinfo_sizet& Xorder_next_index, xinfo_sizet& Xorder_new){
    size_t p = Xorder.size();
    size_t n = Xorder[0].size();
    size_t current_index;
    std::vector<size_t> temp;

    // size_t MAX_SIZE_T = std::numeric_limits<size_t>::max();

    for(size_t i = 0; i < p; i ++ ){
        current_index = Xorder_firstline[i];
        temp.clear();
        while(current_index < UINT_MAX){
            // cout << Xorder[i][current_index] << endl;
            temp.push_back(Xorder[i][current_index]);
            current_index = Xorder_next_index[i][current_index];
        }
        Xorder_new[i] = temp;
        // cout << temp << endl;
    }
    return;
}


void create_y_sort(std::vector<double>& Y_sort, const std::vector<double>& y_std, const xinfo_sizet& Xorder, const xinfo_sizet& Xorder_next_index, const std::vector<size_t>& Xorder_firstline, const size_t& var){
    // recover sorted Y using Xorder linked list object
    // only consider variable var
    size_t current_index = Xorder_firstline[var];

    size_t i = 0;
    // cout << "Xorder_next" << Xorder_next_index[var] << endl;

    // cout << "xoder" << Xorder[var] << endl;

    // cout << "size of Y_sort " << Y_sort.size() << "  ";
    while(current_index < UINT_MAX){
        // cout << "  " << i ;
        // cout << "  current_index " << current_index;       
        // cout << " Xorder " << Xorder[var][current_index];
        Y_sort[i] = y_std[Xorder[var][current_index]];
        current_index = Xorder_next_index[var][current_index];
        i ++;
    }

    return;
}


void create_y_sort_2(std::vector<double>& Y_sort, std::vector<double>& possible_cutpoints, const double* X_std, const std::vector<double>& y_std, const xinfo_sizet& Xorder, const xinfo_sizet& Xorder_next_index, const std::vector<size_t>& Xorder_firstline, const size_t& var, const size_t& N_y){
    // recover sorted Y using Xorder linked list object
    // only consider variable var
    size_t current_index = Xorder_firstline[var];
    size_t i = 0;


    
    while(current_index < UINT_MAX){
        
        possible_cutpoints[i] = *(X_std + N_y * var + Xorder[var][current_index]);

        Y_sort[i] = y_std[Xorder[var][current_index]];
        current_index = Xorder_next_index[var][current_index];
        i ++;

    }

    return;
}



void compute_partial_sum_newXorder(const std::vector<double>& y_std, const xinfo_sizet& Xorder, const xinfo_sizet& Xorder_next_index, const std::vector<size_t>& Xorder_firstline, const size_t& var, const size_t N_y, std::vector<double>& y_cumsum, std::vector<double>& possible_cutpoints, const double* X_std){
    size_t current_index = Xorder_firstline[var];
    size_t i = 0;
    // first element 
    y_cumsum[0] = y_std[Xorder[var][current_index]];
    possible_cutpoints[0] = *(X_std + N_y * var + Xorder[var][current_index]);
    current_index = Xorder_next_index[var][current_index];
    i = 1;


    while(current_index < UINT_MAX){
        possible_cutpoints[i] = *(X_std + N_y * var + Xorder[var][current_index]);
        y_cumsum[i] = y_cumsum[i - 1] + y_std[Xorder[var][current_index]];
        current_index = Xorder_next_index[var][current_index];
        i++;
    }
    return;
}




void create_y_sort_3(std::vector<double>& Y_sort, std::vector<double>& possible_cutpoints, const double* X_std, const std::vector<double>& y_std, const xinfo_sizet& Xorder, const xinfo_sizet& Xorder_next_index, const std::vector<size_t>& Xorder_firstline, const size_t& var, const size_t& N_y, std::vector<size_t>& candidate_index){
    // recover sorted Y using Xorder linked list object
    // only consider variable var
    size_t current_index = Xorder_firstline[var];

    size_t i = 0;
    // cout << "Xorder_next" << Xorder_next_index[var] << endl;

    // cout << "xoder" << Xorder[var] << endl;

    // cout << "size of Y_sort " << Y_sort.size() << "  ";
    size_t index = 0;
    while(current_index < UINT_MAX){
        // cout << "  " << i ;
        // cout << "  current_index " << current_index;       
        // cout << " Xorder " << Xorder[var][current_index];
        Y_sort[i] = y_std[Xorder[var][current_index]];
        current_index = Xorder_next_index[var][current_index];

        if(index < candidate_index.size() && i == (candidate_index[index] - 1)){
            possible_cutpoints[index] = *(X_std + N_y * var + Xorder[var][current_index]);
            index = index + 1;
        }

        i ++;
    }

    return;
}


void compute_partial_sum(std::vector<double>& Y, xinfo_sizet& Xorder, const size_t& var, std::vector<double>& y_cumsum){
    size_t p = Xorder.size();
    size_t N = Xorder[0].size();

    // first element
    y_cumsum[0] = Y[Xorder[var][0]];

    for(size_t q = 1; q < N; q ++ ){
        y_cumsum[q] = y_cumsum[q - 1] + Y[Xorder[var][q]];
    }
    return;
}

