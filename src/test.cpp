
#include "sample_int_crank.h"

using namespace Rcpp;

// [[Rcpp::export(test)]]
void test(int n, int size, Rcpp::NumericVector prob){

    std::vector<double> prob_std(5);

    for(size_t i = 0; i < 5; i++){
        prob_std[i] = prob(i);
    }

    cout << "ok1" << endl;

    std::vector<size_t> output = sample_int_crank2(n, size, prob_std);
    

    cout << "ok2" << endl;

    cout << output << endl;

    return;

}

