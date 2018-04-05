#ifndef GUARD_bart_h
#define GUARD_bart_h

#include <ctime>

#include "tree.h"
#include "treefuns.h"

// [[Rcpp::plugins(cpp11)]]



class forest{
public:
    // friends, main function to train the forest

    friend Rcpp::List train_forest(arma::mat y, arma::mat X, int M, int L, int MC, int N_sweeps);


    // constructor / destructor
    forest();
    forest(size_t m);
    forest(const forest&);
    ~forest();

    // operators
    forest& operator=(const forest&);

    // get, set

    // public methods

//protected:
    size_t m; // number of trees
    std::vector<tree::tree> t; // vector of trees

    // data

};

#endif
