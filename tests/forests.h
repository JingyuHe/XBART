#ifndef GUARD_bart_h
#define GUARD_bart_h

#include <ctime>

#include "tree.h"
#include "treefuns.h"

// [[Rcpp::plugins(cpp11)]]

class Forests
{
    public:
    	Forests();
        Forests(const std::vector<std::vector<tree>> &);

    protected:
        std::vector<std::vector<tree>> trees; // matrix of trees

};

#endif
