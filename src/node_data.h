#ifndef GUARD_node_data_h
#define GUARD_node_data_h

#include <ctime>
#include "common.h"
#include "utility.h"
#include <chrono>

struct NodeData
{
public:
    double sigma;
    size_t N_Xorder; // number of data observations in the current node

    NodeData(){
        sigma = 1.0;
        N_Xorder = 1;
        return;
    }

    NodeData(double sigma, size_t N_Xorder){
        this->sigma = sigma;
        this->N_Xorder = N_Xorder;
        return;
    }

    void update_value(double sigma, size_t N_Xorder){
        this->sigma = sigma;
        this->N_Xorder = N_Xorder;
        return;
    }

    void update_sigma(double sigma){
        this->sigma = sigma;
        return;
    }

    void update_N_Xorder(size_t N_Xorder){
        this->N_Xorder = N_Xorder;
        return;
    }
};

#endif
