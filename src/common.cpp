#include "common.h"



// overload to print vectors and vector<vector>

std::ostream& operator<< (std::ostream& out, const std::vector<double>& v) {
    size_t last = v.size() - 1;
    for(size_t i = 0; i < v.size(); ++i) {
        out << v[i];
        if (i != last) 
            out << ", ";
    }
    return out;
}

std::ostream& operator<< (std::ostream& out, const std::vector<size_t>& v) {
    size_t last = v.size() - 1;
    for(size_t i = 0; i < v.size(); ++i) {
        out << v[i];
        if (i != last) 
            out << ", ";
    }
    return out;
}

std::ostream& operator<< (std::ostream& out, const std::vector< std::vector<double> >& v) {
    size_t last = v.size() - 1;
    for(size_t i = 0; i < v.size(); ++i) {
        out << v[i] << endl;
    }
    return out;
}

std::ostream& operator<< (std::ostream& out, const std::vector< std::vector<size_t> >& v) {
    size_t last = v.size() - 1;
    for(size_t i = 0; i < v.size(); ++i) {
        out << v[i] << endl;
    }
    return out;
}


double sum_vec(std::vector<double>& vec){
    size_t N = vec.size();
    double output = 0;
    for(size_t i = 0; i < N; i ++){
        output = output + vec[i];
    }
    return output;
}


