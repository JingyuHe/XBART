%module xbart_cpp_



%{
#define SWIG_FILE_WITH_INIT
#include "xbart.h"

%}

%include "numpy.i"
%init %{
import_array();
%}

%apply (int DIM1,double* IN_ARRAY1) {(int n,double *a)};
%apply (int DIM1, double* ARGOUT_ARRAY1) {(int size, double *arr)};
%apply (int DIM1,int DIM2,double* IN_ARRAY2) {(int n, int d,double *a)};
%apply (int DIM1,double* IN_ARRAY1) {(int n_y,double *a_y)};
%apply (int DIM1,int DIM2,double* IN_ARRAY2) {(int n_test, int d_test,double *a_test)};

/* Rewrite the high level interface to init constructor */
%pythoncode %{
import collections
%}


%include "xbart.h"




