
%module(package="xbart", moduleimport="import _xbart_cpp_") xbart_cpp_


%{
/* inserts a macro that specifies that the resulting 
C++ file should be built as a python extension */

#define SWIG_FILE_WITH_INIT 
#include "xbart.h"
%}

/* JSON */
%include <std_string.i>

/* Numpy */
%include "numpy.i"
%init %{
import_array();
%}


%apply (int DIM1,double* IN_ARRAY1) {(int n,double *a)};
%apply (int DIM1, double* ARGOUT_ARRAY1) {(int size, double *arr)};
%apply (int DIM1,int DIM2,double* IN_ARRAY2) {(int n, int d,double *a)};
%apply (int DIM1,double* IN_ARRAY1) {(int n_y,double *a_y)};


%include "xbart.h" // Include code for a static version of Python




