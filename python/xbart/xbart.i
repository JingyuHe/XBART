%module xbart_cpp_



%{
#define SWIG_FILE_WITH_INIT
#include "xbart.h"

%}

%include "numpy.i"
%include <std_string.i>
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
%extend XBARTcpp{
%pythoncode %{
def fit_predict(self,x,y,x_test,p_cat=0):
    x_pred = self._fit_predict(x,y,x_test,y.shape[0],p_cat)
    yhats_test = self.get_yhats_test(self.get_N_sweeps()*x_test.shape[0]).reshape((x_test.shape[0],self.get_N_sweeps()),order='C')

    #self.importance = self.get_importance(x.shape[1])
    return yhats_test
%}
%pythoncode %{
def predict(self,x_test):
    x_pred = self._predict(x_test)
    yhats_test = self.get_yhats_test(self.get_N_sweeps()*x_test.shape[0])
    yhats_test = yhats_test.reshape((x_test.shape[0],self.get_N_sweeps()),order='C')
    return yhats_test
%}
%pythoncode %{
def fit(self,x,y,p_cat=0):
    return self._fit(x,y,p_cat)
%}

};

%include "xbart.h"




