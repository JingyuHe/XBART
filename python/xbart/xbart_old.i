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
%extend XBARTcpp{
%pythoncode %{
def __convert_params_check_types(self,params):
### This function converts params to list and 
### It handles the types of params and raises exceptions if needed
### It puts in default values for empty param values 
    import warnings
    from collections import OrderedDict
    DEFAULT_PARAMS = OrderedDict([('M',200),('L',1),("N_sweeps",40)
                        ,("Nmin",1),("Ncutpoints",100) # CHANGE
                        ,("alpha",0.95),("beta",1.25 ),("tau",0.3),# CHANGE
                        ("burnin",15),("mtry",2),("max_depth_num",250), # CHANGE
                        ("draw_sigma",False),("kap",16),("s",4),("verbose",False),
                        ("m_update_sigma",True), ("draw_mu",True),
                        ("parallel",False),("seed",0)])

    list_params = []
    for key,value in DEFAULT_PARAMS.items():
        true_type = type(value)
        new_value = params.get(key,value)
        if not isinstance(new_value,type(value)):  
            if true_type == int:
                if isinstance(new_value,float):
                    if int(new_value) == new_value:
                        new_value = int(new_value)
                        warnings.warn("Value was of " + str(key) + " converted from float to int")
                    else:
                        raise TypeError(str(key) +" should be a positive integer value")
                else:
                    raise TypeError(str(key) +" should be a positive integer")
            elif true_type == float:
                if isinstance(new_value,int):
                    new_value = float(new_value)  
                    ## warnings.warn("Value was of " + str(key) + " converted from int to float")          
                else:
                    raise TypeError(str(key) + " should be a float")
            elif true_type == bool:
                if int(new_value) in [0,1]:
                    new_value = bool(new_value)
                else:    
                    raise TypeError(str(key) + " should be a bool")               
        list_params.append(new_value)             
    return list_params    

def __init__(self,params = {}):

    assert isinstance(params, collections.Mapping), "params must be dictionary like"

    this = __xbart_cpp.new_XBARTcpp(*self.__convert_params_check_types(params))

    # init
    try:
        self.this.append(this)
    except __builtin__.Exception:
        self.this = this   

def __init__(self,num_trees= 200, l = 1 ,num_sweeps = 40, n_min =1,
                num_cutpoints=100,alpha=0.95,beta = 1.25, tau = 0.3,
                burnin = 15, mtry = 1, max_depth_num = 250,
                draw_sigma= False,kap= 16,s = 4,verbose=False,
                m_update_sigma = True, draw_mu = True,
                parallel=False,seed=0):  

    this = __xbart_cpp.new_XBARTcpp(num_trees,l,num_sweeps,n_min,num_cutpoints,
            alpha,beta, tau,burnin, mtry, max_depth_num,draw_sigma,kap,s,
            verbose,m_update_sigma, draw_mu,parallel,seed)  
    # init
    try:
        self.this.append(this)
    except __builtin__.Exception:
        self.this = this  

%}
%pythoncode %{
def predict_2d(self,x):
	x_pred = self.__predict_2d(x,x.shape[0]*x.shape[1])
	return x_pred.reshape(x.shape)
%}
%pythoncode %{
def fit_predict_2d(self,x,y,x_test):
    x_pred = self.__fit_predict(x,y,x_test,y.shape[0])
    yhats_test = self.get_yhats_test(self.get_N_sweeps()*x_test.shape[0]).reshape((x_test.shape[0],self.get_N_sweeps()),order='C')
    return yhats_test
%}
%pythoncode %{
def fit_predict(self,x,y,x_test,p_cat=0):
    x_pred = self.__fit_predict_all(x,y,x_test,y.shape[0],p_cat)
    yhats_test = self.get_yhats_test(self.get_N_sweeps()*x_test.shape[0]).reshape((x_test.shape[0],self.get_N_sweeps()),order='C')

    #self.importance = self.get_importance(x.shape[1])
    return yhats_test
%}
%pythoncode %{
def predict(self,x_test):
    x_pred = self.__predict_all(x_test)
    yhats_test = self.get_yhats_test(self.get_N_sweeps()*x_test.shape[0])
    yhats_test = yhats_test.reshape((x_test.shape[0],self.get_N_sweeps()),order='C')
    return yhats_test
%}
%pythoncode %{
def fit(self,x,y,p_cat=0):
    return self.__fit_all(x,y,p_cat)
%}

};

%include "xbart.h"




