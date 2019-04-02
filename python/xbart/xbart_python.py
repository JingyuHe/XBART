if __name__ == "__main__" and __package__ is None:
    __package__ = "xbart"
from .xbart_cpp_ import XBARTcpp
import collections
import numpy as np

## Optional Import Pandas ## 
try:
	from pandas import DataFrame
	from pandas import Series
except ImportError:
	class DataFrame(object):
		pass
	class Series(object):
		pass
 

class XBART(object):
	def __init__(self,num_trees= 200, l = 1 ,num_sweeps = 40, n_min =1,
				num_cutpoints=100,alpha=0.95, beta = 1.25, tau = 0.3,
                burnin = 15, mtry = 1, max_depth_num = 250,
                draw_sigma= False,kap= 16,s = 4,verbose=False,
                m_update_sigma = True, draw_mu = True,
                parallel=False,seed=0):
		self.__convert_params_check_types(locals())
		self.xbart_cpp = XBARTcpp(num_trees,l,num_sweeps,n_min,num_cutpoints,
			alpha,beta, tau,burnin, mtry, max_depth_num,draw_sigma,kap,s,
			verbose,m_update_sigma, draw_mu,parallel,seed)

	def __add_columns(self,x):
		if isinstance(x,DataFrame):
			self.columns = x.columns
		else:
			self.columns = range(x.shape[1])

	def __update_fit_x_y(self,x,fit_x,y=None,fit_y=None):
		if isinstance(x,DataFrame):
			fit_x = x.values
		if y is not None:
			if isinstance(y,Series):
				fit_y = y.values

	def __check_inputs(self,x,y=None):
		if not isinstance(x,(np.ndarray,DataFrame)):
			raise TypeError(f"x must be numpy array or pandas DataFrame, not type {type(x)}")

		if y is not None: 
			if not isinstance(y,(np.ndarray,Series)):
				raise TypeError(f"y must be numpy array or pandas Series, not type {type(y)}")

	def __convert_params_check_types(self,params):
		### This function converts params to list and 
		### It handles the types of params and raises exceptions if needed
		### It puts in default values for empty param values 
		import warnings
		from collections import OrderedDict
		DEFAULT_PARAMS = OrderedDict([('num_trees',200),('l',1),("num_sweeps",40)
                        ,("n_min",1),("num_cutpoints",100) # CHANGE
                        ,("alpha",0.95),("beta",1.25 ),("tau",0.3),# CHANGE
                        ("burnin",15),("mtry",1),("max_depth_num",250), # CHANGE
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


	def fit(self,x,y,p_cat=0):
		# Checks #
		self.__check_inputs(x,y)
		self.__add_columns(x)
		fit_x = x 
		fit_y = y
		self.__update_fit_x_y(x,fit_x,y,fit_y)

		self.xbart_cpp = self.xbart_cpp._fit(fit_x,fit_y,p_cat)
		return self

	def predict(self,x_test):
		# Check inputs # 
		self.__check_inputs(x_test)
		pred_x = x_test 
		self.__update_fit_x_y(x_test,pred_x)

		# Run Predict
		x_pred = self.xbart_cpp._predict(pred_x)
		# Convert to numpy
		yhats_test = self.xbart_cpp.get_yhats_test(self.xbart_cpp.get_N_sweeps()*pred_x.shape[0])
		# Convert from colum major 
		yhats_test = yhats_test.reshape((pred_x.shape[0],self.xbart_cpp.get_N_sweeps()),order='C')
		return yhats_test

	def fit_predict(self,x,y,x_test,p_cat=0):	
		self.fit(x,y,p_cat)
		return self.predict(x_test)


