import collections
from _xbart_cpp import XBARTcpp
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
 

class PandasXbart(object):
	def __init__(self,num_trees= 200, l = 1 ,num_sweeps = 40, n_min =1,
				num_cutpoints=100,alpha=0.95,beta = 1.25, tau = 0.3,
                burnin = 15, mtry = 1, max_depth_num = 250,
                draw_sigma= False,kap= 16,s = 4,verbose=False,
                m_update_sigma = True, draw_mu = True,
                parallel=False,seed=0):
		#assert isinstance(params, collections.Mapping), "params must be dictionary like"
		#self.params = params
		self.xbart = XBART(num_trees,l,num_sweeps,n_min,num_cutpoints,
			alpha,beta, tau,burnin, mtry, max_depth_num,draw_sigma,kap,s,
			verbose,m_update_sigma, draw_mu,parallel,seed)

	def __add_columns(self,x):
		if isinstance(x,DataFrame):
			self.columns = x.columns
		else:
			self.columns = range(x.shape[1])

	def __update_fit_x_y(self,x,fit_x,y,fit_y):
		if isinstance(x,DataFrame):
			fit_x = x.values
		if isinstance(y,Series):
			fit_y = y.values

	def __check_inputs(self,x,y=None):
		if not isinstance(x,(np.ndarray,DataFrame)):
			raise TypeError("x must be numpy array or pandas DataFrame")

		if (y is not None): 
			if not isinstance(y,(np.ndarray,Series)):
				raise TypeError(f"y must be numpy array or pandas Series, not type {type(y)}")


	def fit(self,x,y,p_cat=0):
		# Checks #
		self.__check_inputs(x,y)
		self.__add_columns(x)
		fit_x = x 
		fit_y = y
		self.__update_fit_x_y(x,y,fit_x,fit_y)

		self.xbart = self.xbart.fit(fit_x,fit_y,p_cat)
		return self

