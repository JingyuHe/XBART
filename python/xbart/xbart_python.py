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
	def __init__(self,num_trees: int = 100, num_sweeps: int = 40, n_min: int = 1,
				num_cutpoints: int = 100,alpha: float = 0.95, beta: float = 1.25, tau = "auto",
                burnin: int = 15, mtry = "auto", max_depth_num: int = 250,
                kap: float = 16.0,s: float = 4.0,verbose: bool = False,
                draw_mu: bool = True,parallel: bool = False,seed: int = 0):

		assert num_sweeps > burnin, "num_sweep must be greater than burnin"
		self.params = dict(num_trees = num_trees,
			num_sweeps = num_sweeps,n_min = n_min,num_cutpoints = num_cutpoints,
			alpha = alpha,beta = beta, tau = tau,burnin = burnin, mtry=mtry, 
			max_depth_num=max_depth_num,kap=kap,s=s,
			verbose=verbose,draw_mu=draw_mu,
			parallel=parallel,seed=seed)
		#self.__check_params(self.params)
		args = self.__convert_params_check_types(**self.params)
		#self.xbart_cpp = XBARTcpp(*args)
		self.xbart_cpp = None

	def __repr__(self):
		items = ("%s = %r" % (k, v) for k, v in self.params.items())
		return str(self.__class__.__name__)+  '(' + str((', '.join(items))) + ")"

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
			raise TypeError("x must be numpy array or pandas DataFrame}")

		if y is not None: 
			if not isinstance(y,(np.ndarray,Series)):
				raise TypeError("y must be numpy array or pandas Series}")
		
	def __check_params(self,params):
		import warnings
		from collections import OrderedDict
		DEFAULT_TYPES = dict([('num_trees',int),("num_sweeps",int)
                        ,("n_min",int),("num_cutpoints",int) # CHANGE
                        ,("alpha",float),("beta",float ),("tau",float),# CHANGE
                        ("burnin",int),("mtry",int),("max_depth_num",int) # CHANGE
                        ,("kap",float),("s",float),("verbose",bool),
                        ("draw_mu",bool),
                        ("parallel",bool),("seed",int)])
		for param,typ in DEFAULT_TYPES.items():
			if not isinstance(params[param],typ):
				try:
					params[param] = typ(params[param])
				except:
					raise TypeError(param +" should be of type "  + str(typ))



	def __update_mtry_tau(self,x):
		if self.params["mtry"] == "auto":
			p = x.shape[1]
			if p < 25:
				self.params["mtry"] = p
			else:
				self.params["mtry"] = int((p)**0.5)
		if self.params["tau"]  == "auto":
			self.params["tau"] = 1/self.params["num_trees"]
		
				
	def __convert_params_check_types(self,**params):
		### This function converts params to list and 
		### It handles the types of params and raises exceptions if needed
		### It puts in default values for empty param values 
		import warnings
		from collections import OrderedDict
		DEFAULT_PARAMS = OrderedDict([('num_trees',100),("num_sweeps",40)
                        ,("n_min",1),("num_cutpoints",100) # CHANGE
                        ,("alpha",0.95),("beta",1.25 ),("tau",0.3),# CHANGE
                        ("burnin",15),("mtry",0),("max_depth_num",250) # CHANGE
                        ,("kap",16.0),("s",4.0),("verbose",False),
                        ("draw_mu",True),
                        ("parallel",False),("seed",0)])
		new_params = DEFAULT_PARAMS.copy()

		#list_params = []
		for key,value in DEFAULT_PARAMS.items():
			true_type = type(value) # Get type
			new_value = params.get(key,value) #
			if not isinstance(new_value,type(value)):  
				if (key in ["mtry","tau"]) and new_value == "auto":
					continue
				elif true_type == int:
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
			#list_params.append(new_value)         
			self.params[key] = new_value    
		#return list_params    


	def fit(self,x,y,p_cat=0):
		# Check inputs #
		self.__check_inputs(x,y)
		self.__add_columns(x)
		fit_x = x 
		fit_y = y

		# Update Values #
		self.__update_fit_x_y(x,fit_x,y,fit_y)
		self.__update_mtry_tau(fit_x)

		# Create xbart_cpp object #
		if self.xbart_cpp is None:
			args = list(self.params.values())
			self.xbart_cpp = XBARTcpp(*args)

		# fit #
		self.xbart_cpp._fit(fit_x,fit_y,p_cat)
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


