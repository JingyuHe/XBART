from __future__ import absolute_import

from .xbart_cpp_ import XBARTcpp
import collections
import numpy as np
import json

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
	'''
	Python extension for Accelerated Bayesian Additive Regression Trees
    Parameters
    ----------
	num_trees : int
        Number of trees in each iteration.
	num_sweeps : int
        Number of sweeps (MCMC draws).
	n_min: int
		Minimum number of samples in each final node.
	num_cutpoints: int
		For continuous variable, number of adaptive cutpoint candidates 
		considered in each split .
	alpha: double
		Tree prior hyperparameter : alpha * (1 + depth) ^ beta.
	beta: double
		Tree prior hyperparameter : alpha * (1 + depth) ^ beta.
	tau: double / "auto"
		Prior for leaf mean variance : mu_j ~ N(0,tau) 
	burnin: int
		Number of sweeps used to burn in - not used for prediction.
	max_depth_num: int
		Represents the maximum size of each tree - size is usually determined via tree prior.
		Use this only when wanting to determnistically cap the size of each tree.
	mtry: int / "auto"
		Number of variables considered at each split - like random forest.
	kap: double
		Prior for sigma :  sigma^2 | residaul ~  1/ sqrt(G) 
		where G ~ gamma( (num_samples + kap) / 2, 2/(sum(residual^2) + s) )
	s: double
		Prior for sigma :  sigma^2 | residaul ~  1/ sqrt(G) 
		where G ~ gamma( (num_samples + kap) / 2, 2/(sum(residual^2) + s) )
	verbose: bool
		Print the progress 
	parallel: bool
		Do computation in parallel
	seed: int
		Random seed, should be a positive integer
	model: str
		"Normal": Regression problems 
				: Classification problems (encode Y \in{ -1,1})
		"Probit": Classification problems (encode Y \in{ -1,1})
		"CLT" : (beta) Classification problems (encode Y \in{ -1,1})
	no_split_penality: double
		Weight of no-split option. The default value in the normal model is log(num_cutpoints). 
		Values should be considered in log scale.
	sample_weights_flag: bool (True)
		To sample weights according to Dirchlet distribution
	
	'''
	def __init__(self,num_trees: int = 100, num_sweeps: int = 40, n_min: int = 1,
				num_cutpoints: int = 100,alpha: float = 0.95, beta: float = 1.25, tau = "auto",
                burnin: int = 15, mtry = "auto", max_depth_num: int = 250,
                kap: float = 16.0,s: float = 4.0,verbose: bool = False,
                draw_mu: bool = True,parallel: bool = False,seed: int = 0,model: str = "Normal",
				no_split_penality = "auto",sample_weights_flag: bool = True):

		assert num_sweeps > burnin, "num_sweep must be greater than burnin"

		MODEL_MAPPINGS = {"Normal":0,"CLT":1,"Probit":2}
		if model in MODEL_MAPPINGS:
			model_num = MODEL_MAPPINGS[model]
		else:
			raise ValueError("model must be either Normal,CLT, or Probit")

		self.model = model
		self.params = dict(num_trees = num_trees,
			num_sweeps = num_sweeps,n_min = n_min,num_cutpoints = num_cutpoints,
			alpha = alpha,beta = beta, tau = tau,burnin = burnin, mtry=mtry, 
			max_depth_num=max_depth_num,kap=kap,s=s,
			verbose=verbose,draw_mu=draw_mu,
			parallel=parallel,seed=seed,model_num=model_num,no_split_penality =no_split_penality,
			sample_weights_flag=sample_weights_flag)
		args = self.__convert_params_check_types(**self.params)
		self._xbart_cpp = None

		# Additional Members
		self.importance = None
		self.sigma_draws = None
		self.is_fit = False

	def __repr__(self):
		items = ("%s = %r" % (k, v) for k, v in self.params.items())
		return str(self.__class__.__name__)+  '(' + str((', '.join(items))) + ")"

	def __add_columns(self,x):
		'''
		Keep columns internally
		'''
		if isinstance(x,DataFrame):
			self.columns = x.columns
		else:
			self.columns = range(x.shape[1])
		self.num_columns = len(self.columns)

	def __update_fit_x_y(self,x,fit_x,y=None,fit_y=None):
		'''
		Convert DataFrame to numpy
		'''
		if isinstance(x,DataFrame):
			fit_x = x.values
		if y is not None:
			if isinstance(y,Series):
				fit_y = y.values

	def __check_input_type(self,x,y=None):
		'''
		Dimension check
		'''
		if not isinstance(x,(np.ndarray,DataFrame)):
			raise TypeError("x must be numpy array or pandas DataFrame")

		if y is not None: 
			if not isinstance(y,(np.ndarray,Series)):
				raise TypeError("y must be numpy array or pandas Series")

	def __check_test_shape(self,x):
		assert x.shape[1] == self.num_columns, "Mismatch on number of columns"

	def __check_params(self,p_cat):
		assert p_cat <= self.num_columns, "p_cat must be <= number of columns"
		assert self.params["mtry"] <= self.num_columns, "mtry must be <= number of columns"

	def __update_mtry_tau_penality(self,x):
		'''
		Handle mtry, tau, and no_split_penality defaults
		'''
		if self.params["mtry"] == "auto":
			self.params["mtry"] = self.num_columns 
		if self.params["tau"]  == "auto":
			self.params["tau"] = 1/self.params["num_trees"]
		
		if self.params["no_split_penality"] == "auto":
			from math import log
			if self.params["model_num"] == 0:
				self.params["no_split_penality"] = log(self.params["num_cutpoints"])
			else:
				self.params["no_split_penality"] = 0.0
		
				
	def __convert_params_check_types(self,**params):
		'''
		This function converts params to list and handles type conversions
		If a wrong type is provided function raises exceptions 
		''' 
		import warnings
		from collections import OrderedDict
		DEFAULT_PARAMS = OrderedDict([('num_trees',100),("num_sweeps",40)
                        ,("n_min",1),("num_cutpoints",100) # CHANGE
                        ,("alpha",0.95),("beta",1.25 ),("tau",0.3),# CHANGE
                        ("burnin",15),("mtry",0),("max_depth_num",250) # CHANGE
                        ,("kap",16.0),("s",4.0),("verbose",False),
                        ("draw_mu",True),
                        ("parallel",False),("seed",0),("model_num",0),("sample_weights_flag",True)])
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
		'''
		Fit XBART model
        Parameters
        ----------
		x : DataFrame or numpy array
            Feature matrix (predictors)
        y : array_like
            Target (response)
		p_cat: int
			Number of features to treat as categorical for cutpoint options. More efficient.
			To use this feature set place the categorical features as the last p_cat columns of x 
		'''

		# Check inputs #
		self.__check_input_type(x,y)
		self.__add_columns(x)
		fit_x = x 
		fit_y = y
		
		# Update Values #
		self.__update_fit_x_y(x,fit_x,y,fit_y)
		self.__update_mtry_tau_penality(fit_x)
		self.__check_params(p_cat)

		# Create xbart_cpp object #
		if self._xbart_cpp is None:
			args = list(self.params.values())
			self._xbart_cpp = XBARTcpp(*args) # Makes C++ object

		# fit #
		self._xbart_cpp._fit(fit_x,fit_y,p_cat)

		# Additionaly Members
		self.importance = self._xbart_cpp._get_importance(fit_x.shape[1])
		self.importance = dict(zip(self.columns,self.importance.astype(int)))
		

		if self.model == "Normal":
			self.sigma_draws = self._xbart_cpp.get_sigma_draw(self.params["num_sweeps"]*self.params["num_trees"])
			# Convert from colum major 
			self.sigma_draws = self.sigma_draws.reshape((self.params["num_sweeps"],self.params["num_trees"]),order='F')
		
		self.is_fit = True
		return self

	def predict(self,x_test,return_mean = True):
		'''
		Predict XBART model
        Parameters
        ----------
		x_test : DataFrame or numpy array
            Feature matrix (predictors)
		return_mean: bool
			If true, will return mean prediction, else will return (n X num_sweeps) "posterior" estimate
	
		Returns
        -------
        prediction : numpy array
		'''

		assert self.is_fit, "Must run fit before running predict"

		# Check inputs # 
		self.__check_input_type(x_test)
		pred_x = x_test 
		self.__check_test_shape(pred_x)
		self.__update_fit_x_y(x_test,pred_x)

		# Run Predict
		x_pred = self._xbart_cpp._predict(pred_x)
		# Convert to numpy
		yhats_test = self._xbart_cpp.get_yhats_test(self.params["num_sweeps"]*pred_x.shape[0])
		# Convert from colum major 
		self.yhats_test = yhats_test.reshape((pred_x.shape[0],self.params["num_sweeps"]),order='C')
		# Compute mean
		self.yhats_mean =  self.yhats_test[:,self.params["burnin"]:].mean(axis=1)

		if return_mean:
			return self.yhats_mean
		else:
			return self.yhats_test

	def fit_predict(self,x,y,x_test,p_cat=0,return_mean=True):	
		'''
		Fit and predict XBART model
        Parameters
        ----------
		x : DataFrame or numpy array
            Feature matrix (predictors)
        y : array_like
            Target (response)
		x_test : DataFrame or numpy array
            Feature matrix (predictors)
		p_cat: int
			Number of features to treat as categorical for cutpoint options. More efficient.
			To use this feature set place the categorical features as the last p_cat columns of x 
		return_mean: bool
			If true, will return mean prediction, else will return (n X num_sweeps) "posterior" estimate
			
		Returns
        -------
        prediction : numpy array
		'''
		self.fit(x,y,p_cat)
		return self.predict(x_test,return_mean)

	def to_json(self,file=None):
		'''
		Serielize XBART model
		Parameters
        ----------
		file: str
			Output path to file. If none, returns string.
		'''
		json_str = self._xbart_cpp._to_json()
		j = json.loads(json_str)
		j["params"] = self.params
		j["num_columns"] = self.num_columns

		if file is not None:
			with open(file, "w") as text_file:
				json.dump(j,text_file)
		else:
			return json_str

	def from_json(self,json_path):
		'''
		Converts serialized file into XBART object
		Parameters
        ----------
		json_path: str
			Path to file.
		'''		
		with open(json_path) as f:
			j = json.load(f)

		self._xbart_cpp = XBARTcpp(json.dumps(j))
		self.is_fit = True
		self.num_columns = j["num_columns"]
		self.params = j["params"]
		return self

