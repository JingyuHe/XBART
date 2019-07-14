
import numpy as np
from collections import OrderedDict
import unittest
import sys
import time

import xbart

def rmse(y1,y2):
	return np.sqrt(np.mean((y1-y2)**2))

class XBARTTesting1(unittest.TestCase):


	def setUp(self):
		self.params = {"num_trees":5,"num_sweeps":2,"num_cutpoints":10,
						"max_depth_num":5,"burnin":1}
		self.model = xbart.XBART(**self.params)
		self.model_2 = xbart.XBART(**self.params)
		n = 100
		self.x = np.random.rand(n)
			

	def test_fit_predict_discrete_2d(self):
		n = 1000
		d = 7
		prob = np.random.uniform(0.2,0.8,d)

		x = np.empty([n,d])
		x[:,0] = np.random.normal(25,10,n)
		for h in range(1,d):
			x[:,h] = np.random.binomial(1,prob[h],n)

  		
		n_test = 1000	
		x_test = np.empty((n_test,d))
		x_test[:,0] = np.random.normal(25,10,n_test)
		for h in range(1,d):
			x_test[:,h] = np.random.binomial(1,prob[h],n_test)


		def discrete_function(x):
			level =  15 - 20*(x[:,0]-25)**2/1500
			level = level + 15*np.logical_and(x[:,2], x[:,4]) -10*np.logical_or(x[:,5] , x[:,6])
			level = level*(2*x[:,3]-1)
			return level

		ftrue = discrete_function(x) 	
		sigma = 3*np.std(ftrue)	

		y = ftrue + sigma*np.random.rand(n)
		y_test = discrete_function(x_test)

		x_copy = x.copy()
		y_copy = y.copy()
		x_test_copy = x_test.copy()

		y_pred = self.model.fit_predict(x,y,x_test,d-1,return_mean=False)

		assert(np.array_equal(x_test_copy, x_test))
		y_hat_2 = self.model.predict(x_test)
		assert(np.array_equal(x_test_copy, x_test))

		self.model_2.fit(x,y,d-1)
		y_pred_3 = self.model_2.predict(x_test,return_mean=False)

		y_hat = y_pred[:,self.params["burnin"]:].mean(axis=1)
		y_hat_3 = y_pred_3[:,self.params["burnin"]:].mean(axis=1)


		print("RMSE XBART:"  + str(rmse(y_hat,y_test)) )
		print("RMSE XBART Pred:"  + str(rmse(y_hat_2,y_test)) )
		print("RMSE XBART Fit Pred Seperate:"  + str(rmse(y_hat_3,y_test)))
		print("RMSE XBART Pred v. Reg:"  + str(rmse(y_hat_2,y_hat)))
		print("RMSE XBART Fit and Pred:"  + str(rmse(y_hat_2,y_hat)))

		self.assertTrue(np.array_equal(y_copy,y))
		self.assertTrue(np.array_equal(x_copy,x))
		self.assertTrue(np.array_equal(x_test_copy,x_test))
		# self.failUnless(np.array_equal(y_pred,y))

	#@unittest.skip("demonstrating skipping")
	def test_to_json(self):
		n = 10000
		d = 10
		x= np.random.rand(n,d)
		y = np.sin(x[:,0]**2)+x[:,1] + np.random.rand(n)

		n_test = 1000
		x_test= np.random.rand(n_test,d)
		y_test = np.sin(x_test[:,0]**2)+x_test[:,1] + np.random.rand(n_test)
		y_pred = self.model.fit_predict(x,y,x_test,return_mean=False)
		y_hat = y_pred[:,self.params["burnin"]:].mean(axis=1)

		x_copy = x.copy()
		y_copy = y.copy()
		x_test_copy = x_test.copy()

		print("RMSE :"  + str(rmse(y_hat,y_test)))
		##print("unique values of prediction:"  +str(np.unique(y_pred)))

		self.assertTrue(np.array_equal(y_copy,y))
		self.assertTrue(np.array_equal(x_copy,x))
		self.assertTrue(np.array_equal(x_test_copy,x_test))

		js = self.model.to_json()
		self.model.to_json("model.xbart")
		self.model.from_json("model.xbart")
		y_pred_json = self.model.predict(x_test,return_mean=False)
		self.assertTrue(np.array_equal(y_pred_json,y_pred))

	def test_z_from_json(self):
		model = xbart.XBART()
		model.from_json("model.xbart")
		n_test = 1000; d = 10
		x_test= np.random.rand(n_test,d)
		y_pred_json = model.predict(x_test,return_mean=False)
		self.assertFalse(np.array_equal(y_pred_json,y_pred_json*0))

			
class XBARTExceptionTesting(unittest.TestCase):

	def test_int_as_bad_float(self):
		with self.assertRaises(TypeError):
			xbart.XBART(num_trees = 5.1)

	def test_int_as_bad_string(self):
		with self.assertRaises(TypeError):
			xbart.XBART(num_trees = 5.1)


	def test_int_as_good_float(self):
		xbart.XBART(num_trees = 5.0)

	def test_float_good_int(self):	
		xbart.XBART(alpha=5)

	def test_float_bad_string(self):
		with self.assertRaises(TypeError):	
			xbart.XBART(alpha="5")

	def test_bool_with_bad_int(self):
		with self.assertRaises(TypeError):
			params = {"parallel":2}
			xbart.XBART(parallel = 2)

	def test_bool_with_bad_float(self):
		with self.assertRaises(TypeError):
			xbart.XBART(parallel= 2.2)
	
	def test_bool_with_bad_string(self):
		with self.assertRaises(TypeError):
			xbart.XBART(parallel = "2")

	def test_bool_with_good_int(self):
		xbart.XBART(parallel = 0)

	def test_bool_with_good_float(self):
		xbart.XBART(parallel = 0.0)

if __name__ == "__main__":
	test_classes_to_run = [XBARTTesting1, XBARTExceptionTesting]

	loader = unittest.TestLoader()

	suites_list = []
	for test_class in test_classes_to_run:
		suite = loader.loadTestsFromTestCase(test_class)
		suites_list.append(suite)

	big_suite = unittest.TestSuite(suites_list)

	runner = unittest.TextTestRunner(verbosity=2)
	results = runner.run(big_suite)
	sys.exit(len(results.errors) + len(results.failures))

