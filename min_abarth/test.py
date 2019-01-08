import abarth
import numpy as np
from collections import OrderedDict
import unittest
import sys
from sklearn.linear_model import LinearRegression



class AbarthTesting1(unittest.TestCase):


	# def setUp(self):
	# 	self.model = abarth.Abarth()

	def setUp(self):
		self.params = OrderedDict([('M',20),('L',1),("N_sweeps",40)
							,("Nmin",1),("Ncutpoints",5)
							,("alpha",0.95),("beta",1.25 ),("tau",.8),("burnin",5),("mtry",2),("max_depth_num",5),
							("draw_sigma",False),("kap",16),("s",4),("verbose",False),("m_update_sigma",False),
							("draw_mu",False),("parallel",False)])
		#self.params["M"] = 50
		self.model = abarth.Abarth(self.params)
		n = 100
		self.x = np.random.rand(n)
		#print self.x.shape


	def test_constructor_m(self):
		self.failUnless(self.model.get_M()==self.params["M"])	

	def test_void_fit_1d(self):
		fit_x = self.model.fit(self.x)
		self.failUnless(fit_x==self.x[-1])
		

	def test_void_fit_2d(self,n=10,d=2):
		X = np.arange(n*d).reshape(n,d)
		#print "Fail unless 2d  X_fit's last elements are equal to X"
		self.failUnless(self.model.fit_x(X) == X[n-1,d-1])

	def test_void_sort_2d(self,n=10,d=2):
		X = np.random.rand(n*d).reshape(n,d)
		#print "Fail unless 2d  X_fit's last elements are equal to X"
		sort_np = np.argsort(X[:,d-1])
		sort_x = self.model.sort_x(X,n).astype(int)
		self.failUnless(all(sort_x == sort_np))
		#self.failUnless(self.model.sort_x(X) == np.argsort(X[:,d-1])[n-1])		

	def test_predict(self):
		n = 100
		x= np.random.rand(n)
		x_pred = self.model.predict(x,n)
		self.failUnless(np.array_equal(x_pred,x))

	#@unittest.skip("demonstrating skipping")
	def test_predict_2(self):
		n = 100
		d = 10
		x= np.random.rand(n,d)
		y_pred = self.model.predict_2d(x)
		self.failUnless(isinstance(x_pred, np.ndarray))
		self.failUnless(np.array_equal(x_pred,x))

	#@unittest.skip("demonstrating skipping")
	
	def test_fit_predict(self):
		n = 1000
		d = 10
		x= np.random.rand(n,d)
		y = np.random.rand(n)

		n_test = 100
		d_test = 10
		x_test= np.random.rand(n_test,d_test)

		y_pred = self.model.fit_predict(x,y,x_test,n)
		self.failUnless(isinstance(y_pred, np.ndarray))
		self.failUnless(np.array_equal(y_pred,y))
	def test_fit_predict_2d(self):
		n = 10000
		d = 10
		x= np.random.rand(n,d)
		y = np.sin(x[:,0]**2)+x[:,1] + np.random.rand(n)

		n_test = 1000
		x_test= np.random.rand(n_test,d)
		y_test = np.sin(x_test[:,0]**2)+x_test[:,1] + np.random.rand(n_test)
		y_pred = self.model.fit_predict_2d(x,y,x_test)
		y_hat = y_pred[:,self.model.get_burnin:].mean(axis=1)
		print(np.sqrt(np.mean((y_hat-y_test)**2)))
		print(y_hat[0:10])
		print(y_test[0:10])
		print(y_pred[0,:])
		print("hello!")
		#y_pred = self.model.fit_predict_2d(x,y,x_test)
		
		# self.failUnless(np.array_equal(y_pred,y))

class AbarthExceptionTesting(unittest.TestCase):

	def test_int_as_bad_float(self):
		with self.assertRaises(TypeError):
			params = {"M":5.1}
			abarth.Abarth(params)

	def test_int_as_bad_string(self):
		with self.assertRaises(TypeError):
			params = {"M":"5.1"}
			abarth.Abarth(params)		

	def test_int_as_good_float(self):
		params = {"M":5.0}
		abarth.Abarth(params)

	def test_float_good_int(self):	
		params = {"alpha":5}
		abarth.Abarth(params)

	def test_float_bad_string(self):
		with self.assertRaises(TypeError):	
			params = {"alpha":"5"}
			abarth.Abarth(params)

	def test_bool_with_bad_int(self):
		with self.assertRaises(TypeError):
			params = {"m_update_sigma":2}
			abarth.Abarth(params)

	def test_bool_with_bad_float(self):
		with self.assertRaises(TypeError):
			params = {"m_update_sigma":2.2}
			abarth.Abarth(params)
	
	def test_bool_with_bad_string(self):
		with self.assertRaises(TypeError):
			params = {"m_update_sigma":"2"}
			abarth.Abarth(params)

	def test_bool_with_good_int(self):
		params = {"m_update_sigma":0}
		abarth.Abarth(params)

	def test_bool_with_good_float(self):
		params = {"m_update_sigma":0.0}
		abarth.Abarth(params)

if __name__ == "__main__":
	test_classes_to_run = [AbarthTesting1, AbarthExceptionTesting]

	loader = unittest.TestLoader()

	suites_list = []
	for test_class in test_classes_to_run:
		suite = loader.loadTestsFromTestCase(test_class)
		suites_list.append(suite)

	big_suite = unittest.TestSuite(suites_list)

	runner = unittest.TextTestRunner(verbosity=2)
	results = runner.run(big_suite)
	sys.exit(len(results.errors) + len(results.failures))

	# suite = unittest.TestLoader().loadTestsFromTestCase(AbarthTesting1)
	# #print "Testing Functions of Abarth Module: \n"
	# result = unittest.TextTestRunner(verbosity=2).run(suite)
	
