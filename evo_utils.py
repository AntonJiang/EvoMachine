import numpy as np
import pandas as pd
import statsmodels as sm
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sn
import random
import enum

class Actiations():
	"""
	Pool of Activation functions

	TODOs:
		variant parameters for activation functions
	"""

	def binary_step(x, a=1):
	    if x<0:
	        return 0
	    else:
	        return a
	
	def linear_function(x, a=1, _coe=0):
    	return a*x + _coe

    def sigmoid_function(x, a=None):
	    z = (1/(1 + np.exp(-x)))
	    return z

	def tanh_function(x, a=None):
	    z = (2/(1 + np.exp(-2*x))) -1
	    return z

	def relu_function(x, a=None):
	    if x<0:
	        return 0
	    else:
	        return x

	def leaky_relu_function(x, a=0.01):
	    if x<0:
	        return a*x
	    else:
	        return x

    def elu_function(x, a):
	    if x<0:
	        return a*(np.exp(x)-1)
	    else:
	        return x

    def swish_function(x, a=None):
    	return x/(1-np.exp(-x))

    def softmax_function(x, a=None):
	    z = np.exp(x)
	    z_ = z/z.sum()
	    return z_
    
	def polynomial_2nd_function(x, a=None, b=None):
		return a*x*x + linear_function(x,b)

	def polynomial_3rd_function(x, a=None, b=None, c=None):
		return a*x*x*x + polynomial_2nd_function(x, b, c)

    single_activation = [binary_step, linear_function, sigmoid_function, tanh_function, \
    	relu_function, leaky_relu_function, elu_function, swish_function, \
    	softmax_function]

   	pool_activation = [lambda x : func(x.sum()) for func in single_activation]

   	single_function = [linear_function, sigmoid_function, leaky_relu_function, swish_function, \
   		softmax_function, polynomial_2nd_function, polynomial_3rd_function]

   	def random_get_functions(size):
		"""
		Return a random activation function
		
		Args:
			size (int) : the size of the functions
		Return 
			(list{Func}) : a list of random activation functions
		"""
		return np.random.choice(single_function, size=size)


	def random_get():
		"""
		Return a random activation function
		
		Args:
		Return 
			(list{Func}) : a list of random activation functions
		"""
		return np.random.choice(pool_activation)

class KillSystem():
	"""
	Util for killing list of Units
	
	Params:
		mercy (float) : likelihood to not kill
		min_survive (0<float<1) : percentage of units must survivein the units
	"""


	def __init__(min_survive):
		self.min_survive = min_survive

	def mark_death(self, true_ouput, input_val, units):
		"""
		Calculate each predicted value and mark units as not right
		
		Args:
			true_output (?) : the desired output
			input_val (?) : the input
			units (list{Unit}) : units ready for inspection
		Return:
			(list{int}) : a list of unique index to kill units
		"""
		size = len(units)
		max_del_size = (1 - self.min_survive)*size
		#TODO: change this:
		del_size = np.random.default_rng().normal(max_del_size/2, max_del_size/5)
		assert(max_del_size == size)
		outputs = [unit.predict(input_val) for unit in units]
		loss = [ mae(output, true_output) for output in outputs]
		loss, index = sorted(zip(loss, np.arange(size)), reverse=True)
		p = np.linalg.norm(np.linspace(del_size, 0, num=del_size))
		index = np.random.choice(index, replace=False, size=del_size, p=p)
		return index 
		
def mae(predictions, targets):
    differences = predictions - targets
    absolute_differences = np.absolute(differences)
    mean_absolute_differences = absolute_differences.mean()
    return mean_absolute_differences

# TODO:
# class Pickle(object):
# 	"""docstring for Pickle"""
# 	def __init__(self, arg):
# 		self.arg = arg
# 		