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
	"""

	def binary_step(x, a=1):
	    if x<0:
	        return 0
	    else:
	        return a
	
	def linear_function(x, a=1):
    	return a*x   

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
    
    single_activation = [binary_step, linear_function, sigmoid_function, tanh_function, \
    	relu_function, leaky_relu_function, elu_function, swish_function, \
    	softmax_function]

   	pool = [lambda x : func(x.sum()) for func in single_activation]

	def random_get(size):
		"""
		Return a random activation function
		
		Args:
			size (int) : the number of functions to be returned

		Return 
			(list{Func}) : a list of random activation functions
		"""
		return np.random.choice(pool)

class KillSystem():
	"""
	Util for killing list of Units
	
	Params:
		mercy (float) : likelihood to not kill
		min_survive (0<float<1) : percentage of units must survivein the units
	"""


	def __init__(self, mercy, min_survive):
		self.mercy = mercy

	def mark_death(self, true_ouput, input, units, population_size):
		"""
		Calculate each predicted value and mark units as not right
		
		Args:
			true_output (?) : the desired output
			input (?) : the input
			units (list{Unit}) : units ready for inspection
			population_size (int) : size of the population
		Return:
			(list{int}) : a list of unique index to kill units

class Pickle(object):
	"""docstring for Pickle"""
	def __init__(self, arg):
		self.arg = arg
		