import numpy as np
import pandas as pd
import statsmodels as sm
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sn
import random
import enum

class KillSystem():.
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
		"""


class ClockCycle():
	"""docstring for ClockCycle"""
	def __init__(self, arg):
		self.arg = arg

class Pickle(object):
	"""docstring for Pickle"""
	def __init__(self, arg):
		self.arg = arg
		