import numpy as np
import pandas as pd
import statsmodels as sm
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sn
import random
import enum

from unit import Unit

class Population():
	"""
	
	Params:
		population_size (int) : size of the population
		population_variant_magnitude (float) : how much the population is mutating
		kill_system (KillSystem) : the type of kill system used
		input_shape (list) : shape of the input
		output_shape (list) : shape of the prediction

	TODOs
		Must:
			1. Refine the gen_weight_variant sensitivity
		Future:
			1. Custom ClockCycle for kill and reproduce
	"""
	def __init__(self, population_size, population_variant_magnitude, kill_system, input_shape, output_shape):
		self.population_size = population_size
		self.population_variant_magnitude = population_variant_magnitude
		self.kill_system = kill_system
		self.input_shape = input_shape
		self.output_shape = output_shape
		self.units = []

		for _ in range(population_size):
			unit = Unit(init_weight_variant_magnitude(), init_probabilities(), population_variant_magnitude)
			self.units.append(unit)

	def predict(self, data):
		"""
		Predict based on input

		Args:
		data (list{?}) : list of input data
		Return:
		(list{?}) : list of predicted values
		"""
	def kill(self):
		"""
		Kill a number of units
		"""
		return

	def produce(self):
		"""
		Reproduce until population is filled again
		"""
		return

	def train(self, label, data):
		"""
		Train the model

		Args:
		label (list{?}) : list of labels
		data (list{?}) : list of input data
		Return
			nill
		"""

	def init_weight_variant_magnitude(self):
		"""
		Generate random weight var based on population variant
		"""
		var = abs(np.random.normal(0, self.population_variant_magnitude))
		return var if var < 1 else 1

	def init_probabilities(self):
		"""
		Generate initial probabilities for each unit
		"""
		return 0
