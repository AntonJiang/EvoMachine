import numpy as np

from evo_utils import KillSystem, ClockCycle, Pickle
from population import Population

class EvoMachine(object):
	""" docstring for EvoMachine
	
	Params:
	populations (list{list{float}}): Variables for each population, population id to parameters
		population_size (int) : number of units in the population
		population_variant_magnitude (float) : the magnitude of mutation within the population
		kill_system (KillSystem) : the specific kill system used for the population
	input_shape (list{int})
	ouput_shape (list{int})

	TODOs:
		0. Shuffle data and add batch training
		1. Allow custom population initiation variables
		2. Different voting mechanism
	"""
	def __init__(self, populations, input_shape, ouput_shape):
		self.populations = []
		self.input_shape = input_shape
		self.ouput_shape = ouput_shape
		
		for population in populations:
			pop = Population(populations[0], populations[1], populations[2], input_shape, ouput_shape)
			self.populations.append(pop)

	def train(self, label, data):
		"""
		Train the model

		Args:
		label (list{?}) : list of labels
		data (list{?}) : list of input data
		Return
			nill
		"""
		assert len(label) == len(data)
		for population in self.populations:
			population.train(label, data)
	
	def predict(self, data):
		"""
		Predict based on input

		Args:
		data (list{?}) : list of input data
		Return:
		(list{?}) : list of predicted values
		"""
		results = [population.predict(data) for population in populations]
		return self.vote(results)

	def vote(self, results):
		"""
		Vote final result among populations

		Args:
		results (list{?}) : the results returned by different populations
		Return:
		({?}) : the average voted result
		"""
		return np.mean(results, axis=0)