import numpy as np
import pandas as pd
import statsmodels as sm
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sn
import random
import enum

from hyper import LivingState



# class Weights():
#     """
#     A wrapper and utility class for weights of each neuron'

#     Param:
#     weights (dict{int:float}): map each connection's id to its weight
#     trends (dict{int:float}): map each connection's id to its trending multiplier
#     """

#     def __init__(self, weights, trends)
#         self.weights = weights
#         self.trends = trends

#     def check_weight_trend(self, death_mult):
#         """
#         Check if any connection is meant to die or reproduce

#         Args:
#         death_mult (float): A multiplier thresh for weight reproduce
        
#         Return:
#         (dict{int:LivingState}): map each connection's id to its living state 
#         """
#         return dict(map(lambda trend: LivingState.DIE if (trend < death_mult) else 
#             LivingState.NORM, self.trends.items()))

#     def gen_new_weights(old_weights, magnitude):
#         percent_change = magnitude/100
#         multiplier = ((np.random.default_rng().uniform(-1, 1, size=len(old_weights.weights))*percent_change) + 1)
#         new_weights = old_weights.weights*multiplier
#         new_trends = old_weights.trends*(new_weights/old_weights.weights)
#         return Weights(new_weights, new_trends)
#         # TODO: Find a way to initialize global mult variabels



class Connections():
	"""
	Each connection is from input neuron to target neuron specific

	Params:
		target_neuron (Neuron) : the target neuron
		weight (float) : the weight multiplier of the current connection
		function (Func) : the specific mathmatical function associated with
							the connection, single input, single output
		state (int) : 1 if the connection is working
	"""
	def __init__(self, target_neurons):
		self.target_neurons = target_neurons
		self.size = len(target_neurons)
        self.weights = gen_new_weights(self.size)
        self.states = np.ones(len(target_neurons))
        self.functions = gen_new_functions()

    def __init__(self, target_neurons, weights, functions, states):
        self.target_neurons = target_neurons
        self.weights = weights
        self.states = states
        self.functions = functions

    def add(self, neurons):
    	"""
    	Add more connections based on neurons

    	Args:
    		neurons (Neurons) : new target neuron for the connection to be added
    	"""

    def gen_new_weights(size):
    	UNDER PROGRESS

   	def gen_new_functions(size):
   		UNDER PROGRESS

    def check_weight_trend(self, death_mult):
        # Return 0 for normal 1 for death
        # Upper level will kill connections accordingly when init the neuron
        return self.weight.check_weight_trend(death_mult)

class InputNeuron(Neuron):
	"""
	The First Neuron in a Unit, a wrapper for the input neuron

	Params:
		neuron (Neuron) : the basic neuron
	"""
	def __init__(self, input_shape, **kwargs):
		super.__init__(**kwargs)
		self.input_shape = input_shape

class OutputNeuron(Neuron):
	"""
	The Last Neuron in a Unit, a wrapper for the output neuron

	Params:
		neuron (Neuron) : the basic neuron
	"""
	def __init__(self, output_shape, **kwargs):
		super.__init__(**kwargs)
		self.output_shape = output_shape


class Neuron():
	"""
	A basic neuron

	Params:
		id (int) : the unique id for the neuron
		distance (0<float<1): the distance of the neuron to the input neuron
		activation (Function) : the function to calcuated the activation function
		connections (dict{Neurons:int}) : dict of neurons to weights that it connects to
		
		pro
		
	"""
    def __init__(self, distance, activation, connections, meta_variant, input_count=0):
        self.distance = distance
        self.activation = activation
        self.connections = connections
        self.meta_variant = meta_variant
        self.input_count = input_count
        self.temp_input = []

    def update_input_count(self, delta):
    	self.input_count += delta

    def update_connections(self, neurons):
    	"""
    	Extend the current connection with target neurons
    	Generate properties similar to existing connections
    	"""
    	assert len(connections) >=1
	    for target_neuron in neurons:
            target_neuron.update_input_count(1);
    	
  		self.connections.add(neurons)

    def gen_variant_value(old_value, magnitude):
        percent_change = magnitude/100
        multiplier = ((np.random.default_rng().uniform(-1, 1, size=1)*percent_change) + 1)
        return old_value*multiplier

    def produce(self):
        ### Return list of child producd
        UNDER PROGRESS
        return

    def output(self, single_input):
        # Input ordered by
        self.temp_input.append(single_input)
        if (len(self.temp_input) != self.input_count):
        	return

       	functioned_output = [func(self.activation(temp_input)) for func in self.connections.functions]
       	outputs = self.connections.states.multiply(functioned_output).multiply(self.connections.weights())
       	
       	for neuron, output in zip(self.connections.target_neurons, outputs):
       		neuron.output(output)
       	#Clear the calculation memory
        temp_input = []