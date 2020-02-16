import numpy as np
import pandas as pd
import statsmodels as sm
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sn
import random
import enum


class LivingState(enum.Enum):
    DIE = -1
    NORM = 0

class Weights():
    """
    A wrapper and utility class for weights of each neuron'

    Param:
    weights (dict{int:float}): map each connection's id to its weight
    trends (dict{int:float}): map each connection's id to its trending multiplier
    """

    def __init__(self, weights, trends)
        self.weights = weights
        self.trends = trends

    def check_weight_trend(self, death_mult):
        """
        Check if any connection is meant to die or reproduce

        Args:
        death_mult (float): A multiplier thresh for weight reproduce
        
        Return:
        (dict{int:LivingState}): map each connection's id to its living state 
        """
        return dict(map(lambda trend: LivingState.DIE if (trend < death_mult) else 
            LivingState.NORM, self.trends.items()))

    def gen_new_weights(old_weights, magnitude):
        percent_change = magnitude/100
        multiplier = ((np.random.default_rng().uniform(-1, 1, size=len(old_weights.weights))*percent_change) + 1)
        new_weights = old_weights.weights*multiplier
        new_trends = old_weights.trends*(new_weights/old_weights.weights)
        return Weights(new_weights, new_trends)
        # TODO: Find a way to initialize global mult variabels



class Connections():
    def __init__(self, weights, states, functions):
        self.weights = weights
        self.states = states
        self.functions = functions
        self.size = len(states)

    def gen_new_connection(self, magnitude, drop_rate):
        new_states = np.random.default_rng().uniform(size=size) < drop_rate
        new_functions = Connections.gen_new_functions(self.functions, magnitude)
        new_weights = Weights.gen_new_weights(self.weights, magnitude)
        return Connections(new_weights, new_states, new_functions)

    def gen_new_functions(old_functions, magnitude):
        # TODO: New Functions varies from old_functions by magnitude
        # NEED: A global pool of functions:

    def get_weights(self):
        return self.weights.weights
    def get_states(self):
        return self.states
    def get_functions(self):
        return self.functions
    def get_connections(self):
        return np.stack([self.weights,weights, self.states, self.functions], axis=1)
    def check_weight_trend(self, death_mult):
        # Return 0 for normal 1 for death
        # Upper level will kill connections accordingly when init the neuron
        return self.weights.check_weight_trend(death_mult)

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
    def __init__(self, id, activation, connections, natural_death_thresh, random_death_proba,
                 natural_produce_thresh, random_produce_proba):
        self.id = id
        self.activation = activation
        self.connections = connections
        self.natural_death_thresh = natural_death_thresh
        self.random_death_prob = random_death_prob

        self.additional_produce_std = additional_produce_std

        self.produce_variation_mag = produce_variation_mag
        self.produce_thresh = produce_thresh

        self.connection_drop_rate = connection_drop_rate
        self.calcuated_output = None

    def gen_new_activation(old_activation, magnitude):
        #Gen new activation function based on global activation function tool
        return 0
    def gen_variant_value(old_value, magnitude):
        percent_change = magnitude/100
        multiplier = ((np.random.default_rng().uniform(-1, 1, size=1)*percent_change) + 1)
        return old_value*multiplier

    def produce(self):
        ### Return list of child producd
        produce_num = 1 + round(abs(np.random.default_rng().normal(0, additional_produce_std))/2) # Diveid 2 account for inverted mean
        children = []
        magnitude = self.produce_variation_mag
        for _ in range(produce_num):
            new_connection = self.connections.gen_new_connection(magnitude, self.connection_drop_rate)
            new_activation = gen_new_activation(self.activation, magnitude)
            child = Neuron(activation=new_activation, connections=new_connection, natural_death_thresh=gen_variant_value(self.natural_death_thresh, magnitude),
                random_death_prob=gen_variant_value(self.random_death_prob, magnitude), additional_produce_std=gen_variant_value(self.additional_produce_std, magnitude),
                produce_thresh=gen_variant_value(self.produce_thresh, magnitude), connection_drop_rate=gen_variant_value(self.connection_drop_rate, magnitude))
            children.append(child)
        return children

    def output(self, inputs):
        # Input ordered by  
        values = [] 
        for func, value in zip(self.connections.get_functions):
            values.append(func(value))
        return self.activation(np.array(values).multiply(self.connections.get_weights() * self.connections.get_states()))