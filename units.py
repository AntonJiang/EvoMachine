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
    def __init__(self, id, activation, connections, natural_death_thresh, random_death_prob,
                 additional_produce_std, produce_variation_mag, produce_thresh, connection_drop_rate):
        self.id = id
        self.activation = activation
        self.connections = connections
        self.natural_death_thresh = natural_death_thresh
        self.random_death_prob = random_death_prob

        self.additional_produce_std = additional_produce_std

        self.produce_variation_mag = produce_variation_mag
        self.produce_thresh = produce_thresh

        self.connection_drop_rate = connection_drop_rate

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

class Layer():
    def __init__(self, neurons, neuron_num, pre_layer, next_layer, additional_produce_std)
        self.neurons = neurons # A list of existing neurons
        self.neuron_num = neuron_num
        self.pre_layer = pre_layer
        self.next_layer = next_layer
        self.additional_produce_std = additional_produce_std

    def output(self, inputs):

    def produce(self):
        #Output New layers
        produce_num = 1 + round(abs(np.random.default_rng().normal(0, self.additional_produce_std))/2)
        for _ in range(produce_num):
            new_neurons = [neuron.produce() for neuron in self.neurons]
 
class InputLayer(Layer):
    def __init__(self, drop_rate, **kwargs):
        self.drop_rate = drop_rate
        super().__init__(**kwargs)
        self.pre_layer = None

class OutputLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.next_layer = None

class HiddenLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class Brain():
    def __init__(self, num_hidden, hidden_layer_produce_prob):
        self.num_hidden = num_hidden
        self.hidden_layer_produce_prob = hidden_layer_produce_prob
        return 0
        
    def predict(self, inputs):
        # TODO: perform the calculations
        return 0

    def produce(self)

class Unit():
    def __init__(self,rand_produce_prob, produce_variant_std,neuron_info):
        ### Bith of init
        self.rand_produce_prob = rand_produce_prob
        self.produce_variant_std = produce_variant_std
        self.neuron_info = neuron_info
        self.brain = self.create_brain()
        return self
    def create_brain(self):
        # TODO: Initialize the brain
        return 0
    
    def gen_random_brain(self):
        # TODO: fully randomized brain states
        return self.create_brain()
    
    def reproduce(self):
        if (random.uniform(0,1) <= self.rand_produce_prob):
            return self.gen_random_brain()
        # TODO: variant of current brain states
        return 0
    
    def predict(inputs):
        return self.brain.predict(inputs)