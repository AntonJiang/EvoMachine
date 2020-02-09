import numpy as np
import pandas as pd
import statsmodels as sm
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sn
import random
import enum

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
    """
    Class for each individual computational unit

    Param:
        weight_variant_magnitude (float) : the variance magnitude for changing the weights
        probabilities (list{float}) :           
            layer_produce_prob (float) : the probability for one layer to produce
            layer_del_prob (float) : the probability for one layer to disappear
            neuron_produce_prob (float) : the probability for one neuron to produce
            neuron_del_prob (float) : the probability for one neuron to disappear
            connection_produce_prob (float) : the probability for one connection to produce
            connection_del_prob (float) : the probability for one connection to disappear
        meta_variant_magnitude (float) : the magnitude of how much all variables change
        brain (Brain) : the NN layer structure

    TODOs:
        Must:
            1. Initialize probabilities
            2. Initialize brain
        future:
            1. Make each weight have their own variant magnitude
    """
    def __init__(self, weight_variant_magnitude, probabilities, meta_variant_magnitude):
        assert weight_variant_magnitude <= 1 and weight_variant_magnitude >= 0
        self.weight_variant_magnitude = weight_variant_magnitude
        self.probabilities = probabilities
        self.meta_variant_magnitude = meta_variant_magnitude
        self.brain = self.create_brain(?)

    def create_brain(?):
        return 0

    def reproduce(self):
        """
        Produce a variant copy
        
        Params:
        
        Return:
            (Unit) : the variant copy
        """
        return 0
    
    def predict(inputs):
        """
        Calculate result based on inputs
        """
        return self.brain.predict(inputs)