import numpy as np
import pandas as pd
import statsmodels as sm
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sn
import random
import enum

from hyper import Hyperparam
from neuron import Neuron

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
        
        num_hidden (int) : the number of hidden layers
        num_neurons (list{int}) : a list of neuron in each hidden layer

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
        self.meta_variant_magnitude = meta_variant_magnitude

        self.input_shape = Hyperparam.input_shape
        self.output_shape = Hyperparam.output_shape
        self.neurons = []


    def _product(list):
        p = 1
        for i in list:
            p *= i
        return p

    def reproduce(self):
        """
        Produce a single variant copy
        
        Args:
            None
        Return:
            (Unit) : the variant copy
        """
        return 0
    
    def predict(self, inputs):
        """
        Calculate result based on inputs

        Args:
            inputs (?) : a single input
        Return:
            (?) : the calculated result
        """
        return self.brain.predict(inputs)