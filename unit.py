import numpy as np
import pandas as pd
import statsmodels as sm
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sn
import random
import enum

from hyper import Hyperparam
from evo_utils import Activation
from neuron import Neuron

class Unit():
    """
    Class for each individual computational unit

    Param:
        probabilities (list{float}) :           
            neuron_produce_prob (float) : the probability for one neuron to produce
            neuron_del_prob (float) : the probability for one neuron to disappear
            connection_produce_prob (float) : the probability for one connection to produce
            connection_del_prob (float) : the probability for one connection to disappear
        meta_variant_magnitude (float) : the magnitude of how much all variables change
        
        num_hidden (int) : the number of hidden layers
        neurons (list{Neurons})) : a list of neurons with their properties

    TODOs:
        Must:
            1. Initialize probabilities
        future:
            1. Make each weight have their own variant magnitude
    """
    def __init__(self, probabilities, meta_variant_magnitude):
        self.meta_variant_magnitude = meta_variant_magnitude

        self.input_shape = Hyperparam.input_shape
        self.output_shape = Hyperparam.output_shape
        self.neurons = []
        neuron_count = Hyperparam.mean_neuron_num + np.random.normal(0, meta_variant_magnitude)

        distances = np.sort(np.random.random(size=neuron_count), kind='quicksort')

        # Initalize distance and basic parameters
        # First Neuron is the inputLayer
        for index, distance in zip(range(neuron_count), distances):
            Neuron_class = Neuron
            if (index == 0):
                Neuron_class = InputNeuron
            else if (index == neuron_count-1):
                Neuron_class = OutputNeuron    
            self.neurons.append(Neuron_class(distance=distance, activation=Activation.random_get(), \
                connections=Connections([]), meta_variant=meta_variant_magnitude))

        # Initialize connections based on distances
        for index, neuron in zip(range(neuron_count), self.neurons):
            if neuron.input_count == 0 and index != 0:
                # If there are no input, then find set an input
                input_connections = np.random.choice(neurons[:index], np.random.randint(low=1, high=index))
                for input_neuron in input_connections:
                    input_neuron.update_connections([neuron])

            else:
            #Find output connections
                target_neurons = np.random.choice(neurons[index+1:], np.random.randint(low=1, high=neuron_count - index))
                neuron.update_connections(target_neurons)

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
        UNDER PROGRESS
        return 0
    
    def predict(self, inputs):
        """
        Calculate result based on inputs

        Args:
            inputs (?) : a single input
        Return:
            (?) : the calculated result
        """
        UNDER PROGRESS
        return