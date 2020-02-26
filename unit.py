import numpy as np
import pandas as pd
import statsmodels as sm
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sn
import random
import copy

from hyper import Hyperparam
from evo_utils import Activation
from neuron import Neuron

class Unit():
    """
    Class for each individual computational unit

    Param:
        probabilities (list{float}) :           
            neuron_produce_percent (float) : the probability for one neuron to produce
            neuron_del_percent (float) : the probability for one neuron to disappear
            connection_produce_percent (float) : the probability for one connection to produce
            connection_del_percent (float) : the probability for one connection to disappear
        meta_variant_magnitude (float) : the magnitude of how much all variables change

    TODOs:
        future:
            1. Make each weight have their own variant magnitude
    """
    def __init__(self, probabilities, meta_variant_magnitude):
        self.probabilities = probabilities
        self.meta_variant_magnitude = meta_variant_magnitude

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
                connections=Connections([]), meta_variant=meta_variant_magnitude, probabilities=probabilities))

        # Initialize connections based on distances
        for index, neuron in zip(range(neuron_count), self.neurons):
            if (neuron.input_count == 0) or (index == neuron_count-1 and _product(neuron.output_shape) > neuron.input_count):
                # If there are no input, then find set an input
                # Or is the output neuron with output shape greater than input
                input_connections = np.random.choice(neurons[:index], np.random.randint(low=1, high=index))
                for input_neuron in input_connections:
                    input_neuron.update_connections([neuron])

            #Find output connections
            target_neurons = np.random.choice(neurons[index+1:], np.random.randint(low=1, high=neuron_count - index))
            neuron.update_connections(target_neurons)

    def __init__(self, probabilities, meta_variant_magnitude, neurons):
        self.probabilities = probabilities
        self.meta_variant_magnitude = meta_variant_magnitude

        self.neurons = neurons

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
        #TODOs: varies probability and meta_variant by meta_variant
        #TODOs: custom scheme for selected neurons to change
        rng = np.random.default_rng()

        og_size = len(self.neurons)
        indexes = np.arange(og_size)
        mother_index = rng.choice(indexes, size=_deter_size(rng, self.probabilities.neuron_produce_percent, \
                                                        Hyperparam.neuron_produce_percent_variant, og_size), \
            replace=False, shuffle=False)
        survive_index = rng.choice(indexes, size=og_size - _deter_size(rng, self.probabilities.neuron_del_percent, \
                                                        Hyperparam.neuron_del_percent_variant, og_size), \
            replace=False, shuffle=False)        

        new_neurons = []
        for i in mother_index:
            new_neurons.append(self.neurons[i].produce(connection_produce_percent, connection_del_percent))
        for i in survive_index:
            new_neurons.append(copy.deepcopy(self.neurons[i]))

        new_unit = Unit(self.probabilities, self.meta_variant_magnitude, new_neurons)
        return new_unit
    
    def _deter_size(rng, mean, scale, og_size):
        size = int(rng.normal(prob, scale)*og_size)
        if size == 0:
            return 1
        if size == og_size:
            return og_size -1

    def predict(self, inputs):
        """
        Calculate result based on inputs

        Args:
            inputs (?) : a single input
        Return:
            (?) : the calculated result
        """
        self.neurons[0].output(inputs)
        return self.neurons[neuron_count].output_val