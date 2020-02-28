import numpy as np
import pandas as pd
import statsmodels as sm
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sn
import random
import copy
import evo_utils as e_utils

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
            neruon_update_percent (float) : the probability for neuron to update
        meta_variant_magnitude (float) : the magnitude of how much all variables change

    TODOs:
        future:
            1. Make each weight have their own variant magnitude
            2. Change Unit connection initiation to uniform for all distance, then check for unconnected !
    """
    def __init__(self, probabilities, meta_variant_magnitude):
        self.probabilities = probabilities
        self.meta_variant_magnitude = meta_variant_magnitude

        self.neurons = []
        neuron_count = Hyperparam.mean_neuron_num + np.random.normal(0, meta_variant_magnitude)

        distances = np.sort(np.random.default_rng().random(size=neuron_count), kind='quicksort')

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

        new_unit = copy.deepcopy(self)


        # Update existing neurons with new weights
        update_index = rng.choice(np.arange(len(new_unit.neurons)), size=e_utils._deter_size(rng, self.probabilities.neruon_update_percent, \
                                                        Hyperparam.neruon_update_percent_variant, len(new_neurons)), \
            replace=False)

        [new_unit.neurons[i].update(self.neurons) for i in update_index]


        # Kill old neurons
        dying_indexes = rng.choice(indexes, size=e_utils._deter_size(rng, self.probabilities.neuron_del_percent, \
                                                        Hyperparam.neuron_del_percent_variant, og_size), \
            replace=False)        
        
        [new_unit.neurons[i].self_destruct() for i in dying_indexes]
        self.refactor_graph()

        # Produce new neurons
        mother_indexes = rng.choice(indexes, size=e_utils._deter_size(rng, self.probabilities.neuron_produce_percent, \
                                                        Hyperparam.neuron_produce_percent_variant, og_size), \
            replace=False)
                                                                #TODOs: Ulgy parameters
        [new_unit.neurons.append(new_unit.neurons[i].produce(self.neurons)) for i in mother_indexes]

        ### Maybe it's not neccesary to sort, instead, change update to accomandate for unsorted neurons
        # sorted(new_unit.neurons, key=lambda neuron : neuron.distance)
        
        return new_unit

    def refactor_graph(self):
        """
        Run the unit and clean out any died connections and establish isolated connections
        """
        TODO

        return new_unit
    


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