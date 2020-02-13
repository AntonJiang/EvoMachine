import numpy as np
import pandas as pd
import statsmodels as sm
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sn
import random
import enum

class Layer():
	"""
	Each Lyaer

	"""

    def __init__(self, neuron_num, pre_layer=None, next_layer=None)
        """
        Initialize Each layer and its connections with the previous layer
        
        """
        self.neuron_num = neuron_num
        self.pre_layer = pre_layer
        self.next_layer = next_layer

    def output(self, inputs):

    def produce(self):
        #Output New layers

class InputLayer(Layer):
    def __init__(self, drop_rate, **kwargs):
        self.drop_rate = drop_rate
        super().__init__(**kwargs)
        
class OutputLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
class HiddenLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
