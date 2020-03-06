import numpy
"""
connection_produce_percent (float) : the probability for one connection to produce
connection_del_percent (float) : the probability for one connection to disappear
"""
unit_base_hidden_layer = 5
unit_base_layer_neuron = 5

input_shape = None
output_shape = None

input_layer_drop_rate = 0.1

mean_neuron_num = 20

min_survive_percent = 0.3

neuron_del_percent = 0.1
neuron_produce_percent = 0.1
neuron_update_percent = 0.2


neuron_produce_percent_variant = 0.03

neuron_del_percent_variant = 0.03

neuron_update_percent_variant = 0.2

connection_update_percent = 0.4

connection_update_percent_variant = 0.05

connection_update_ratio = 0.8

connection_life_percent = 0.4

connection_life_percent_variant = 0.05

connection_state_flip_percent = 0.1

connection_function_percent = 0.05

connection_function_percent_variant = 0.01

connection_reference_weight = 10

verbose = 10


rng = numpy.random.default_rng(100)

def output_activation(x):
    return x
