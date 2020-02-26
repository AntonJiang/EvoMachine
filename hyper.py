
class Hyperparam:
	unit_base_hidden_layer = 5
    unit_base_layer_neuron = 5

    input_shape = None
    output_shape = None

    input_layer_drop_rate = 0.1

   	mean_neuron_num = None
   	
   	output_activation = lambda x : x

   	min_survive_percent = 0.3

   	neuron_produce_percent_variant = 0.03

   	neuron_del_percent_variant = 0.03

