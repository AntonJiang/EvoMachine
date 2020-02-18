import enum

class LivingState(enum.Enum):
    DIE = -1
    NORM = 0


class Hyperparam:
	unit_base_hidden_layer = 5
    unit_base_layer_neuron = 5

    input_shape = None
    output_shape = None

    input_layer_drop_rate = 0.1

   	mean_neuron_num = None
   	