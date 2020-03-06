import numpy as np
import hyper
import copy
import evo_utils
from connections import Connections


def gen_variant_value(old_value, magnitude):
    percent_change = magnitude / 100
    multiplier = ((hyper.rng.uniform(-1, 1, size=1) * percent_change) + 1)
    return old_value * multiplier


class Neuron:
    """
    A basic neuron

    Params:
        id (int) : the unique id for the neuron
        distance (0<float<1): the distance of the neuron to the input neuron
        activation (Function) : the function to calculated the activation function
        connections (Connections) : dict of neurons to weights that it connects to
        meta_variant (float) : the magnitude of how much all variables change
        input_count (int) : the number of inputs connected
        temp_input (list{?}) : the values of currently calculated inputs

    TODO: Combine all attributes into one multidimensional array
    """

    def __init__(self, distance, activation, connections, meta_variant, input_neurons=None):
        self.distance = distance
        self.activation = activation
        self.connections = connections
        self.meta_variant = meta_variant
        self.input_count = len(input_neurons)
        # TODO, optimize refactor, change input neurons to dict with unique input count
        self.input_neurons = input_neurons
        self.temp_input = []

    def __str__(self):
        return self.distance

    def del_input_i(self, index):
        del self.input_neurons[index]
        self.input_count -= 1

    def del_input(self, neuron):
        self.input_neurons.remove(neuron)
        self.input_count -= 1

    def update_input_neurons(self, neurons):
        self.input_neurons.extend(neurons)
        self.input_count += len(neurons)

    def update_input_neuron(self, neuron):
        self.input_neurons.append(neuron)
        self.input_count += 1

    def update_input_count(self, delta):
        self.input_count += delta

    def delete_connection_i(self, index):
        self.connections.delete_i(index)

    def delete_connection(self, neuron):
        self.connections.delete(neuron)

    def update_connections(self, neurons):
        """
        Extend the current connection with target neurons
        Generate properties similar to existing connections
        """
        self.connections.add(neurons)

    def produce(self, meta_neurons):
        """
        Return a variant copy of the neuron
        """
        if hyper.verbose > 4:
            print(f'Producing Neuron')
        new_connection = Connections(np.copy(self.connections.get_target_neurons()),
                                     np.copy(self.connections.get_weights()),
                                     np.copy(self.connections.get_functions()),
                                     np.copy(self.connections.get_states()))

        # Child will have a new distance
        new_distance = hyper.rng.random()

        new_neuron = Neuron(new_distance, self.activation, new_connection, self.meta_variant,
                            input_neurons=copy.copy(self.input_neurons))

        # Check for Distance mismatch of connections
        for index in range(len(new_connection.get_target_neurons()) - 1, -1, -1):
            if new_distance > new_connection.get_target_neurons()[index].distance:
                # Remove the target connection
                new_connection.delete_i(index)
            else:
                new_connection.get_target_neurons()[index].update_input_neurons([new_neuron])

        # Check for distance mismatch for inputs
        for index in range(len(new_neuron.input_neurons) - 1, -1, -1):
            if new_distance < new_neuron.input_neurons[index].distance:
                new_neuron.del_input_i(index)
            else:
                new_neuron.input_neurons[index].update_connections([new_neuron])
        new_neuron.update(meta_neurons)
        evo_utils.display_neuron(new_neuron)

        return new_neuron

    def self_destruct(self):
        """
        Delete all the connections involved with the neuron
        """
        # Delete outgoing connections
        for target_neuron in self.connections.get_target_neurons():
            target_neuron.del_input(self)
        for input_neuron in self.input_neurons:
            input_neuron.delete_connection(self)

    def update(self, meta_neurons):
        """
        Update the connections of the neuron

        """

        if hyper.rng.random() > 1.0 / self.meta_variant:
            self.activation = evo_utils.random_get()

        self.connections.update(self.meta_variant, meta_neurons, self)

        self.meta_variant = hyper.rng.normal(self.meta_variant, self.meta_variant / 3)

    def output(self, single_input):
        # Input ordered by
        self.temp_input.append(single_input)
        if len(self.temp_input) != self.input_count:
            return

        functioned_output = [func(self.activation(self.temp_input)) for func in self.connections.get_functions()]
        outputs = self.connections.get_states() * functioned_output * self.connections.get_weights()

        for neuron, output in zip(self.connections.get_target_neurons(), outputs):
            neuron.output(output)
        # Clear the calculation memory
        self.temp_input = []


class InputNeuron(Neuron):
    """
    The First Neuron in a Unit, a wrapper for the input neuron

    Params:
        input_shape (list{int}) : the input shape
        ** kwargs

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_count = 1
        self.activation = None
        self.input_aggregate_multiplier = None

    def output(self, inputs):
        inputs = np.tile(np.array(inputs).ravel(), self.input_aggregate_multiplier)

        functioned_output = [func(output) for output, func in zip(inputs, self.connections.get_functions())]
        outputs = self.connections.get_states() * functioned_output * self.connections.get_weights()

        for neuron, output in zip(self.connections.get_target_neurons(), outputs):
            neuron.output(output)


class OutputNeuron(Neuron):
    """
    The Last Neuron in a Unit, a wrapper for the output neuron

    Params:
        output_shape (list{int}) : the output shape
        activation (Func) : unique activation for output
        output_val (?) : the final output value 
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.output_val = None
        self.activation = hyper.output_activation
        self.output_aggregate_multiplier = None

    def output(self, single_input):
        self.temp_input.append(single_input)
        if len(self.temp_input) != self.input_count:
            return

        self.output_val = np.reshape(
            self.activation(evo_utils.aggregate(np.array(self.temp_input),
                                                self.output_aggregate_multiplier)), hyper.output_shape)

        self.temp_input = []
