import hyper
import copy
import numpy as np
import evo_utils
from neuron import Neuron, InputNeuron, OutputNeuron
from connections import Connections


class Unit:
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

    def __init__(self, probabilities, meta_variant_magnitude, neurons=None):
        if neurons is None:
            neurons = []
        self.probabilities = probabilities
        self.meta_variant_magnitude = meta_variant_magnitude
        self.neurons = neurons

    def setup(self):
        neuron_count = hyper.mean_neuron_num

        distances = np.sort(np.random.default_rng().random(size=neuron_count), kind='quicksort')
        distances[0] = -1
        distances[-1] = 1

        # Initialize distance and basic parameters
        # Initialize input neurons
        for index, distance in enumerate(distances):
            Neuron_class = Neuron
            input_neurons = []

            if index == 0:
                Neuron_class = InputNeuron
            else:
                input_neurons = np.random.choice(self.neurons[:index], np.random.randint(low=1, high=index + 1))
                if index == len(distances) - 1:
                    Neuron_class = OutputNeuron

            self.neurons.append(Neuron_class(distance=distance, activation=evo_utils.random_get(),
                                             connections=Connections([]), meta_variant=self.meta_variant_magnitude,
                                             input_neurons=input_neurons))

            for input_neuron in input_neurons:
                input_neuron.update_connections([self.neurons[-1]])

        # Initialize connections based on distances
        for index, neuron in enumerate(self.neurons[:-1]):
            # Find output connections
            target_neurons = np.random.choice(self.neurons[index + 1:],
                                              np.random.randint(low=1, high=neuron_count - index))
            neuron.update_connections(target_neurons)

        self.refactor_output()
        self.refactor_input()

    @property
    def reproduce(self):
        """
        Produce a single variant copy
        
        Args:
        Return:
            (Unit) : the variant copy
        """
        # TODOs: varies probability and meta_variant by meta_variant
        # TODOs: custom scheme for selected neurons to change
        rng = np.random.default_rng()
        og_size = len(self.neurons)
        indexes = np.arange(og_size)

        new_unit = copy.deepcopy(self)

        # Kill old neurons
        dying_indexes = rng.choice(indexes[1:-1], size=evo_utils.deter_size(rng, self.probabilities.neuron_del_percent,
                                                                            hyper.neuron_del_percent_variant, og_size),
                                   replace=False)

        [new_unit.neurons[i].self_destruct() for i in sorted(dying_indexes, reverse=True)]

        # Produce new neurons
        mother_indexes = rng.choice(indexes[1:-1],
                                    size=evo_utils.deter_size(rng, self.probabilities.neuron_produce_percent,
                                                              hyper.neuron_produce_percent_variant, og_size),
                                    replace=False)

        [new_unit.neurons.append(new_unit.neurons[i].produce(self.neurons)) for i in mother_indexes]

        # Update existing neurons with new weights
        update_index = rng.choice(np.arange(len(new_unit.neurons)),
                                  size=evo_utils.deter_size(rng, self.probabilities.neruon_update_percent,
                                                            hyper.neuron_update_percent_variant,
                                                            len(new_unit.neurons)),
                                  replace=False)

        [new_unit.neurons[i].update(self.neurons) for i in update_index]

        new_unit.refactor_graph()

        return new_unit

    def refactor_graph(self):
        """
        Run the unit and clean out any died connections\

        1. Fix no input neurons --
        2. Fix no output neurons --
        5. Fix InputNeuron, OutputNeuron shape mismatch --
        """
        for index, neuron in enumerate(self.neurons[1:-1]):
            if neuron.input_count == 0:
                input_neurons = np.random.choice(self.neurons[:index + 1], np.random.randint(low=1, high=index + 2))
                neuron.update_input_neurons(input_neurons)
                [input_neuron.update_connections([neuron]) for input_neuron in input_neurons]
            if neuron.connections.size == 0:
                target_neurons = np.random.choice(self.neurons[index + 1:],
                                                  np.random.randint(low=1, high=len(self.neurons) - index))
                neuron.update_connections([target_neurons])
                [target_neuron.update_input_neurons([neuron]) for target_neuron in target_neurons]

        self.refactor_output()
        self.refactor_input()

    def refactor_output(self):
        out = self.neurons[-1]
        out_size = evo_utils.product(hyper.output_shape)
        curr_size = out.input_count
        delta_size = evo_utils.delta_size(curr_size, out_size, 1)
        rng = np.random.default_rng()

        if delta_size >= 0:
            # If need more connections:
            input_neurons = rng.choice(self.neurons[1:-1], size=delta_size)
            out.update_input_neurons(input_neurons)
            for input_neuron in input_neurons:
                input_neuron.update_connections([out])
        else:
            # If wants less connectios
            no_del_count, del_index = evo_utils.no_del_inputs(delta_size, self)
            if curr_size - no_del_count + delta_size <= 0:
                # Duplicate Code
                # If can't have less connections
                input_neurons = rng.choice(self.neurons[1:-1], size=delta_size)
                out.update_input_neurons(input_neurons)
                for input_neuron in input_neurons:
                    input_neuron.update_connections([out])
            else:
                # If able to have less connections
                del_index = sorted(rng.choice(del_index, size=-delta_size), reverse=True)
                for i in del_index:
                    out.input_neurons[i].delete_connection(out)
                    out.del_input_i(i)

    def refactor_input(self):
        rng = np.random.default_rng()
        input_neuron = self.neurons[0]
        in_size = evo_utils.product(hyper.input_shape)
        curr_size = input_neuron.connections.size
        delta_size = evo_utils.delta_size(curr_size, in_size, 0)

        if delta_size >= 0:
            # If need more connections:
            connection_neurons = rng.choice(self.neurons[1:-1], size=delta_size)
            input_neuron.update_connections(connection_neurons)
            for connection_neuron in connection_neurons:
                connection_neuron.update_input_neurons([input_neuron])
        else:
            # If wants less connections
            no_del_count, del_index = evo_utils.no_del_outputs(delta_size, self)
            if curr_size - no_del_count + delta_size <= 0:
                # Duplicate Again
                # If can't have less connections:
                connection_neurons = rng.choice(self.neurons[1:-1], size=delta_size)
                input_neuron.update_connections(connection_neurons)
                for connection_neuron in connection_neurons:
                    connection_neuron.update_input_neurons([input_neuron])
            else:
                # If able to have less connections
                del_index = sorted(rng.choice(del_index, size=-delta_size), reverse=True)
                for i in del_index:
                    input_neuron.connections.target_neurons[i].del_input(input_neuron)
                    input_neuron.delete_connection_i(i)

    def predict(self, inputs):
        """
        Calculate result based on inputs

        Args:
            inputs (?) : a single input
        Return:
            (?) : the calculated result
        """
        self.neurons[0].output(inputs)
        return self.neurons[len(self.neurons)].output_val
