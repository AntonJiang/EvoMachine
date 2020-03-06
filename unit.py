import hyper
import copy
import numpy as np
import evo_utils
from neuron import Neuron, InputNeuron, OutputNeuron
from connections import Connections
from operator import attrgetter

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

    def __init__(self, meta_variant_magnitude, neurons=None):
        if neurons is None:
            neurons = []
        self.meta_variant_magnitude = meta_variant_magnitude
        self.neurons = neurons

    def setup(self):
        neuron_count = hyper.mean_neuron_num

        distances = np.sort(hyper.rng.random(size=neuron_count), kind='quicksort')
        distances[0] = -1
        distances[-1] = 1

        # Initialize distance and basic parameters
        # Initialize input neurons
        for index, distance in enumerate(distances):
            neuron_class = Neuron
            input_neurons = []

            if index == 0:
                neuron_class = InputNeuron
            else:
                input_neurons = hyper.rng.choice(self.neurons[:index],
                                                 hyper.rng.integers(low=1, high=index + 1)).tolist()
                if index == len(distances) - 1:
                    neuron_class = OutputNeuron

            self.neurons.append(neuron_class(distance=distance, activation=evo_utils.random_get(),
                                             connections=Connections([]), meta_variant=self.meta_variant_magnitude,
                                             input_neurons=input_neurons))

            for input_neuron in input_neurons:
                input_neuron.update_connections([self.neurons[index]])
        # Initialize connections based on distances
        for index, neuron in enumerate(self.neurons[:-1]):
            # Find output connections
            target_neurons = hyper.rng.choice(self.neurons[index + 1:],
                                              hyper.rng.integers(low=1, high=neuron_count - index))
            neuron.update_connections(target_neurons)
            for target_neuron in target_neurons:
                target_neuron.update_input_neuron(neuron)
        if hyper.verbose > 3:
            print('    Finished Initialization')
        self.refactor_output()
        self.refactor_input()
        evo_utils.display_unit(self)

    def reproduce(self):
        """
        Produce a single variant copy

        Args:
        Return:
            (Unit) : the variant copy
        """
        # TODOs: varies probability and meta_variant by meta_variant
        # TODOs: custom scheme for selected neurons to change
        if hyper.verbose > 3:
            print(f'Producing Unit {self}')
        og_size = len(self.neurons)
        indexes = np.arange(og_size)

        new_unit = copy.deepcopy(self)

        evo_utils.display_unit(self)

        # Kill old neurons
        dying_indexes = hyper.rng.choice(indexes[1:-1], size=evo_utils.deter_size(hyper.neuron_del_percent,
                                                                                  hyper.neuron_del_percent_variant,
                                                                                  og_size),
                                         replace=False)
        for i in sorted(dying_indexes, reverse=True):
            new_unit.neurons[i].self_destruct()
            del new_unit.neurons[i]

        evo_utils.display_unit(new_unit)
        # Produce new neurons
        mother_indexes = hyper.rng.choice(indexes[1:-1],
                                          size=evo_utils.deter_size(hyper.neuron_produce_percent,
                                                                    hyper.neuron_produce_percent_variant, og_size),
                                          replace=False)

        [new_unit.neurons.append(new_unit.neurons[i].produce(new_unit.neurons)) for i in mother_indexes]
        new_unit.neurons.sort(key=attrgetter('distance'))
        evo_utils.display_unit(new_unit)

        # Update existing neurons with new weights
        update_index = hyper.rng.choice(np.arange(len(new_unit.neurons)),
                                        size=evo_utils.deter_size(hyper.neuron_update_percent,
                                                                  hyper.neuron_update_percent_variant,
                                                                  len(new_unit.neurons)),
                                        replace=False)

        [new_unit.neurons[i].update(self.neurons) for i in update_index]
        evo_utils.display_unit(new_unit)
        new_unit.refactor_graph()
        evo_utils.display_unit(new_unit)
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
                input_neurons = hyper.rng.choice(self.neurons[:index+1], hyper.rng.integers(low=1, high=index + 2))
                neuron.update_input_neurons(input_neurons)
                [input_neuron.update_connections([neuron]) for input_neuron in input_neurons]
            if neuron.connections.size == 0:
                target_neurons = hyper.rng.choice(self.neurons[index + 2:],
                                                  hyper.rng.integers(low=1, high=len(self.neurons) - index))
                neuron.update_connections(target_neurons)
                [target_neuron.update_input_neurons([neuron]) for target_neuron in target_neurons]

        self.refactor_output()
        self.refactor_input()

    def refactor_output(self):
        if hyper.verbose > 3:
            print('        Refactoring Output')
        out = self.neurons[-1]
        out_size = evo_utils.product(hyper.output_shape)
        curr_size = out.input_count
        delta_size, multiplier = evo_utils.delta_size(curr_size, out_size)
        out.output_aggregate_multiplier = multiplier

        if delta_size >= 0:
            # If need more connections:
            if hyper.verbose > 3:
                print(f'        Need {delta_size} More Connections, Having {curr_size}. ')
            input_neurons = hyper.rng.choice(self.neurons[1:-1], size=delta_size)
            out.update_input_neurons(input_neurons)
            for input_neuron in input_neurons:
                input_neuron.update_connections([out])
        else:
            # If wants less connections
            no_del_count, del_index = evo_utils.no_del_inputs(delta_size, self)
            if curr_size - no_del_count + delta_size <= 0:
                # Duplicate Code
                # If can't have less connections
                if hyper.verbose > 3:
                    print(f'        Need {delta_size} More Connections, Having {curr_size}. ')
                input_neurons = hyper.rng.choice(self.neurons[1:-1], size=delta_size)
                out.update_input_neurons(input_neurons)
                for input_neuron in input_neurons:
                    input_neuron.update_connections([out])
            else:
                # If able to have less connections
                if hyper.verbose > 3:
                    print(f'        Need {-delta_size} Less Connections, Having {curr_size}. ')
                del_index = sorted(hyper.rng.choice(del_index, size=-delta_size), reverse=True)
                for i in del_index:
                    out.input_neurons[i].delete_connection(out)
                    out.del_input_i(i)

    def refactor_input(self):
        if hyper.verbose > 3:
            print('        Refactoring Input')
        input_neuron = self.neurons[0]
        in_size = evo_utils.product(hyper.input_shape)
        curr_size = input_neuron.connections.size
        delta_size, multiplier = evo_utils.delta_size(curr_size, in_size)
        input_neuron.input_aggregate_multiplier = multiplier

        if delta_size >= 0:
            # If need more connections:
            if hyper.verbose > 3:
                print(f'        Need {delta_size} More Connections, Having {curr_size}. ')
            connection_neurons = hyper.rng.choice(self.neurons[1:-1], size=delta_size)
            input_neuron.update_connections(connection_neurons)
            for connection_neuron in connection_neurons:
                connection_neuron.update_input_neurons([input_neuron])
        else:
            # If wants less connections
            no_del_count, del_index = evo_utils.no_del_outputs(delta_size, self)
            if curr_size - no_del_count + delta_size <= 0:
                # Duplicate Again
                # If can't have less connections:
                if hyper.verbose > 3:
                    print(f'        Need {delta_size} More Connections, Having {curr_size}. ')
                connection_neurons = hyper.rng.choice(self.neurons[1:-1], size=delta_size)
                input_neuron.update_connections(connection_neurons)
                for connection_neuron in connection_neurons:
                    connection_neuron.update_input_neurons([input_neuron])
            else:
                # If able to have less connections
                if hyper.verbose > 3:
                    print(f'        Need {-delta_size} Less Connections, Having {curr_size}. ')
                del_index = sorted(hyper.rng.choice(del_index, size=-delta_size), reverse=True)
                for i in del_index:
                    input_neuron.connections.target_neurons[i].del_input(input_neuron)
                    input_neuron.delete_connection_i(i)

    def predict(self, index, inputs):
        """
        Calculate result based on inputs

        Args:
            inputs (?) : a single input
        Return:
            (?) : the calculated result
        """
        if hyper.verbose > 2:
            print(f'Predicting Unit {self}')
        self.neurons[0].output(inputs)
        return self.neurons[len(self.neurons) - 1].output_val
