import hyper
import numpy as np
from neuron import Neuron, InputNeuron, OutputNeuron
from evo_utils import Activations
from connections import Connections
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
    def __init__(self, probabilities, meta_variant_magnitude, neurons=[]):
        self.probabilities = probabilities
        self.meta_variant_magnitude = meta_variant_magnitude
        self.neurons = []


    def setup(self):
        neuron_count = (hyper.mean_neuron_num)

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
                input_neurons = np.random.choice(self.neurons[:index], np.random.randint(low=1, high=index+1))
                if index == len(distances) - 1:
                    Neuron_class = OutputNeuron

            self.neurons.append(Neuron_class(distance=distance, activation=Activations.random_get(), \
                connections=Connections([]), meta_variant=self.meta_variant_magnitude, input_neurons=input_neurons))

            for input_neuron in input_neurons:
                input_neuron.update_connections([self.neurons[-1]])


        # Initialize connections based on distances
        for index, neuron in enumerate(self.neurons[:-1]):
            #Find output connections
            target_neurons = np.random.choice(self.neurons[index+1:], np.random.randint(low=1, high=neuron_count - index))
            neuron.update_connections(target_neurons)

        refactor_ouput(self)
        refactor_input(self)


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

        # Kill old neurons
        dying_indexes = rng.choice(indexes[1:-1], size=e_utils._deter_size(rng, self.probabilities.neuron_del_percent, \
                                                        hyper.neuron_del_percent_variant, og_size), \
            replace=False)
        
        [new_unit.neurons[i].self_destruct() for i in sorted(dying_indexes, reverse=True)]
        

        # Produce new neurons
        mother_indexes = rng.choice(indexes[1:-1], size=e_utils._deter_size(rng, self.probabilities.neuron_produce_percent, \
                                                        hyper.neuron_produce_percent_variant, og_size), \
            replace=False)

        [new_unit.neurons.append(new_unit.neurons[i].produce(self.neurons)) for i in mother_indexes]
        
        # Update existing neurons with new weights
        update_index = rng.choice(np.arange(len(new_unit.neurons)), size=e_utils._deter_size(rng, self.probabilities.neruon_update_percent, \
                                                        hyper.neruon_update_percent_variant, len(new_neurons)), \
            replace=False)

        [new_unit.neurons[i].update(self.neurons) for i in update_index]




        new_unit.refactor_graph()
        ### Maybe it's not neccesary to sort, instead, change update to accomandate for unsorted neurons
        # sorted(new_unit.neurons, key=lambda neuron : neuron.distance)
        
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
                input_neurons = np.random.choice(neurons[:index+1], np.random.randint(low=1, high=index+2))
                neuron.update_input_neurons(input_neurons)
                [input_neuron.update_connections([neuron]) for input_neuron in input_neurons]
            if neuron.connections.size == 0:
                target_neurons = np.random.choice(neurons[index+1:], np.random.randint(low=1, high=neuron_count - index))
                neuron.update_connections([target_neurons])
                [target_neuron.update_input_neurons([neuron]) for target_neuron in target_neurons]

        refactor_ouput(self)
        refactor_input(self)



    def refactor_ouput(self):
        out = self.neurons[-1]
        out_size = _product(hyper.output_shape)
        curr_size = out.input_count
        delta_size = _delta_size(curr_size, out_size, 1)

        if delta_size >= 0:
            # If need more connections:
            input_neurons = np.random.choice(self.neurons[1:-1], size=delta_size)
            out.update_input_neurons(input_neurons)
            for input_neuron in input_neurons:
                input_neuron.update_connections([out])
        else:
            # If wants less connectios
            no_del_count, del_index = _no_del_inputs(delta_size, self)
            if curr_size - no_del_count + delta_size <= 0:
                # Duplicate Code
                # If can't have less connections
                input_neurons = np.random.choice(self.neurons[1:-1], size=delta_size)
                out.update_input_neurons(input_neurons)
                for input_neuron in input_neurons:
                    input_neuron.update_connections([out])
            else:
                # If able to have less connections
                del_index = sorted(np.random.default_rng().choice(del_index, size=-delta_size), reverse=True)
                for i in del_index:
                    out.input_neurons[i].delete_connection(out)
                    out.del_input_i(i)
    
    def refactor_input(self):
        rng = np.random.default_rng()
        input_neuron = self.neuron[0]
        in_size = _product(hyper.input_shape)
        curr_size = input_neuron.connections.size
        delta_size = _delta_size(curr_size, in_size, 0)

        if delta_size >= 0:
            # If need more connections:
            connection_neurons = rng.choice(self.neurons[1:-1], size=delta_size)
            input_neuron.update_connections(connection_neurons)
            for connection_neuron in connection_neurons:
                connection_neuron.update_input_neurons([input_neuron])
        else:
            # If wants less connections
            no_del_count, del_index = _no_del_outputs(delta_size, self)
            if curr_size - no_del_count + delta_size <= 0:
                #Duplicate Again
                # If can't have less connections:
                connection_neurons = rng.choice(self.neurons[1:-1], size=delta_size)
                input_neuron.update_connections(connection_neurons)
                for connection_neuron in connection_neurons:
                    connection_neuron.update_input_neurons([input_neuron])
            else:
                # If able to have less connections
                del_index = sorted(np.random.default_rng().choice(del_index, size=-delta_size), reverse=True)
                for i in del_index:
                    input_neuron.connections.target_neurons[i].del_input(input_neuron)
                    input_neuron.delete_connection_i(i)


    def _no_del_outputs(delta_size, unit):
        input_neuron = unit.neurons[-1]
        connections = input_neuron.connections
        if connections.size + delta_size <= 0:
            return True

        no_del_count = 0
        del_index = []
        for index, connection_neuron in enumerate(connections.target_neurons):
            if connection_neuron.input_neurons.count(input_neuron) == connection_neuron.input_count:
                no_del_count += 1
            else:
                del_index.append(index)

        return no_del_count, del_index

    def _no_del_inputs(delta_size, unit):
        out = unit.neurons[-1]
        input_neurons = out.input_neurons
        if len(input_neurons) + delta_size <= 0:
            return True
        no_del_count = 0
        del_index = []
        for index, input_neuron in enumerate(input_neurons):
            if input_neuron.connections.target_neurons.count(out) == input_neuron.connections.size:
                no_del_count += 1
            else:
                del_index.append(index)

        return no_del_count, del_index

    def _delta_size(curr_size, want_size, flag):
        multiplier = round(1.0*curr_size/want_size)
        if multiplier <= 0:
            multiplier = 1
        if flag == 1:
            hyper.output_aggregate_multiplier = multiplier
        if flag == 0:
            hyper.input_aggregate_multiplier = multiplier
        delta_size = want_size*multiplier - curr_size
        return delta_size

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