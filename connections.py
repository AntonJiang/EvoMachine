import numpy as np
import hyper

from evo_utils import Activations

class Connections():
    """
    Each connection is from input neuron to target neuron specific

    Params:
        target_neuron (Neuron) : the target neuron
        weight (float) : the weight multiplier of the current connection
        function (Func) : the specific mathmatical function associated with
                            the connection, single input, single output
        state (int) : 1 if the connection is working
    """

    def __init__(self, target_neurons, weights=None, functions=None, states=None):
        self.target_neurons = target_neurons
        self.size = len(target_neurons)

        if weights is None:
            self.weights = Connections.gen_new_weights(self.size)
        else:
            self.weights = weights
        
        if functions is None:
            self.functions = Activations.random_get_functions(self.size)
        else:
            self.functions = functions

        if states is None:
            self.states = np.ones(len(target_neurons))
        else:
            self.states = states

    def delete(self, index):
        """
        Delete the connection at the giving index if possible
        """
        self.size -= 1
        del self.weights[index]
        del self.states[index]
        del self.functions[indx]
        del self.target_neurons[index]

    def add(self, neurons):
        """
        Add more connections based on neurons

        Args:
            neurons (Neurons) : new target neuron for the connection to be added
        """
        addition_size = len(neurons)
        self.weights = np.hstack((self.weights, Connections.gen_new_weights(addition_size, self.weights)))
        self.size += addition_size
        self.states.extend(np.ones(addition_size))
        self.functions = Activations.random_get_functions(self.size)
        self.target_neurons.extend(neurons)

    def update(self, meta_variant, neuron, neurons, distance):
        """
        Update the connection in some degree, delete some connection, add some more connections
        """
        rng = np.random.default_rng()

        # Kill some connections
        death_index = rng.choice(np.arange(self.size), size=e_utils._deter_size(rng, hyper.connection_life_percent/2, \
                                                        hyper.connection_life_percent_variant/2, self.size), \
            replace=False)
        for i in sorted(death_index, reverse=True):
            self.delete(i, neuron)
            self.target_neurons[i].del_input(neuron)
        

        update_index = rng.choice(np.arange(self.size), size=e_utils._deter_size(rng, hyper.connection_update_percent, \
                                                        hyper.connection_update_percent_variant, self.size), \
            replace=False)

        for i in update_index:
            self._update(i, meta_variant)


        birth_neurons = rng.choice(neurons, size=e_utils._deter_size(rng, hyper.connection_life_percent/2, \
                                                        hyper.connection_life_percent_variant/2, self.size), \
            replace=False)
        
        birth_neurons = [neuron for neuron in birth_neurons if distance < neuron.distance ]
        self.add(birth_neurons)

    def _update(self, index, meta_variant):
        """
        Update at the given index
        """
        rng = np.random.default_rng()
        self.weights = [rng.normal(weight, weight/3*(meta_variant/10)) for weight in self.weights]
        ### Able to optimize this:
        self.states = [_flip(state) for state in self.states if (rng.random() < hyper.connection_state_flip_percent)]
        function_index = rng.choice(np.arange(self.size), size=e_utils._deter_size(rng, hyper.connection_function_percent, \
                                                        hyper.connection_function_percent_variant, self.size), \
            replace=False)
        for func, index in zip(Activations.random_get_functions(len(function_index)), function_index):
            self.functions[index] = func

    def _flip(num):
        if num == 1:
            return 0
        return 1

    def gen_new_weights(size, reference_weight=[]):
        """
        Generate Weights based on Sample Weights
        
        Args:
            size (int) : the size of weights need to be generated
            sample_weight (list{float}) : sample weights for reference
        
        Return:
            (list{float}) : new weights of size
        """
        mean = np.mean(reference_weight)
        new_weights = np.random.default_rng().normal(mean, np.std(reference_weight) ,size)
        return new_weights

    def check_weight_trend(self, death_mult):
        # Return 0 for normal 1 for death
        # Upper level will kill connections accordingly when init the neuron
        #TODOS
        return 0