import numpy as np
import hyper
import evo_utils


def _flip(num):
    if num == 1:
        return 0
    return 1


def _gen_new_weights(size, reference_weight=None):
    """
    Generate Weights based on Sample Weights

    Args:
        size (int) : the size of weights need to be generated
        reference_weight (list{float}) : sample weights for reference

    Return:
        (list{float}) : new weights of size
    """
    if size == 0:
        return []

    if reference_weight is None or len(reference_weight) == 0:
        reference_weight = [hyper.connection_reference_weight]
    mean = np.mean(reference_weight)
    new_weights = hyper.rng.normal(mean, np.std(reference_weight), size=size)
    return new_weights


class Connections:
    """
    Each connection is from input neuron to target neuron specific

    Params:
        target_neuron (Neuron) : the target neuron
        weight (float) : the weight multiplier of the current connection
        function (Func) : the specific mathematical function associated with
                            the connection, single input, single output
        state (int) : 1 if the connection is working
    """

    def __init__(self, target_neurons, weights=None, functions=None, states=None):
        self.size = len(target_neurons)

        if weights is None:
            weights = _gen_new_weights(self.size)

        if functions is None:
            functions = evo_utils.random_get_functions(self.size)

        if states is None:
            states = np.ones(self.size)

        self.attr = np.array([target_neurons, weights, states, functions])

    def get_functions(self):
        return self.attr[3]

    def get_states(self):
        return self.attr[2]

    def get_weights(self):
        return self.attr[1]

    def get_target_neurons(self):
        return self.attr[0]

    def delete(self, neuron):

        self.size -= 1
        self.attr = np.delete(self.attr, obj=np.argwhere(self.attr[0] == neuron)[0][0], axis=1)

    def delete_i(self, index):
        """
        Delete the connection at the giving index if possible
        """
        self.size -= 1
        self.attr = np.delete(self.attr, obj=index, axis=1)

    def add(self, neurons):
        """
        Add more connections based on neurons

        Args:
            neurons (Neurons) : new target neuron for the connection to be added
        """
        addition_size = len(neurons)
        self.attr = np.concatenate((self.attr, np.array([neurons, _gen_new_weights(addition_size, self.get_weights()),
                                                np.ones(addition_size),
                                                         evo_utils.random_get_functions(addition_size)])),
                                   axis=1)
        self.size += addition_size

    def update(self, meta_variant, neurons, self_neuron):
        """
        Update the connection in some degree, delete some connection, add some more connections

        Args:
            meta_variant : the meta variant
            neurons : list of neurons with chance to connect with
            self_neuron : the self neuron
        """

        # Kill some connections
        if self.size > 1:
            death_index = hyper.rng.choice(np.arange(self.size),
                                     size=evo_utils.deter_size(hyper.connection_life_percent / 2,
                                                               hyper.connection_life_percent_variant / 2, self.size),
                                     replace=False)
            if len(death_index) == self.size:
                death_index = death_index[0]

            for i in sorted(death_index, reverse=True):
                self.get_target_neurons()[i].del_input(self_neuron)
                self.delete_i(i)

            update_index = hyper.rng.choice(np.arange(self.size),
                                      size=evo_utils.deter_size(hyper.connection_update_percent,
                                                                hyper.connection_update_percent_variant, self.size),
                                      replace=False)

            for i in update_index:
                self._update(i, meta_variant)

        birth_neurons = hyper.rng.choice(neurons, size=evo_utils.deter_size(hyper.connection_life_percent / 2,
                                                                      hyper.connection_life_percent_variant / 2,
                                                                      self.size),
                                   replace=False)

        birth_neurons = [neuron for neuron in birth_neurons if self_neuron.distance < neuron.distance]
        self.add(birth_neurons)
        [neuron.update_input_neuron(self_neuron) for neuron in birth_neurons]

    def _update(self, index, meta_variant):
        """
        Update at the given index
        """
        self.get_weights()[index] = hyper.rng.normal(self.get_weights()[index],
                                               self.get_weights()[index] / 3 * (meta_variant / 10))
        # Able to optimize this:
        if hyper.rng.random() < hyper.connection_state_flip_percent:
            self.get_states()[index] = _flip(self.get_states()[index])

        if hyper.rng.random() < hyper.connection_function_percent:
            self.get_functions()[index] = evo_utils.random_get_functions(1)[0]
