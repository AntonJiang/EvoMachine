import numpy as np
import hyper


# TODO:
# class Pickle(object):
# 	"""docstring for Pickle"""
# 	def __init__(self, arg):
# 		self.arg = arg
#

def display_neuron(neuron):
    print(f'**** Neuron Graph ****')
    print(f'Target Neurons:')
    for target_neuron in neuron.connections.get_target_neurons():
        print(f'{target_neuron.distance}')
    print('Input Neurons')
    for input_neuron in neuron.input_neurons:
        print(f'{input_neuron.distance}')

def display_unit(unit):
    print(f'---------- Unit Graph ----------')
    print(f' {unit=} {unit.meta_variant_magnitude=}')
    connection_dict = dict()

    for neuron in unit.neurons:
        # Check for broken out connection
        distance = neuron.distance
        temp_dict = dict(input=[], output=[])
        # Print all outputs
        for target_neuron in neuron.connections.get_target_neurons():
            temp_dict['output'].append(_tuple(target_neuron))

        # Use input to verify outputs and print input if input no outputs
        for input_neuron in neuron.input_neurons:
            if _tuple(neuron) in connection_dict[str(input_neuron.distance)]['output']:
                connection_dict[str(input_neuron.distance)]['output'].remove(_tuple(neuron))
            else:
                temp_dict['input'].append(_tuple(input_neuron))

        connection_dict[str(distance)] = temp_dict
    for key, item in connection_dict.items():
        print(f'{key} : {item}')
    print(f'---------- Finished -----------')


def _tuple(neuron):
    return neuron.distance, neuron


def no_del_outputs(delta, unit):
    input_neuron = unit.neurons[-1]
    connections = input_neuron.connections
    if connections.size + delta <= 0:
        return True

    no_del_count = 0
    del_index = []
    for index, connection_neuron in enumerate(connections.target_neurons):
        if connection_neuron.input_neurons.count(input_neuron) == connection_neuron.input_count:
            no_del_count += 1
        else:
            del_index.append(index)

    return no_del_count, del_index


def no_del_inputs(delta, unit):
    out = unit.neurons[-1]
    input_neurons = out.input_neurons
    if len(input_neurons) + delta <= 0:
        return True
    no_del_count = 0
    del_index = []
    for index, input_neuron in enumerate(input_neurons):
        if len(np.unqiue(input_neuron.connections.get_target_neurons())) == 1:
            no_del_count += 1
        else:
            del_index.append(index)

    return no_del_count, del_index


def delta_size(curr_size, want_size):
    multiplier = round(1.0 * curr_size / want_size)
    if multiplier <= 0:
        multiplier = 1
    delta = want_size * multiplier - curr_size
    return delta, multiplier


def product(lst):
    p = 1
    for i in lst:
        p *= i
    return p


def deter_size(mean, scale, og_size):
    size = int(hyper.rng.normal(mean, scale) * og_size)
    if size == 0:
        return 1
    if size == og_size:
        return og_size - 1
    return size


def aggregate(inputs, multiplier):
    return inputs.reshape(multiplier, -1).sum(axis=0) / multiplier


def random_get():
    """
    Return a random activation functions

    Args:
    Return
        (list{Func}) : a list of random activation functions
    """
    return hyper.rng.choice(pool_activation)


def random_get_functions(size):
    """
    Return a random activation functions

    Args:
        size (int) : the size of the functions
    Return
        (list{Func}) : a list of random activation functions
    """
    if size == 0:
        return []
    return hyper.rng.choice(single_function, size=size)


"""
Pool of Activation functions

TODOs:
    variant parameters for activation functions
"""


def _binary_step(x, a=1):
    if x < 0:
        return 0
    else:
        return a


def _linear_function(x, a=1.0, _coe=0.0):
    return a * x + _coe


def _sigmoid_function(x, a=None):
    if x > 100:
        x = 100
    z = (1 / (1 + np.exp(-x)))
    return z


def _tanh_function(x, a=None):
    if x > 100:
        x = 100
    z = (2 / (1 + np.exp(-2 * x))) - 1
    return z


def _relu_function(x, a=None):
    if x < 0:
        return 0
    else:
        return x


def _leaky_relu_function(x, a=0.01):
    if x < 0:
        return a * x
    else:
        return x


def _elu_function(x, a):
    # Cap at 1000
    if x > 100:
        x = 100
    if x < 0:
        return a * (np.exp(x) - 1)
    else:
        return x


def _swish_function(x, a=None):
    if x == 0:
        return -1
    if x > 100:
        x = 100
    return x / (1 - np.exp(-x))


def _softmax_function(x, a=None):
    if x > 100:
        x = 100
    z = np.exp(x)
    z_ = z / z.sum()
    return z_


def _polynomial_2nd_function(x, a=1.0, b=1.0):
    return a * x * x + _linear_function(x, b)


def _polynomial_3rd_function(x, a=1.0, b=1.0, c=1.0):
    return a * x * x * x + _polynomial_2nd_function(x, b, c)


single_activation = [_binary_step, _linear_function, _sigmoid_function, _tanh_function,
                     _relu_function, _leaky_relu_function, _elu_function, _swish_function]

pool_activation = [lambda x: func(sum(x)) for func in single_activation]

single_function = [_linear_function, _sigmoid_function, _leaky_relu_function, _swish_function,
                   _polynomial_2nd_function, _polynomial_3rd_function]


def mae(predictions, targets):
    differences = predictions - targets
    absolute_differences = np.absolute(differences)
    mean_absolute_differences = absolute_differences.mean()
    return mean_absolute_differences


class KillSystem:
    """
    Util for killing list of Units

    Params:
        mercy (float) : likelihood to not kill
        min_survive (0<float<1) : percentage of units must survivein the units
    """

    def __init__(self, min_survive):
        self.min_survive = min_survive

    def mark_death(self, true_output, input_val, units):
        """
        Calculate each predicted value and mark units as not right

        Args:
            true_output (?) : the desired output
            input_val (?) : the input
            units (list{Unit}) : units ready for inspection
        Return:
            (list{int}) : a list of unique index to kill units
        """
        size = len(units)
        max_del_size = (1 - self.min_survive) * size
        del_size = int(hyper.rng.normal(max_del_size / 3, max_del_size / 5))

        assert del_size < size and max_del_size < size

        outputs = [unit.predict(index, input_val) for index, unit in enumerate(units)]
        loss = np.array([mae(output, true_output) for output in outputs])
        index = np.argsort([-loss, np.arange(size)], axis=1)[0]
        if hyper.verbose > 1:
            print(f'Max Loss {loss[index[0]]}, Min Loss {loss[index[-1]]}, Median Loss {loss[index[int(size / 2)]]}')
        p = np.linspace(1, 0, num=size)
        p = p / np.linalg.norm(p, ord=1)
        index = hyper.rng.choice(index, replace=False, size=del_size, p=p)
        return index
