import numpy as np
import hyper


# TODO:
# class Pickle(object):
# 	"""docstring for Pickle"""
# 	def __init__(self, arg):
# 		self.arg = arg
# 		

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
        if input_neuron.connections.target_neurons.count(out) == input_neuron.connections.size:
            no_del_count += 1
        else:
            del_index.append(index)

    return no_del_count, del_index


def delta_size(curr_size, want_size, flag):
    multiplier = round(1.0 * curr_size / want_size)
    if multiplier <= 0:
        multiplier = 1
    if flag == 1:
        hyper.output_aggregate_multiplier = multiplier
    if flag == 0:
        hyper.input_aggregate_multiplier = multiplier
    delta = want_size * multiplier - curr_size
    return delta


def product(lst):
    p = 1
    for i in lst:
        p *= i
    return p


def deter_size(rng, mean, scale, og_size):
    size = int(rng.normal(mean, scale) * og_size)
    if size == 0:
        return 1
    if size == og_size:
        return og_size - 1


def aggregate(inputs, multiplier):
    return inputs.reshape(multiplier, -1).sum(axis=0) / multiplier


def random_get():
    """
    Return a random activation functions

    Args:
    Return
        (list{Func}) : a list of random activation functions
    """
    return np.random.choice(pool_activation)


def random_get_functions(size):
    """
    Return a random activation functions

    Args:
        size (int) : the size of the functions
    Return
        (list{Func}) : a list of random activation functions
    """
    return np.random.choice(single_function, size=size)


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


def _linear_function(x, a=1, _coe=0):
    return a * x + _coe


def _sigmoid_function(x, a=None):
    z = (1 / (1 + np.exp(-x)))
    return z


def _tanh_function(x, a=None):
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
    if x < 0:
        return a * (np.exp(x) - 1)
    else:
        return x


def _swish_function(x, a=None):
    return x / (1 - np.exp(-x))


def _softmax_function(x, a=None):
    z = np.exp(x)
    z_ = z / z.sum()
    return z_


def _polynomial_2nd_function(x, a=None, b=None):
    return a * x * x + _linear_function(x, b)


def _polynomial_3rd_function(x, a=None, b=None, c=None):
    return a * x * x * x + _polynomial_2nd_function(x, b, c)


single_activation = [_binary_step, _linear_function, _sigmoid_function, _tanh_function,
                     _relu_function, _leaky_relu_function, _elu_function, _swish_function,
                     _softmax_function]

pool_activation = [lambda x: func(x.sum()) for func in single_activation]

single_function = [_linear_function, _sigmoid_function, _leaky_relu_function, _swish_function,
                   _softmax_function, _polynomial_2nd_function, _polynomial_3rd_function]


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
        # TODO: change this:
        del_size = np.random.default_rng().normal(max_del_size / 2, max_del_size / 5)
        assert (max_del_size == size)
        outputs = [unit.predict(input_val) for unit in units]
        loss = [mae(output, true_output) for output in outputs]
        loss, index = sorted(zip(loss, np.arange(size)), reverse=True)
        p = np.linalg.norm(np.linspace(del_size, 0, num=del_size))
        index = np.random.choice(index, replace=False, size=del_size, p=p)
        return index
