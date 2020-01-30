class WeightsProperties():
    def __init__(self, weights, past_weights, weights_trend, weights_counter, difficulty_to_death, difficulty_to_reproduce)
        self.difficulty_to_reproduce = difficulty_to_reproduce
        self.difficulty_to_death = difficulty_to_death
        self.weights_trend = weights_trend
        self.weights_counter = weights_counter
    def check_weight_trend(self):
        


class Connections():
    def __init__(self, weights, weights_prop, states, functions):
        self.weights = weights
        self.weights_prop = weights_prop
        self.states = states
        self.functions = functions

    def new_connection(self, magnitude):
    #TODO generate connection based on mag  
        return 0
    def get_weights(self):
        return self.weights
    def get_states(self):
        return self.states
    def get_functions(self):
        return self.functions
    def get_connections(self):
        return np.stack([self.weights, self.states, self.functions], axis=1)
    def check_weight_trend(self):
        #Return 0 normal trend, 1 for up_trend, -1 for down trend, 2 for reproduce, -2 for death
        return self.weights_prop.check_weight_trend()

class Neurons():
    def __init__(self, connections, natural_death_thresh, random_death_prob,
                 random_produce_prob, produce_variation_mag, produce_thresh, random_produce_mag):
        self.activation = activation
        self.connections = connections
        self.natural_death_thresh = natural_death_thresh
        self.random_death_prob = random_death_prob
        self.random_produce_prob = random_produce_prob
        self.produce_variation_mag = produce_variation_mag
        self.produce_thresh = produce_thresh
        self.random_produce_mag = random_produce_mag
    def check_death(self):
        return self.connections.check_death()

    def check

    def output(self, inputs):
        # Input ordered by neurons
        values = []
        for func, value in zip(self.connections.get_functions):
            values.append(func(value))
        return self.activation(np.array(values).multiply(self.connections.get_weights() * self.connections.get_states()))

class InputLayer(Layers):
    def __init__(self, drop_rate, **kwargs):
        self.drop_rate = drop_rate
        super().__init__(**kwargs)
        self.pre_layer = None

class OutputLayer(Layers):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.next_layer = None

class HiddenLayer(Layers):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class Layers():
    def __init__(self, neuron_num, pre_layer, next_layer)
        self.neuron_num = neuron_num
        self.pre_layer = pre_layer
        self.next_layer = next_layer
  
    def output(self):
        