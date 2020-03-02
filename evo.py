import numpy as np
import pandas as pd

from evo_utils import KillSystem
from population import Population
import hyper

np.seterr(all='raise')


def main():
    train_data = pd.read_csv("data/random-linear-regression/train.csv")

    population = [100, 10.0]
    populations = [population for _ in range(10)]
    label = np.array(train_data['y'])
    data = np.array(train_data['x'])

    model = EvoMachine(populations, (1, 1), (1, 1))


def vote(results):
    """
    Vote final result among populations

    Args:
    results (list{?}) : the results returned by different populations
    Return:
    ({?}) : the average voted result
    """
    return np.mean(results, axis=0)


class EvoMachine(object):
    """ docstring for EvoMachine
    Params:
    populations (list{list{float}}): Variables for each population, population id to parameters
        population_size (int) : number of units in the population
        population_variant_magnitude (float) : the magnitude of mutation within the population
    input_shape (list{int})
    ouput_shape (list{int})

    TODOs:
        0. Shuffle data and add batch training
        1. Allow custom population initiation variables
        2. Different voting mechanism
    """

    def __init__(self, populations, input_shape, output_shape):
        self.populations = []
        # TODOS: Set custom min_survice_percent
        hyper.input_shape = input_shape
        hyper.output_shape = output_shape
        for population in populations:
            pop = Population(population[0], population[1], KillSystem(hyper.min_survive_percent))
            self.populations.append(pop)

    def train(self, label, data):
        """
        Train the model

        Args:
        label (list{?}) : list of labels
        data (list{?}) : list of input data
        Return
        """
        assert len(label) == len(data)
        for population in self.populations:
            population.train(label, data)

    def predict(self, data):
        """
        Predict based on input

        Args:
        data (list{?}) : list of input data
        Return:
        (list{?}) : list of predicted values
        """
        results = [population.predict(data) for population in self.populations]
        return vote(results)


if __name__ == "__main__":
    main()
