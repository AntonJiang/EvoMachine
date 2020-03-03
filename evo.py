import numpy as np
import pandas as pd
from evo_utils import KillSystem
from population import Population
import hyper

np.seterr(all='raise')


def main():
    print("Starting Evolution Machine")
    train_data = pd.read_csv("data/random-linear-regression/train.csv")
    num_population = 1
    num_unit = 30
    meta = 10.0
    hyper.verbose = 10

    population = [num_unit, meta]
    populations = [population for _ in range(num_population)]
    label = np.array(train_data['y'])
    data = np.array(train_data['x'])


    model = EvoMachine(populations, (1, 1), (1, 1))
    model.train(np.array(label), np.array(data))
    print(model.predict([10, 10, 10, 10, 10]))


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
        if hyper.verbose > 0:
            print(f'Generating Population {len(populations)=}')
        self.populations = []
        # TODOS: Set custom min_survice_percent
        hyper.input_shape = input_shape
        hyper.output_shape = output_shape
        for index, population in enumerate(populations):
            if hyper.verbose > 1:
                print(f'Generating Population {index} with {population[0]} units')
            pop = Population(population[0], population[1], KillSystem(hyper.min_survive_percent))
            self.populations.append(pop)
        if hyper.verbose > 0:
            print('Finished Generating Populations.')

    def train(self, label, data):
        """
        Train the model

        Args:
        label (list{?}) : list of labels
        data (list{?}) : list of input data
        Return
        """
        if hyper.verbose > 0:
            print(f'Starting Training {len(data)=} {len(label)=}')
        assert len(label) == len(data)
        for index, population in enumerate(self.populations):
            if hyper.verbose > 1:
                print(f'Training population {index}')
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
