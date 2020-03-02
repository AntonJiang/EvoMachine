from unit import Unit
import numpy as np


def vote(results):
    """
    Vote final result among units

    Args:
        results (list{?}) : the results returned by different units
    Return:
        ({?}) : the average voted result
    """
    return np.mean(results, axis=0)


class Population:
    """
    Params:
        population_size (int) : size of the population
        population_variant_magnitude (float) : how much the population is mutating
        kill_system (KillSystem) : the type of kill system used
        units (list{Unit}) : a list of units within the population

    TODOs
        Must:
            1. Refine the gen_weight_variant sensitivity
        Future:
            1. Custom ClockCycle for kill and reproduce
            2. Custom number of label per killing == batch training
            3. Implement mercy in killing, so that it doesn't kill everything
            4. Implement ordered units based on performance
                a. Implement non-uniform distribution for performant units
                b. More efficient unit produce by specificing num need to produce for each unit
    """

    def __init__(self, population_size, population_variant_magnitude, kill_system):
        self.population_variant_magnitude = population_variant_magnitude
        self.kill_system = kill_system
        self.units = []

        for _ in range(population_size):
            unit = Unit([0.1, 0.1, 0.2], population_variant_magnitude)
            unit.setup()
            self.units.append(unit)

    def train(self, labels, data):
        """
        Train the model

        Args:
            labels (list{?}) : list of labels
            data (list{?}) : list of input data
        """
        # Kill off units, using one data entry per killing
        for y, x in zip(labels, data):
            self.kill([y], [x])
            self.produce()

    def kill(self, labels, data):
        """
        Kill a number of units

        Args:
            labels (list{?}) : a list of labels for training the units
            data (list{?}) : a list of x value for prediction
        """
        for y, x in zip(labels, data):
            death_index = self.kill_system.mark_death(y, x, self.units)
            for i in sorted(death_index, reverse=True):
                del self.units[i]

    def produce(self):
        """
        Reproduce until population is filled again
        """
        num_to_produce = len(self.units) - len(self.units)
        produce_index = np.random.choice(self.units, size=num_to_produce, replace=True, p=None)
        self.units.extend([self.units[index].reproduce for index in produce_index])

    def predict(self, data):
        """
        Predict based on input

        Args:
            data (list{?}) : list of input data
        Return:
            (list{?}) : list of predicted values
        """
        results = [vote([unit.predict(entry) for unit in self.units]) for entry in data]
        return results
