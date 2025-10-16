import json

import numpy as np
from matplotlib import pyplot as plt

from tiny_gp import tiny_gp_methods, Hist, Individual, FitnessFunction, Operation, FUN_2ARG


class TinyGP:
    """
    TinyGP.
    """
    max_length: int = 10000
    population_size: int = 100000
    depth: int = 5
    generations: int = 100
    tournament_size: int = 2
    min_random: float = -5
    max_random: float = 5
    var_number: int = 1
    constant_count: int = 100
    permutation_per_node: float = 0.05
    crossover_probability: float = 0.9
    targets: np.ndarray = None
    fitness_cases: int | None = None
    seed: int = -1
    hist: Hist = Hist()
    best_individual: Individual | None = None
    fitness_function: FitnessFunction = FitnessFunction.MAE
    operations: set[Operation] = {Operation.ADD, Operation.SUB, Operation.MUL, Operation.DIV}
    goal_fitness: float = 1e-5
    java_path: str | None = None

    def __init__(
            self,
            *,
            max_length: int = 10000,
            population_size: int = 100000,
            depth: int = 5,
            generations: int = 100,
            tournament_size: int = 2,
            min_random: float = -5,
            max_random: float = 5,
            constant_count: int = 100,
            mutation_probability: float = 0.05,
            crossover_probability: float = 0.9,
            fitness_function: FitnessFunction = FitnessFunction.MAE,
            operations: set[Operation] | None = None,
            goal_fitness: float = 1e-5
    ):
        """
        Sets the parameters of the evolution.
        :param max_length: Maximum length of the individual
        :param population_size: Population size
        :param depth: Maximum initial amount of operations in an individual
        :param generations: Maximum number of generations in the simulation
        :param tournament_size: How many individuals compete for the best and the worst individual
        :param min_random: Constant's lower bound
        :param max_random: Constant's upper bound
        :param constant_count: Amount of constants
        :param mutation_probability: Probability of mutation
        :param crossover_probability: Probability to make an offspring
        :param fitness_function: Function to calculate error
        :param operations: Operations used by the simulated individuals
        :param goal_fitness: Maximum allowed error
        """
        if operations is None:
            operations = {Operation.ADD, Operation.SUB, Operation.MUL, Operation.DIV}
        else:
            if len(operations) <= 1:
                raise ValueError("You need to provide at least one operation.")
            test = {opp for opp in operations if opp.name in FUN_2ARG}
            if len(test) == 0:
                raise ValueError("You need to provide at least one 2 argument operation.")

        self.max_length = max_length
        self.population_size = population_size
        self.depth = depth
        self.generations = generations
        self.tournament_size = tournament_size
        self.min_random = min_random
        self.max_random = max_random
        self.constant_count = constant_count
        self.permutation_per_node = mutation_probability
        self.crossover_probability = crossover_probability
        self.fitness_function = fitness_function
        self.operations = operations
        self.goal_fitness = goal_fitness

    @staticmethod
    def set_java_path(path: str):
        TinyGP.java_path = path

    def fit(self, targets: np.ndarray, random_state: int = -1) -> Hist:
        """
        Run simulation.
        :param targets: Array with rows equal number of fitness cases,
                        columns equal the number variables + last column for the result of the function
        :param random_state: Seed for RNG
        :return: History
        """
        return tiny_gp_methods.fit(self, targets, random_state)

    def _map_operations(self) -> dict[str, int]:
        """
        :return: Dictionary mapping operation name to it's integer value,
                 also adds FSET_END and FSET_2ARG_END
        """
        return tiny_gp_methods.map_operations(self)

    def to_json(self) -> str:
        """Serialize the TinyGP instance to JSON."""
        data = {
            "max_length": self.max_length,
            "population_size": self.population_size,
            "depth": self.depth,
            "generations": self.generations,
            "tournament_size": self.tournament_size,
            "min_random": self.min_random,
            "max_random": self.max_random,
            "var_number": self.var_number,
            "constant_count": self.constant_count,
            "mutation_probability": self.permutation_per_node,
            "crossover_probability": self.crossover_probability,
            "targets": self.targets.tolist(),
            "fitness_cases": self.fitness_cases,
            "seed": self.seed,
            "hist": self.hist.model_dump(),
            "best_individual": (
                self.best_individual.model_dump() if self.best_individual else None
            ),
        }
        return json.dumps(data, indent=4)

    def save_json(self, file: str):
        """Serialize and save the instance to JSON."""
        with open(file, 'w') as f:
            f.write(self.to_json())

    @classmethod
    def from_json(cls, file: str) -> "TinyGP":
        """Parse a JSON and return a TinyGP instance."""
        with open(file, 'r') as f:
            data = json.loads(f.read())

        instance = cls()

        if "targets" in data:
            instance.targets = np.asarray(data.pop("targets"))

        if "hist" in data:
            instance.hist = Hist.model_validate(data.pop("hist"))
            for e in instance.hist.entries:
                e.best_individual.operations = instance._map_operations()  # add operations manually, ugh

        if "best_individual" in data and data["best_individual"]:
            instance.best_individual = Individual.model_validate(data.pop("best_individual"))
            instance.best_individual.operations = instance._map_operations()  # add operations manually, ugh

        for attr in cls.__dict__.keys():
            if attr in data:
                setattr(instance, attr, data[attr])

        return instance

    def __str__(self):
        return (f"seed={self.seed}\n"
                f"max_length={self.max_length}\n"
                f"population_size={self.population_size}\n"
                f"depth={self.depth}\n"
                f"crossover_probability={self.crossover_probability}\n"
                f"mutation_probability={self.permutation_per_node}\n"
                f"min_random={self.min_random}\n"
                f"max_random={self.max_random}\n"
                f"generations={self.generations}\n"
                f"tournament_size={self.tournament_size}\n"
                f"best_individual={self.best_individual}")

    def evaluate(self, variables: list[float] | np.ndarray) -> float | np.ndarray:
        """Predict result from variables using the best individual."""
        return self.best_individual.evaluate(variables)

    def plot(self, targets: np.ndarray | None = None):
        """Plot the target function and the evaluation."""
        if targets is None:
            targets = self.targets

        x = targets[:, 0]
        y = targets[:, 1]

        plt.figure(figsize=(8, 5))

        plt.plot(x, y, label="Target Function", color="blue", linewidth=2)

        plt.scatter(x, self.evaluate(x.reshape(-1, 1)), label="Evaluated Points", color="red", marker='o')

        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Target Function vs Evaluated Points")
        plt.legend()
        plt.grid(True)
        plt.show()

