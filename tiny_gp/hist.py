import numpy as np
from matplotlib import pyplot as plt
from pydantic import BaseModel

from tiny_gp.entry import Entry


class Hist(BaseModel):
    """
    History of simulation.
    """
    entries: list[Entry] = list()

    def __init__(self, entries: list[Entry] | None = None):
        super().__init__(
            entries=entries if entries else list()
        )

    def append(self, entry: Entry):
        self.entries.append(entry)

    def __str__(self):
        return "\n".join([str(e) for e in self.entries])

    def __repr__(self):
        return self.__str__()

    @property
    def generation(self) -> np.ndarray:
        return np.asarray([e.gen for e in self.entries])

    @property
    def avg_fitness(self) -> np.ndarray:
        return np.asarray([e.avg_fitness for e in self.entries])

    @property
    def best_fitness(self) -> np.ndarray:
        return np.asarray([e.best_fitness for e in self.entries])

    @property
    def avg_size(self) -> np.ndarray:
        return np.asarray([e.avg_size for e in self.entries])

    def plot(self):
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 3, 1)
        plt.plot(self.generation, self.avg_fitness, label="Average Fitness", marker='o')
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.title("Average Fitness over Generations")
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 3, 2)
        plt.plot(self.generation, self.best_fitness, label="Best Fitness", marker='s')
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.title("Best Fitness over Generations")
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 3, 3)
        plt.plot(self.generation, self.avg_size, label="Average Size", color='orange', marker='^')
        plt.xlabel("Generation")
        plt.ylabel("Average Size")
        plt.title("Average Program Size over Generations")
        plt.grid(True)

        plt.tight_layout()
        plt.show()
