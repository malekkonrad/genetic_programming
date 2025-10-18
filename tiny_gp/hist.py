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
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        def remove_outliers(data):
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            mask = (data >= lower_bound) & (data <= upper_bound)
            return mask
        
        gen = self.generation
        avg_fit = self.avg_fitness
        best_fit = self.best_fitness
        avg_sz = self.avg_size
        
        axes[0, 0].plot(gen, avg_fit, label="Average Fitness", marker='o')
        axes[0, 0].set_xlabel("Generation")
        axes[0, 0].set_ylabel("Fitness")
        axes[0, 0].set_title("Average Fitness over Generations")
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        axes[0, 1].plot(gen, best_fit, label="Best Fitness", marker='s')
        axes[0, 1].set_xlabel("Generation")
        axes[0, 1].set_ylabel("Fitness")
        axes[0, 1].set_title("Best Fitness over Generations")
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        axes[0, 2].plot(gen, avg_sz, label="Average Size", color='orange', marker='^')
        axes[0, 2].set_xlabel("Generation")
        axes[0, 2].set_ylabel("Average Size")
        axes[0, 2].set_title("Average Program Size over Generations")
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        # bez outlierów
        mask_avg_fit = remove_outliers(avg_fit)
        axes[1, 0].plot(gen[mask_avg_fit], avg_fit[mask_avg_fit], label="Average Fitness (no outliers)", marker='o')
        axes[1, 0].set_xlabel("Generation")
        axes[1, 0].set_ylabel("Fitness")
        axes[1, 0].set_title("Average Fitness over Generations (No Outliers)")
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        mask_best_fit = remove_outliers(best_fit)
        axes[1, 1].plot(gen[mask_best_fit], best_fit[mask_best_fit], label="Best Fitness (no outliers)", marker='s')
        axes[1, 1].set_xlabel("Generation")
        axes[1, 1].set_ylabel("Fitness")
        axes[1, 1].set_title("Best Fitness over Generations (No Outliers)")
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        mask_avg_sz = remove_outliers(avg_sz)
        axes[1, 2].plot(gen[mask_avg_sz], avg_sz[mask_avg_sz], label="Average Size (no outliers)", color='orange', marker='^')
        axes[1, 2].set_xlabel("Generation")
        axes[1, 2].set_ylabel("Average Size")
        axes[1, 2].set_title("Average Program Size over Generations (No Outliers)")
        axes[1, 2].legend()
        axes[1, 2].grid(True)

        plt.tight_layout()
        plt.show()
