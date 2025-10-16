from pydantic import BaseModel

from tiny_gp.individual import Individual


class Entry(BaseModel):
    """
    Singular entry of the History.
    """
    gen: int
    avg_fitness: float
    best_fitness: float
    avg_size: float
    best_individual: Individual

    def __init__(self, gen: int, avg_fitness: float, best_fitness: float, avg_size: float, best_individual: Individual):
        super().__init__(
            gen=gen,
            avg_fitness=avg_fitness,
            best_fitness=best_fitness,
            avg_size=avg_size,
            best_individual=best_individual
        )

    def __str__(self):
        return (f"-----Generation: {self.gen} -----\n"
                f"Average fitness: {self.avg_fitness}\n"
                f"Best fitness: {self.best_fitness}\n"
                f"Average size: {self.avg_size}\n"
                f"Best individual: {self.best_individual}")

    def __repr__(self):
        return self.__str__()
