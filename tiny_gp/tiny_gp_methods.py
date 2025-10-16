import re
import subprocess
import threading

import numpy as np
from py4j.java_gateway import JavaGateway
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tiny_gp import TinyGP

from tiny_gp import Hist, Entry, Individual, DIVISION_CUT_OUT, EXPONENT_CUT_OUT, FSET_START, Operation, FUN_1ARG

from pathlib import Path

module_dir = Path(__file__).resolve().parent


def map_operations(self: "TinyGP") -> dict[str, int]:
    operations: dict[str, int] = dict()
    i: int = FSET_START
    for opp in Operation:
        if opp in self.operations:
            operations[opp.name] = i
            i += 1

    FSET_END = i - 1
    FSET_2ARG_END = FSET_END

    for opp, id_ in operations.items():
        if opp in FUN_1ARG:
            FSET_2ARG_END = id_ - 1
            break

    for opp in Operation:
        if opp not in self.operations:
            operations[opp.name] = i
            i += 1

    operations["FSET_END"] = FSET_END
    operations["FSET_2ARG_END"] = FSET_2ARG_END
    return operations


def fit(self: "TinyGP", targets: np.ndarray, seed: int = -1):
    self.seed = seed
    self.targets = targets
    self.var_number = targets.shape[1] - 1
    self.fitness_cases = targets.shape[0]
    self.hist = Hist()  # reset history
    assert self.var_number + self.constant_count <= FSET_START, f"Sum of variable count and constant count must be less than {FSET_START}."
    # Operations
    operations = map_operations(self)

    pattern = re.compile(r"//TAG\{(.*?)}")

    def repl(match):
        var_name = match.group(1)
        return str(variables.get(var_name))

    variables = {
        "MAX_LEN": int(self.max_length),
        "DIVISION_CUT_OUT": float(DIVISION_CUT_OUT),
        "EXPONENT_CUT_OUT": float(EXPONENT_CUT_OUT),
        "POPSIZE": int(self.population_size),
        "DEPTH": int(self.depth),
        "GENERATIONS": int(self.generations),
        "TSIZE": int(self.tournament_size),
        "minrandom": float(self.min_random),
        "maxrandom": float(self.max_random),
        "varnumber": int(self.var_number),
        "fitnesscases": int(self.fitness_cases),
        "randomnumber": int(self.constant_count),
        "PMUT_PER_NODE": float(self.permutation_per_node),
        "CROSSOVER_PROB": float(self.crossover_probability),
        "seed": int(self.seed),
        "targets": np.array2string(self.targets, separator=',').replace("[", "{").replace("]", "}"),
        "fitness_function": self.fitness_function,
        "goal_fitness": self.goal_fitness,
        "FSET_START": FSET_START,
    } | operations  # add operations

    with open(f"{module_dir}/tiny_gp_java/TinyGP.java", 'r', encoding='utf-8') as f:
        content = f.read()

    compiled = pattern.sub(repl, content)

    with open(f"{module_dir}/tiny_gp_java/TinyGP_compiled.java", 'w', encoding='utf-8') as f:
        f.write(compiled)

    def run_java():
        """
        start TinyGP.java
        """
        process = subprocess.Popen(
            [self.java_path, "-Dfile.encoding=UTF-8", "-cp", f"{module_dir}/tiny_gp_java/py4j-0.10.9.9.jar",
             f"{module_dir}/tiny_gp_java/TinyGP_compiled.java"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8'
        )
        for line in process.stdout:
            if "\\r" in line:
                print(line.replace("\\r", "\r").rstrip("\n"), end="")
            else:
                print(line, end="")

    gateway: JavaGateway | None = None
    thread: threading.Thread | None = None

    try:
        thread = threading.Thread(target=run_java)
        thread.start()

        gateway = JavaGateway()
        tiny_gp = gateway.entry_point
        tiny_gp.evolve()

        x = list(tiny_gp.getX())

        for h in tiny_gp.getHist():
            self.hist.append(Entry(
                int(h.getGen()),
                float(h.getAvg_fitness()),
                float(h.getBest_fitness()),
                float(h.getAvg_size()),
                Individual(list(h.getBest_individual()), x, self.var_number)
            ))
    finally:
        if gateway:
            gateway.shutdown()
        if thread:
            thread.join()

    for e in self.hist.entries:
        e.best_individual.operations = operations

    self.best_individual = self.hist.entries[-1].best_individual  # save the best

    return self.hist
