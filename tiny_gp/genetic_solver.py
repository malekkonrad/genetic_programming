import os
import json

import numpy as np
from typing import *
import matplotlib.pyplot as plt

from tiny_gp.hist import Hist
from tiny_gp.operations import Operation
from tiny_gp.tiny_gp import TinyGP


class GeneticSolver:

    targets_to_compare = None

    def __init__(self, data_file: str,):
        self.data_file_name = data_file
        self.data_file_path = f'./data/{data_file}.dat'
        
    def solve(self, operations: Optional[list[Operation]] = None, generations: Optional[int] = 30) -> Hist:

        targets = list()

        with open(self.data_file_path, "r") as f:
            line = f.readline()
            varnumber, randomnumber, minrandom, maxrandom, fitnesscases = [float(s) for s in line.split()]
            varnumber = int(varnumber)
            randomnumber = int(randomnumber)
            fitnesscases = int(fitnesscases)
            for line in f:
                targets.extend([float(s) for s in line.split()])
                
        targets_np = np.array(targets)
        targets_np = targets_np.reshape([fitnesscases,varnumber+1]) 

        self.targets_to_compare = targets_np

        self.tiny_gp = TinyGP(
            constant_count=randomnumber,
            min_random=minrandom,
            max_random=maxrandom,
            operations=operations,
            generations=generations,
            # operations={Operation.ADD, Operation.MUL}  # you can provide custom operation (at least one must be 2 argument function)
        )
        # FIXME to stop the evolution at will you have to restart the jupyter kernel 
        # hist = tiny_gp.fit(targets_np, random_state=3)  # YES random_state!!!!
        self.hist = self.tiny_gp.fit(targets_np)

        self._save_hist()
        self._save_gp()

        return self.hist



    def plot(self):
        self.tiny_gp.plot()
    

    def _save_gp(self):
        PATH = './gps'
        if not os.path.exists(PATH):
            os.makedirs(PATH)
        self.tiny_gp.save_json(f'{PATH}/gp_{self.data_file_name}.json')


    def _save_hist(self):
        PATH = './hists'
        if not os.path.exists(PATH):
            os.makedirs(PATH)
            
        x = self.hist.model_dump()
        with open(f'{PATH}/hist_{self.data_file_name}.json', 'w') as f:
            json.dump(x, f)


    def evaluate(self, vars):
        return self.tiny_gp.evaluate(vars)
    


    def compare_with_another(self, solver2: "GeneticSolver"):
        x = self.targets_to_compare[:, 0]
        y = solver2.targets_to_compare[:, 1]

        plt.figure(figsize=(8, 5))

        plt.plot(x, y, label="Target Function", color="blue", linewidth=2)

        plt.scatter(x, self.evaluate(x.reshape(-1, 1)), label="Evaluated Points without exp", color="red", marker='o')
        plt.scatter(x, solver2.evaluate(x.reshape(-1, 1)), label="Evaluated Points with exp", color="green", marker='o')

        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Target Function vs Evaluated Points")
        plt.legend()
        plt.grid(True)
        plt.show()