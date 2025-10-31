from tiny_gp import TinyGP

if __name__ == "__main__":
    gp = TinyGP.from_json("gps/gp_problem10_a.json")
    print(gp.best_individual.operations)
