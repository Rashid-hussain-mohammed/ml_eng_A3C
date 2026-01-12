from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

from nsga2_problem import VehicleOptimizationProblem


def run_optimization(seed=1):
    problem = VehicleOptimizationProblem()

    algorithm = NSGA2(
        pop_size=120,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=20),
        mutation=PM(eta=25),
        eliminate_duplicates=True
    )

    termination = get_termination("n_gen", 200)

    result = minimize(
        problem,
        algorithm,
        termination,
        seed=seed,
        save_history=False,
        verbose=False
    )

    return result



if __name__ == "__main__":
    res = run_optimization()
    print("Optimization completed.")
