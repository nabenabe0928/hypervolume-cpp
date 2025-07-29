import time

import hvcpp
import numpy as np
from optuna._hypervolume import compute_hypervolume
from optuna.study._multi_objective import _is_pareto_front

n_objectives = 5
rng = np.random.RandomState(42)
X_unique = np.unique(rng.normal(size=(100, n_objectives)), axis=0)
sorted_pareto_sols = X_unique[_is_pareto_front(X_unique, True)]
ref_point = np.full(n_objectives, 10.0)
print(hvcpp.compute_hypervolume(sorted_pareto_sols, ref_point))
print(compute_hypervolume(sorted_pareto_sols, ref_point, assume_pareto=True))
runtime_python = 0.0
runtime_cpp = 0.0
for _ in range(10):
    X_unique = np.unique(rng.normal(size=(1000, n_objectives)), axis=0)
    sorted_pareto_sols = X_unique[_is_pareto_front(X_unique, True)]
    start = time.time()
    hvcpp.compute_hypervolume(sorted_pareto_sols, ref_point)
    runtime_cpp += time.time() - start
    start = time.time()
    compute_hypervolume(sorted_pareto_sols, ref_point, assume_pareto=True)
    runtime_python += time.time() - start

print(f"{runtime_cpp=}, {runtime_python=}")
