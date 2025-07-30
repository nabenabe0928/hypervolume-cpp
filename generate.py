import numpy as np
import hvcpp
import optuna


rng = np.random.RandomState(42)
X = np.unique(rng.normal(size=(100, 4)), axis=0)
pareto_sols = X[optuna.study._multi_objective._is_pareto_front(X, True)]
hv = optuna._hypervolume.compute_hypervolume(pareto_sols, np.full(4, 10.0), assume_pareto=True)
print(pareto_sols, hv, hvcpp.compute_hypervolume(pareto_sols, np.full(4, 10.0)))
s = "{"
for i, vs in enumerate(pareto_sols):
    s += "{"
    for j, v in enumerate(vs):
        s += f"{v.item()}," if j < len(vs) - 1 else f"{v.item()}"
    s += "}," if i < len(pareto_sols) - 1 else "}"
s += "};"
print(s)
