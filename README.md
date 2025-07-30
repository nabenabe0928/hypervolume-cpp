# Hypervolume calculation in C++

Please build the C++ file so that you can call our module from your Python project:

```shell
$ git clone git@github.com:nabenabe0928/hypervolume-cpp.git
$ cd hypervolume-cpp
$ mkdir build && cd build
$ cmake ..
$ make
# Most likely, under the `site-packages` directory.
$ mv hvcpp.cpython-* <path/to/your/python/lib>
```

> [!NOTE]
> This implementation is the C++ version of the [Optuna](https://github.com/optuna/optuna/blob/master/optuna/_hypervolume/wfg.py) WFG implementation.

## Example

```python
import hvcpp
import numpy as np
from optuna.study._multi_objective import _is_pareto_front


n_objectives = 4
rng = np.random.RandomState(42)
# The input array must be sorted in the first objective.
X_unique = np.unique(rng.normal(size=(100, n_objectives)), axis=0)
# The input array should only contain the Pareto solutions. (Otherwise, the computation becomes inefficient.)
sorted_pareto_sols = X_unique[_is_pareto_front(X_unique, True)]
ref_point = np.full(n_objectives, 10.0)
print(hvcpp.compute_hypervolume(sorted_pareto_sols, ref_point))
```
