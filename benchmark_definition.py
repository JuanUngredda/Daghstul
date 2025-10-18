import math

import torch
from botorch.test_functions import Rosenbrock, Michalewicz, Ackley

# ---------- BENCHMARK DEFINITIONS ----------
BENCHMARKS = {
    "Rosenbrock_6D": {
        "fn": Rosenbrock(dim=6),
        "dim": 6,
        "lb": torch.full((6,), -2.0, dtype=torch.double),
        "ub": torch.full((6,), 2.0, dtype=torch.double),
    },
    "Michalewicz_10D": {
        "fn": Michalewicz(dim=10),
        "dim": 10,
        "lb": torch.zeros(10, dtype=torch.double),
        "ub": torch.full((10,), math.pi, dtype=torch.double),
    },
    "Ackley_100D": {
        "fn": Ackley(dim=100),
        "dim": 100,
        "lb": torch.full((100,), -32.768, dtype=torch.double),
        "ub": torch.full((100,), 32.768, dtype=torch.double),
    },
}
