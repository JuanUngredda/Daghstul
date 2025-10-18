import time
from pathlib import Path

import torch

from bo_loop import run_bo_problem

dtype = torch.double

if __name__ == "__main__":
    experiments = {
        "Rosenbrock_6D": {"total_samples": 100, "initial_samples": 10},
        "Michalewicz_10D": {"total_samples": 300, "initial_samples": 15},
        "Ackley_100D": {"total_samples": 500, "initial_samples": 20},
    }

    acqf_type = "EI"
    base_folder = Path(f"results/{acqf_type}")
    base_folder.mkdir(exist_ok=True)

    seeds = range(1, 31)

    for name, cfg in experiments.items():
        for seed in seeds:
            print(f"\nRunning {name}, seed={seed} (samples={cfg['total_samples']})...")
            t0 = time.time()
            out_file = base_folder / f"{name}_seed{seed}_results.json"

            try:
                res = run_bo_problem(
                    problem_name=name,
                    total_samples=cfg["total_samples"],
                    initial_samples=cfg["initial_samples"],
                    acquisition=acqf_type,
                    batch_size=1,
                    seed=seed,
                    save_path=str(out_file),
                )
                print(f"Saved results to {out_file} (took {time.time() - t0:.1f}s)")
            except Exception as e:
                print(f"Error running {name} with seed {seed}: {e}")