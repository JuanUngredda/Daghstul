import json
from pathlib import Path

import torch
from botorch.optim import optimize_acqf
from scipy.stats import qmc

from benchmark_definition import BENCHMARKS
from utils import train_model, create_acquisition_function, evaluate_batch, PosteriorMean

data_type = torch.double


def run_bo_problem(problem_name,
                   total_samples,
                   initial_samples=10,
                   acquisition="EI",
                   kernel="Matern52",
                   batch_size=1,
                   seed=12345,
                   save_path="results.json"):
    torch.manual_seed(int(seed))
    prob = BENCHMARKS[problem_name]
    d, lb, ub, fn = prob["dim"], prob["lb"], prob["ub"], prob["fn"]

    if initial_samples >= total_samples:
        raise ValueError("initial_samples must be < total_samples")
    bo_iterations = total_samples - initial_samples

    out = save_general_info(acquisition,
                            batch_size,
                            bo_iterations,
                            d,
                            initial_samples,
                            kernel,
                            lb,
                            problem_name,
                            seed,
                            total_samples, ub)
    sampler = qmc.LatinHypercube(d=d, seed=int(seed))
    X_init = torch.tensor(sampler.random(n=initial_samples), dtype=data_type)

    y_init = evaluate_batch(fn, X_init, lb, ub)

    save_initial_samples(X_init, lb, out, ub, y_init)

    X_all, y_all = X_init.clone(), y_init.clone()

    for it in range(1, bo_iterations + 1):
        gp = train_model(X_all, y_all)
        acquisition_function = create_acquisition_function(acquisition=acquisition, gp=gp, y_all=y_all)
        candidate = compute_next_candidate(batch_size, d, acquisition_function, gp)
        y_next = evaluate_batch(fn, candidate, lb, ub)
        X_all = torch.cat([X_all, candidate])
        y_all = torch.cat([y_all, y_next])

        # --- Save iteration info ---
        iter_block = {"iterations": it, "batch_size": batch_size, "sampled_locations": []}
        save_sampled_locations(candidate, iter_block, lb, ub, y_next)
        out["search_iterations"].append(iter_block)

        print(f"[{problem_name}] Iter {it}/{bo_iterations} -> f={y_next.squeeze().tolist()}")
        # --- Save JSON ---
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as fh:
            json.dump(out, fh, indent=2)
    return out


def save_general_info(acquisition, batch_size, bo_iterations, d, initial_samples, kernel, lb, problem_name, seed,
                      total_samples, ub):
    return {
        "problem_metadata": {
            "test_function": problem_name,
            "dimensionality": d,
            "lower_bounds": str(lb.tolist()),
            "upper_bounds": str(ub.tolist()),
            "replication_number": int(seed)
        },
        "algorithm_parameters": {
            "acquisition_function": acquisition,
            "kernel": kernel,
            "initial_samples": initial_samples,
            "BO_iterations": bo_iterations,
            "total_samples": total_samples,
            "batch_size": batch_size,
            "other_params": {},
            "seed": int(seed)
        },
        "extra_info": {
            "team_notes": f"Basic BO with {acquisition} using torch-only implementation",
            "code_reference": "",
            "other_files": []
        },
        "initial_samples": {
            "batch_size": batch_size,
            "sampled_locations": []
        },
        "search_iterations": []
    }


def save_sampled_locations(candidate, iter_block, lb, ub, y_next):
    for x, y in zip(candidate, y_next):
        x_orig = (lb + (ub - lb) * x).tolist()
        iter_block["sampled_locations"].append({
            "locations": x_orig,
            "evaluations": y.item()})


def save_initial_samples(X_init, lb, out, ub, y_init):
    for i, (x, y) in enumerate(zip(X_init, y_init), start=1):
        x_orig = (lb + (ub - lb) * x).tolist()
        out["initial_samples"]["sampled_locations"].append(
            {"iterations": i, "locations": x_orig, "evaluations": y.item()})


def compute_next_candidate(batch_size, d, ei, gp):
    bounds_unit = torch.stack([torch.zeros(d, dtype=torch.double), torch.ones(d, dtype=torch.double)])
    argmax_gp, max_gp = compute_best_using_smart_init(gp, bounds_unit, ei)
    candidate, acq_value = optimize_acqf(acq_function=ei,
                                         bounds=bounds_unit,
                                         q=batch_size,
                                         num_restarts=20,
                                         raw_samples=512)
    if max_gp > acq_value:
        return argmax_gp
    return candidate


def compute_best_using_smart_init(model, bounds, ei):
    x_dim = bounds.shape[1]
    best_x, min_mean = optimize_acqf(
        acq_function=PosteriorMean(model),
        bounds=bounds,
        q=1,
        num_restarts=20,
        raw_samples=512,
    )
    candidate, acq_value = optimize_acqf(
        acq_function=ei,
        bounds=bounds,
        q=1,
        num_restarts=20,
        raw_samples=512,
        batch_initial_conditions=best_x
    )

    return candidate.reshape(1, x_dim), acq_value
