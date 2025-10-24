from __future__ import annotations

import torch
from botorch.acquisition import ExpectedImprovement, AnalyticAcquisitionFunction, qKnowledgeGradient
from botorch.exceptions import UnsupportedError
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.model import Model
from botorch.models.transforms import Standardize
from botorch.posteriors import Posterior
from botorch.sampling.normal import NormalMCSampler
from botorch.utils import t_batch_mode_transform
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor
from torch.quasirandom import SobolEngine


def evaluate_batch(func, X_normalized: torch.Tensor, lb: torch.Tensor, ub: torch.Tensor):
    """
    Evaluate the function on the *original scale*.
    The GP, acquisition, and optimizer work in [0,1]^d space.
    """
    X_original = lb + (ub - lb) * X_normalized
    y = func(X_original)
    return y.reshape(-1, 1).to(dtype=X_normalized.dtype, device=X_normalized.device)


def create_acquisition_function(acquisition, gp, y_all):
    if acquisition == "EI":
        best_f = y_all.min().item()
        return ExpectedImprovement(model=gp, best_f=best_f, maximize=True)
    elif acquisition == "KG":
        objective_samples = 7
        sampler = QuantileSampler(number_of_objective_samples=objective_samples)
        return qKnowledgeGradient(gp, num_fantasies=objective_samples, sampler=sampler)
    else:
        raise NotImplementedError(f"Acquisition function '{acquisition}' is not implemented")


def train_model(X_all, y_all):
    gp = SingleTaskGP(X_all, y_all, outcome_transform=Standardize(m=1))
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    return gp


class PosteriorMean(AnalyticAcquisitionFunction):
    def __init__(self, model: Model) -> None:
        super().__init__(model=model)

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        posterior = self.model.posterior(X)
        mean = posterior.mean.squeeze(-1)
        return mean.squeeze(-1)


class QuantileSampler(NormalMCSampler):
    def __init__(self, number_of_objective_samples: int, seed: int | None = None,
                 number_of_fantasies_for_constraints=torch.Size, **kwargs: torch.Any) -> None:
        super().__init__(torch.Size([number_of_objective_samples]), seed, **kwargs)
        self.number_of_fantasies_for_constraints = number_of_fantasies_for_constraints
        self.number_of_fantasies_for_objective = number_of_objective_samples

    def _construct_base_samples(self, posterior: Posterior) -> None:
        target_shape = self._get_collapsed_shape(posterior=posterior)
        if self.base_samples is None or self.base_samples.shape != target_shape:
            base_collapsed_shape = target_shape[len(self.sample_shape):]
            output_dim = base_collapsed_shape.numel()
            if output_dim > SobolEngine.MAXDIM:
                raise UnsupportedError(
                    "SobolQMCSampler only supports dimensions "
                    f"`q * o <= {SobolEngine.MAXDIM}`. Requested: {output_dim}"
                )
            base_samples = self.draw_quantiles(
                device=posterior.device,
            )
            base_samples = base_samples.view(target_shape)
            self.register_buffer("base_samples", base_samples)
        self.to(device=posterior.device, dtype=posterior.dtype)

    def draw_quantiles(self, device):
        quantiles_z = (torch.arange(self.number_of_fantasies_for_objective) + 0.5) * (
                1 / self.number_of_fantasies_for_objective)
        normal = torch.distributions.Normal(0, 1)
        z_vals = normal.icdf(quantiles_z)
        return z_vals.to(device=device)
