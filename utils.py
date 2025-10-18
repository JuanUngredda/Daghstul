import torch
from botorch.acquisition import ExpectedImprovement, AnalyticAcquisitionFunction
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.model import Model
from botorch.models.transforms import Standardize
from botorch.utils import t_batch_mode_transform
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor


def evaluate_batch(func, X_normalized: torch.Tensor, lb: torch.Tensor, ub: torch.Tensor):
    """
    Evaluate the function on the *original scale*.
    The GP, acquisition, and optimizer work in [0,1]^d space.
    """
    X_original = lb + (ub - lb) * X_normalized
    y = func(X_original)
    return y.reshape(-1, 1)


def create_acquisition_function(acquisition, gp, y_all):
    if acquisition == "EI":
        best_f = y_all.min().item()
        return ExpectedImprovement(model=gp, best_f=best_f, maximize=False)
    else:
        raise NotImplementedError(f"Acquisition function '{acquisition}' is not implemented")


def train_model(X_all, y_all):
    gp = SingleTaskGP(X_all, y_all, outcome_transform=Standardize(m=1))
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    return gp


class PosteriorMean(AnalyticAcquisitionFunction):
    r"""Analytic acquisition function that simply returns the posterior mean."""

    def __init__(self, model: Model) -> None:
        super().__init__(model=model)

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        """
        Evaluate the posterior mean at X.

        Args:
            X: A `(b) x 1 x d`-dim Tensor of `(b)` t-batches of `d`-dim design points.

        Returns:
            A `(b)`-dim Tensor of posterior mean values.
        """
        posterior = self.model.posterior(X)
        mean = posterior.mean.squeeze(-1)
        return -mean.squeeze(-1)
