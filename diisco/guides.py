"""
This module implements the abstract base class for a Guide
(Variational Distribution) and particular instantiations of it.

Unfortunately, due the requirement of pyro that every sample site
in the model must be matched by a sample site in the guide, this
module is tightly coupled with the model module and both should
be examined in tandem.

See Guide base class for interface.
"""
import torch
import pyro
import pyro.distributions as dist
from pyro.distributions import constraints
from torch.distributions.utils import vec_to_tril_matrix

import diisco.names as names

from diisco.constants import EPSILON

GUIDE_REGISTRY = {}


def register_guide(name):
    """
    Decorator to register a new guide.
    """

    def register_guide_cls(cls):
        if name in GUIDE_REGISTRY:
            raise ValueError(
                "Cannot register duplicate guide ({})".format(name)
            )
        if not issubclass(cls, Guide):
            raise ValueError(
                "Guide ({}: {}) must extend the Guide class".format(
                    name, cls.__name__
                )
            )
        GUIDE_REGISTRY[name] = cls
        return cls

    return register_guide_cls


class Guide:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def W_mean(self):
        """
        Should return the mean of the W distribution.
        at observed timepoints. The shaped of the returned tensor
        should be (n_timepoints, n_cell_types, n_cell_types)
        """
        raise NotImplementedError

    @property
    def F_mean(self):
        """
        Should return the mean of the F distribution.
        at observed timepoints. The shaped of the returned tensor
        should be (n_timepoints, n_cell_types, 1)
        """
        raise NotImplementedError


@register_guide("multivariate_normal_factorized")
class MultivariateNormalFactorized(Guide):
    """
    Guide for the DIISCO algorithm implementing a factorized normal
    distribution via Cholesky parametrization. Described below.

    Details:
        We use q(W, f) = q(W)q(f) as the guide.
        q(W) = prod_{i=1}^{n_cell_types} prod_{j=1}^{n_cell_types} N(W_{ij})`
        $q(f) = prod_{i=1}^{n_cell_types} N(f_i)`
        Every q(W_{ij}) and q(f_i) are independent of each other but they are
        mulitvariate normal distributions coupled through time.
    """

    # Names of the parameters in the guide.
    F_MEAN = "f_mean"
    F_CHOLESKY_DIAG = "f_cholesky_diag"
    F_CHOLESKY_OFF_DIAG = "f_cholesky_off_diag"
    W_MEAN = "w_mean"
    W_CHOLESKY_DIAG = "w_cholesky_diag"
    W_CHOLESKY_OFF_DIAG = "w_cholesky_off_diag"

    def __init__(
        self, n_cell_types, n_time_points, prior_f_mean, *args, **kwargs
    ):
        # Check that the model has n_cell_types and n_timepoints
        # and raise an error if it doesn't.

        self.n_cell_types = n_cell_types
        self.n_timepoints = n_time_points
        self.prior_f_mean = prior_f_mean

    def __call__(
        self, timepoints: torch.Tensor, proportions: torch.Tensor = None
    ) -> None:
        """
        :param timepoints: Tensor of timepoints for the samples.
            shape: (n_timepoints, 1)
        :param proportions: Tensor of proportions for the samples.
            shape: (n_timepoints, n_cell_types)

        Note: This function is tightly coupled with the model and should have
        the same signature as the model's `forward` function.
        """
        n_cell_types = self.n_cell_types
        n_timepoints = self.n_timepoints

        n_cholesky_params_w = int(n_timepoints * (n_timepoints + 1) / 2)
        n_cholesky_params_f = int(n_timepoints * (n_timepoints + 1) / 2)

        n_cholesky_diag_params_w = n_timepoints
        n_cholesky_diag_params_f = n_timepoints

        cholesky_params_w = pyro.param(
            self.W_CHOLESKY_OFF_DIAG,
            torch.zeros(n_cell_types, n_cell_types, n_cholesky_params_w),
        )
        cholesky_params_f = pyro.param(
            self.F_CHOLESKY_OFF_DIAG,
            torch.zeros(n_cell_types, n_cholesky_params_f),
        )

        cholesky_diag_params_w = pyro.param(
            self.W_CHOLESKY_DIAG,
            torch.abs(torch.randn(n_cholesky_diag_params_w)) * EPSILON,
            constraint=constraints.softplus_positive,
        )
        cholesky_diag_params_f = pyro.param(
            self.F_CHOLESKY_DIAG,
            torch.abs(torch.randn(n_cholesky_diag_params_f)) * EPSILON,
            constraint=constraints.softplus_positive,
        )

        cholesky_w = vec_to_tril_matrix(cholesky_params_w)
        cholesky_f = vec_to_tril_matrix(cholesky_params_f)

        # Replace the diagonal with the cholesky diagonal parameters
        cholesky_w[
            :, :, torch.arange(n_timepoints), torch.arange(n_timepoints)
        ] = cholesky_diag_params_w
        cholesky_f[
            :, torch.arange(n_timepoints), torch.arange(n_timepoints)
        ] = cholesky_diag_params_f
        covariances_w = torch.matmul(cholesky_w, cholesky_w.transpose(-1, -2))
        covariances_f = torch.matmul(cholesky_f, cholesky_f.transpose(-1, -2))

        # Add small constant to the diagonal to make sure the
        # covariance matrix is positive definite
        covariances_f = covariances_f + torch.eye(n_timepoints) * EPSILON
        covariances_w = covariances_w + torch.eye(n_timepoints) * EPSILON

        mean_w = pyro.param(
            self.W_MEAN,
            torch.randn(n_cell_types, n_cell_types, n_timepoints) * EPSILON,
        )
        mean_f = pyro.param(self.F_MEAN, self.prior_f_mean).squeeze(-1)

        with pyro.plate("cell_types_outer_W", n_cell_types, dim=-2):
            with pyro.plate("cell_types_inner_W", n_cell_types, dim=-1):
                pyro.sample(
                    names.W, dist.MultivariateNormal(mean_w, covariances_w)
                )

        with pyro.plate("node_plate", n_cell_types, dim=-1):
            pyro.sample(names.F, dist.MultivariateNormal(mean_f, covariances_f))

    @property
    def W_mean(self):
        params = pyro.get_param_store()
        W_mean = params[self.W_MEAN]
        W_mean = W_mean.permute(2, 0, 1)
        assert W_mean.shape == (
            self.n_timepoints,
            self.n_cell_types,
            self.n_cell_types,
        )
        return W_mean

    @property
    def F_mean(self):
        params = pyro.get_param_store()
        F_mean = params[self.F_MEAN].permute(1, 0, 2)
        assert F_mean.shape == (self.n_timepoints, self.n_cell_types, 1)
        return F_mean
