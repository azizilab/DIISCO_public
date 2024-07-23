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
from sklearn.linear_model import LinearRegression

import diisco.names as names

from diisco.constants import EPSILON

GUIDE_REGISTRY = {}


def register_guide(name):
    """
    Decorator to register a new guide.
    """

    def register_guide_cls(cls):
        if name in GUIDE_REGISTRY:
            raise ValueError("Cannot register duplicate guide ({})".format(name))
        if not issubclass(cls, Guide):
            raise ValueError(
                "Guide ({}: {}) must extend the Guide class".format(name, cls.__name__)
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

    def B_MEAN(self):
        """
        Should return the mean of the B distribution.
        at observed timepoints. The shaped of the returned tensor
        should be (n_timepoints, n_cell_types, 1)
        """
        raise NotImplementedError

    def _hypers_guide(self, hypers_init_vals, *args, **kwargs):
        """
        We use the same guide for the hyperparameters of every
        possible guide. Namely we use a MAP estimate of the
        hyperparameters.
        """
        init_lengthscale_w = torch.tensor(hypers_init_vals[names.LENGTHSCALE_W])
        w_lengthscale = pyro.param(
            names.LENGTHSCALE_W + "_param",
            init_lengthscale_w,
            constraint=constraints.positive,
        )
        w_lengthscale = pyro.sample(names.LENGTHSCALE_W, dist.Delta(w_lengthscale))


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
    B_MEAN = "b_mean"
    B_CHOLESKY_DIAG = "b_cholesky_diag"
    B_CHOLESKY_OFF_DIAG = "b_cholesky_off_diag"

    def __init__(
        self,
        n_cell_types,
        n_time_points,
        prior_f_mean,
        use_bias=False,
        lambda_matrix=None,
        hypers_init_vals=None,
        timepoints=None,
        proportions=None,
        *args,
        **kwargs
    ):
        # Check that the model has n_cell_types and n_timepoints
        # and raise an error if it doesn't.

        self.n_cell_types = n_cell_types
        self.n_timepoints = n_time_points
        self.prior_f_mean = prior_f_mean
        self.use_bias = use_bias
        self.hypers_init_vals = hypers_init_vals
        print("guide hyperparameters", self.hypers_init_vals)

        if (
            timepoints is not None
            and proportions is not None
            and lambda_matrix is not None
        ):
            init_W, init_B = self._init_params(timepoints, proportions, lambda_matrix)
            self.init_W = init_W
            self.init_B = init_B

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
        # Call the guide for the hyperparameters.
        # This is an effectfull
        self._hypers_guide(hypers_init_vals=self.hypers_init_vals)

        n_cell_types = self.n_cell_types
        n_timepoints = self.n_timepoints

        # How many parameters we should have for the cholesky decomposition
        # of the covariance matrix for each coordinate through time.
        n_cholesky_params_w = int(n_timepoints * (n_timepoints + 1) / 2)
        n_cholesky_params_b = int(n_timepoints * (n_timepoints + 1) / 2)
        n_cholesky_params_f = int(n_timepoints * (n_timepoints + 1) / 2)

        n_cholesky_diag_params_w = n_timepoints
        n_cholesky_diag_params_b = n_timepoints
        n_cholesky_diag_params_f = n_timepoints

        cholesky_params_w = pyro.param(
            self.W_CHOLESKY_OFF_DIAG,
            torch.zeros(n_cell_types, n_cell_types, n_cholesky_params_w),
        )
        cholesky_params_b = pyro.param(
            self.B_CHOLESKY_OFF_DIAG,
            torch.zeros(n_cell_types, 1, n_cholesky_params_b),
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
        cholesky_diag_params_b = pyro.param(
            self.B_CHOLESKY_DIAG,
            torch.abs(torch.randn(n_cholesky_diag_params_b)) * EPSILON,
            constraint=constraints.softplus_positive,
        )

        cholesky_w = vec_to_tril_matrix(cholesky_params_w)
        cholesky_f = vec_to_tril_matrix(cholesky_params_f)
        cholesky_b = vec_to_tril_matrix(cholesky_params_b)

        # Replace the diagonal with the cholesky diagonal parameters
        cholesky_w[
            :, :, torch.arange(n_timepoints), torch.arange(n_timepoints)
        ] = cholesky_diag_params_w
        cholesky_b[
            :, :, torch.arange(n_timepoints), torch.arange(n_timepoints)
        ] = cholesky_diag_params_b
        cholesky_f[
            :, torch.arange(n_timepoints), torch.arange(n_timepoints)
        ] = cholesky_diag_params_f

        covariances_w = torch.matmul(cholesky_w, cholesky_w.transpose(-1, -2))
        covariances_f = torch.matmul(cholesky_f, cholesky_f.transpose(-1, -2))
        covariances_b = torch.matmul(cholesky_b, cholesky_b.transpose(-1, -2))

        # Add small constant to the diagonal to make sure the
        # covariance matrix is positive definite
        covariances_f = covariances_f + torch.eye(n_timepoints) * EPSILON
        covariances_w = covariances_w + torch.eye(n_timepoints) * EPSILON
        covariances_b = covariances_b + torch.eye(n_timepoints) * EPSILON

        mean_w = pyro.param(
            self.W_MEAN,
            self.init_W
            + torch.randn(n_cell_types, n_cell_types, n_timepoints) * EPSILON**2,
        )
        mean_b = pyro.param(
            self.B_MEAN,
            self.init_B + torch.randn(n_cell_types, 1, n_timepoints) * EPSILON**2,
        )
        mean_f = pyro.param(self.F_MEAN, self.prior_f_mean).squeeze(-1)

        with pyro.plate("cell_types_outer_W", n_cell_types, dim=-2):
            with pyro.plate("cell_types_inner_W", n_cell_types, dim=-1):
                pyro.sample(names.W, dist.MultivariateNormal(mean_w, covariances_w))
        with pyro.plate("node_plate", n_cell_types, dim=-1):
            pyro.sample(names.F, dist.MultivariateNormal(mean_f, covariances_f))

        if self.use_bias:
            with pyro.plate("cell_types_outer_B", n_cell_types, dim=-2):
                with pyro.plate("single_inner_B", 1, dim=-1):
                    pyro.sample(names.B, dist.MultivariateNormal(mean_b, covariances_b))

    def _init_params(self, timepoints, proportions, lambda_matrix):
        """
        Performs a simple linear regression to initialize the
        parameters of the parameters of the guide.

        Params:
        -------
        timepoints: torch.Tensor
            shape: (n_timepoints, 1)
        proportions: torch.Tensor
            shape (n_timepoints, n_cell_types)
        lambda_matrix: torch.Tensor
            shape (n_cell_types, n_cell_types)

        Returns:
        --------
        init_W: torch.Tensor
            shape: (n_cell_types, n_cell_types, n_timepoints)
        init_B: torch.Tensor
            shape: (n_cell_types, n_timepoints)
        """
        n_cell_types = proportions.shape[1]
        n_timepoints = proportions.shape[0]

        init_W = torch.zeros(n_cell_types, n_cell_types, n_timepoints)
        init_B = torch.zeros(n_cell_types, n_timepoints)

        for cell_type in range(n_cell_types):
            # Fit a linear regression to initialize the parameters
            # of the guide.

            # We zero out the cell types so that the model
            # cant use them.
            x = proportions * lambda_matrix[cell_type, :]
            y = proportions[:, cell_type]
            model = LinearRegression(fit_intercept=self.use_bias)
            model.fit(x, y)

            # We have collinearity and these values could be anything
            # so we just set them to zero.
            w = torch.tensor(model.coef_)
            w = w.flatten() * lambda_matrix[cell_type, :].flatten()
            if self.use_bias:
                b = torch.tensor(model.intercept_)
                init_B[cell_type, :] = b

            init_W[cell_type, :, :] = w.unsqueeze(-1)

        init_B.unsqueeze_(1)
        assert init_W.shape == (n_cell_types, n_cell_types, n_timepoints)
        assert init_B.shape == (n_cell_types, 1, n_timepoints)

        return init_W, init_B

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

    @property
    def B_mean(self):
        params = pyro.get_param_store()
        B_mean = params[self.B_MEAN].permute(2, 0, 1)
        assert B_mean.shape == (self.n_timepoints, self.n_cell_types, 1)
        return B_mean


@register_guide("diagonal_normal")
class DiagonalNormal(Guide):
    """
    Guide for the DIISCO algorithm implementing a single
    normal distribution for each factor with diagonal covariance.
    """

    # Names of the parameters in the guide.
    F_MEAN = "f_mean"
    F_CHOLESKY_DIAG = "f_cholesky_diag"
    F_CHOLESKY_OFF_DIAG = "f_cholesky_off_diag"
    W_MEAN = "w_mean"
    W_CHOLESKY_DIAG = "w_cholesky_diag"
    W_CHOLESKY_OFF_DIAG = "w_cholesky_off_diag"
    B_MEAN = "b_mean"
    B_CHOLESKY_DIAG = "b_cholesky_diag"
    B_CHOLESKY_OFF_DIAG = "b_cholesky_off_diag"

    def __init__(
        self,
        n_cell_types,
        n_time_points,
        prior_f_mean,
        use_bias=False,
        lambda_matrix=None,
        hypers_init_vals=None,
        timepoints=None,
        proportions=None,
        *args,
        **kwargs
    ):
        # Check that the model has n_cell_types and n_timepoints
        # and raise an error if it doesn't.

        self.n_cell_types = n_cell_types
        self.n_timepoints = n_time_points
        self.prior_f_mean = prior_f_mean
        self.use_bias = use_bias
        self.hypers_init_vals = hypers_init_vals

        if (
            timepoints is not None
            and proportions is not None
            and lambda_matrix is not None
        ):
            init_W, init_B = self._init_params(timepoints, proportions, lambda_matrix)
            self.init_W = init_W
            self.init_B = init_B

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
        # Call the guide for the hyperparameters.
        # This is an effectfull function.
        self._hypers_guide(hypers_init_vals=self.hypers_init_vals)

        n_cell_types = self.n_cell_types
        n_timepoints = self.n_timepoints

        n_cholesky_diag_params_w = n_timepoints * n_cell_types**2
        n_cholesky_diag_params_b = n_timepoints * n_cell_types
        n_cholesky_diag_params_f = n_timepoints * n_cell_types

        cholesky_diag_params_w = (
            pyro.param(
                self.W_CHOLESKY_DIAG,
                torch.abs(torch.randn(n_cholesky_diag_params_w)) * EPSILON,
                constraint=constraints.softplus_positive,
            )
            + EPSILON
        )
        cholesky_diag_params_w = cholesky_diag_params_w.reshape(
            n_cell_types, n_cell_types, n_timepoints
        )

        cholesky_diag_params_f = (
            pyro.param(
                self.F_CHOLESKY_DIAG,
                torch.abs(torch.randn(n_cholesky_diag_params_f)) * EPSILON,
                constraint=constraints.softplus_positive,
            )
            + EPSILON
        )
        cholesky_diag_params_f = cholesky_diag_params_f.reshape(
            n_cell_types, n_timepoints
        )

        cholesky_diag_params_b = (
            pyro.param(
                self.B_CHOLESKY_DIAG,
                torch.abs(torch.randn(n_cholesky_diag_params_b)) * EPSILON,
                constraint=constraints.softplus_positive,
            )
            + EPSILON
        )
        cholesky_diag_params_b = cholesky_diag_params_b.reshape(
            n_cell_types, 1, n_timepoints
        )

        mean_w = pyro.param(
            self.W_MEAN,
            self.init_W
            + torch.randn(n_cell_types, n_cell_types, n_timepoints) * EPSILON**2,
        )
        mean_b = pyro.param(
            self.B_MEAN,
            self.init_B + torch.randn(n_cell_types, 1, n_timepoints) * EPSILON**2,
        )
        mean_f = pyro.param(self.F_MEAN, self.prior_f_mean).squeeze(-1)

        with pyro.plate("cell_types_outer_W", n_cell_types, dim=-2):
            with pyro.plate("cell_types_inner_W", n_cell_types, dim=-1):
                W = pyro.sample(
                    names.W,
                    dist.Normal(mean_w, cholesky_diag_params_w).to_event(1),
                )
                assert W.shape == (n_cell_types, n_cell_types, n_timepoints)

        with pyro.plate("node_plate", n_cell_types, dim=-1):
            F = pyro.sample(
                names.F, dist.Normal(mean_f, cholesky_diag_params_f).to_event(1)
            )
            assert F.shape == (n_cell_types, n_timepoints)

        if self.use_bias:
            with pyro.plate("cell_types_outer_B", n_cell_types, dim=-2):
                with pyro.plate("single_inner_B", 1, dim=-1):
                    B = pyro.sample(
                        names.B,
                        dist.Normal(mean_b, cholesky_diag_params_b).to_event(1),
                    )
                assert B.shape == (n_cell_types, 1, n_timepoints)

    def _init_params(self, timepoints, proportions, lambda_matrix):
        """
        Performs a simple linear regression to initialize the
        parameters of the parameters of the guide.

        Params:
        -------
        timepoints: torch.Tensor
            shape: (n_timepoints, 1)
        proportions: torch.Tensor
            shape (n_timepoints, n_cell_types)
        lambda_matrix: torch.Tensor
            shape (n_cell_types, n_cell_types)

        Returns:
        --------
        init_W: torch.Tensor
            shape: (n_cell_types, n_cell_types, n_timepoints)
        init_B: torch.Tensor
            shape: (n_cell_types, n_timepoints)
        """
        n_cell_types = proportions.shape[1]
        n_timepoints = proportions.shape[0]

        init_W = torch.zeros(n_cell_types, n_cell_types, n_timepoints)
        init_B = torch.zeros(n_cell_types, n_timepoints)

        for cell_type in range(n_cell_types):
            # Fit a linear regression to initialize the parameters
            # of the guide.

            # We zero out the cell types so that the model
            # cant use them.
            x = proportions * lambda_matrix[cell_type, :]
            y = proportions[:, cell_type]
            model = LinearRegression(fit_intercept=self.use_bias)
            model.fit(x, y)

            # We have collinearity and these values could be anything
            # so we just set them to zero.
            w = torch.tensor(model.coef_)
            w = w.flatten() * lambda_matrix[cell_type, :].flatten()
            if self.use_bias:
                b = torch.tensor(model.intercept_)
                init_B[cell_type, :] = b

            init_W[cell_type, :, :] = w.unsqueeze(-1)

        init_B.unsqueeze_(1)
        assert init_W.shape == (n_cell_types, n_cell_types, n_timepoints)
        assert init_B.shape == (n_cell_types, 1, n_timepoints)

        # TODO: We are not using linear regression to initialize the
        # parameters of the guide. We should remove the code here then.
        init_W = torch.randn(n_cell_types, n_cell_types, n_timepoints) * 0.1
        init_B = torch.randn(n_cell_types, 1, n_timepoints) * 0.1

        return init_W, init_B

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

    @property
    def B_mean(self):
        params = pyro.get_param_store()
        B_mean = params[self.B_MEAN].permute(2, 0, 1)
        assert B_mean.shape == (self.n_timepoints, self.n_cell_types, 1)
        return B_mean
