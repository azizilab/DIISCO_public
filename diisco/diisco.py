"""
Contains the main class for the DIISCO algorithm.
"""


from functools import partial
from typing import Callable, Dict
from collections import defaultdict

import pyro
import pyro.distributions as dist
import torch
from pyro import poutine
from pyro.distributions import constraints
from tqdm import tqdm

import diisco.constants as constants
import diisco.names as names
import diisco.guides as guides
from diisco.gaussian_process import GaussianProcessRegressor
from diisco.math_ops import rbf_kernel, is_psd
from diisco.utils import shape_and_rate_from_range
from diisco.prior_f_model import PriorFModel


class DIISCO:
    default_hypers_to_optim = [
        names.LENGTHSCALE_W,
    ]
    all_hypers = [
        names.SIGMA_Y,
        names.SIGMA_W,
        names.SIGMA_F,
        names.LENGTHSCALE_W,
        names.LENGTHSCALE_F,
        names.VARIANCE_W,
        names.VARIANCE_F,
    ]

    def __init__(
        self,
        lambda_matrix: torch.Tensor,
        hypers_init_vals: dict = None,
        use_bias: bool = False,
        verbose: bool = False,
        verbose_freq: int = 100,
    ):
        """
        Initialize the DIISCO model.
        :param lambda_matrix: A matrix of shape (n_cell_types, n_cell_types) that
            contains the values used for regularization in the kernel matrix.
        :param use_bias: Whether to use a bias term in the model. The bias term
            will have the hyper-parameters as those used for the W matrix.
        :param hypers_init_vals: A dictionary that contains the initial values for
            the hyperparameters. If any hyperparameter is not specified the default
            value will be used. See the constants module for default value and valid
            hyperparameter names.
        :param verbose: Whether to print the loss during training.
        :param verbose_freq: How often to print the loss during the training run.
        """
        self.model = None
        self.guide = None
        self.svi = None
        self.losses = []
        self.device = None  # Currently not used

        self.lambda_matrix = lambda_matrix
        self.use_bias = use_bias

        self.train_timepoints = None
        self.train_proportions = None

        hypers_init_vals = hypers_init_vals or {}
        self._check_user_hypers_init_vals(hypers_init_vals)
        self.hypers_init_vals = self._get_full_hypers_init_vals(hypers_init_vals)

        self.prior_model = None

        self._prior_is_set = False  # Flag to check if the prior has been set
        self._is_fit = False  # Flag to check if the model has been fit
        self.prior_f_mean = None  # The means to use for the emission model
        self.prior_f_cov = (
            None  # The covariance to use for the variational distribution
        )

        self.verbose = verbose
        self.verbose_freq = verbose_freq
        self.sub_size = None

    def fit(
        self,
        timepoints: torch.Tensor,
        proportions: torch.Tensor,
        n_iter: int = 1000,
        lr: float = 0.1,
        guide: str = "MultivariateNormal",
        subsample_size: int = None,
        hypers_to_optim: dict = None,
    ):
        """
        This function fits the DIISCO model fully to the data. It is
        equivalent to normalizing, setting the prior, and then running variational
        inference on the model.
        :param timepoints: A tensor of shape (n_timepoints, 1) that contains
            the timepoints at which the cell proportions were measured.
        :param proportions: A tensor of shape (n_timepoints, n_cell_types) that
             contains the cell proportions at each timepoint.
        :param n_iter: The number of iterations to run variational inference for.
        :param lr: The learning rate to use for variational inference.
        :param guide: The guide to use for variational inference. Currently only
            "MultivariateNormal" and "MultivariateNormalFactorized" and "DiagonalNormal"
            are supported.
        :param hypers_to_optimize: A list of hyperparameters to optimize to optimize
            using variational inference. This feature is very experimental and it
            probably best left unused.
        """
        self.subsample_size = subsample_size

        self.train_timepoints = timepoints
        self.train_proportions = proportions

        self.n_cell_types = proportions.shape[1]
        self.n_timepoints = proportions.shape[0]

        pyro.clear_param_store()

        self.fit_and_set_f_prior_params(
            timepoints=timepoints,
            proportions=proportions,
            hypers=self.hypers_init_vals,
        )
        self.n_cell_types = proportions.shape[1]
        self.n_timepoints = proportions.shape[0]

        hypers_to_optim = hypers_to_optim or constants.DEFAULT_HYPERS_TO_OPTIM
        all_hypers = constants.DEFAULT_HYPERS.keys()
        hypers_to_block = [h for h in all_hypers if h not in hypers_to_optim]
        self.model = pyro.poutine.block(self._model, hide=hypers_to_block)

        self.guide = self._get_guide(guide, self.model)
        self.svi = pyro.infer.SVI(
            model=self.model,
            guide=self.guide,
            optim=pyro.optim.Adam({"lr": lr}),
            loss=pyro.infer.Trace_ELBO(),
        )

        self.losses = []
        self._is_fit = True
        for i in range(n_iter):
            loss = self.svi.step(timepoints, proportions)
            loss = loss / self.n_timepoints * self.n_cell_types * self.n_cell_types
            self.losses.append(loss)
            if self.verbose and i % self.verbose_freq == 0:
                print("[iteration %04d] loss: %.4f" % (i + 1, loss))

    def _get_model_hypers(self) -> dict:
        """
        Runs the sample sites associated with the model hyperparameters
        and returns a dictionary with the values of the hyperparameters.

        Importantly, this function is effectful.
        """
        sigma_w_init = torch.tensor(self.hypers_init_vals[names.SIGMA_W])
        sigma_w = pyro.param(
            names.SIGMA_W, sigma_w_init, constraint=constraints.positive
        )
        sigma_y_init = torch.tensor(self.hypers_init_vals[names.SIGMA_Y])
        sigma_y = pyro.param(
            names.SIGMA_Y, sigma_y_init, constraint=constraints.positive
        )
        w_variance_init = torch.tensor(self.hypers_init_vals[names.VARIANCE_W])
        w_variance = pyro.param(
            names.VARIANCE_W, w_variance_init, constraint=constraints.positive
        )

        w_lenghtscale_range = self.hypers_init_vals[names.LENGTHSCALE_W_RANGE]
        w_lengthscale_mean = self.hypers_init_vals[names.LENGTHSCALE_W]
        # We will sample from a gamma and use a normal approximation
        # so that 90% of the samples are within the range.
        upper_bound = w_lengthscale_mean + w_lenghtscale_range / 2
        lower_bound = w_lengthscale_mean - w_lenghtscale_range / 2
        alpha, beta = shape_and_rate_from_range(lower_bound, upper_bound, 0.9)
        w_length_scale = pyro.sample(names.LENGTHSCALE_W, dist.Gamma(alpha, beta))

        return {
            names.SIGMA_W: sigma_w,
            names.SIGMA_Y: sigma_y,
            names.LENGTHSCALE_W: w_length_scale,
            names.VARIANCE_W: w_variance,
        }

    def _model(self, timepoints, proportions=None):
        """
        Model for the DIISCO algorithm.
        :param timepoints: Tensor of timepoints for the samples.
             shape: (n_timepoints, 1)
        :param proportions: Tensor of proportions for the samples.
            shape: (n_timepoints, n_cell_types)
        """
        # We use a dictionary to store the parameters of the model
        # so that they can be easily accessed later as a return value.
        n_cell_types = self.n_cell_types
        n_timepoints = self.n_timepoints

        hypers = self._get_model_hypers()

        sigma_w = hypers[names.SIGMA_W]
        sigma_y = hypers[names.SIGMA_Y]
        lengthscale_w = hypers[names.LENGTHSCALE_W]
        variance_w = hypers[names.VARIANCE_W]
        w_covariance = rbf_kernel(
            timepoints,
            timepoints,
            length_scale=lengthscale_w,
            variance=variance_w,
        )
        f_covariance = self.prior_f_cov
        f_mean = self.prior_f_mean

        assert is_psd(w_covariance), w_covariance

        with pyro.plate("cell_types_outer_W", n_cell_types, dim=-2):
            with pyro.plate("cell_types_inner_W", n_cell_types, dim=-1):
                w_covariance = w_covariance + torch.eye(n_timepoints) * sigma_w**2
                W = pyro.sample(
                    names.W,
                    dist.MultivariateNormal(torch.zeros(n_timepoints), w_covariance),
                    infer=dict(
                        baseline={
                            "use_decaying_avg_baseline": True,
                            "baseline_beta": 0.95,
                        }
                    ),
                )
                assert W.shape == (n_cell_types, n_cell_types, n_timepoints)
        assert self.lambda_matrix.shape == (n_cell_types, n_cell_types)
        W = W * self.lambda_matrix.unsqueeze(-1)

        if self.use_bias:
            b_covariance = rbf_kernel(
                timepoints,
                timepoints,
                length_scale=lengthscale_w,
                variance=variance_w,
            )
            b_covariance = b_covariance + torch.eye(n_timepoints) * sigma_w**2
            with pyro.plate("cell_types_outer_B", n_cell_types, dim=-2):
                with pyro.plate("single_inner_B", 1, dim=-1):
                    B = pyro.sample(
                        names.B,
                        dist.MultivariateNormal(
                            torch.zeros(n_timepoints), b_covariance
                        ),
                    )
                    assert B.shape == (n_cell_types, 1, n_timepoints)

        with pyro.plate("node_plate", n_cell_types, dim=-1):
            assert f_mean.shape == (n_cell_types, n_timepoints, 1)
            assert f_covariance.shape == (
                n_cell_types,
                n_timepoints,
                n_timepoints,
            )
            f_mean = f_mean.squeeze(-1)
            f = pyro.sample(
                names.F,
                dist.MultivariateNormal(f_mean, f_covariance),
                infer=dict(
                    baseline={
                        "use_decaying_avg_baseline": True,
                        "baseline_beta": 0.95,
                    }
                ),
            )
            assert f.shape == (n_cell_types, n_timepoints)

        subsample_size = self.subsample_size if self.subsample_size else None
        subsample_size = subsample_size if proportions is not None else None

        with pyro.plate(
            "data_plate_outer",
            n_timepoints,
            dim=-2,
            subsample_size=subsample_size,
        ) as ind:
            with pyro.plate("data_plate_inner", n_cell_types, dim=-1):
                W = W.permute(2, 0, 1)  # reorder for batch matrix multiplication
                f = f.permute(1, 0)  # reorder for batch matrix multiplication
                means = torch.bmm(W, f.unsqueeze(-1)).squeeze(-1)
                assert means.shape == (n_timepoints, n_cell_types)

                if self.use_bias:
                    B = B.permute(2, 0, 1).squeeze(-1)
                    means = means + B

                if subsample_size:
                    proportions = proportions[ind]
                    means = means[ind]
                y = pyro.sample(names.Y, dist.Normal(means, sigma_y), obs=proportions)

    def score(
        self,
        timepoints: torch.Tensor,
        proportions: torch.Tensor,
        samples: int = 1000,
    ) -> float:
        """
        Score the model on the test data. Higher is better.
        :param timepoints: Tensor of timepoints for the samples.
            shape: (n_timepoints, 1)
        :param proportions: Tensor of proportions for the samples.
            shape: (n_timepoints, n_cell_types)
        """
        n_cell_types = self.n_cell_types
        n_timepoints = timepoints.shape[0]
        assert timepoints.shape == (n_timepoints, 1)
        assert proportions.shape == (n_timepoints, n_cell_types)

        samples = self.sample(timepoints, samples)
        proportions = proportions.unsqueeze(0)

        return torch.mean(torch.sum((samples - proportions) ** 2, dim=-1)).item()

    def sample_observed_proportions(self, n_samples: int = 1000) -> torch.Tensor:
        """
        Sample proportions to the model at the observed timepoints.
        To do this the model uses exclusively the parameters learned
        from variational inference and does not use a GP to interpolate.
        """
        y = []
        for _ in range(n_samples):
            guide_trace = poutine.trace(self.guide).get_trace(self.train_timepoints)
            trained_model = poutine.replay(self.model, trace=guide_trace)
            trained_model_trace = poutine.trace(trained_model).get_trace(
                self.train_timepoints
            )
            y.append(trained_model_trace.nodes[names.Y]["value"])
        return torch.stack(y)

    def sample_f_prior(
        self, timepoints: torch.Tensor, n_samples: int = 1000
    ) -> torch.Tensor:
        """
        We sample f from the prior models learned from the data.
        :param timepoints: Tensor of timepoints for the samples.
            shape: (n_timepoints, 1)
        :param n_samples: Number of samples to draw from the prior.

        :return: Tensor of shape (n_samples, n_cell_types, n_timepoints)
        """
        if self.prior_f_mean is None or self.prior_f_cov is None:
            raise ValueError("The prior model must be fit before sampling.")

        n_cell_types, _, _ = self.prior_f_mean.shape
        n_timepoints = timepoints.shape[0]

        mean, cov = self.prior_model.predict(timepoints)
        distribution = dist.MultivariateNormal(mean.squeeze(-1), cov)
        samples = distribution.sample((n_samples,))
        assert samples.shape == (n_samples, n_cell_types, n_timepoints)
        return samples

    def sample(
        self,
        timepoints: torch.Tensor,
        n_samples: int = 1000,
        n_samples_per_latent: int = 1,
        include_emission_variance: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict the proportions for the given timepoints.
        :param timepoints: Tensor of timepoints for the samples.
            shape: (n_timepoints, 1)
        :param n_samples: Number of samples to draw from the posterior.
        :param sample_reuse_rate: Number of times to reuse the same
            latent sample. This is helpful for speeding up the sampling
            process. The higher the number the faster the sampling but
            the less accurate the results.
        Returns:
            A  dictionary with the samples for each parameter. The keys are:
            names.W, names.F, names.Y and the values are tensors of shape
                (n_samples, n_timepoints, n_cell_types, n_cell_types),
                (n_samples, n_timepoints, n_cell_types, 1)
                (n_samples, n_timepoints, n_cell_types)
                respectively.
        """
        # This function samples the steps from the posterior according to the
        # following algorithm (see the paper for more details):
        # 1. Use the variational family to sample W, B, and F at observed timepoints
        # 2. Use the GP to obtain samples P(W_unobserved | W_observed) and P(B_unobserved | B_observed)
        # 3. Use a GP to obtain samples P(F_unobserved | F_observed). Importantly, this GP
        #    has a mean and covariance that is not zero (because we originally condition it on the observed
        #    samples so the new prior is equal to the original posterior). To achieve the same effect we sample
        #    P(F_unobserved | F_latent_observed, F_real_observed) and then we sample F_latent_unobserved.
        # 4. Use the sampled W, B, and F to sample Y.
        # 5. Repeat steps 2-4 n_samples_per_latent times.
        samples = defaultdict(list)
        pbar = tqdm(range(n_samples))

        samples_so_far = 0
        while samples_so_far < n_samples:
            guide_trace = poutine.trace(self.guide).get_trace(self.train_timepoints)
            trained_model = poutine.replay(self._model, trace=guide_trace)
            model_trace = poutine.trace(trained_model).get_trace(self.train_timepoints)

            W_sampled = model_trace.nodes[names.W]["value"]
            f_sampled = model_trace.nodes[names.F]["value"]
            B_sampled = model_trace.nodes[names.B]["value"] if self.use_bias else None

            w_length_scale = model_trace.nodes[names.LENGTHSCALE_W]["value"]
            w_variance = model_trace.nodes[names.VARIANCE_W]["value"]

            sigma_y = model_trace.nodes[names.SIGMA_Y]["value"]
            sigma_w = model_trace.nodes[names.SIGMA_W]["value"]

            if not include_emission_variance:
                sigma_y = torch.ones_like(sigma_y) * 0.001

            assert W_sampled.shape == (
                self.n_cell_types,
                self.n_cell_types,
                self.n_timepoints,
            )
            assert f_sampled.shape == (self.n_cell_types, self.n_timepoints)
            if self.use_bias:
                assert B_sampled.shape == (
                    self.n_cell_types,
                    1,
                    self.n_timepoints,
                )

            W_predict = torch.zeros(
                (
                    n_samples_per_latent,
                    self.n_cell_types,
                    self.n_cell_types,
                    timepoints.shape[0],
                )
            )
            # Compute P(W_unobserved | W_observed)
            for i in range(self.n_cell_types):
                for j in range(self.n_cell_types):
                    gpr_w_kernel = partial(
                        rbf_kernel,
                        length_scale=w_length_scale,
                        variance=w_variance,
                    )

                    regressor = GaussianProcessRegressor(gpr_w_kernel, sigma_y=sigma_w)
                    train_timepoints = self.train_timepoints.view(-1, 1)
                    train_w = W_sampled[i, j, :].view(-1, 1)
                    timepoints = timepoints.view(-1, 1)

                    assert train_timepoints.shape == (self.n_timepoints, 1)
                    assert train_w.shape == (self.n_timepoints, 1)
                    regressor.fit(train_timepoints, train_w)
                    w_sample = regressor.sample(
                        timepoints, n_samples=n_samples_per_latent
                    )
                    assert w_sample.shape == (
                        n_samples_per_latent,
                        timepoints.shape[0],
                        1,
                    ), w_sample.shape

                    W_predict[:, i, j, :] = w_sample.squeeze(-1)

            # Compute P(B_unobserved | B_observed)
            if self.use_bias:
                B_predict = torch.zeros(
                    (
                        n_samples_per_latent,
                        self.n_cell_types,
                        1,
                        timepoints.shape[0],
                    )
                )
                for i in range(self.n_cell_types):
                    gpr_b_kernel = partial(
                        rbf_kernel,
                        length_scale=w_length_scale,
                        variance=w_variance,
                    )

                    regressor = GaussianProcessRegressor(gpr_b_kernel, sigma_y=sigma_w)
                    train_timepoints = self.train_timepoints.view(-1, 1)
                    train_b = B_sampled[i, :, :].view(-1, 1)
                    timepoints = timepoints.view(-1, 1)

                    assert train_timepoints.shape == (self.n_timepoints, 1)
                    assert train_b.shape == (self.n_timepoints, 1)
                    regressor.fit(train_timepoints, train_b)
                    b_sample = regressor.sample(
                        timepoints, n_samples=n_samples_per_latent
                    )
                    assert b_sample.shape == (
                        n_samples_per_latent,
                        timepoints.shape[0],
                        1,
                    ), b_sample.shape
                    B_predict[:, i, :, :] = b_sample.squeeze(-1)

            # Sample the fs at the unobserved timepoints by updating the prior
            all_train_timepoints = torch.cat(
                [self.train_timepoints, self.train_timepoints],
                dim=0,
            )
            all_observed_proportions = torch.cat(
                [self.train_proportions, f_sampled.T],
                dim=0,
            )

            prior_updated_model = PriorFModel(
                self.hypers_init_vals[names.LENGTHSCALE_F],
                self.hypers_init_vals[names.VARIANCE_F],
                self.hypers_init_vals[names.SIGMA_F],
            )
            prior_updated_model.fit(
                all_train_timepoints,
                all_observed_proportions,
            )
            f_mean, f_cov = prior_updated_model.predict(timepoints)
            delta_diag = torch.eye(timepoints.shape[0]).unsqueeze(0)
            f_cov = (
                # f_cov + delta_diag * self.hypers_init_vals[names.SIGMA_F] ** 2
                f_cov
                + delta_diag * 0.01
            )
            f_predict = dist.MultivariateNormal(f_mean.squeeze(-1), f_cov).sample(
                (n_samples_per_latent,)
            )
            assert f_predict.shape == (
                n_samples_per_latent,
                self.n_cell_types,
                timepoints.shape[0],
            )

            W_predict = W_predict * self.lambda_matrix.unsqueeze(-1)
            W_predict = W_predict.permute(0, 3, 1, 2)
            assert W_predict.shape == (
                n_samples_per_latent,
                timepoints.shape[0],
                self.n_cell_types,
                self.n_cell_types,
            )
            f_predict = f_predict.permute(0, 2, 1).unsqueeze(-1)
            assert f_predict.shape == (
                n_samples_per_latent,
                timepoints.shape[0],
                self.n_cell_types,
                1,
            ), "f_predict.shape: {}".format(f_predict.shape)

            if self.use_bias:
                B_predict = B_predict.permute(0, 3, 1, 2)
                assert B_predict.shape == (
                    n_samples_per_latent,
                    timepoints.shape[0],
                    self.n_cell_types,
                    1,
                ), "B_predict.shape: {}".format(B_predict.shape)

            y_predict_mean = torch.matmul(W_predict, f_predict).squeeze(-1)
            y_predict_mean = (
                y_predict_mean + B_predict.squeeze(-1)
                if self.use_bias
                else y_predict_mean
            )
            y_predict_cov = torch.eye(self.n_cell_types) * sigma_y**2
            y_predict_cov = y_predict_cov.unsqueeze(0)

            y_sampled = dist.MultivariateNormal(
                y_predict_mean,
                y_predict_cov,
            ).sample()

            samples[names.W].append(W_predict)
            samples[names.F].append(f_predict)
            samples[names.Y].append(y_sampled)
            if self.use_bias:
                samples[names.B].append(B_predict)

            # Update the progress bar
            samples_so_far += n_samples_per_latent
            pbar.update(n_samples_per_latent)
        pbar.close()

        # Convert the samples to tensors and get rid of
        # the extra samples
        for key in samples:
            samples[key] = torch.cat(samples[key], dim=0)

        return samples

    def fit_and_set_f_prior_params(
        self,
        timepoints: torch.Tensor,
        proportions: torch.Tensor,
        hypers: dict,
    ) -> None:
        """
        Set the prior for the model for f by fitting a GP to the
        proportions.

        :param timepoints: Tensor of timepoints for the samples.
            shape: (n_timepoints, 1)
        :param proportions: Tensor of proportions for the samples.
            shape: (n_timepoints, n_cell_types)
        """
        length_scale = hypers[names.LENGTHSCALE_F]
        variance = hypers[names.VARIANCE_F]
        sigma_f = hypers[names.SIGMA_F]
        self.prior_model = PriorFModel(
            length_scale=length_scale, variance=variance, sigma_y=sigma_f
        )
        self.prior_model.fit(timepoints, proportions)

        self.prior_f_mean, self.prior_f_cov = self.prior_model.predict(timepoints)
        self._prior_is_set = True

    def get_means(self, timepoints: torch.Tensor) -> Dict[str, torch.Tensor]:
        n_train_timepoints = self.train_timepoints.shape[0]
        n_cell_types = self.n_cell_types

        guide_trace = poutine.trace(self.guide).get_trace(self.train_timepoints)
        trained_model = poutine.replay(self._model, trace=guide_trace)
        model_trace = poutine.trace(trained_model).get_trace(self.train_timepoints)

        W_sampled = self.guide.W_mean
        assert W_sampled.shape == (
            n_train_timepoints,
            n_cell_types,
            n_cell_types,
        )

        f_sampled = self.guide.F_mean
        assert f_sampled.shape == (n_train_timepoints, n_cell_types, 1)

        w_length_scale = model_trace.nodes[names.LENGTHSCALE_W]["value"]
        w_variance = model_trace.nodes[names.VARIANCE_W]["value"]

        sigma_w = model_trace.nodes[names.SIGMA_W]["value"]

        W_predict = torch.zeros((timepoints.shape[0], n_cell_types, n_cell_types))
        for i in range(self.n_cell_types):
            for j in range(self.n_cell_types):
                gpr_w_kernel = partial(
                    rbf_kernel,
                    length_scale=w_length_scale,
                    variance=w_variance,
                )

                regressor = GaussianProcessRegressor(gpr_w_kernel, sigma_y=sigma_w)
                train_timepoints = self.train_timepoints.view(-1, 1)
                train_w = W_sampled[:, i, j].view(-1, 1)
                timepoints = timepoints.view(-1, 1)

                assert train_timepoints.shape == (self.n_timepoints, 1)
                assert train_w.shape == (self.n_timepoints, 1)
                regressor.fit(train_timepoints, train_w)
                w_predicted_mean, _ = regressor.predict(timepoints)
                assert w_predicted_mean.shape == (timepoints.shape[0], 1)

                W_predict[:, i, j] = w_predicted_mean.squeeze()

        # Compute P(B_unobserved | B_observed)
        B_predict = torch.zeros((timepoints.shape[0], n_cell_types, 1))
        if self.use_bias:
            for i in range(self.n_cell_types):
                gpr_b_kernel = partial(
                    rbf_kernel,
                    length_scale=w_length_scale,
                    variance=w_variance,
                )

                regressor = GaussianProcessRegressor(gpr_b_kernel, sigma_y=sigma_w)
                train_timepoints = self.train_timepoints.view(-1, 1)
                B_sampled = self.guide.B_mean
                train_b = B_sampled[:, i].view(-1, 1)
                timepoints = timepoints.view(-1, 1)

                assert train_timepoints.shape == (self.n_timepoints, 1)
                assert train_b.shape == (self.n_timepoints, 1)
                regressor.fit(train_timepoints, train_b)
                b_predicted_mean, _ = regressor.predict(timepoints)
                assert b_predicted_mean.shape == (timepoints.shape[0], 1)

                B_predict[:, i] = b_predicted_mean

        # Sample the fs at the unobserved timepoints by updating the prior
        all_train_timepoints = torch.cat(
            [self.train_timepoints, self.train_timepoints],
            dim=0,
        )

        all_observed_proportions = torch.cat(
            [self.train_proportions, f_sampled.squeeze(-1)],
            dim=0,
        )

        prior_updated_model = PriorFModel(
            self.hypers_init_vals[names.LENGTHSCALE_F],
            self.hypers_init_vals[names.VARIANCE_F],
            self.hypers_init_vals[names.SIGMA_F],
        )
        prior_updated_model.fit(
            all_train_timepoints,
            all_observed_proportions,
        )
        f_predict, _ = prior_updated_model.predict(timepoints)
        f_predict = f_predict.permute(1, 0, 2)

        W_predict = W_predict * self.lambda_matrix.unsqueeze(0)
        y_predict_mean = torch.matmul(W_predict, f_predict).squeeze(-1)
        if self.use_bias:
            print("B_predict.shape: {}".format(B_predict.shape))
            print("y_predict_mean.shape: {}".format(y_predict_mean.shape))
            y_predict_mean = y_predict_mean + B_predict.squeeze(-1)
        means = {
            names.W: W_predict.detach(),
            names.F: f_predict.squeeze(-1).detach(),
            names.Y: y_predict_mean.detach(),
            names.B: B_predict.detach() if self.use_bias else None,
        }
        return means

    def get_W_mean(self, n_samples: int = 1) -> torch.Tensor:
        """
        Estimate the mean of W by sampling from the posterior.
        Returned tensor:
            shape:(n_cell_types, n_cell_types, n_timepoints)
        """
        W_samples = []
        for _ in range(n_samples):
            guide_trace = poutine.trace(self.guide).get_trace(self.train_timepoints)
            trained_model = poutine.replay(self._model, trace=guide_trace)
            model_trace = poutine.trace(trained_model).get_trace(self.train_timepoints)
            W_sampled = model_trace.nodes[names.W]["value"]
            W_sampled = W_sampled * self.lambda_matrix.unsqueeze(-1)
            W_samples.append(W_sampled)

        W_mean = torch.stack(W_samples).mean(0)
        W_mean = W_mean.permute(2, 0, 1)
        assert W_mean.shape == (
            self.n_timepoints,
            self.n_cell_types,
            self.n_cell_types,
        )
        return W_mean

    def get_f_mean(self, n_samples: int = 1) -> torch.Tensor:

        f_samples = []
        for _ in range(n_samples):
            guide_trace = poutine.trace(self.guide).get_trace(self.train_timepoints)
            trained_model = poutine.replay(self._model, trace=guide_trace)
            model_trace = poutine.trace(trained_model).get_trace(self.train_timepoints)
            f_sampled = model_trace.nodes[names.F]["value"]
            f_samples.append(f_sampled)

        f_mean = torch.stack(f_samples).mean(0)
        f_mean = f_mean.permute(1, 0)
        assert f_mean.shape == (self.n_timepoints, self.n_cell_types)
        return f_mean

    def _check_user_hypers_init_vals(self, hypers_init_vals: dict) -> None:
        """
        Makes sure that only the hyperparameters that are specified in the model are
        passed to the model.
        """
        # Check that a dictionary is passed
        if not isinstance(hypers_init_vals, dict):
            raise TypeError("hypers_init_vals must be a dictionary.")

        # Check that the keys in the dictionary are valid
        keys_in_model = set(constants.DEFAULT_HYPERS.keys())
        keys_in_init_vals = set(hypers_init_vals.keys())

        if not keys_in_init_vals.issubset(keys_in_model):
            raise ValueError(
                "The keys in hypers_init_vals must be a subset of the keys in the model."
            )

    def _get_full_hypers_init_vals(self, hypers_init_vals: dict) -> dict:
        """
        Returns a dictionary with the hyperparameters that are specified in the model.
        If the hyperparameters are not specified in the model, they are set to None.
        """
        hypers = {}
        for key in constants.DEFAULT_HYPERS.keys():
            default_val = constants.DEFAULT_HYPERS[key]
            hypers[key] = hypers_init_vals.get(key, default_val)
        return hypers

    def _get_guide(self, guide_name: str, model: Callable) -> Callable:
        """
        Returns the guide function corresponding to the name passed.
        """
        if guide_name == "AutoNormal":
            return pyro.infer.autoguide.AutoDiagonalNormal(model)
        if guide_name == "MultivariateNormal":
            return pyro.infer.autoguide.AutoMultivariateNormal(model)
        elif guide_name == "MultivariateNormalFactorized":
            return guides.MultivariateNormalFactorized(
                self.n_cell_types,
                self.n_timepoints,
                self.prior_f_mean,
                use_bias=self.use_bias,
                lambda_matrix=self.lambda_matrix,
                hypers_init_vals=self.hypers_init_vals,
                timepoints=self.train_timepoints,
                proportions=self.train_proportions,
            )
        elif guide_name == "DiagonalNormal":
            return guides.DiagonalNormal(
                self.n_cell_types,
                self.n_timepoints,
                self.prior_f_mean,
                use_bias=self.use_bias,
                lambda_matrix=self.lambda_matrix,
                hypers_init_vals=self.hypers_init_vals,
                timepoints=self.train_timepoints,
                proportions=self.train_proportions,
            )
        else:
            raise ValueError(
                "The guide must be either 'MultivariateNormal' or 'MultivariateNormalFactorized'or 'DiagonalNormal'."
            )
