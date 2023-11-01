import torch
from functools import partial
from diisco.math_ops import rbf_kernel
from diisco.gaussian_process import GaussianProcessRegressor


class PriorFModel:
    def __init__(self, length_scale, variance, sigma_y):
        self.length_scale = length_scale
        self.variance = variance
        self.sigma_y = sigma_y

        self.kernel = partial(
            rbf_kernel, length_scale=length_scale, variance=variance
        )
        self.models = None

        self.is_fitted = False

    def fit(self, timepoints, proportions):
        """
        Fit the model to the data.
        :param timepoints: torch.tensor of shape (n_samples, 1)
        :param proportions: torch.tensor of shape (n_samples, n_proportions)
        :return:
        """
        n_samples, n_proportions = proportions.shape
        if not timepoints.shape == (n_samples, 1):
            raise ValueError(
                "The timepoints must be a column vector of shape (n_samples, 1)."
            )

        self.models = []
        for target in proportions.T:
            model = self._fit_target(timepoints, target)
            self.models.append(model)

        self.is_fitted = True

    def _fit_target(self, timepoints, target):
        """
        Fit a single target.
        :param timepoints: torch.tensor of shape (n_samples, 1)
        :param target: torch.tensor of shape (n_samples,)
        :return:
        """
        model = GaussianProcessRegressor(self.kernel, self.sigma_y)

        # Make sure that the target is a column vector
        target = target.reshape(-1, 1)
        model.fit(timepoints, target)
        return model

    def predict(self, timepoints):
        """
        Predict the latent features for the given timepoints.
        :param timepoints: torch.tensor of shape (n_samples, 1)
        :return: means, covariances of shape (n_targets, n_samples, 1)
            and (n_targets, n_samples, n_samples)
        """
        assert self.is_fitted

        n_samples, n_targets = len(timepoints), len(self.models)
        means = torch.zeros(n_targets, n_samples, 1)
        covariances = torch.zeros(n_targets, n_samples, n_samples)

        for i, model in enumerate(self.models):
            mean, covariance = model.predict(timepoints)
            means[i] = mean
            covariances[i] = covariance

        assert means.shape == (n_targets, n_samples, 1)
        assert covariances.shape == (n_targets, n_samples, n_samples)
        return means, covariances
