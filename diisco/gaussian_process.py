"""
Simple module containing a poorman's version of a Gaussian Process Regression.
The module is based on the implementation of the Gaussian Process Regression
as described in the book "Gaussian Processes for Machine Learning" by
Carl Edward Rasmussen and Christopher K. I. Williams. The implementation
follows Algorithm 2.1  on page 19.
book: http://gaussianprocess.org/gpml/chapters/RW.pdf
"""

import torch

from diisco.utils import make_psd


class GaussianProcessRegressor:
    def __init__(self, kernel, sigma_y=1e-6):
        self.kernel = kernel
        self.sigma_y = sigma_y
        self.noise = sigma_y**2

        self._fit = False
        self.X = None
        self.y = None
        self.L = None
        self.alpha = None

    def fit(self, X, y):
        """
        Fit the Gaussian Process Regression model to the data.
        :param X: The input data. shape (n_samples, n_features)
        :param y: The output data. shape (n_samples, n_targets)
        """
        self.X = X
        self.y = y
        K = self.kernel(X, X) + self.noise * torch.eye(X.shape[0])

        self.L = torch.linalg.cholesky(K)
        self.alpha = torch.cholesky_solve(y, self.L)  # L.T @ L @ alpha = y

    def predict(self, X):
        """
        Predict the output for the given input data.
        :param X: The input data. shape (n_samples, n_features)
        :return: The predicted output
            :mean: The mean of the predicted output. shape (n_samples, n_targets)
            :var: The variance of the predicted output. shape (n_samples, n_samples)
        """
        K_star = self.kernel(self.X, X)
        assert K_star.shape[0] == self.X.shape[0]
        K_star_t = K_star.T
        mean = K_star_t @ self.alpha
        assert mean.shape[0] == X.shape[0]
        v = torch.cholesky_solve(K_star, self.L)
        var = self.kernel(X, X) - K_star_t @ v
        var = make_psd(var)

        assert mean.shape == (X.shape[0], self.y.shape[1])
        assert var.shape == (X.shape[0], X.shape[0])
        return mean, var

    def sample(self, X, n_samples=1, with_cov=False):
        """
        Sample the output for the given input data.
        :param X: The input data. shape (n_samples, n_features)
        :param n_samples: The number of samples to draw. Default is 1.
        :return: The sampled output. shape (n_samples, n_targets, dim_targets)
        """
        mean, var = self.predict(X)
        assert mean.shape == (X.shape[0], self.y.shape[1])
        assert var.shape == (X.shape[0], X.shape[0])
        if not with_cov:
            dig_var = torch.diag(var)
            var = torch.diag(dig_var)
            var.clamp_(min=0)
        var = var + self.noise * torch.eye(var.shape[0])
        samples = torch.distributions.MultivariateNormal(
            mean.squeeze(), var
        ).sample((n_samples,))
        samples = samples.view(n_samples, X.shape[0], self.y.shape[1])
        assert samples.shape == (
            n_samples,
            X.shape[0],
            self.y.shape[1],
        ), samples.shape
        return samples
