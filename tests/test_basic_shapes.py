from diisco import DIISCO
import diisco.names as names
import pytest
import torch


@pytest.mark.parametrize(
    "use_bias, guide",
    [
        (True, "MultivariateNormal"),
        (True, "MultivariateNormalFactorized"),
        (True, "DiagonalNormal"),
        (False, "MultivariateNormal"),
        (False, "MultivariateNormalFactorized"),
        (False, "DiagonalNormal"),
    ],
)
def test_basic_shapes_work(use_bias, guide):
    """
    Simple test to match that there are no errors running
    the code with basic shapes under a simple variety of
    conditions
    """
    # Add determinism
    torch.manual_seed(0)

    n_timepoints = 3
    n_cell_types = 2

    lambda_matrix = torch.eye(n_cell_types)
    timepoints = (
        torch.arange(n_timepoints).reshape(-1, 1) + torch.randn(n_timepoints, 1) * 0.1
    )
    cell_types = torch.randn(n_timepoints, n_cell_types)
    n_iter = 2

    model = DIISCO(lambda_matrix=lambda_matrix, use_bias=use_bias, verbose=False)

    model.fit(timepoints, cell_types, n_iter=n_iter, guide=guide)

    # Check that the samples are generated with the
    # correct shapes
    n_samples = 3
    samples = model.sample(timepoints, n_samples=n_samples)
    W_samples = samples[names.W]
    F_samples = samples[names.F]
    B_samples = samples[names.B]
    Y_samples = samples[names.Y]

    assert W_samples.shape == (n_samples, n_timepoints, n_cell_types, n_cell_types)
    assert F_samples.shape == (n_samples, n_timepoints, n_cell_types, 1)
    assert Y_samples.shape == (n_samples, n_timepoints, n_cell_types)
    if use_bias:
        assert B_samples.shape == (n_samples, n_timepoints, n_cell_types, 1)

    # Check that the predictions are generated with the
    # correct shapes
    # TODO: Add support for additional guides.
    if guide == "MultivariateNormalFactorized":
        means = model.get_means(timepoints)
        W_means = means[names.W]
        F_means = means[names.F]
        B_means = means[names.B]
        Y_means = means[names.Y]

        assert W_means.shape == (n_timepoints, n_cell_types, n_cell_types)
        assert F_means.shape == (n_timepoints, n_cell_types, 1)
        assert Y_means.shape == (n_timepoints, n_cell_types)
        if use_bias:
            assert B_means.shape == (n_timepoints, n_cell_types, 1)
