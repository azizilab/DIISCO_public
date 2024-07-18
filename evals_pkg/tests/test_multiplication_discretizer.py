from evals.models import Model, LinearModel, RollingLinearModel
from evals.discretization import AbsoluteValueDiscretizer

import pytest
from jaxtyping import Float, Bool
import numpy as np
from numpy import ndarray
from sklearn.metrics import accuracy_score


@pytest.mark.parametrize(
    "model_class, linear_data, model_config",
    [
        (LinearModel, True, {}),
        (RollingLinearModel, True, {"min_points_per_regression": 20}),
        (RollingLinearModel, False, {"min_points_per_regression": 20}),
    ],
)
def test_sanity_models(model_class: Model, linear_data: bool, model_config: dict):
    n_timepoints = 100
    n_cells = 4
    std_deviations = 1
    count_zeros = True

    model = model_class(**model_config)
    t, y, is_active_truth, is_active_used = _generate_dummy_data(
        n_cells, n_timepoints, linear=linear_data
    )
    model.fit(t, y, is_active_used)

    obs_interactions = model.predict_obs_interactions()

    discretizer = AbsoluteValueDiscretizer(std_deviations, count_zeros)
    discretization = discretizer(t, y, obs_interactions)
    assert discretization.shape == (n_timepoints, n_cells, n_cells)

    is_active_truth = np.repeat(is_active_truth[None, ...], n_timepoints, axis=0)

    accuracy = accuracy_score(is_active_truth.flatten(), discretization.flatten())

    assert accuracy > 0.999, "Accuracy is too low : " + _get_error_msg(
        is_active_truth, discretization
    )


def _get_error_msg(
    is_active_truth: Bool[ndarray, "n_cells n_cells"],
    discretization: Bool[ndarray, "n_timepoints n_cells n_cells"],
) -> str:
    """
    Prints an descriptive error message for the test. If
    is_active_truth and discretization are not equal, it will
    return a string with the indices of the cells where the
    two matrices differ. Otherwise, it will return an empty string.
    """

    msg = ""
    for timepoint in range(discretization.shape[0]):
        for out_cell in range(discretization.shape[1]):
            for in_cell in range(discretization.shape[2]):
                if (
                    is_active_truth[timepoint, out_cell, in_cell]
                    != discretization[timepoint, out_cell, in_cell]
                ):
                    msg += f"Timepoint {timepoint}, cell {in_cell} -> cell {out_cell}\n"
                    msg += (
                        f"Expected: {is_active_truth[timepoint, out_cell, in_cell]}\n"
                    )
                    msg += f"Got: {discretization[timepoint, out_cell, in_cell]}\n \n"
    return msg


def _generate_dummy_data(
    n_cells: int,
    n_timepoints: int,
    linear: bool = True,
    seed: int = 0,
    noise: float = 0.05,
) -> tuple[Float[ndarray, " n_timepoints"], Float[ndarray, "n_timepoints n_cells"]]:
    """
    We generate dummy data with the following properties:
        - The only interactions are between cell 0 and cell 1
        - For all the other cells the interactions are zero
        - The time is uniformly sampled
        - The data is centered and scaled

    Parameters
    ----------
    n_cells : int
        Number of cells
    n_timepoints : int
        Number of timepoints
    linear : bool
        Whether the relationship between cell 0
        and cell 1 is linear or not

    Returns
    -------
    t : np.ndarray
        The time points at which the data was sampled
    y : np.ndarray
        The observed values of the cells
    is_active_true: np.ndarray
        A matrix containing 1 if the edge is active, 0 otherwise. This
        the actual true matrix of interactions
    is_active_used: np.ndarray
        A matrix containing 1 if the edge is active, 0 otherwise.
        This is the matrix of interactions that the model should use when training.
        It is much more sparse than is_active_true
    """
    np.random.seed(seed)

    t = np.linspace(0, 1, n_timepoints) + 3
    y = np.random.randn(n_timepoints, n_cells) * noise
    y[:, 0] = y[:, 0] + t

    # The only interactions are between cell 0 and cell 1
    y[:, 1] = 2 * y[:, 0] if linear else (2 * y[:, 0]) ** 2 * t

    # Add noise
    y += noise * np.random.randn(n_timepoints, n_cells)

    # Center
    y = (y - np.mean(y, axis=0)) / np.std(y, axis=0)

    # Create the is_active matrix
    is_active_true = np.zeros((n_cells, n_cells))
    is_active_true[0, 1] = 1
    is_active_true[1, 0] = 1

    # Create the is_active matrix
    # We zero out the diagonal and the first column
    # for the ones where it is not used to avoid colinearity issues
    is_active_used = np.ones((n_cells, n_cells)) - np.eye(n_cells)
    is_active_used[2:, 0] = 0

    return t, y, is_active_true, is_active_used
