from evals.models import Model, LinearModel, RollingLinearModel
import pytest
from jaxtyping import Float, Bool
import numpy as np
from numpy import ndarray
from sklearn.metrics import r2_score


@pytest.mark.parametrize(
    "model_class, linear_data, model_config",
    [
        (LinearModel, True, {}),
        (RollingLinearModel, True, {"min_points_per_regression": 20, "use_bias": False}),
        (RollingLinearModel, False, {"min_points_per_regression": 20, "use_bias": False}),
    ],
)
def test_sanity_models(model_class: Model, linear_data: bool, model_config: dict):
    n_timepoints = 100
    n_cells = 4

    model = model_class(**model_config)
    t, y, is_active_true, is_active_used = _generate_dummy_data(n_cells, n_timepoints, linear=linear_data)

    model.fit(t, y, is_active_used)

    # Check that the predictions have the right shape
    y_train_pred = model.predict_y_train()
    assert y_train_pred.shape == (n_timepoints, n_cells)

    # Check that predictions are good for the first two cells
    y_pred_0 = y_train_pred[:, 0].flatten()
    y_true_0 = y[:, 0].flatten()
    assert r2_score(y_true_0, y_pred_0) > 0.9

    y_pred_1 = y_train_pred[:, 1].flatten()
    y_true_1 = y[:, 1].flatten()
    assert r2_score(y_true_1, y_pred_1) > 0.9

    # Check that the interactions have the correct shape
    interactions = model.predict_interactions(t, y)
    assert interactions.shape == (n_timepoints, n_cells, n_cells)

    # Check that the interactions at known timepoints have the correct shape
    obs_interactions = model.predict_obs_interactions()
    assert obs_interactions.shape == (n_timepoints, n_cells, n_cells)


    # Check that the interactions are correct
    # the only interactions that should be non-zero are between cell 0 and cell 1
    threshold = 0.5
    for timepoint in range(n_timepoints):
        for out_cell in range(n_cells):
            for in_cell in range(n_cells):
                interaction_coeff = interactions[timepoint, out_cell, in_cell]

                if is_active_used[out_cell, in_cell] == 0:
                    assert interaction_coeff == 0, f"Expected 0 but got {interaction_coeff}"

                msg = f"Observed interaction between cell {out_cell} and cell {in_cell} "
                msg += f"at timepoint {timepoint} is {interaction_coeff}, but expected {is_active_true[out_cell, in_cell]}"
                if is_active_true[out_cell, in_cell] == 1:
                    assert np.abs(interaction_coeff) > threshold, msg
                else :
                    assert np.abs(interaction_coeff) < threshold, msg


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
