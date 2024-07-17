from jaxtyping import Float
from numpy import ndarray

def get_percentage_contributions(
    w_matrix: Float[ndarray, "n_timepoints n_cells n_cells"],
    cell_obs: Float[ndarray, "n_timepoints n_cells"],
    independent_contributions: Float[ndarray, "n_timepoints n_cells"] = None,
) -> Float[ndarray, "n_timepoints n_cells n_cells"]:
    """
    Returns for every coordinate at every timepoint the percentage contribution of that
    coordinate to the total value to predict. In other words if
    value_to_predict[t, i, j] == 0.5
    this means that at time t removing the contribution of coordinate j to predict the value of i
    would result in reduction of 50% of the value to predict.
    """
    assert w_matrix.ndim == 3
    n_timepoints, n_cells, _ = w_matrix.shape
    assert cell_obs.shape == (n_timepoints, n_cells)
    assert independent_contributions is None or independent_contributions.shape == (n_timepoints, n_cells)


    # (time, out_cell, in_cell1 * in_cell2)
    coordinate_contributions = w_matrix * cell_obs[:, None, :]
    assert coordinate_contributions.shape == (n_timepoints, n_cells, n_cells)

    # The value to predict is cell_obs or cell_obs - independent_contributions
    # depending on whetehr the regression has a constant value or not.
    value_to_predict = cell_obs if independent_contributions is None else cell_obs - independent_contributions
    assert value_to_predict.shape == (n_timepoints, n_cells)

    # Get the per coordinate contribution over total value to predict
    percentage_contributions = coordinate_contributions / value_to_predict[:, :, None]
    assert percentage_contributions.shape == (n_timepoints, n_cells, n_cells)
    return percentage_contributions