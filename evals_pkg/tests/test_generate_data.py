from evals.generate_data import generate_data, SampledData
# import linear regression from sklearn
from sklearn.linear_model import LinearRegression
import numpy as np



def test_generate_data():
    np.random.seed(0)

    n_blocks = 2
    n_indepenent_per_block = 2
    n_dependent_per_block = 1
    n_timepoints = 100
    noise = 0.0001
    length_scale = 100 # should be like linear regression
    seed = 0
    p_active = 0.1

    sampled_data : SampledData = generate_data(
        n_blocks=n_blocks,
        n_indepenent_per_block=n_indepenent_per_block,
        n_dependent_per_block=n_dependent_per_block,
        n_timepoints=n_timepoints,
        noise=noise,
        length_scale=length_scale,
        seed=seed,
        p_active=p_active
    )

    # check that the data is generated correctly
    block_size = n_indepenent_per_block + n_dependent_per_block
    total_cells = n_blocks * block_size

    # check the shapes
    assert sampled_data.weights_matrix.shape == (n_timepoints, total_cells, total_cells)
    assert sampled_data.is_active_matrix.shape == (n_timepoints, total_cells, total_cells)
    assert sampled_data.observed_matrix.shape == (n_timepoints, total_cells)
    assert sampled_data.timepoints.shape == (n_timepoints,)

    # Check that is_active_matrix is the same across all timepoints
    assert np.all(sampled_data.is_active_matrix == sampled_data.is_active_matrix[0])

    # Check that the weight matrix is similar across all timepoints
    for i in range(total_cells):
        for j in range(total_cells):
            assert np.std(sampled_data.weights_matrix[:, i, j]) < 0.1

    # Check each cell is the result of what it should be roughtly
    for t in range(n_timepoints):
        for i in range(total_cells):
            predicted_val = (sampled_data.weights_matrix[t, i].flatten() @ sampled_data.observed_matrix[t])
            assert np.abs(predicted_val - sampled_data.observed_matrix[t, i]) < 0.1