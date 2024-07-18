from evals.generate_data import (
    generate_data,
    SampledData,
    compute_reachability_matrix,
    create_true_interactions_from_dependent_computations,
)

# import linear regression from sklearn
from sklearn.linear_model import LinearRegression
import numpy as np


def test_generate_data():
    np.random.seed(0)

    n_blocks = 2
    n_indepenent_per_block = 2
    n_dependent_per_block = 1
    n_timepoints = 100
    noise = 0.001
    length_scale = 100  # should be like linear regression
    seed = 0
    p_active = 0.1
    flip_prob_active = 0.1
    flip_prob_inactive = 0.1

    sampled_data: SampledData = generate_data(
        n_blocks=n_blocks,
        n_indepenent_per_block=n_indepenent_per_block,
        n_dependent_per_block=n_dependent_per_block,
        n_timepoints=n_timepoints,
        noise=noise,
        length_scale=length_scale,
        seed=seed,
        p_active=p_active,
        flip_prob_active=flip_prob_active,
        flip_prob_inactive=flip_prob_inactive,
    )

    # check that the data is generated correctly
    block_size = n_indepenent_per_block + n_dependent_per_block
    total_cells = n_blocks * block_size

    # check the shapes
    assert sampled_data.weights.shape == (n_timepoints, total_cells, total_cells)
    assert sampled_data.true_interactions.shape == (
        n_timepoints,
        total_cells,
        total_cells,
    )
    assert sampled_data.observations.shape == (n_timepoints, total_cells)
    assert sampled_data.standardized_observations.shape == (n_timepoints, total_cells)
    assert sampled_data.timepoints.shape == (n_timepoints,)
    assert sampled_data.model_prior.shape == (total_cells, total_cells)

    # Check that the weight matrix is similar across all timepoints
    for i in range(total_cells):
        for j in range(total_cells):
            assert np.std(sampled_data.weights[:, i, j]) < 0.1

    # Check each cell is roughtly the result of what it should be
    for t in range(n_timepoints):
        for i in range(total_cells):
            predicted_val = (
                sampled_data.weights[t, i].flatten() @ sampled_data.observations[t]
            )
            msg = f"Predicted value {predicted_val} is too far from the observed value {sampled_data.observations[t, i]}"
            msg += f" at timepoint {t} and cell {i}"
            assert np.abs(predicted_val - sampled_data.observations[t, i]) < 0.01, msg


def test_matrix_reachability():
    """
    Simple sanity check to make sure that function using
    reachability matrices works.
    """

    # First three nodes are connected in a line
    # Fourth node is not connected to anything
    adj_matrix = np.array([[1, 1, 0, 0], [1, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 1]])

    true_reach_matrix = np.array(
        [[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0], [0, 0, 0, 1]]
    )

    reach_matrix = compute_reachability_matrix(adj_matrix)
    print(reach_matrix)
    print(true_reach_matrix)
    assert np.all(reach_matrix == true_reach_matrix)


def test_create_true_interactions_from_dependent_computations():
    w_matrix = np.array(
        [
            [
                [1, 0, 0, 0],  # Independent
                [1, 0, 1, 0],  # Depends on 0 and 2
                [0, 0, 1, 0],  # Independent
                [0, 0, 0, 1],  # Independent
            ],
            [
                [1, 0, 0, 0],  # Independent
                [1, 0, 0.01, 0],  # Depends on 0 (2 is too small)
                [0, 0, 1, 0],  # Independent
                [0, 0, 0, 1],  # Independent
            ],
        ]
    )

    true_interaction_matrix = np.array(
        [
            [
                [0, 1, 1, 0],  # Independent
                [1, 0, 1, 0],  # Depends on 0 and 2
                [1, 1, 0, 0],  # Independent
                [0, 0, 0, 0],  # Independent
            ],
            [
                [0, 1, 0, 0],  # Independent
                [1, 0, 0, 0],  # Depends on 0 (2 is too small)
                [0, 0, 0, 0],  # Independent
                [0, 0, 0, 0],  # Independent
            ],
        ]
    )

    interaction_matrix = create_true_interactions_from_dependent_computations(
        w_matrix, threshold=0.5
    )
    assert np.all(interaction_matrix == true_interaction_matrix)
