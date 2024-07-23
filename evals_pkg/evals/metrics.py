"""
Contains functions for computing metrics for evaluating how well a model is performing
on a particular dataset.
"""
from dataclasses import dataclass
from jaxtyping import Int, Float
from numpy import ndarray

from evals.discretization import AbsoluteValueDiscretizer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)

import numpy as np


MIN_STDS = 0.0
MAX_STDS = 5
N_STDS = int((MAX_STDS - MIN_STDS) * 2 + 1) # (0, 0.5, 1, 1.5, ..., 5)


@dataclass
class InteractionMetrics:
    """
    Contains the metrics for evaluating the performance of a model in predicting interactions.
    All metrics that depend on a threshold are computed for a range of thresholds
    depending on the discretization.
    """

    # The stds to use for discretization
    stds: Float[ndarray, " n_stds"]
    accuracies: Float[ndarray, " n_stds"]
    precisions: Float[ndarray, " n_stds"]
    recalls: Float[ndarray, " n_stds"]
    f1_scores: Float[ndarray, " n_stds"]
    auc: float
    prc_auc: float
    symmetrical_auc: float  # same as auc but we symmetrize the interactions
    symmetrical_prc_auc: float  # same as prc_auc but we symmetrize the interactions
    # The index of the best std according to the AUC
    best_std: int
    # True, transformed and non-transformed interactions
    # are not flattened
    true_interactions: list[int]
    transformed_interactions: list[int]
    symmetrical_transformed_interactions: list[int]



@dataclass
class ObservationMetrics:
    """
    Contains simple metrics for evaluating the performance of a model in predicting the observations.

    This is not the main objective of the model. Therefore it is ok to not
    do well here. However it serves as a sanity check and debugging
    tool
    """

    r2: float
    mse: float
    mae: float
    rmse: float


def evaluate_predicted_interactions(
    true_interactions: Int[ndarray, " n_timepoints n_cells n_cells"],
    interaction_score: Float[ndarray, " n_timepoints n_cells n_cells"],
    timepoints: Float[ndarray, " n_timepoints"],
    observations: Float[ndarray, " n_timepoints n_cells"],
) -> InteractionMetrics:
    """
    Computes several metrics to evaluate the performance of the model in predicting interactions.

    Parameters
    ----------
    interaction_score : np.ndarray
        A score indicating the strength of the interaction between cells.
        This is not yet passed through a discretizer.
    true_interactions : np.ndarray
        A boolean matrix (0 or 1) indicating whether the interaction is active
        at that timepoint or not.

    Returns
    -------
    metrics : InteractionMetrics
        A dataclass containing the metrics for evaluating the performance of the model.
        See the InteractionMetrics class for more information.
    """
    n_timepoints, n_cells, _ = interaction_score.shape
    assert true_interactions.shape == (n_timepoints, n_cells, n_cells)
    assert interaction_score.shape == true_interactions.shape
    assert timepoints.shape == (n_timepoints,)
    assert observations.shape == (n_timepoints, n_cells)

    # Compute the metrics
    accuracies = np.zeros(N_STDS)
    precisions = np.zeros(N_STDS)
    recalls = np.zeros(N_STDS)
    f1_scores = np.zeros(N_STDS)
    stds = np.linspace(MIN_STDS, MAX_STDS, N_STDS)

    for std_num, std in enumerate(stds):
        discretizer = AbsoluteValueDiscretizer(std_deviations=std)
        discretization = discretizer(timepoints, observations, interaction_score)

        accuracies[std_num] = accuracy_score(
            true_interactions.flatten(), discretization.flatten()
        )
        precisions[std_num] = precision_score(
            true_interactions.flatten(), discretization.flatten()
        )
        recalls[std_num] = recall_score(
            true_interactions.flatten(), discretization.flatten()
        )
        f1_scores[std_num] = f1_score(
            true_interactions.flatten(), discretization.flatten()
        )

    # Compute the AUC
    transformed_interactions = discretizer.transform_interactions(
        timepoints, observations, interaction_score
    )
    symmetrical_transformed_interactions = np.maximum(
        transformed_interactions, transformed_interactions.transpose(0, 2, 1)
    )
    auc = roc_auc_score(true_interactions.flatten(), transformed_interactions.flatten())
    prc_auc = average_precision_score(
        true_interactions.flatten(), transformed_interactions.flatten()
    )
    symmetrical_auc = roc_auc_score(
        true_interactions.flatten(), symmetrical_transformed_interactions.flatten()
    )
    symmetrical_prc_auc = average_precision_score(
        true_interactions.flatten(), symmetrical_transformed_interactions.flatten()
    )
    best_std = np.argmax(f1_scores)

    return InteractionMetrics(
        stds=stds,
        accuracies=accuracies,
        precisions=precisions,
        recalls=recalls,
        f1_scores=f1_scores,
        auc=auc,
        prc_auc=prc_auc,
        best_std=best_std,
        symmetrical_auc=symmetrical_auc,
        symmetrical_prc_auc=symmetrical_prc_auc,
        true_interactions=true_interactions.flatten().tolist(),
        transformed_interactions=transformed_interactions.flatten().tolist(),
        symmetrical_transformed_interactions= symmetrical_transformed_interactions.flatten().tolist()
    )



def evaluate_predicted_observations(
    true_observations: Float[ndarray, " n_timepoints n_cells"],
    predicted_observations: Float[ndarray, " n_timepoints n_cells"],
) -> ObservationMetrics:

    r2 = r2_score(true_observations.flatten(), predicted_observations.flatten())
    mse = mean_squared_error(
        true_observations.flatten(), predicted_observations.flatten()
    )
    mae = mean_absolute_error(
        true_observations.flatten(), predicted_observations.flatten()
    )
    rmse = np.sqrt(mse)

    return ObservationMetrics(r2=r2, mse=mse, mae=mae, rmse=rmse)
