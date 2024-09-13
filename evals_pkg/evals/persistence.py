from dataclasses import dataclass
import json
import numpy as np


@dataclass
class RunResults:
    """
    Contains all of the results from a single run of a model.
    These are the results that should be saved
    """

    # Metrics pertaining the values of y
    r2: float  # R2 score of the model
    mse: float  # Mean squared error of the model
    rmse: float
    mae: float

    # Metrics pertaining the interactions
    # All the lists have the same length and correspond to using a
    # threshold of that length
    r2_W: list[float]
    rmse_W: list[float]
    stds: list[float]
    accuracies: list[float]
    precisions: list[float]
    recalls: list[float]
    f1_scores: list[float]
    auc: float
    prc_auc: float
    symmetrical_auc: float
    symmetrical_prc_auc: float
    # holds the predictions of the interactions
    true_interactions: list[int]
    transformed_interactions: list[int]
    symmetrical_transformed_interactions: list[int]
    predicted_interactions: list[list[list[float]]]  # n_timepoints x n_cells x n_cells
    predicted_observations: list[list[float]]  # n_timepoints x n_cells

    # Metrics pertaining the dataset
    n_cells: int
    n_timepoints: int
    n_independent_per_block: int
    n_dependent_per_block: int
    noise: float
    p_active: float
    flip_prob_active: float
    flip_prob_inactive: float
    threshold_for_active: float
    seed: int

    # Dataset stuff
    weights: list[list[list[float]]] # n_timepoints x n_cells x n_cells
    standardized_observations : list[list[float]] # n_timepoints x n_cells
    observations : list[list[float]] # n_timepoints x n_cells
    timepoints : list[float] # n_timepoints

    # metric pertaining the model itself
    model_name: str
    config: dict
    run_name: str


def save_run_results(run_results: RunResults, path: str):
    """
    Saves the run results to a json file.
    Assumes that path doesn't have yet the extension
    """
    run_results = _convert_arrays_to_list(run_results)
    if not path.endswith(".json"):
        path += ".json"
    with open(path, "w") as f:
        json.dump(run_results.__dict__, f)


def load_run_results(path: str) -> RunResults:
    """
    Loads the run results from a json file
    """
    with open(path, "r") as f:
        data = json.load(f)
    return RunResults(**data)


def print_run_results(run_results: RunResults):
    bars = "*" * 80 + "\n"
    space_between_results = "\n \n"

    run_results = _convert_arrays_to_list(run_results)
    run_results_str = space_between_results
    run_results_str += bars
    run_results_str += "Run Results\n"
    run_results_str += bars
    for key, value in run_results.__dict__.items():
        # if its a list and its too long we don't print it
        if isinstance(value, list) and len(value) > 15:
            value = value[:15]

        run_results_str += f"{key}: {value}\n"
    run_results_str += bars
    run_results_str += space_between_results
    print(run_results_str)


def _convert_arrays_to_list(run_results: RunResults):
    """
    Ensures that there are no arrays in the dataclass
    """
    run_results_dict = run_results.__dict__
    for key, value in run_results_dict.items():
        if isinstance(value, np.ndarray):
            run_results_dict[key] = value.tolist()
    return RunResults(**run_results_dict)
