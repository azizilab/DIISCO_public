from dataclasses import dataclass
import json


@dataclass
class RunResults:
    """
    Contains all of the results from a single run of a model.
    These are the results that should be saved
    """

    # Metrics pertaining the values of y
    r2_y: float  # R2 score of the model
    mse_y: float  # Mean squared error of the model

    # Metrics pertaining the interactions
    f1_score_interactions: float
    accuracy_interactions: float
    precision_interactions: float
    recall_interactions: float
    roc_auc_interactions: float

    # Metrics pertaining the dataset
    n_cells: int
    n_timepoints: int
    n_independent_per_block: int
    noise: float

    # metric pertaining the model itself
    model_name: str
    config: dict
    run_name: str


def save_run_results(run_results: RunResults, path: str):
    """
    Saves the run results to a json file
    """
    with open(path, "w") as f:
        json.dump(run_results.__dict__, f)


def load_run_results(path: str) -> RunResults:
    """
    Loads the run results from a json file
    """
    with open(path, "r") as f:
        data = json.load(f)
    return RunResults(**data)
