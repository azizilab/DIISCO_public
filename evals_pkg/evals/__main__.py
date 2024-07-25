"""
Main code for evaluating the model
"""

import argparse
from evals.models import get_models_dict
from evals.models import Model
from evals.generate_data import SampledData, generate_data
from evals.persistence import (
    RunResults,
    save_run_results,
    load_run_results,
    print_run_results,
)
from evals.utils import get_default_parameters
from evals.metrics import (
    evaluate_predicted_interactions,
    InteractionMetrics,
    evaluate_predicted_observations,
    ObservationMetrics,
)
from evals.models import DiiscoModel

import os
import namesgenerator
import numpy as np
import uuid
import torch


# import torch


TRAIN_SIZE = 0.5


def parse_args() -> argparse.Namespace:
    """
    See command line arguments from the script
    for more information on the arguments
    """
    parser = argparse.ArgumentParser(description="Description of your script.")

    # --------------------------------------------------
    # General arguments: Positional
    # --------------------------------------------------

    parser.add_argument(
        "job_dir", type=str, help="Where to save the results of the evaluation"
    )

    # --------------------------------------------------
    # General arguments: Not positional
    # --------------------------------------------------

    msg = "Experiment name: If none empty a directory with this name will be created"
    msg += "in the job-dir to save the results of the experiment."
    parser.add_argument("--name", type=str, default="", help=msg)

    # --------------------------------------------------
    # Arguments for generating data
    # --------------------------------------------------

    msg = "Numbers of  independent groups of cells interacting"
    msg += "with each other"
    parser.add_argument("--n-blocks", type=int, default=3, help=msg)

    msg = "The number of cells that are generated by a separate GP process and don't"
    msg += "depend on any other cell per block"
    parser.add_argument("--n-independent", type=int, default=3, help=msg)

    msg = "The number of cells in each block that depend on other cells in the block"
    parser.add_argument("--n-dependent", type=int, default=3, help=msg)

    msg = "The number of timepoints to generate data for"
    parser.add_argument("--n-timepoints", type=int, default=100, help=msg)

    msg = "The amount of noise to add to the data"
    parser.add_argument("--noise", type=float, default=0.01, help=msg)

    msg = "The length scale of the GP kernel"
    parser.add_argument("--length-scale", type=float, default=100, help=msg)

    msg = "The length scale of the weight matrix"
    parser.add_argument("--weights-length-scale", type=float, default=100, help=msg)

    msg = "The seed to use for generating the data"
    parser.add_argument("--seed", type=int, default=0, help=msg)

    msg = "The probability that an interaction between an independent and dependent cell is active"
    msg += "At least one interaction must be active always"
    parser.add_argument("--p-active", type=float, default=0.1, help=msg)

    msg = "The probability that an active interaction becomes inactive"
    parser.add_argument("--flip-prob-active", type=float, default=0.1, help=msg)

    msg = "The probability that an inactive interaction becomes active"
    parser.add_argument("--flip-prob-inactive", type=float, default=0.1, help=msg)

    msg = "The threshold for an interaction to be considered active"
    parser.add_argument("--threshold", type=float, default=0.1, help=msg)

    msg = "Whether to not save the results of the experiment"
    parser.add_argument("--no-save", action="store_true", help=msg)

    msg = "Any additional hyper-parameters for the model to be used. "
    msg += "Should be in the form <parameter-name> <parameter-value>"
    parser.add_argument("--model-parameters", nargs="*", help=msg)

    # --------------------------------------------------
    # Arguments for evaluating the model
    # --------------------------------------------------
    msg = "Which model to use for the experiment"
    parser.add_argument(
        "--model",
        type=str,
        choices=list(get_models_dict().keys()),
        default=list(get_models_dict().keys())[0],
        help=msg,
    )

    args = parser.parse_args()
    return args


def generate_run_name() -> str:
    """
    Create a random name for the run
    The name of the file to be saved is given by a random name generator
    of the form "<adjective>-<noun>", e.g. "cute-babage"+ a random uuid.
    """
    run_name = namesgenerator.get_random_name()
    uuid_ = uuid.uuid4().hex
    return f"{run_name}-{uuid_}"


def create_and_return_save_pth(job_dir: str, experiment_name: str, run_name) -> str:
    """
    Create a directory to save the results of the experiment.

    If name is not empty, a directory with this name will be created in
    the job dir to save the results of the experiment. Otherwise, the job dir
    will be used to save the results of the experiment.

    Function ensure that all directories in the path are created. The save pth
    does not include the file extension.
    """
    if experiment_name != "":
        save_pth = os.path.join(job_dir, experiment_name)
    else:
        save_pth = job_dir

    os.makedirs(save_pth, exist_ok=True)

    save_pth = os.path.join(save_pth, run_name)
    return save_pth


def make_model_config(model_cls: Model, args: argparse.Namespace) -> dict:
    """
    Gets the model configuration, including default parameters
    and any neccesary extra configurations that depend on args

    Returns
    -------
    config: dict
        The configuration for the model.
    """
    config = get_default_parameters(model_cls)

    # Check if the class is a DiiscoModel
    if model_cls == DiiscoModel:
        assert "w_length_scale" in config
        config["w_length_scale"] = args.weights_length_scale

        assert "y_length_scale" in config
        config["y_length_scale"] = args.length_scale

        assert "y_sigma" in config
        config["y_sigma"] = args.noise /2

    # Add additional model parameters
    if args.model_parameters is not None:
        for i in range(0, len(args.model_parameters), 2):
            # Assume that the model parameters are in the form --<parameter-name> <parameter-value>
            key = args.model_parameters[i]
            value = args.model_parameters[i + 1]
            assert key in config, f"Parameter {key} not found in the model configuration"
            value_type = type(config[key])
            if value_type == bool:
                config[key] = value == "True"
            else:
                config[key] = value_type(value)

    return config


def set_seed(seed: int):
    """
    Set the seed for numpy and pytorch
    """
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    args = parse_args()
    set_seed(args.seed)

    dataset: SampledData = generate_data(
        n_blocks=args.n_blocks,
        n_independent_per_block=args.n_independent,
        n_dependent_per_block=args.n_dependent,
        n_timepoints=args.n_timepoints,  # We will split the data into train and test
        noise=args.noise,
        length_scale=args.length_scale,
        weights_length_scale=args.weights_length_scale,
        seed=args.seed,
        p_active=args.p_active,
        flip_prob_active=args.flip_prob_active,
        flip_prob_inactive=args.flip_prob_inactive,
        threshold_for_active=args.threshold,
    )

    model_cls = get_models_dict()[args.model]
    config = make_model_config(model_cls, args)
    model = model_cls(**config)

    model.fit(
        t=dataset.timepoints, y=dataset.observations, is_active=dataset.model_prior
    )

    interaction_score = model.predict_obs_interactions()
    interaction_metrics: InteractionMetrics = evaluate_predicted_interactions(
        true_interactions=dataset.true_interactions,
        interaction_score=interaction_score,
        observations=dataset.observations,
        timepoints=dataset.timepoints,
    )
    # print("\n\ninteraction_score", interaction_score.astype(int)[0])
    # print("\n\ntrue_interactions", dataset.true_interactions.astype(int)[0])

    true_observations = dataset.observations
    predicted_observations = model.predict_y_train()

    # print("\n\ntrue_observations", true_observations)
    # print("\n\npredicted_observations", predicted_observations)

    observation_metrics: ObservationMetrics = evaluate_predicted_observations(
        true_observations=true_observations,
        predicted_observations=predicted_observations,
    )

    run_name = generate_run_name()
    save_pth = create_and_return_save_pth(args.job_dir, args.name, run_name=run_name)

    run_results = RunResults(
        # Metrics pertaining the values of y
        r2=observation_metrics.r2,
        mse=observation_metrics.mse,
        rmse=observation_metrics.rmse,
        mae=observation_metrics.mae,
        # Metrics pertaining the interactions
        stds=interaction_metrics.stds,
        accuracies=interaction_metrics.accuracies,
        precisions=interaction_metrics.precisions,
        recalls=interaction_metrics.recalls,
        f1_scores=interaction_metrics.f1_scores,
        auc=interaction_metrics.auc,
        prc_auc=interaction_metrics.prc_auc,
        symmetrical_auc=interaction_metrics.symmetrical_auc,
        symmetrical_prc_auc=interaction_metrics.symmetrical_prc_auc,
        true_interactions=interaction_metrics.true_interactions,
        transformed_interactions=interaction_metrics.transformed_interactions,
        symmetrical_transformed_interactions=interaction_metrics.symmetrical_transformed_interactions,
        # Metrics pertaining the dataset
        n_cells=dataset.observations.shape[1],
        n_timepoints=dataset.observations.shape[0],
        n_independent_per_block=args.n_independent,
        n_dependent_per_block=args.n_dependent,
        noise=args.noise,
        p_active=args.p_active,
        flip_prob_active=args.flip_prob_active,
        flip_prob_inactive=args.flip_prob_inactive,
        threshold_for_active=args.threshold,
        seed=args.seed,
        # Dataset stuff
        weights=dataset.weights.tolist(),
        standardized_observations=dataset.standardized_observations.tolist(),
        observations=dataset.observations.tolist(),
        timepoints=dataset.timepoints.tolist(),
        # Stuff pertaining the model itself
        model_name=args.model,
        config=config,
        run_name=run_name,
    )
    if not args.no_save:
        save_run_results(run_results, save_pth)
    print_run_results(run_results)


if __name__ == "__main__":
    main()
