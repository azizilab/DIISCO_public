"""
Code for generating a table for the paper.

Assumption is that the job-dir has the structure of the following:
    job-dir/
        experiment1/
            result_1.json
            result_2.json
        experiment2/
            result_1.json
            result_2.json
        ...
"""
import argparse
from evals.persistence import load_run_results, RunResults, print_run_results
import os
import numpy as np
from sklearn.metrics import roc_curve
from matplotlib import pyplot as plt
import re


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--job-dir",
        type=str,
        required=True,
        help="Path to the job directory",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Path to the output file",
    )
    return parser.parse_args()

def get_data(job_dir: str) -> dict[str,list[RunResults]]:
    """
    Returns
    -------
    List of list of RunResults. Each entry in the outter list corresponds to an experiment.
    each entry in the inner list corresponds to a run in that experiment.
    """
    data = {}
    for experiment in os.listdir(job_dir):
        experiment_path = os.path.join(job_dir, experiment)
        runs = []
        for run in os.listdir(experiment_path):
            run_path = os.path.join(experiment_path, run)
            run_results = load_run_results(run_path)
            runs.append(run_results)
        data[experiment] = runs
    return data


def make_model_name(run: RunResults) -> str:
    model_name = run.model_name
    if model_name == "DiiscoModel":
        model_name += "_lr_" + str(run.config["lr"])
    else:
        if run.config["ignore_is_active"] == True:
            model_name += ""
        else:
            model_name += " w/ Prior"

    # Add a space before each capital letter
    model_name = " ".join(re.findall('[A-Z][^A-Z]*', model_name))
    return model_name

def format_name(name: str) -> str:
    name = name.replace("_", " ")
    # capitalize first letter
    name = name[0].upper() + name[1:]
    return name

def make_roc_curves(
    data: dict[str,list[RunResults]]
    )->  dict[str, dict[str, (list[float], list[float])]]:

    """
    Constructs an ROC curve for each model in each experiment.
    The same seed is used for all models in the same experiment.

    Parameters
    ----------
    data: dict[str,list[RunResults]]
        The data to be used for the table.
        The key is the experiment name and the value is a list of RunResults.
    """

    # First we construct a nice table using dictionaries
    rows = {} # {experiment_name: {model_name: (fpr, tpr)}}

    for experiment_name, experiment_data in data.items():
        model_dict = {} # {model_identifier: {metric: [values]}}
        seed = experiment_data[0].seed # only one seed per experiment

        for run in experiment_data:
            if run.seed != seed:
                continue  # ignore run so that all have the same seed

            model_name = make_model_name(run)
            true_interactions = run.true_interactions
            symmetrical_transformed_interactions = run.symmetrical_transformed_interactions
            fpr, tpr, _ = roc_curve(true_interactions, symmetrical_transformed_interactions)
            model_dict[model_name] = (fpr, tpr)

        rows[experiment_name] = model_dict
    return rows

def plot_and_save_roc_curves(
    rows: dict[str, dict[str, (list[float], list[float])]],
    output_path: str,
    ):
    """
    Makes one plot per experiment.
    """

    for experiment_name, experiment_data in rows.items():
        fig, ax = plt.subplots()
        for model_name, (fpr, tpr) in experiment_data.items():
            ax.plot(fpr, tpr, label=model_name)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curves for {format_name(experiment_name)}")
        ax.legend()

        # save to pdf
        os.makedirs(output_path, exist_ok=True)
        fig.savefig(os.path.join(output_path, f"{experiment_name}.pdf"))


def main():
    args = parse_args()
    data = get_data(args.job_dir)
    rows = make_roc_curves(data)
    plot_and_save_roc_curves(rows, args.out_dir)




if __name__ == "__main__":
    main()