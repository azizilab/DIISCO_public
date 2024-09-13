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


Y_TO_PLOT_PER_SCENARIO = 4

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



def plot_and_save_datasets(
    data : dict[str, list[RunResults]],
    output_path: str,
    ):
    """
    Plots and saves the datasets to the output path.
    Each dataset has Y_TO_PLOT_PER_SCENARIO cells plotted.
    The cells are randomly selected for each scenario.
    """
    cell_colors = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"]
    for data_name, experiment_data in data.items():
        np.random.seed(0)

        standardized_observations = experiment_data[0].standardized_observations
        timepoints = np.array(experiment_data[0].timepoints)
        seed = experiment_data[0].seed

        standardized_observations = np.array(standardized_observations)
        n_timepoints, n_cells = len(timepoints), len(standardized_observations[0])
        cell_selection = np.random.choice(n_cells, Y_TO_PLOT_PER_SCENARIO, replace=False)

        fig, ax = plt.subplots(figsize=(10, 10))
        sorted_timepoints = np.argsort(timepoints)
        timepoints = timepoints[sorted_timepoints]

        # First plot the standardized observations
        for i, cell in enumerate(cell_selection):
            cell_color = cell_colors[i]
            cells = standardized_observations[sorted_timepoints, cell]
            ax.scatter(timepoints, cells, label=f"Cell {cell}", color=cell_color)

        # Now plot the predictions for the same cells
        for run in experiment_data:
            model_name = make_model_name(run)

            if "Diisco"  not in model_name:
                continue
            if run.seed != seed:
                continue

            predictions = np.array(run.predicted_observations)
            predictions = predictions[sorted_timepoints]
            for i, cell in enumerate(cell_selection):
                cell_color = cell_colors[i]
                ax.scatter(timepoints, predictions[:, cell], label=f"{model_name} Cell {cell}", color=cell_color, marker="x")


        # Add the legend
        ax.legend()

        ax.set_title(f"Standardized Observations for {data_name}")
        os.makedirs(output_path, exist_ok=True)

        ax.set_xlabel("Time")
        ax.set_ylabel("Standardized Observations")
        fig.savefig(os.path.join(output_path, f"{data_name}.pdf"))

def main():
    args = parse_args()
    data = get_data(args.job_dir)
    plot_and_save_datasets(data, args.out_dir)






if __name__ == "__main__":
    main()