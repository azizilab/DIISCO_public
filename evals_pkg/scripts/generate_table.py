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

METRICS = [
    "AUC",
    "F1",
    "Prec",
    "Rec",
]

# Some functions that take the metric
# and return the function to be used for the extraction
# from the RunResults object
extract_funcs = {
    "AUC": lambda run: run.symmetrical_auc,
    # get the one with best f1 score
    "F1": lambda run: run.f1_scores[np.argmax(run.f1_scores)],
    "Prec": lambda run: run.precisions[np.argmax(run.f1_scores)],
    "Rec": lambda run: run.recalls[np.argmax(run.f1_scores)],
}

METRIC_TO_CAPTION = {
    "AUC": "AUC",
    "F1": "F1 Score",
    "Prec": "Precision",
    "Rec": "Recall",
}



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--job-dir",
        type=str,
        required=True,
        help="Path to the job directory",
    )
    #parser.add_argument(
    #    "--output-path",
    #    type=str,
    #    required=True,
    #    help="Path to the output file",
    #)
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
        model_name +=  str(run.config["ignore_is_active"])
    return model_name

def format_name(name: str) -> str:
    return name.replace("_", " ")

def make_dicts_for_table(
    data: dict[str,list[RunResults]]
    )->  dict[str, dict[str, dict[str, list[float]]]]:

    """
    Constructs two dictionaries that will be used to generate the table.
    See the return value for more information.

    Parameters
    ----------
    data: dict[str,list[RunResults]]
        The data to be used for the table.
        The key is the experiment name and the value is a list of RunResults.
    Returns
    -------
    Rows: dict[str, dict[Tuple[str, frozenset], dict[str, list[float]]]]
        The structure is {experiment_name: {model_identifier: {metric: [values]}}
        The model identifier is a unique tuple that identifies the model.
    """

    # First we construct a nice table using dictionaries
    rows = {} # {experiment_name: {model_name: {metric: [values]}}}

    for experiment_name, experiment_data in data.items():
        model_dict = {} # {model_identifier: {metric: [values]}}

        for run in experiment_data:
            model_name = make_model_name(run)
            model_config = run.config

            # Bookkeeping
            if model_name not in model_dict:
                model_dict[model_name] = {metric: [] for metric in METRICS}

            for metric in METRICS:
                model_dict[model_name][metric].append(extract_funcs[metric](run))

        rows[experiment_name] = model_dict
    return rows

def should_be_bolded(
    experiment_data: dict[str, dict[str, list[float]]],
    model_name: str,
    metric: str,
    ) -> bool:
    """
    Returns whether the value should be bolded which is the case if the value
    is at least as good as any other value in the experiment.
    """

    mean = np.mean(experiment_data[model_name][metric])
    std_err = np.std(experiment_data[model_name][metric]) / np.sqrt(len(experiment_data[model_name][metric]))

    for other_model_name, other_model_data in experiment_data.items():
        if other_model_name == model_name:
            continue
        other_mean = np.mean(other_model_data[metric])
        other_std_err = np.std(other_model_data[metric]) / np.sqrt(len(other_model_data[metric]))
        if mean + std_err < other_mean - other_std_err:
            return False
    return True

def make_table(
    rows: dict[str, dict[str, dict[str, list[float]]]],
    metrics: list[str],
) -> str:
    """
    Constructs a table from the rows and model_identifier_to_name dictionaries.
    The table is a multi-hierarchical table in latex.
    """
    first_row = next(iter(rows.values()))
    n_models = len(first_row)
    n_metrics = len(metrics)
    n_experiments = len(rows)
    model_names = list(first_row.keys())

    table = "\\begin{table}[!h]\n"
    table += "\\centering\n"
    table += f"\\caption{{Model Performance: {METRIC_TO_CAPTION[metrics[0]]}}}\n"
    table += "\\begin{tabular}{" + "c" * (n_models * n_metrics + 1) + "}\n"
    table += "\\toprule\n"
    #table += "\\multirow{1}{*}{} "a

    # Add the Big Metric names
    #for metric in metrics:
    #    table += f"& \\multicolumn{{{n_models}}}{{c}}{{{metric}}}"
    #table += "\\\\\n"

    # Add midrule
    #for i in range(n_metrics):
    #    start = 2 + i * n_models
    #    end = start + n_models
    #    table += f"\\cmidrule(lr){{{start}-{end}}} "
    #table += "\n"

    # Add the model names
    for _ in range(n_metrics):
        for model_name in model_names:
            table += f"& {format_name(model_name)} "
    table += "\\\\\n"
    table += "\\midrule\n"

    # Add the data
    # sort rows by experiment name
    sorted_experiment_names = sorted(rows.keys())
    for experiment_name in sorted_experiment_names:
        experiment_data = rows[experiment_name]
        table += f"\\textbf{{{format_name(experiment_name)}}} "

        for metric in metrics:
            for model_name in model_names:
                metric_mean = np.mean(experiment_data[model_name][metric])
                metric_std = np.std(experiment_data[model_name][metric]) / np.sqrt(len(experiment_data[model_name][metric]))
                metric_val = f"& {metric_mean:.2f} $\\pm$ {metric_std:.2f} "
                metric_val_bold = f"& \\textbf{{{metric_mean:.2f}}} $\\pm$ {metric_std:.2f} "
                table += metric_val_bold if should_be_bolded(experiment_data, model_name, metric) else metric_val
        table += "\\\\\n"
    table += "\\bottomrule\n"
    table += "\\end{tabular}\n"
    table += "\\end{table}\n"

    return table



def main():
    args = parse_args()
    data = get_data(args.job_dir)
    rows = make_dicts_for_table(data)
    for metric in METRICS:
        table = make_table(rows, [metric])
        print(table)


if __name__ == "__main__":
    main()