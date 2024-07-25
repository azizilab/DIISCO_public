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
from sklearn.metrics import roc_curve, precision_recall_curve
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

def make_prec_rec_curves(
    data: dict[str,list[RunResults]]
    )->  dict[str, dict[str, (list[float], list[float], list[float])]]:

    """
    Constructs an precision recall curve for each model in each experiment.
    The same seed is used for all models in the same experiment.

    Parameters
    ----------
    data:
        {experiment_name:
            {
                model_name:
                    (fpr, tpr_mean, tpr_std)
            }
        }
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
            prec, rec, _ = precision_recall_curve(true_interactions, symmetrical_transformed_interactions)
            print(prec, rec)

            if model_name not in model_dict:
                model_dict[model_name] = [(prec, rec)]
            else:
                model_dict[model_name].append((prec, rec))


        # The fpr is different per seed,
        # So first we create a single big fpr per model
        # Then extrapolate the tpr for each fpr
        # Then we average the tpr for each fpr
        for model_name, prec_rec_list in model_dict.items():
            collective_prec = np.concatenate([prec for prec, _ in prec_rec_list])
            collective_prec = sorted(set(collective_prec))


            # extrapolate tpr for each fpr
            collective_rec = []
            for prec, rec in prec_rec_list:
                collective_rec.append(np.interp(collective_prec, prec, rec))

            # average tpr for each fpr
            collective_rec = np.mean(collective_rec, axis=0)
            collective_rec_std = np.std(collective_rec, axis=0) / np.sqrt(len(prec_rec_list))
            model_dict[model_name] = (collective_prec, collective_rec, collective_rec_std)

        rows[experiment_name] = model_dict
    return rows

def plot_and_save_prec_rec_curves(
    rows: dict[str, dict[str, (list[float], list[float])]],
    output_path: str,
    ):
    """
    Makes one plot per experiment.
    """
    sorted_model_names = sorted(set(model_name for model_dict in rows.values() for model_name in model_dict.keys()))
    colors = plt.cm.get_cmap("tab20", len(sorted_model_names))

    for experiment_name, experiment_data in rows.items():
        fig, ax = plt.subplots()
        for mi, model_name in enumerate(sorted_model_names):
            if model_name in experiment_data:
                prec, rec_mean, tpr_std = experiment_data[model_name]
                ax.plot(prec, rec_mean, label=model_name, color=colors(mi))
                #ax.fill_between(fpr, tpr_mean - tpr_std, tpr_mean + tpr_std, color=colors(mi), alpha=0.3)
        ax.set_xlabel("Precision")
        ax.set_ylabel("Recall")
        ax.set_title(f"Precision Recall Curves for {format_name(experiment_name)}")
        ax.legend()

        # save to pdf
        os.makedirs(output_path, exist_ok=True)
        fig.savefig(os.path.join(output_path, f"{experiment_name}.pdf"))


def main():
    args = parse_args()
    data = get_data(args.job_dir)
    rows = make_prec_rec_curves(data)
    plot_and_save_prec_rec_curves(rows, args.out_dir)




if __name__ == "__main__":
    main()