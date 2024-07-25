"""
Simple script that generates a commands.txt file to be used alongside parallel.
It also prints a simple tex file with the commands that will be run.

> commands.txt

It also prints a simple tex file with the specific settings that
will be run.
"""
import os
import pandas as pd

JOB_DIR = "results_full_easier"
FILE_NAME = "commands.txt"


# Flip probs
LOW_MISPECIFICATION_FLIP_PROB_ACTIVE = 0.01
HIGH_MISPECIFICATION_FLIP_PROB_ACTIVE = 0.2
LOW_MISPECIFICATION_FLIP_PROB_INACTIVE = 0.1
HIGH_MISPECIFICATION_FLIP_PROB_INACTIVE = 0.5

# Number of time pointsand blocks
N_TIMEPOINTS_SMALL = 12
N_TIMEPOINTS_LARGE = 60
N_BLOCKS = 2
N_INDEPENDENT = 5
N_DEPENDENT = 4
NOISE = 0.1

# Dynamics constants
WEIGHTS_LENGTHSCALE_LINEAR = 10
WEIGHTS_LENGTHSCALE_SLOW = 0.6
WEIGHTS_LENGTHSCALE_FAST = 0.3
LENGTHS_SCALE_LINEAR = 0.1
LENGTHS_SCALE_SLOW = 0.1
LENGTHS_SCALE_FAST = 0.1


# Connectivity constants
LOW_CONNECTIVITY_PROB = 0.2
HIGH_CONNECTIVITY_PROB = 0.5

# Other constants
SEED = 42
# REMOVE Diisco for now
MODELS = ["LinearModel"]#, "RollingLinearModel"]#, "DiiscoModel"]
DIISCO_MODEL_PARAMS = [
    {
        "--model-parameters": "lr 0.01",
    },
    {
        "--model-parameters": "lr 0.005",
    }
]
OTHER_MODEL_PARAMS = [
    {
        "--model-parameters": "ignore_is_active False",
    },
    {
        "--model-parameters": "ignore_is_active True",
    }
]


seeds = [43, 44,]# 45, 46, 47]

############################################
# Scenarios
############################################

arg_names_to_tex ={
    # write string to be used in tex
    "--n-blocks": "n_{\\text{blocks}}",
    "--n-independent": "\(n_{\\text{ind}}",
    "--n-dependent": "n_{\\text{dep}}",
    "--noise": "Noise level \\sigma",
    "--flip-prob-active": "p_{\\text{flip-active}}",
    "--flip-prob-inactive": "p_{\\text{flip-inactive}}",
    "--weights-length-scale": "l_{w}",
    "--length-scale": "l_{y}",
    "--p-active": "p_{\\text{act}}",
    "--n-timepoints": "n_{\\text{timepoints}}",

}


constant_all_scenarios = {
    "--n-blocks": N_BLOCKS,
    "--n-independent": N_INDEPENDENT,
    "--n-dependent": N_DEPENDENT,
    "--noise": NOISE,
}



scenarios = [
# Scenario 1: Simple linear regression  (low misspecification- small dataset)
{
    "--name": "scenario_1",
    "--flip-prob-active": LOW_MISPECIFICATION_FLIP_PROB_ACTIVE,
    "--flip-prob-inactive": LOW_MISPECIFICATION_FLIP_PROB_INACTIVE,
    "--weights-length-scale": WEIGHTS_LENGTHSCALE_LINEAR,
    "--length-scale": LENGTHS_SCALE_LINEAR,
    "--p-active": LOW_CONNECTIVITY_PROB,
    "--n-timepoints": N_TIMEPOINTS_SMALL,
    "description": "Simple linear regression  (low misspecification- small dataset)"
},


# Scenario 2: Simple linear regression  (high misspecification- small dataset)
{
    "--name": "scenario_2",
    "--flip-prob-active": HIGH_MISPECIFICATION_FLIP_PROB_ACTIVE,
    "--flip-prob-inactive": HIGH_MISPECIFICATION_FLIP_PROB_INACTIVE,
    "--weights-length-scale": WEIGHTS_LENGTHSCALE_LINEAR,
    "--length-scale": LENGTHS_SCALE_LINEAR,
    "--p-active": LOW_CONNECTIVITY_PROB,
    "--n-timepoints": N_TIMEPOINTS_SMALL,
    "description": "Simple linear regression  (high misspecification- small dataset)"
},

# Scenario 3: Slow dynamics (low misspecification- small dataset)
{
    "--name": "scenario_3",
    "--flip-prob-active": LOW_MISPECIFICATION_FLIP_PROB_ACTIVE,
    "--flip-prob-inactive": LOW_MISPECIFICATION_FLIP_PROB_INACTIVE,
    "--weights-length-scale": WEIGHTS_LENGTHSCALE_SLOW,
    "--length-scale": LENGTHS_SCALE_SLOW,
    "--p-active": LOW_CONNECTIVITY_PROB,
    "--n-timepoints": N_TIMEPOINTS_SMALL,
    "description": "Slow dynamics (low misspecification- small dataset)"
},


# Scenario 4: Slow dynamics (high misspecification- small dataset)
{
    "--name": "scenario_4",
    "--flip-prob-active": HIGH_MISPECIFICATION_FLIP_PROB_ACTIVE,
    "--flip-prob-inactive": HIGH_MISPECIFICATION_FLIP_PROB_INACTIVE,
    "--weights-length-scale": WEIGHTS_LENGTHSCALE_SLOW,
    "--length-scale": LENGTHS_SCALE_SLOW,
    "--p-active": LOW_CONNECTIVITY_PROB,
    "--n-timepoints": N_TIMEPOINTS_SMALL,
    "description": "Slow dynamics (high misspecification- small dataset)"
},


# Scenario 5: Fast dynamics (low misspecification- small dataset)
{
    "--name": "scenario_5",
    "--flip-prob-active": LOW_MISPECIFICATION_FLIP_PROB_ACTIVE,
    "--flip-prob-inactive": LOW_MISPECIFICATION_FLIP_PROB_INACTIVE,
    "--weights-length-scale": WEIGHTS_LENGTHSCALE_FAST,
    "--length-scale": LENGTHS_SCALE_FAST,
    "--p-active": LOW_CONNECTIVITY_PROB,
    "--n-timepoints": N_TIMEPOINTS_SMALL,
    "description": "Fast dynamics (low misspecification- small dataset)"
},


# Scenario 6: Fast dynamics (high misspecification- small dataset)
{
    "--name": "scenario_6",
    "--flip-prob-active": HIGH_MISPECIFICATION_FLIP_PROB_ACTIVE,
    "--flip-prob-inactive": HIGH_MISPECIFICATION_FLIP_PROB_INACTIVE,
    "--weights-length-scale": WEIGHTS_LENGTHSCALE_FAST,
    "--length-scale": LENGTHS_SCALE_FAST,
    "--p-active": LOW_CONNECTIVITY_PROB,
    "--n-timepoints": N_TIMEPOINTS_SMALL,
    "description": "Fast dynamics (high misspecification- small dataset)"
},


# Scenario 7: High-connectivity (low misspecification- small dataset)
{
    "--name": "scenario_7",
    "--flip-prob-active": LOW_MISPECIFICATION_FLIP_PROB_ACTIVE,
    "--flip-prob-inactive": LOW_MISPECIFICATION_FLIP_PROB_INACTIVE,
    "--weights-length-scale": WEIGHTS_LENGTHSCALE_SLOW,
    "--length-scale": LENGTHS_SCALE_SLOW,
    "--p-active": HIGH_CONNECTIVITY_PROB,
    "--n-timepoints": N_TIMEPOINTS_SMALL,
    "description": "High-connectivity (low misspecification- small dataset)"
},


# Scenario 8: High-connectivity (high misspecification- small dataset)
{
    "--name": "scenario_8",
    "--flip-prob-active": HIGH_MISPECIFICATION_FLIP_PROB_ACTIVE,
    "--flip-prob-inactive": HIGH_MISPECIFICATION_FLIP_PROB_INACTIVE,
    "--weights-length-scale": WEIGHTS_LENGTHSCALE_SLOW,
    "--length-scale": LENGTHS_SCALE_SLOW,
    "--p-active": HIGH_CONNECTIVITY_PROB,
    "--n-timepoints": N_TIMEPOINTS_SMALL,
    "description": "High-connectivity (high misspecification- small dataset)"
},

# Scenario 9: Simple linear regression  (low misspecification- large dataset)
{
    "--name": "scenario_9",
    "--flip-prob-active": LOW_MISPECIFICATION_FLIP_PROB_ACTIVE,
    "--flip-prob-inactive": LOW_MISPECIFICATION_FLIP_PROB_INACTIVE,
    "--weights-length-scale": WEIGHTS_LENGTHSCALE_LINEAR,
    "--length-scale": LENGTHS_SCALE_LINEAR,
    "--p-active": LOW_CONNECTIVITY_PROB,
    "--n-timepoints": N_TIMEPOINTS_LARGE,
    "description": "Simple linear regression  (low misspecification- large dataset)"
},


# Scenario 10: Simple linear regression  (high misspecification- large dataset)
{
    "--name": "scenario_10",
    "--flip-prob-active": HIGH_MISPECIFICATION_FLIP_PROB_ACTIVE,
    "--flip-prob-inactive": HIGH_MISPECIFICATION_FLIP_PROB_INACTIVE,
    "--weights-length-scale": WEIGHTS_LENGTHSCALE_LINEAR,
    "--length-scale": LENGTHS_SCALE_LINEAR,
    "--p-active": LOW_CONNECTIVITY_PROB,
    "--n-timepoints": N_TIMEPOINTS_LARGE,
    "description": "Simple linear regression  (high misspecification- large dataset)"
},

# Scenario 11: Slow dynamics (low misspecification- large dataset)
{
    "--name": "scenario_11",
    "--flip-prob-active": LOW_MISPECIFICATION_FLIP_PROB_ACTIVE,
    "--flip-prob-inactive": LOW_MISPECIFICATION_FLIP_PROB_INACTIVE,
    "--weights-length-scale": WEIGHTS_LENGTHSCALE_SLOW,
    "--length-scale": LENGTHS_SCALE_SLOW,
    "--p-active": LOW_CONNECTIVITY_PROB,
    "--n-timepoints": N_TIMEPOINTS_LARGE,
    "description": "Slow dynamics (low misspecification- large dataset)"
},


# Scenario 12: Slow dynamics (high misspecification- large dataset)
{
    "--name": "scenario_12",
    "--flip-prob-active": HIGH_MISPECIFICATION_FLIP_PROB_ACTIVE,
    "--flip-prob-inactive": HIGH_MISPECIFICATION_FLIP_PROB_INACTIVE,
    "--weights-length-scale": WEIGHTS_LENGTHSCALE_SLOW,
    "--length-scale": LENGTHS_SCALE_SLOW,
    "--p-active": LOW_CONNECTIVITY_PROB,
    "--n-timepoints": N_TIMEPOINTS_LARGE,
    "description": "Slow dynamics (high misspecification- large dataset)"
},


# Scenario 13: Fast dynamics (low misspecification- large dataset)
{
    "--name": "scenario_13",
    "--flip-prob-active": LOW_MISPECIFICATION_FLIP_PROB_ACTIVE,
    "--flip-prob-inactive": LOW_MISPECIFICATION_FLIP_PROB_INACTIVE,
    "--weights-length-scale": WEIGHTS_LENGTHSCALE_FAST,
    "--length-scale": LENGTHS_SCALE_FAST,
    "--p-active": LOW_CONNECTIVITY_PROB,
    "--n-timepoints": N_TIMEPOINTS_LARGE,
    "description": "Fast dynamics (low misspecification- large dataset)"
},


# Scenario 14: Fast dynamics (high misspecification- large dataset)
{
    "--name": "scenario_14",
    "--flip-prob-active": HIGH_MISPECIFICATION_FLIP_PROB_ACTIVE,
    "--flip-prob-inactive": HIGH_MISPECIFICATION_FLIP_PROB_INACTIVE,
    "--weights-length-scale": WEIGHTS_LENGTHSCALE_FAST,
    "--length-scale": LENGTHS_SCALE_FAST,
    "--p-active": LOW_CONNECTIVITY_PROB,
    "--n-timepoints": N_TIMEPOINTS_LARGE,
    "description": "Fast dynamics (high misspecification- large dataset)"
},


# Scenario 15: High-connectivity (low misspecification- large dataset)
{
    "--name": "scenario_15",
    "--flip-prob-active": LOW_MISPECIFICATION_FLIP_PROB_ACTIVE,
    "--flip-prob-inactive": LOW_MISPECIFICATION_FLIP_PROB_INACTIVE,
    "--weights-length-scale": WEIGHTS_LENGTHSCALE_SLOW,
    "--length-scale": LENGTHS_SCALE_SLOW,
    "--p-active": HIGH_CONNECTIVITY_PROB,
    "--n-timepoints": N_TIMEPOINTS_LARGE,
    "description": "High-connectivity (low misspecification- large dataset)"

},


# Scenario 16: High-connectivity (high misspecification- large dataset)
{
    "--name": "scenario_16",
    "--flip-prob-active": HIGH_MISPECIFICATION_FLIP_PROB_ACTIVE,
    "--flip-prob-inactive": HIGH_MISPECIFICATION_FLIP_PROB_INACTIVE,
    "--weights-length-scale": WEIGHTS_LENGTHSCALE_SLOW,
    "--length-scale": LENGTHS_SCALE_SLOW,
    "--p-active": HIGH_CONNECTIVITY_PROB,
    "--n-timepoints": N_TIMEPOINTS_LARGE,
    "description": "High-connectivity (high misspecification- large dataset)"
},
]

def generate_latex_table_from_df(df: pd.DataFrame, caption: str = "Model Performance: AUC") -> str:
    table = "\\begin{table}[!h]\n"
    table += "\\centering\n"
    table += f"\\caption{{{caption}}}\n"
    table += "\\begin{tabular}{" + "c" * (len(df.columns) + 1) + "}\n"
    table += "\\toprule\n"

    # Header
    table += " & " + " & ".join(df.columns) + " \\\\\n"
    table += "\\midrule\n"

    # Rows
    for index, row in df.iterrows():
        formatted_row = " & ".join(
            f"\\textbf{{{val}}}" if (isinstance(val, str) and val.startswith("\\textbf{")) or val == row.max() else str(val) for val in row
        )
        table += f"\\textbf{{{index}}} & {formatted_row} \\\\\n"

    table += "\\bottomrule\n"
    table += "\\end{tabular}\n"
    table += "\\end{table}\n"

    return table

def make_tex(scenarios, arg_names_to_tex, constant_all_scenarios):
    """
    Makes
    """
    caption =  "Scenario settings"
    n_rows = len(scenarios)
    n_cols = len(scenarios[0])
    col_names =  [arg_names_to_tex[key] for key in scenarios[0].keys() if key in arg_names_to_tex]


    table = "\\begin{table}[!h]\n"
    table += "\\centering\n"
    table += f"\\caption{{{caption}}}\n"
    table += "\\begin{tabular}{" + "c" * (len(col_names) + 1) + "}\n"
    table += "\\toprule\n"

    # Header

    table += "& $ " + " $&$ ".join(col_names) + " \\\\\n"
    table += "\\midrule\n"

    # Rows
    for scenario in scenarios: # one row per scenario
        name = scenario["--name"].replace("_", " ")
        name = name[0].upper() + name[1:]
        table += f"{name} & "
        for key, value in scenario.items():
            if key in arg_names_to_tex:
                table += f"{value} & "
        table = table[:-2] +  "\\\\\n"

    table += "\\bottomrule\n"
    table += "\\end{tabular}\n"
    table += "\\end{table}\n"

    return table

def rm_description(scenarios):
    for scenario in scenarios:
        del scenario["description"]
    return scenarios


# clean later
tex_table_with_scenarios = make_tex(scenarios, arg_names_to_tex, constant_all_scenarios)
print(tex_table_with_scenarios)
scenarios = rm_description(scenarios)


# Delete the file if it exists
if os.path.exists(FILE_NAME):
    os.remove(FILE_NAME)

for seed in seeds:
    for scenario in scenarios:
        for model in MODELS:
            command = f"python3 evals/__main__.py {JOB_DIR} --model {model}"
            for key, value in constant_all_scenarios.items():
                command += f" {key} {value}"
            for key, value in scenario.items():
                command += f" {key} {value}"
            command += f" --seed {seed}"

            if model == "DiiscoModel": # We need to add the DIISCO_MODEL_PARAMS and create one command for each one
                model_specific_params = DIISCO_MODEL_PARAMS
            else:
                model_specific_params = OTHER_MODEL_PARAMS

            for params in model_specific_params:
                model_command = command
                for key, value in params.items():
                    model_command += f" {key} {value}"
                with open(FILE_NAME, "a") as f:
                    f.write(model_command + "\n")








