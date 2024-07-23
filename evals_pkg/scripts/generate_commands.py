"""
Simple script that generates a commands.txt file to be used alongside parallel.

Usage:
> python3 generate_commands.py # generates commands.txt
> parallel -j 30 < commands.txt # runs the commands in parallel (just to cap the number of parallel jobs)
"""
import os

JOB_DIR = "results_full"
FILE_NAME = "commands.txt"


# Flip probs
LOW_MISPECIFICATION_FLIP_PROB_ACTIVE = 0.01
HIGH_MISPECIFICATION_FLIP_PROB_ACTIVE = 0.2
LOW_MISPECIFICATION_FLIP_PROB_INACTIVE = 0.2
HIGH_MISPECIFICATION_FLIP_PROB_INACTIVE = 0.5

# Number of timepoints and blocks
N_TIMEPOINTS_SMALL = 12
N_TIMEPOINTS_LARGE = 60
N_BLOCKS = 2
N_INDEPENDENT = 5
N_DEPENDENT = 4
NOISE = 0.2

# Dynamics constants
WEIGHTS_LENGTHSCALE_LINEAR = 10
WEIGHTS_LENGTHSCALE_SLOW = 0.4
WEIGHTS_LENGTHSCALE_FAST = 0.1
LENGTHS_SCALE_LINEAR = 0.05
LENGTHS_SCALE_SLOW = 0.05
LENGTHS_SCALE_FAST = 0.05


# Connectivity constants
LOW_CONNECTIVITY_PROB = 0.1
HIGH_CONNECTIVITY_PROB = 0.5

# Other constants
SEED = 42
# REMOVE Diisco for now
MODELS = ["LinearModel", "RollingLinearModel", "DiiscoModel"]
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


seeds = [43, 44, 45, 46, 47]

############################################
# Scenarios
############################################
constant_all_scenarios = {
    "--n-blocks": N_BLOCKS,
    "--n-independent": N_INDEPENDENT,
    "--n-dependent": N_DEPENDENT,
    "--noise": NOISE,
}



scenarios = [
# Scenario 1: Simple linear regression  (low mispecification - small dataset)
{
    "--name": "scenario_1",
    "--flip-prob-active": LOW_MISPECIFICATION_FLIP_PROB_ACTIVE,
    "--flip-prob-inactive": LOW_MISPECIFICATION_FLIP_PROB_INACTIVE,
    "--weights-length-scale": WEIGHTS_LENGTHSCALE_LINEAR,
    "--length-scale": LENGTHS_SCALE_LINEAR,
    "--p-active": LOW_CONNECTIVITY_PROB,
    "--n-timepoints": N_TIMEPOINTS_SMALL,

},


# Scenario 2: Simple linear regression  (high mispecification - small dataset)
{
    "--name": "scenario_2",
    "--flip-prob-active": HIGH_MISPECIFICATION_FLIP_PROB_ACTIVE,
    "--flip-prob-inactive": HIGH_MISPECIFICATION_FLIP_PROB_INACTIVE,
    "--weights-length-scale": WEIGHTS_LENGTHSCALE_LINEAR,
    "--length-scale": LENGTHS_SCALE_LINEAR,
    "--p-active": LOW_CONNECTIVITY_PROB,
    "--n-timepoints": N_TIMEPOINTS_SMALL,
},

# Scenario 3: Slow dynamics (low mispecification - small dataset)
{
    "--name": "scenario_3",
    "--flip-prob-active": LOW_MISPECIFICATION_FLIP_PROB_ACTIVE,
    "--flip-prob-inactive": LOW_MISPECIFICATION_FLIP_PROB_INACTIVE,
    "--weights-length-scale": WEIGHTS_LENGTHSCALE_SLOW,
    "--length-scale": LENGTHS_SCALE_SLOW,
    "--p-active": LOW_CONNECTIVITY_PROB,
    "--n-timepoints": N_TIMEPOINTS_SMALL,
},


# Scenario 4: Slow dynamics (high mispecification - small dataset)
{
    "--name": "scenario_4",
    "--flip-prob-active": HIGH_MISPECIFICATION_FLIP_PROB_ACTIVE,
    "--flip-prob-inactive": HIGH_MISPECIFICATION_FLIP_PROB_INACTIVE,
    "--weights-length-scale": WEIGHTS_LENGTHSCALE_SLOW,
    "--length-scale": LENGTHS_SCALE_SLOW,
    "--p-active": LOW_CONNECTIVITY_PROB,
    "--n-timepoints": N_TIMEPOINTS_SMALL,
},


# Scenario 5: Fast dynamics (low mispecification - small dataset)
{
    "--name": "scenario_5",
    "--flip-prob-active": LOW_MISPECIFICATION_FLIP_PROB_ACTIVE,
    "--flip-prob-inactive": LOW_MISPECIFICATION_FLIP_PROB_INACTIVE,
    "--weights-length-scale": WEIGHTS_LENGTHSCALE_FAST,
    "--length-scale": LENGTHS_SCALE_FAST,
    "--p-active": LOW_CONNECTIVITY_PROB,
    "--n-timepoints": N_TIMEPOINTS_SMALL,
},


# Scenario 6: Fast dynamics (high mispecification - small dataset)
{
    "--name": "scenario_6",
    "--flip-prob-active": HIGH_MISPECIFICATION_FLIP_PROB_ACTIVE,
    "--flip-prob-inactive": HIGH_MISPECIFICATION_FLIP_PROB_INACTIVE,
    "--weights-length-scale": WEIGHTS_LENGTHSCALE_FAST,
    "--length-scale": LENGTHS_SCALE_FAST,
    "--p-active": LOW_CONNECTIVITY_PROB,
    "--n-timepoints": N_TIMEPOINTS_SMALL,
},


# Scenario 7: High-connectivity (low mispecification - small dataset)
{
    "--name": "scenario_7",
    "--flip-prob-active": LOW_MISPECIFICATION_FLIP_PROB_ACTIVE,
    "--flip-prob-inactive": LOW_MISPECIFICATION_FLIP_PROB_INACTIVE,
    "--weights-length-scale": WEIGHTS_LENGTHSCALE_SLOW,
    "--length-scale": LENGTHS_SCALE_SLOW,
    "--p-active": HIGH_CONNECTIVITY_PROB,
    "--n-timepoints": N_TIMEPOINTS_SMALL,
},


# Scenario 8: High-connectivity (high mispecification - small dataset)
{
    "--name": "scenario_8",
    "--flip-prob-active": HIGH_MISPECIFICATION_FLIP_PROB_ACTIVE,
    "--flip-prob-inactive": HIGH_MISPECIFICATION_FLIP_PROB_INACTIVE,
    "--weights-length-scale": WEIGHTS_LENGTHSCALE_SLOW,
    "--length-scale": LENGTHS_SCALE_SLOW,
    "--p-active": HIGH_CONNECTIVITY_PROB,
    "--n-timepoints": N_TIMEPOINTS_SMALL,
},

# Scenario 9: Simple linear regression  (low mispecification - large dataset)
{
    "--name": "scenario_9",
    "--flip-prob-active": LOW_MISPECIFICATION_FLIP_PROB_ACTIVE,
    "--flip-prob-inactive": LOW_MISPECIFICATION_FLIP_PROB_INACTIVE,
    "--weights-length-scale": WEIGHTS_LENGTHSCALE_LINEAR,
    "--length-scale": LENGTHS_SCALE_LINEAR,
    "--p-active": LOW_CONNECTIVITY_PROB,
    "--n-timepoints": N_TIMEPOINTS_LARGE,
},


# Scenario 10: Simple linear regression  (high mispecification - large dataset)
{
    "--name": "scenario_10",
    "--flip-prob-active": HIGH_MISPECIFICATION_FLIP_PROB_ACTIVE,
    "--flip-prob-inactive": HIGH_MISPECIFICATION_FLIP_PROB_INACTIVE,
    "--weights-length-scale": WEIGHTS_LENGTHSCALE_LINEAR,
    "--length-scale": LENGTHS_SCALE_LINEAR,
    "--p-active": LOW_CONNECTIVITY_PROB,
    "--n-timepoints": N_TIMEPOINTS_LARGE,
},

# Scenario 11: Slow dynamics (low mispecification - large dataset)
{
    "--name": "scenario_11",
    "--flip-prob-active": LOW_MISPECIFICATION_FLIP_PROB_ACTIVE,
    "--flip-prob-inactive": LOW_MISPECIFICATION_FLIP_PROB_INACTIVE,
    "--weights-length-scale": WEIGHTS_LENGTHSCALE_SLOW,
    "--length-scale": LENGTHS_SCALE_SLOW,
    "--p-active": LOW_CONNECTIVITY_PROB,
    "--n-timepoints": N_TIMEPOINTS_LARGE,
},


# Scenario 12: Slow dynamics (high mispecification - large dataset)
{
    "--name": "scenario_12",
    "--flip-prob-active": HIGH_MISPECIFICATION_FLIP_PROB_ACTIVE,
    "--flip-prob-inactive": HIGH_MISPECIFICATION_FLIP_PROB_INACTIVE,
    "--weights-length-scale": WEIGHTS_LENGTHSCALE_SLOW,
    "--length-scale": LENGTHS_SCALE_SLOW,
    "--p-active": LOW_CONNECTIVITY_PROB,
    "--n-timepoints": N_TIMEPOINTS_LARGE,
},


# Scenario 13: Fast dynamics (low mispecification - large dataset)
{
    "--name": "scenario_13",
    "--flip-prob-active": LOW_MISPECIFICATION_FLIP_PROB_ACTIVE,
    "--flip-prob-inactive": LOW_MISPECIFICATION_FLIP_PROB_INACTIVE,
    "--weights-length-scale": WEIGHTS_LENGTHSCALE_FAST,
    "--length-scale": LENGTHS_SCALE_FAST,
    "--p-active": LOW_CONNECTIVITY_PROB,
    "--n-timepoints": N_TIMEPOINTS_LARGE,
},


# Scenario 14: Fast dynamics (high mispecification - large dataset)
{
    "--name": "scenario_14",
    "--flip-prob-active": HIGH_MISPECIFICATION_FLIP_PROB_ACTIVE,
    "--flip-prob-inactive": HIGH_MISPECIFICATION_FLIP_PROB_INACTIVE,
    "--weights-length-scale": WEIGHTS_LENGTHSCALE_FAST,
    "--length-scale": LENGTHS_SCALE_FAST,
    "--p-active": LOW_CONNECTIVITY_PROB,
    "--n-timepoints": N_TIMEPOINTS_LARGE,
},


# Scenario 15: High-connectivity (low mispecification - large dataset)
{
    "--name": "scenario_15",
    "--flip-prob-active": LOW_MISPECIFICATION_FLIP_PROB_ACTIVE,
    "--flip-prob-inactive": LOW_MISPECIFICATION_FLIP_PROB_INACTIVE,
    "--weights-length-scale": WEIGHTS_LENGTHSCALE_SLOW,
    "--length-scale": LENGTHS_SCALE_SLOW,
    "--p-active": HIGH_CONNECTIVITY_PROB,
    "--n-timepoints": N_TIMEPOINTS_LARGE,

},


# Scenario 16: High-connectivity (high mispecification - large dataset)
{
    "--name": "scenario_16",
    "--flip-prob-active": HIGH_MISPECIFICATION_FLIP_PROB_ACTIVE,
    "--flip-prob-inactive": HIGH_MISPECIFICATION_FLIP_PROB_INACTIVE,
    "--weights-length-scale": WEIGHTS_LENGTHSCALE_SLOW,
    "--length-scale": LENGTHS_SCALE_SLOW,
    "--p-active": HIGH_CONNECTIVITY_PROB,
    "--n-timepoints": N_TIMEPOINTS_LARGE,
},
]

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








