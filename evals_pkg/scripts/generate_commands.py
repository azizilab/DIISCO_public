"""
Simple script that generates a commands.txt file to be used alongside parallel.

Usage:
> python3 generate_commands.py # generates commands.txt
> parallel -j 30 < commands.txt # runs the commands in parallel (just to cap the number of parallel jobs)
"""
import os

JOB_DIR = "results"
FILE_NAME = "commands.txt"


# Flip probs
LOW_MISPECIFICATION_FLIP_PROB_ACTIVE = 0.01
HIGH_MISPECIFICATION_FLIP_PROB_ACTIVE = 0.2
LOW_MISPECIFICATION_FLIP_PROB_INACTIVE = 0.1
HIGH_MISPECIFICATION_FLIP_PROB_INACTIVE = 0.3

# Number of timepoints and blocks
N_TIMEPOINTS = 80
N_BLOCKS = 2
N_INDEPENDENT = 4
N_DEPENDENT = 4
NOISE = 0.1

# Dynamics constants
WEIGHTS_LENGTHSCALE_LINEAR = 10
WEIGHTS_LENGTHSCALE_SLOW = 2
WEIGHTS_LENGTHSCALE_FAST = 0.8
LENGTHS_SCALE_LINEAR = 0.1
LENGTHS_SCALE_SLOW = 0.1
LENGTHS_SCALE_FAST = 0.1


# Connectivity constants
LOW_CONNECTIVITY_PROB = 0.1
HIGH_CONNECTIVITY_PROB = 0.5

# Other constants
SEED = 42
MODELS = ["LinearModel", "RollingLinearModel", "DiiscoModel"]
DIISCO_MODEL_PARAMS = [
    {
        "--model-parameters": "lr 0.01",
    },
    {
        "--model-parameters": "lr 0.001",
    },
    {
        "--model-parameters": "lr 0.0001",
    },
    {
        "--model-parameters": "lr 0.00001",
    }
]


############################################
# Scenarios
############################################
constant_all_scenarios = {
    "--n-timepoints": N_TIMEPOINTS,
    "--n-blocks": N_BLOCKS,
    "--n-independent": N_INDEPENDENT,
    "--n-dependent": N_DEPENDENT,
    "--noise": NOISE,
    "--seed": SEED,
}



scenarios = [
# Scenario 1: Simple linear regression  (low mispecification)
{
    "--name": "scenario_1",
    "--flip-prob-active": LOW_MISPECIFICATION_FLIP_PROB_ACTIVE,
    "--flip-prob-inactive": LOW_MISPECIFICATION_FLIP_PROB_INACTIVE,
    "--weights-length-scale": WEIGHTS_LENGTHSCALE_LINEAR,
    "--length-scale": LENGTHS_SCALE_LINEAR,
    "--p-active": LOW_CONNECTIVITY_PROB,

},


# Scenario 2: Simple linear regression  (high mispecification)
{
    "--name": "scenario_2",
    "--flip-prob-active": HIGH_MISPECIFICATION_FLIP_PROB_ACTIVE,
    "--flip-prob-inactive": HIGH_MISPECIFICATION_FLIP_PROB_INACTIVE,
    "--weights-length-scale": WEIGHTS_LENGTHSCALE_LINEAR,
    "--length-scale": LENGTHS_SCALE_LINEAR,
    "--p-active": LOW_CONNECTIVITY_PROB,
},

# Scenario 3: Slow dynamics (low mispecification)
{
    "--name": "scenario_3",
    "--flip-prob-active": LOW_MISPECIFICATION_FLIP_PROB_ACTIVE,
    "--flip-prob-inactive": LOW_MISPECIFICATION_FLIP_PROB_INACTIVE,
    "--weights-length-scale": WEIGHTS_LENGTHSCALE_SLOW,
    "--length-scale": LENGTHS_SCALE_SLOW,
    "--p-active": LOW_CONNECTIVITY_PROB,
},


# Scenario 4: Slow dynamics (high mispecification)
{
    "--name": "scenario_4",
    "--flip-prob-active": HIGH_MISPECIFICATION_FLIP_PROB_ACTIVE,
    "--flip-prob-inactive": HIGH_MISPECIFICATION_FLIP_PROB_INACTIVE,
    "--weights-length-scale": WEIGHTS_LENGTHSCALE_SLOW,
    "--length-scale": LENGTHS_SCALE_SLOW,
    "--p-active": LOW_CONNECTIVITY_PROB,
},


# Scenario 5: Fast dynamics (low mispecification)
{
    "--name": "scenario_5",
    "--flip-prob-active": LOW_MISPECIFICATION_FLIP_PROB_ACTIVE,
    "--flip-prob-inactive": LOW_MISPECIFICATION_FLIP_PROB_INACTIVE,
    "--weights-length-scale": WEIGHTS_LENGTHSCALE_FAST,
    "--length-scale": LENGTHS_SCALE_FAST,
    "--p-active": LOW_CONNECTIVITY_PROB,
},


# Scenario 6: Fast dynamics (high mispecification)
{
    "--name": "scenario_6",
    "--flip-prob-active": HIGH_MISPECIFICATION_FLIP_PROB_ACTIVE,
    "--flip-prob-inactive": HIGH_MISPECIFICATION_FLIP_PROB_INACTIVE,
    "--weights-length-scale": WEIGHTS_LENGTHSCALE_FAST,
    "--length-scale": LENGTHS_SCALE_FAST,
    "--p-active": LOW_CONNECTIVITY_PROB,
},


# Scenario 7: High-connectivity (low mispecification)
{
    "--name": "scenario_7",
    "--flip-prob-active": LOW_MISPECIFICATION_FLIP_PROB_ACTIVE,
    "--flip-prob-inactive": LOW_MISPECIFICATION_FLIP_PROB_INACTIVE,
    "--weights-length-scale": WEIGHTS_LENGTHSCALE_SLOW,
    "--length-scale": LENGTHS_SCALE_SLOW,
    "--p-active": HIGH_CONNECTIVITY_PROB,

},


# Scenario 8: High-connectivity (high mispecification)
{
    "--name": "scenario_8",
    "--flip-prob-active": HIGH_MISPECIFICATION_FLIP_PROB_ACTIVE,
    "--flip-prob-inactive": HIGH_MISPECIFICATION_FLIP_PROB_INACTIVE,
    "--weights-length-scale": WEIGHTS_LENGTHSCALE_SLOW,
    "--length-scale": LENGTHS_SCALE_SLOW,
    "--p-active": HIGH_CONNECTIVITY_PROB,
},

]

# Delete the file if it exists
if os.path.exists(FILE_NAME):
    os.remove(FILE_NAME)


for scenario in scenarios:
    for model in MODELS:
        command = f"python3 evals/__main__.py {JOB_DIR} --model {model}"
        for key, value in constant_all_scenarios.items():
            command += f" {key} {value}"
        for key, value in scenario.items():
            command += f" {key} {value}"

        if model == "DiiscoModel": # We need to add the DIISCO_MODEL_PARAMS and create one command for each one
            for diisco_params in DIISCO_MODEL_PARAMS:
                diisco_command = command
                for key, value in diisco_params.items():
                    diisco_command += f" {key} {value}"
                with open(FILE_NAME, "a") as f:
                    f.write(diisco_command + "\n")
        else:
            with open(FILE_NAME, "a") as f:
                f.write(command + "\n")









