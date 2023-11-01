"""
This module contains various constants used throughout the project
such as default values for hyperparameters, etc.
"""
import diisco.names as names

# This specifies both the default values for the hyperparameters
# and the valid hyperparameter names.

DEFAULT_HYPERS = {
    names.LENGTHSCALE_W: 100,
    names.LENGTHSCALE_W_RANGE: 3,
    names.LENGTHSCALE_F: 100,
    names.VARIANCE_W: 1.0,
    names.VARIANCE_F: 1.0,
    names.SIGMA_Y: 0.2,
    names.SIGMA_W: 0.1,
    names.SIGMA_F: 0.2,
    names.PRIOR_VARIANCE_F: 1.0,
}

DEFAULT_HYPERS_TO_OPTIM = []

# This is a constant used for numerical stability.
EPSILON = 1e-10
