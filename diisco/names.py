"""
Simple module for storing the names of the different strings that need to be used in
the specification of pyro models. This is done done to avoid typos and to use the
interpreter to check for errors in consistency.
"""

# Names for the different parameters of the model
LENGTHSCALE_W = "lengthscale_w"  # lengthscale for the weights
LENGTHSCALE_W_RANGE = "lengthscale_w_range"  # the range of all values of the lengthscale for the weights
LENGTHSCALE_F = "lengthscale_f"  # lengthscale for the latent features
VARIANCE_W = "variance_w"  # constant C in C * K(x, x) for the weights
VARIANCE_F = "variance_f"  # constant C in C * K(x, x) for the latent features
PRIOR_VARIANCE_F = "prior_variance_f"  # variance used for fitting prior
SIGMA_Y = "sigma_y"  # Constant C in K(x, x) + C * I i
SIGMA_W = "sigma_w"  # Constant C in K(x, x) + C * I
SIGMA_F = "sigma_f"  # Constant C in K(x, x) + C * I
W = "W"  # Weights
F = "F"  # Latent features
Y = "Y"  # Output
B = "B"  # Bias
