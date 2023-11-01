import torch

import diisco.utils as utils


def is_psd(mat):
    is_symmetric = bool((mat == mat.T).all())
    is_positive_definite = bool(torch.all(torch.symeig(mat)[0] > 0))
    return is_symmetric and is_positive_definite


def rbf_kernel(
    x1: torch.Tensor,
    x2: torch.Tensor,
    length_scale: float,
    variance: float = 1.0,
) -> torch.Tensor:
    """
    Compute the RBF kernel between x1 and x2.
    Based on Pyro's implementation of the RBF kernel.
    :param x1: First input. shape: (n1, d)
    :param x2: Second input. shape: (n2, d)
    :param length_scale: Length scale of the kernel.
    :return: Kernel matrix. shape: (n1, n2)
    """
    x1_scaled = x1 / length_scale
    x2_scaled = x2 / length_scale
    dists = torch.cdist(x1_scaled, x2_scaled, p=2)
    covariance = variance * torch.exp(-0.5 * dists**2)
    covariance = utils.make_psd(covariance)
    return covariance
