import torch
from scipy.stats import norm


def flat_vect_to_lower_triangular_matrix(vect, n, batch_size):
    """
    Convert a flat vector to a lower triangular matrix.
    :param vect: The flat vector.
    :param n: The number of rows and columns of the matrix.
    :return: The lower triangular matrix.
    """
    assert vect.shape[0] == int(n * (n + 1) / 2) * batch_size
    # A tensor of dimensions (batch_size, n, n)
    lower_triangular_matrix = torch.zeros((batch_size, n, n))

    bool_mask = torch.tril(torch.ones((n, n))).bool()
    bool_mask = bool_mask.repeat(batch_size, 1, 1)
    lower_triangular_matrix[bool_mask] = vect
    return lower_triangular_matrix


def make_psd(mat):
    """
    Make a matrix positive semidefinite.
    :param mat: The matrix to make positive semidefinite. shape (batch_size, n, n)
    or (n, n)
    :return: The positive semidefinite matrix.
    """

    # Add a small constant to the diagonal to make sure the matrix is positive definite
    small_constant = 0.00001

    if mat.shape[0] == mat.shape[1]:
        if len(mat.shape) == 3:
            mat = (
                mat
                + torch.eye(mat.shape[-1]).repeat(mat.shape[0], 1, 1) * small_constant
            )
        else:
            mat = mat + torch.eye(mat.shape[-1]) * small_constant

    return mat


def make_symmetric(mat):
    """
    Make a matrix symmetric alongside the largest square block.
    :param mat: The matrix to make symmetric. shape (batch_size, n_1, n_2)
    or (n_1, n_1)
    :return: The symmetric matrix.
    """
    if mat.shape[-1] == mat.shape[-2]:
        return (mat + mat.transpose(-1, -2)) / 2
    else:
        min_dim = min(mat.shape[-1], mat.shape[-2])
        block = mat[..., :min_dim, :min_dim]
        block = (block + block.transpose(-1, -2)) / 2
        mat[..., :min_dim, :min_dim] = block
        return mat


def shape_and_rate_from_range(
    lower_bound: float, upper_bound: float, confidence_interval: float
) -> (float, float):
    """
    Computes the shape and rate parameters of a Gamma distribution so
    that approximately the given confidence interval is covered with
    the coverage specified by the "confidence_interval" parameter.

    The function approximates a gamma with a normal distribution
    with the same mean and variance and then computes the shape and
    rate parameters of the gamma distribution that cover the same
    confidence interval as the normal distribution.
    """
    tail = (1 - confidence_interval) / 2
    c = norm.ppf(1 - tail)

    range = upper_bound - lower_bound
    mean = (upper_bound + lower_bound) / 2
    beta = 2 * c**2 * (2 * mean) / range**2
    alpha = c**2 * (2 * mean) ** 2 / range**2
    return alpha, beta
