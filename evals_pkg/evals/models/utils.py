from typing import Float, Int
from numpy import ndarray

def check_fit_inputs(t: Float[ndarray, "n_timepoints"], Y: Float[ndarray, "n_timepoints n_cells"], is_active: Int[ndarray, "n_cells n_cells"]) -> None:
    if t.ndim != 1:
        raise ValueError(f"t should be a 1D array, but got shape {t.shape}")
    if Y.ndim != 2:
        raise ValueError(f"Y should be a 2D array, but got shape {Y.shape}")
    if is_active.ndim != 2:
        raise ValueError(f"is_active should be a 2D array, but got shape {is_active.shape}")
    if Y.shape[0] != t.shape[0]:
        raise ValueError(f"Y.shape[0] should be equal to t.shape[0], but got {Y.shape[0]} and {t.shape[0]}")
    if Y.shape[1] != is_active.shape[0]:
        raise ValueError(f"Y.shape[1] should be equal to is_active.shape[0], but got {Y.shape[1]} and {is_active.shape[0]}")
    if is_active.shape[0] != is_active.shape[1]:
        raise ValueError(f"is_active should be a square matrix, but got shape {is_active.shape}")
