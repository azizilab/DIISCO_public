"""
Contains code for various forms of discretization.
"""

from abc import ABC, abstractmethod
from jaxtyping import Int, Float
from numpy import ndarray
import numpy as np
import einops


class Discretizer(ABC):
    """
    Abstract class defining the API for discretizing the output
    of an interaction matrix. This is useful for models that
    predict continuous values for the interactions.

    The user of this class should implement the discretize method

    Parameters
    ----------
    std_deviations : float
        Deviations from the mean to consider as active.
    count_zeros : bool
        Whether to count zeros for the computation of the standard deviation.
    """

    def __init__(self, std_deviations: float = 1, count_zeros: bool = False):
        self.std_deviations = std_deviations
        self.count_zeros = count_zeros

    def __call__(
        self,
        t: Float[ndarray, " n_timepoints"],
        y: Float[ndarray, " n_timepoints, n_cells"],
        interactions: Float[ndarray, " n_timepoints, n_cells, n_cells"],
    ) -> Int[ndarray, " n_timepoints, n_cells, n_cells"]:
        self.check_shapes(t, y, interactions)
        transformed_interactions = self.transform_interactions(t, y, interactions)
        return self.discretize(transformed_interactions)

    def __repr__(self):
        return self.__class__.__name__

    def discretize(
        self,
        transformed_interactions: Float[ndarray, " n_timepoints, n_cells, n_cells"],
    ) -> Int[ndarray, " n_timepoints, n_cells, n_cells"]:
        """
        Recieves, timepoints, observed values of the cells, and the predicted interactions
        and returns a boolean matrix (0 or 1) indicating whether the interaction is active
        at that timepoint or not.
        """

        abs_interactions = np.abs(transformed_interactions)

        threshold = self.std_deviations * abs_interactions.mean()
        if not self.count_zeros:
            values = abs_interactions.flatten()
            values = values[values != 0]
            threshold = self.std_deviations * values.std()

        return (abs_interactions > threshold).astype(int)

    @abstractmethod
    def transform_interactions(
        self,
        t: Float[ndarray, " n_timepoints"],
        y: Float[ndarray, " n_timepoints, n_cells"],
        interactions: Float[ndarray, " n_timepoints, n_cells, n_cells"],
    ) -> Float[ndarray, " n_timepoints, n_cells, n_cells"]:
        """
        Tranforms the interactions before discretizing them.
        This can be done via a wide variety of methods. For example,
        shapley values, or some other form of normalization.
        """

    def _check_shapes(
        self,
        t: Float[ndarray, " n_timepoints"],
        y: Float[ndarray, " n_timepoints, n_cells"],
        interactions: Float[ndarray, " n_timepoints, n_cells, n_cells"],
    ) -> None:
        """
        Check that the input shapes are as expected.
        """

        assert t.ndim == 1
        assert y.ndim == 3
        assert interactions.ndim == 3

        n_timepoints, n_cells, _ = y.shape
        assert t.shape == (n_timepoints,)
        assert interactions.shape == (n_timepoints, n_cells, n_cells)


class MultiplicationDiscretizer(Discretizer):
    """
    Discretizes the interaction by assuming that w_i * y_i
    provides the strength of the interaction between cell i and cell j
    when predicting cell j.

    Parameters
    ----------
    std_deviations : float
        Deviations from the mean to consider as active.
    count_zeros : bool
        Whether to count zeros for the computation of the standard deviation.
    """

    def __init__(self, std_deviations: float = 1, count_zeros: bool = False):
        super().__init__(std_deviations, count_zeros)

    def transform_interactions(
        self,
        t: Float[ndarray, " n_timepoints"],
        y: Float[ndarray, " n_timepoints, n_cells"],
        interactions: Float[ndarray, " n_timepoints, n_cells, n_cells"],
    ) -> Float[ndarray, " n_timepoints, n_cells, n_cells"]:

        self._check_shapes(t, y, interactions)
        n_timepoints, n_cells = y.shape

        # transformed_interactions[timepoint, cell, other_cell] = interactions[timepoint, cell, other_cell] * y[timepoint, other_cell]
        transformed_interactions = einops.einsum(
            interactions,
            y,
            "timepoint out_cell in_cell, timepoint in_cell -> timepoint out_cell in_cell",
        )
        assert transformed_interactions.shape == (n_timepoints, n_cells, n_cells)
        return transformed_interactions
