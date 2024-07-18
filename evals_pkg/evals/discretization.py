"""
Contains code for various forms of discretization.
"""

from abc import ABC, abstractmethod
from jaxtyping import Int, Float
from numpy import ndarray
import numpy as np
import einops

# import gmm from sklearn
from sklearn.mixture import GaussianMixture
from sklearn.covariance import EllipticEnvelope


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
    standardize : bool
        Whether to standardize the interactions by the total value. In other words the
        the strength of the interaction is measured by w_i * y_i / y_j
    """

    def __init__(
        self,
        std_deviations: float = 1,
        count_zeros: bool = True,
        standardize: bool = True,
    ):
        self.std_deviations = std_deviations
        self.count_zeros = count_zeros
        self.standardize = standardize

    def __call__(
        self,
        t: Float[ndarray, " n_timepoints"],
        y: Float[ndarray, " n_timepoints n_cells"],
        interactions: Float[ndarray, " n_timepoints n_cells n_cells"],
    ) -> Int[ndarray, " n_timepoints n_cells n_cells"]:
        self._check_shapes(t, y, interactions)
        transformed_interactions = self.transform_interactions(t, y, interactions)
        return self.discretize(transformed_interactions)

    def __repr__(self):
        return self.__class__.__name__

    def discretize(
        self,
        transformed_interactions: Float[ndarray, " n_timepoints n_cells n_cells"],
    ) -> Int[ndarray, " n_timepoints n_cells n_cells"]:
        """
        Recieves, timepoints, observed values of the cells, and the predicted interactions
        and returns a boolean matrix (0 or 1) indicating whether the interaction is active
        at that timepoint or not.

        Discretizes the interactions by computing the std and choosing the interactions
        that are above a certain number of std deviations from the mean.
        """
        # Just for safety
        transformed_interactions = np.abs(transformed_interactions)
        flat_cell_interactions = transformed_interactions.flatten()
        if self.count_zeros:
            cell_std = flat_cell_interactions.std()
        else:

            values = flat_cell_interactions[flat_cell_interactions != 0]
            cell_std = values.std()

        return (transformed_interactions > self.std_deviations * cell_std).astype(int)

    @abstractmethod
    def transform_interactions(
        self,
        t: Float[ndarray, " n_timepoints"],
        y: Float[ndarray, " n_timepoints n_cells"],
        interactions: Float[ndarray, " n_timepoints n_cells n_cells"],
    ) -> Float[ndarray, " n_timepoints n_cells n_cells"]:
        """
        Tranforms the interactions before discretizing them.
        This can be done via a wide variety of methods. For example,
        shapley values, or some other form of normalization.
        """

    def _check_shapes(
        self,
        t: Float[ndarray, " n_timepoints"],
        y: Float[ndarray, " n_timepoints n_cells"],
        interactions: Float[ndarray, " n_timepoints n_cells n_cells"],
    ) -> None:
        """
        Check that the input shapes are as expected.
        """

        assert t.ndim == 1
        assert y.ndim == 2
        assert interactions.ndim == 3

        n_timepoints, n_cells = y.shape
        assert t.shape == (n_timepoints,)
        assert interactions.shape == (n_timepoints, n_cells, n_cells)


class AbsoluteValueDiscretizer(Discretizer):
    """
    Discretizes the interaction by taking the absolute values
    of the interactions. This is only a good idea if the interactions
    the data is centered and scaled. Otherwise, the magnitudes of
    the interactions will be uninformative.

    Parameters
    ----------
    std_deviations : float
        Deviations from the mean to consider as active.
    count_zeros : bool
        Whether to count zeros for the computation of the standard deviation.
    standardize : bool
        Whether to standardize the interactions by the total value. In other words the
        the strength of the interaction is measured by w_i * y_i / y_j
    """

    def __init__(
        self,
        std_deviations: float = 1,
        count_zeros: bool = True,
        standardize: bool = True,
    ):
        super().__init__(std_deviations, count_zeros, standardize)

    def transform_interactions(
        self,
        t: Float[ndarray, " n_timepoints"],
        y: Float[ndarray, " n_timepoints n_cells"],
        interactions: Float[ndarray, " n_timepoints n_cells n_cells"],
    ) -> Float[ndarray, " n_timepoints n_cells n_cells"]:

        self._check_shapes(t, y, interactions)
        return np.abs(interactions)
