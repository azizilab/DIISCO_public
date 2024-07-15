from abc import ABC, abstractmethod
from numpy import ndarray
from jaxtyping import Float


class Model(ABC):
    """
    Defines the abstract interface for a model  for the task of learning latent
    interaction matrix W(t) from a sequence of observed cell-counts Y(t).
    """

    @abstractmethod
    def fit(
        self,
        t: Float[ndarray, " n_timepoints"],
        Y: Float[ndarray, "n_timepoints n_cells"],
    ) -> None:
        """
        Parameters
        ----------
        t : np.ndarray
            The time points at which the data was sampled
        Y : np.ndarray
            The observed values of the cells
        """
        pass

    @abstractmethod
    def predict_w_mean(
        self,
        t: Float[ndarray, " n_timepoints"],
        Y: Float[ndarray, "n_timepoints n_cells"] = None,
    ) -> Float[ndarray, "n_timepoints n_cells n_cells"]:
        """
        Parameters
        ----------
        t : np.ndarray
            The time points at which to predict. These don't have to be the
            same as the time points used to fit the model.
        Y : np.ndarray
            The observed values of the cells at time t. Not every model needs to use this
            and ideally no model should use this so that the model can be used to predict
            points where there is no data.
        Returns
        -------
        W_mean : np.ndarray
            The mean of the weights matrix at time t. In other words
            W_mean[t, i, j] is the mean of the weight from cell j to cell i at time t.
            where the "weight" can be any reaonsable and consistent measure quantifying
            the strength of the interaction between cell j and cell i.
        """
        pass

    def predict_obs_w_mean(
        self,
    ) -> Float[ndarray, "n_timepoints n_cells n_cells"]:
        """
        Returns
        -------
        W_mean: np.ndarray
            Returns the w_mean matrix quantifying the strength of the interaction
            between cell j and cell i. but only for the timepoints which were
            observed during the fit phase.
        """
        pass

    @property
    @abstractmethod
    def can_predict_unobserved(self) -> bool:
        """
        Returns
        -------
        can_predict_unobserved : bool
            Returns whether the model can predict unobserved timepoints.
            If true the model can use predict_w_mean to predict timepoints.
            Otherwise the model can only predict timepoints which were observed
            and hence only predict_obs_w_mean can be used.
        """
        pass
