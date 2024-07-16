from abc import ABC, abstractmethod
from numpy import ndarray
from jaxtyping import Float, Bool


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
        is_active: Bool[ndarray, "n_cells n_cells"],
    ) -> None:
        """
        Parameters
        ----------
        t : np.ndarray
            The time points at which the data was sampled
        Y : np.ndarray
            The observed values of the cells
        is_active:
            A matrix containing 1 if the edge is active, 0 otherwise
        """
        pass

    @abstractmethod
    def predict_interactions(
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
        interaction : np.ndarray
            Interaction matrix at time t.
            interaction[t, i, j] is the mean weight from cell j to cell i at time t.
            where the "weight" can be any reaonsable and consistent measure quantifying
            the strength of the interaction between cell j and cell i.
        """
        pass

    def predict_obs_interactions(
        self,
    ) -> Float[ndarray, "n_timepoints n_cells n_cells"]:
        """
        Returns
        -------
        interaction: np.ndarray
            Returns the interaction matrix quantifying the strength of the interaction
            between cell j and cell i. but only for the timepoints which were
            observed during the fit phase.
        """
        pass


AVAILABLE_MODELS = {}


# create a decorator for registering models
def register_model(cls):
    AVAILABLE_MODELS[cls.__name__] = cls
    return cls


def get_models_dict() -> dict[str, Model]:
    """
    Returns the dictionary of available models.
    The keys are the names of the models and the values are the model classes.
    """
    return AVAILABLE_MODELS
