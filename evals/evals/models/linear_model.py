"""
Simple linear regression model used for testing purposes.
The model, learns a simple linear relationship between the input and output.
"""

from evals.models.base import Model, register_model
from jaxtyping import Float, Int
from numpy import ndarray
from typing import Optional
from sklearn.linear_model import LinearRegression
import shap
import numpy as np


def _check_fit_inputs(t: Float[ndarray, "n_timepoints"], Y: Float[ndarray, "n_timepoints n_cells"], is_active: Int[ndarray, "n_cells n_cells"]) -> None:
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


@register_model
class LinearModel(Model):
    """
    Defines the abstract interface for a model  for the task of learning latent
    interaction matrix W(t) from a sequence of observed cell-counts Y(t).

    This model learns to predict y(t)_i = W(t)_i @ y(t)_{-i}. Additionally
    a is_active matrix can be used to only use certain cells to predict the
    value of cell i.
    """

    def __init__(self):
        self._models : list[LinearRegression] = None # len(models) == n_cells
        self._is_active : Int[ndarray, "n_cells n_cells"] = None
        self._n_cells : int = None
        self._n_timepoints : int = None
        self._is_fitted : bool = False

        self._t_train : Float[ndarray, " n_timepoints"] = None
        self._Y_train : Float[ndarray, "n_timepoints n_cells"] = None

    def fit(
        self,
        t: Float[ndarray, " n_timepoints"],
        Y: Float[ndarray, "n_timepoints n_cells"],
        is_active: Int[ndarray, "n_cells n_cells"],
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
        _check_fit_inputs(t, Y, is_active)

        self._n_timepoints = t.shape[0]
        self._n_cells = Y.shape[1]
        self._is_active = is_active

        for cell in range(self._n_cells):
            model = LinearRegression()
            lm_X = Y[:, [i for i in range(self._n_cells) if is_active[cell, i] == 1]]
            lm_Y = Y[:, cell]
            model.fit(lm_X, lm_Y)
            self._models.append(model)

        self._is_fitted = True

    def predict_interaction(
        self,
        t: Float[ndarray, " n_timepoints"],
        Y: Float[ndarray, "n_timepoints n_cells"] = None,
    ) -> Float[ndarray, "n_timepoints n_cells n_cells"]:
        """
        Returns Shapley values for the model at time t.

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
            The mean of the weights matrix at time t. In other words
            interaction[t, i, j] is the mean of the weight from cell j to cell i at time t.
            where the "weight" can be any reaonsable and consistent measure quantifying
            the strength of the interaction between cell j and cell i.
        """
        if not self._is_fitted:
            raise ValueError("The model has not been fitted yet.")
        if Y is None:
            raise ValueError(f"Y should be provided to predict_w_mean for the {self.__class__.__name__} model.")

        shap_values = np.zeros((self._n_timepoints, self._n_cells, self._n_cells))

        for cell in range(self._n_cells):
            active_cell_idx = [i for i in range(self._n_cells) if self._is_active[cell, i] == 1]
            lm_X = Y[:, active_cell_idx]
            X100 = shap.utils.sample(lm_X, 100)
            explainer = shap.Explainer(self._models[cell].predict, X100)
            cell_shap_values = explainer.shap_values(lm_X)
            assert cell_shap_values.shape == (self._n_timepoints, lm_X.shape[1])
            shap_values[:, cell, active_cell_idx] = cell_shap_values

        return shap_values


    def predict_obs_interaction(
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
        if not self._is_fitted:
            raise ValueError("The model has not been fitted yet.")

        return self.predict_interaction(self._t_train, self._Y_train)


    @property
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
        return True

