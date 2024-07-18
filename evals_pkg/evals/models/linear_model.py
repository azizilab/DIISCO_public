"""
Simple linear regression model used for testing purposes.
The model, learns a simple linear relationship between the input and output.
"""

from evals.models.base import Model, register_model
from jaxtyping import Float, Int
from numpy import ndarray
from typing import Optional
from sklearn.linear_model import LinearRegression
import numpy as np
from evals.models.utils import check_fit_inputs


@register_model
class LinearModel(Model):
    """
    Defines the abstract interface for a model  for the task of learning latent
    interaction matrix W(t) from a sequence of observed cell-counts Y(t).

    This model learns to predict y(t)_i = W(t)_i @ y(t)_{-i}. Additionally
    a is_active matrix can be used to only use certain cells to predict the
    value of cell i.

    Parameters
    ----------
    use_bias : bool
        Whether to use a bias term in the linear model.
    """

    def __init__(self, use_bias: bool = False):
        self._use_bias = use_bias

        self._models: list[LinearRegression] = []  # len(models) == n_cells
        self._is_active: Int[ndarray, "n_cells n_cells"] = None
        self._n_cells: int = None
        self._n_timepoints: int = None
        self._is_fitted: bool = False

        self._t_train: Float[ndarray, " n_timepoints"] = None
        self._y_train: Float[ndarray, "n_timepoints n_cells"] = None

    def fit(
        self,
        t: Float[ndarray, " n_timepoints"],
        y: Float[ndarray, "n_timepoints n_cells"],
        is_active: Int[ndarray, "n_cells n_cells"],
    ) -> None:
        """
        Parameters
        ----------
        t : np.ndarray
            The time points at which the data was sampled
        y : np.ndarray
            The observed values of the cells
        is_active:
            A matrix containing 1 if the edge is active, 0 otherwise
        """
        check_fit_inputs(t, y, is_active)

        self._n_timepoints = t.shape[0]
        self._n_cells = y.shape[1]
        self._is_active = is_active

        self._t_train = t
        self._y_train = y

        for cell in range(self._n_cells):
            model = LinearRegression(fit_intercept=self._use_bias)
            lm_X = y[:, [i for i in range(self._n_cells) if is_active[cell, i] == 1]]
            lm_y = y[:, cell]
            model.fit(lm_X, lm_y)
            self._models.append(model)

        self._is_fitted = True

    def predict_interactions(
        self,
        t: Float[ndarray, " n_timepoints"],
        y: Float[ndarray, "n_timepoints n_cells"] = None,
    ) -> Float[ndarray, "n_timepoints n_cells n_cells"]:
        """
        Returns Shapley values for the model at time t.

        Parameters
        ----------
        t : np.ndarray
            The time points at which to predict. These don't have to be the
            same as the time points used to fit the model.
        y : np.ndarray
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
        if y is None:
            raise ValueError(
                f"y should be provided to predict_w_mean for the {self.__class__.__name__} model."
            )

        interaction_values = np.zeros(
            (self._n_timepoints, self._n_cells, self._n_cells)
        )

        for cell in range(self._n_cells):
            active_cell_idx = [
                i for i in range(self._n_cells) if self._is_active[cell, i] == 1
            ]
            interaction_values[:, cell, active_cell_idx] = self._models[cell].coef_

        return interaction_values

    def predict_obs_interactions(
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

        return self.predict_interactions(self._t_train, self._y_train)

    def predict_y_train(self) -> Float[ndarray, "n_timepoints n_cells"]:
        """
        Returns the predicted values of the cells at the timepoints
        used to fit the model.
        """
        if not self._is_fitted:
            raise ValueError("The model has not been fitted yet.")

        y_pred = np.zeros((self._n_timepoints, self._n_cells))
        for cell in range(self._n_cells):
            active_cell_idx = [
                i for i in range(self._n_cells) if self._is_active[cell, i] == 1
            ]
            y_pred[:, cell] = self._models[cell].predict(
                self._y_train[:, active_cell_idx]
            )

        return y_pred
