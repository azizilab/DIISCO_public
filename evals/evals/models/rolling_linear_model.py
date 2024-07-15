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


@register_model
class RollingLinearModel(Model):
    """
    Defines the abstract interface for a model  for the task of learning latent
    interaction matrix W(t) from a sequence of observed cell-counts Y(t).

    This model learns to predict y(t)_i = W(t)_i @ y(t)_{-i}. Additionally
    a is_active matrix can be used to only use certain cells to predict the
    value of cell i.
    """

    def __init__(self,
        min_points_per_regression: int = 3,
        num_searches_per_timepoint: int = 10,
    ):

        self.min_points_per_regression = min_points_per_regression
        self.num_searches_per_timepoint = num_searches_per_timepoint

        self._is_active : Int[ndarray, "n_cells n_cells"] = None
        self._n_cells : int = None
        self._n_timepoints : int = None
        self._is_fitted : bool = False

        self._best_epsilon_per_cell : Float[ndarray, "n_cells"] = None

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

        self._is_active = is_active
        self._Y_train = Y
        self._t_train = t

        self._n_timepoints = t.shape[0]
        self._n_cells = Y.shape[1]
        self._is_fitted = True

        self._best_epsilon_per_cell = np.zeros(self._n_cells)

        for cell in range(self._n_cells):
            is_active_cell_idx = [i for i in range(self._n_cells) if is_active[cell, i] == 1]
            lm_X = Y[:, is_active_cell_idx]
            lm_Y = Y[:, cell]

            best_loss = np.inf
            t_range = np.max(t) - np.min(t)

            # We go for a range of possible epsilon and try the model with each
            # of them. We keep the epsilon that gives the best loss.
            for epsilon in np.linspace(0, t_range / 2, self.num_searches_per_timepoint):
                Y_pred = _looc_preds_for_rolling_linear_model(
                                        X=lm_X,
                                        Y=lm_Y,
                                        timepoints=t,
                                        epsilon=epsilon,
                                        min_points_per_regression=self.min_points_per_regression
                )
                loss = np.mean((Y_pred - lm_Y) ** 2)
                if loss < best_loss:
                    best_loss = loss
                    self._best_epsilon_per_cell[cell] = epsilon


    def predict_train_Y(
        self,
    ) -> Float[ndarray, "n_timepoints n_cells"]:
        """
        """
        Y_train_pred = np.zeros((self._n_timepoints, self._n_cells))
        t_train = self._t_train

        for cell in range(self._n_cells):
            is_active_cell_idx = [i for i in range(self._n_cells) if self._is_active[cell, i] == 1]
            lm_X = self._Y_train[:, is_active_cell_idx]
            lm_Y = self._Y_train[:, cell]

            Y_train_pred[:, cell] = _looc_preds_for_rolling_linear_model(
                X=lm_X,
                Y=lm_Y,
                timepoints=t_train,
                epsilon=self._best_epsilon_per_cell[cell],
                min_points_per_regression=self.min_points_per_regression
            )

        return Y_train_pred


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
            lm_X_train = self._Y_train[:, active_cell_idx]
            lm_X_pred = Y[:, active_cell_idx]
            lm_Y_train = self._Y_train[:, cell]
            for t_num, t_instance in enumerate(t):
                model = _get_rolling_linear_model(
                                t_to_pred=t_instance,
                                X_train=lm_X_train,
                                Y_train=lm_Y_train,
                                t_train=self._t_train,
                                epsilon=self._best_epsilon_per_cell[cell],
                                min_points_per_regression=self.min_points_per_regression
                )
                X100 = shap.utils.sample(lm_X_train, 100)
                explainer = shap.Explainer(model.predict, X100)
                cell_shap_values = explainer.shap_values(lm_X_pred[t_num].reshape(1, -1))
                assert cell_shap_values.shape == (1, lm_X_pred.shape[1])
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


def _get_datapoints_around_timepoint(
    t_instance: float,
    X: Float[ndarray, "n_timepoints n_active_cells"],
    Y: Float[ndarray, "n_timepoints"],
    timepoints: Float[ndarray, "n_timepoints"],
    epsilon: float,
    min_points_per_regression: int,
) -> tuple[
        Float[ndarray, "n_used_timepoints n_active_cells"],
        Float[ndarray, "n_used_timepoints"],
        Float[ndarray, "n_used_timepoints"],
    ]:
    """
    Returns a tuple with the X and Y datapoints whose  time point is in the interval
    (t_instance - epsilon, t_instance + epsilon) or the closest `min_points_per_regression`

    Parameters
    ----------
    t_instance : float
        The time point to get the data points around
    X : np.ndarray
        The independent variables
    Y : np.ndarray
        The dependent variables
    timepoints : np.ndarray
        The timepoints of the data associated with X and Y
    epsilon : float
        The half width of the interval around t_instance
    min_points_per_regression : int
        The minimum number of points to use for the regression.

    Returns
    -------
    X_used : np.ndarray
        The independent variables to use for the regression
    Y_used : np.ndarray
        The dependent variables to use for the regression
    timepoints_used : np.ndarray
        The timepoints of the data associated with X_used and Y_used
    """
    if X.shape[0] != Y.shape[0]:
        msg = f"X and Y should have the same number of timepoints, "
        msg += f"but got {X.shape[0]} and {Y.shape[0]}"
        raise ValueError(msg)
    if X.shape[0] != timepoints.shape[0]:
        msg = f"X and timepoints should have the same number of timepoints, "
        msg += f"but got {X.shape[0]} and {timepoints.shape[0]}"
        raise ValueError(msg)
    if epsilon < 0:
        raise ValueError(f"epsilon should be positive, but got {epsilon}")
    if min_points_per_regression > X.shape[0]:
        msg = f"min_points_per_regression should be less than the number of timepoints, "
        msg += f"but got {min_points_per_regression} and {X.shape[0]}"
        raise ValueError(msg)

    distances = np.abs(timepoints - t_instance)
    points_in_interval = distances <= epsilon
    num_point_in_interval = np.sum(points_in_interval)

    points_to_use : Int[ndarray, "n_used_timepoints"]
    if num_point_in_interval >= min_points_per_regression:
        points_to_use = np.where(points_in_interval)[0]
    else:
        closest_points = np.argsort(distances)[:min_points_per_regression]
        points_to_use = closest_points

    X_used = X[points_to_use]
    Y_used = Y[points_to_use]
    timepoints_used = timepoints[points_to_use]

    assert X_used.shape[0] == Y_used.shape[0]
    assert X_used.shape[0] == timepoints_used.shape[0]
    assert len(points_to_use) == X_used.shape[0]
    return X_used, Y_used, timepoints_used

def _get_rolling_linear_model(
    t_to_pred: float,
    X_train: Float[ndarray, "n_timepoints n_active_cells"],
    Y_train: Float[ndarray, "n_timepoints"],
    t_train: Float[ndarray, "n_timepoints"],
    epsilon: float,
    min_points_per_regression: int,
) -> LinearRegression:
    """
    """
    X_used, Y_used, _ = _get_datapoints_around_timepoint(
                            t_instance=t_to_pred,
                            X=X_train,
                            Y=Y_train,
                            timepoints=t_train,
                            epsilon=epsilon,
                            min_points_per_regression=min_points_per_regression
                        )

    model = LinearRegression()
    model.fit(X_used, Y_used)
    return model


def _predict_one_point_with_rolling_linear_model(
    X_to_pred: Float[ndarray, "n_active_cells"],
    t_to_pred: float,
    X_train: Float[ndarray, "n_timepoints n_active_cells"],
    Y_train: Float[ndarray, "n_timepoints"],
    t_train: Float[ndarray, "n_timepoints"],
    epsilon: float,
    min_points_per_regression: int,
) -> Float[ndarray, " "]:
    """
    Trains a linear model using the data in X_train and Y_train and
    predicts the values of Y for the timepoints in t_to_pred using the
    data in X_to_pred. The linear model is trained only with points
    which are in the interval (t_instance - epsilon, t_instance + epsilon)
    or the closest `min_points_per_regression` points, whichever is larger.
    """
    model = _get_rolling_linear_model(
                t_to_pred=t_to_pred,
                X_train=X_train,
                Y_train=Y_train,
                t_train=t_train,
                epsilon=epsilon,
                min_points_per_regression=min_points_per_regression
            )
    y_pred = model.predict(X_to_pred.reshape(1, -1))
    return y_pred.flatten()[0]


def _looc_preds_for_rolling_linear_model(
    X: Float[ndarray, "n_timepoints n_active_cells"],
    Y: Float[ndarray, "n_timepoints"],
    timepoints: Float[ndarray, "n_timepoints"],
    epsilon: float,
    min_points_per_regression: int,
) -> Float[ndarray, "n_timepoints"]:
    """

    """
    Y_pred = np.zeros(Y.shape)

    # We will essentially do a leave-one-out cross validation
    # to predict the value of each cell at each timepoint.
    for i, t_instance in enumerate(timepoints):
        X_without_i = np.delete(X, i, axis=1)
        Y_without_i = np.delete(Y, i, axis=1)
        timepoints_without_i = np.delete(timepoints, i)

        y_pred = _predict_one_point_with_rolling_linear_model(
            X_to_pred=X[i],
            t_to_pred=t_instance,
            X_train=X_without_i,
            Y_train=Y_without_i,
            t_train=timepoints_without_i,
            epsilon=epsilon,
            min_points_per_regression=min_points_per_regression
        )
        Y_pred[i] = y_pred

    return Y_pred

