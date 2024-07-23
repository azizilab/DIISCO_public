"""
Simple linear regression model used for testing purposes.
The model, learns a simple linear relationship between the input and output.
"""

from evals.models.base import Model, register_model
from jaxtyping import Float, Int
from numpy import ndarray
from typing import Optional
from sklearn.linear_model import LinearRegression, Ridge
import shap
import numpy as np
from evals.models.utils import check_fit_inputs


@register_model
class RollingLinearModel(Model):
    """
    Defines the abstract interface for a model  for the task of
    learning latent intercations between cells using a rolling linear model.

    The model works as follows. At each timpoint t there is a separate local
    linear model that predicts the value of cell i using the values of the other
    cells at time t. It is local in the sense that it only uses training data
    from a small interval around time t. Then we use Shapley values to quantify
    the strength of the interaction between cell j and cell i at time t.

    The size of the interval around time t is controlled by an epsilon parameter
    which is determined cell-wise during the fitting phase by performing a grid
    search with leave-one-out cross validation.

    Parameters
    ----------
    min_points_per_regression : int
        The minimum number of points to use for any given linear model. Essentially,
        the epsilon parameter is grown until we have at least this many points to
        use for the regression when predicting the value of cell i.
    num_searches_per_timepoint : int
        Determines the number of epsilon values to try during the grid search
        for each cell. The search range for epsilon is (0, t_range / 2) where t_range
        is the range of the timepoints provided during the fitting phase.
    """

    def __init__(
        self,
        min_points_per_regression: int = 3,
        num_searches_per_timepoint: int = 10,
        use_bias: bool = False,
        ignore_is_active: bool = False,
    ):

        self.min_points_per_regression = min_points_per_regression
        self.num_searches_per_timepoint = num_searches_per_timepoint
        self.use_bias = use_bias
        self.ignore_is_active = ignore_is_active

        self._is_active: Int[ndarray, "n_cells n_cells"] = None
        self._n_cells: int = None
        self._n_timepoints: int = None
        self._is_fitted: bool = False

        self._best_epsilon_per_cell: Float[ndarray, " n_cells"] = None

        self._t_train: Float[ndarray, " n_timepoints"] = None
        self._y_train: Float[ndarray, "n_timepoints n_cells"] = None

    def fit(
        self,
        t: Float[ndarray, " n_timepoints"],
        y: Float[ndarray, "n_timepoints n_cells"],
        is_active: Int[ndarray, "n_cells n_cells"]=None,
    ) -> None:
        """
        Fits the model to the data by saving the observed values of the cells
        and timepoints while performing a grid search to find the best epsilon
        for each cell by using leave-one-out cross validation.

        Parameters
        ----------
        t : np.ndarray
            The time points at which the data was sampled
        y : np.ndarray
            The observed values of the cells
        is_active:
            A matrix containing 1 if the edge is active, 0 otherwise
        """
        if self.ignore_is_active:
            is_active = np.ones((y.shape[1], y.shape[1]))

        check_fit_inputs(t, y, is_active)

        self._is_active = is_active
        self._y_train = y
        self._t_train = t

        self._n_timepoints = t.shape[0]
        self._n_cells = y.shape[1]
        self._is_fitted = True

        self._best_epsilon_per_cell = np.zeros(self._n_cells)

        for cell in range(self._n_cells):
            is_active_cell_idx = [
                i for i in range(self._n_cells) if is_active[cell, i] == 1
            ]
            lm_X = y[:, is_active_cell_idx]
            lm_y = y[:, cell]

            best_loss = np.inf
            t_range = np.max(t) - np.min(t)

            # We go for a range of possible epsilon and try the model with each
            # of them. We keep the epsilon that gives the best loss.
            for epsilon in np.linspace(0, t_range / 2, self.num_searches_per_timepoint):
                y_pred = _looc_preds_for_rolling_linear_model(
                    X=lm_X,
                    y=lm_y,
                    timepoints=t,
                    epsilon=epsilon,
                    min_points_per_regression=self.min_points_per_regression,
                    use_bias=self.use_bias,
                )
                loss = np.mean((y_pred - lm_y) ** 2)
                if loss < best_loss:
                    best_loss = loss
                    self._best_epsilon_per_cell[cell] = epsilon

    def predict_y_train(
        self,
    ) -> Float[ndarray, "n_timepoints n_cells"]:
        """
        Predicts the observed values of the cells at the timepoints used during the
        fitting phase.

        This method is useful for debugging, testing, and for evaluating the quality
        of the learned model on training data.
        """
        if not self._is_fitted:
            raise ValueError("The model has not been fitted yet.")

        y_train_pred = np.zeros((self._n_timepoints, self._n_cells))
        t_train = self._t_train

        for cell in range(self._n_cells):
            is_active_cell_idx = [
                i for i in range(self._n_cells) if self._is_active[cell, i] == 1
            ]
            lm_X = self._y_train[:, is_active_cell_idx]
            lm_y = self._y_train[:, cell]

            y_train_pred[:, cell] = _looc_preds_for_rolling_linear_model(
                X=lm_X,
                y=lm_y,
                timepoints=t_train,
                epsilon=self._best_epsilon_per_cell[cell],
                min_points_per_regression=self.min_points_per_regression,
                use_bias=self.use_bias,
            )

        return y_train_pred

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
            lm_X_train = self._y_train[:, active_cell_idx]
            lm_y_train = self._y_train[:, cell]
            for t_num, t_instance in enumerate(t):
                model = _get_rolling_linear_model(
                    t_to_pred=t_instance,
                    X_train=lm_X_train,
                    y_train=lm_y_train,
                    t_train=self._t_train,
                    epsilon=self._best_epsilon_per_cell[cell],
                    min_points_per_regression=self.min_points_per_regression,
                    use_bias=self.use_bias,
                )
                interaction_values[t_num, cell, active_cell_idx] = model.coef_

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


def _get_datapoints_around_timepoint(
    t_instance: float,
    X: Float[ndarray, "n_timepoints n_active_cells"],
    y: Float[ndarray, " n_timepoints"],
    timepoints: Float[ndarray, " n_timepoints"],
    epsilon: float,
    min_points_per_regression: int,
) -> tuple[
    Float[ndarray, "n_used_timepoints n_active_cells"],
    Float[ndarray, " n_used_timepoints"],
    Float[ndarray, " n_used_timepoints"],
]:
    """
    Returns a tuple with the X and y datapoints whose  time point is in the interval
    (t_instance - epsilon, t_instance + epsilon) or the closest `min_points_per_regression`

    Parameters
    ----------
    t_instance : float
        The time point to get the data points around
    X : np.ndarray
        The independent variables
    y : np.ndarray
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
    y_used : np.ndarray
        The dependent variables to use for the regression
    timepoints_used : np.ndarray
        The timepoints of the data associated with X_used and y_used
    """
    if X.shape[0] != y.shape[0]:
        msg = "X and y should have the same number of timepoints, "
        msg += f"but got {X.shape[0]} and {y.shape[0]}"
        raise ValueError(msg)
    if X.shape[0] != timepoints.shape[0]:
        msg = "X and timepoints should have the same number of timepoints, "
        msg += f"but got {X.shape[0]} and {timepoints.shape[0]}"
        raise ValueError(msg)
    if epsilon < 0:
        raise ValueError(f"epsilon should be positive, but got {epsilon}")
    if min_points_per_regression > X.shape[0]:
        msg = "min_points_per_regression should be less than the number of timepoints, "
        msg += f"but got {min_points_per_regression} and {X.shape[0]}"
        raise ValueError(msg)

    distances = np.abs(timepoints - t_instance)
    points_in_interval = distances <= epsilon
    num_point_in_interval = np.sum(points_in_interval)

    points_to_use: Int[ndarray, " n_used_timepoints"] = None
    if num_point_in_interval >= min_points_per_regression:
        points_to_use = np.where(points_in_interval)[0]
    else:
        closest_points = np.argsort(distances)[:min_points_per_regression]
        points_to_use = closest_points

    X_used = X[points_to_use]
    y_used = y[points_to_use]
    timepoints_used = timepoints[points_to_use]

    assert X_used.shape[0] == y_used.shape[0]
    assert X_used.shape[0] == timepoints_used.shape[0]
    assert len(points_to_use) == X_used.shape[0]
    return X_used, y_used, timepoints_used


def _get_rolling_linear_model(
    t_to_pred: float,
    X_train: Float[ndarray, "n_timepoints n_active_cells"],
    y_train: Float[ndarray, " n_timepoints"],
    t_train: Float[ndarray, " n_timepoints"],
    epsilon: float,
    min_points_per_regression: int,
    use_bias: bool,
) -> LinearRegression:
    """ """
    X_used, y_used, _ = _get_datapoints_around_timepoint(
        t_instance=t_to_pred,
        X=X_train,
        y=y_train,
        timepoints=t_train,
        epsilon=epsilon,
        min_points_per_regression=min_points_per_regression,
    )

    model = LinearRegression(fit_intercept=use_bias)
    model.fit(X_used, y_used)
    return model


def _predict_one_point_with_rolling_linear_model(
    X_to_pred: Float[ndarray, " n_active_cells"],
    t_to_pred: float,
    X_train: Float[ndarray, "n_timepoints n_active_cells"],
    y_train: Float[ndarray, " n_timepoints"],
    t_train: Float[ndarray, " n_timepoints"],
    epsilon: float,
    min_points_per_regression: int,
    use_bias: bool,
) -> Float[ndarray, " "]:
    """
    Trains a linear model using the data in X_train and y_train and
    predicts the values of y for the timepoints in t_to_pred using the
    data in X_to_pred. The linear model is trained only with points
    which are in the interval (t_instance - epsilon, t_instance + epsilon)
    or the closest `min_points_per_regression` points, whichever is larger.
    """
    model = _get_rolling_linear_model(
        t_to_pred=t_to_pred,
        X_train=X_train,
        y_train=y_train,
        t_train=t_train,
        epsilon=epsilon,
        min_points_per_regression=min_points_per_regression,
        use_bias=use_bias,
    )
    y_pred = model.predict(X_to_pred.reshape(1, -1))
    return y_pred.flatten()[0]


def _looc_preds_for_rolling_linear_model(
    X: Float[ndarray, "n_timepoints n_active_cells"],
    y: Float[ndarray, " n_timepoints"],
    timepoints: Float[ndarray, " n_timepoints"],
    epsilon: float,
    min_points_per_regression: int,
    use_bias: bool,
) -> Float[ndarray, " n_timepoints"]:
    """ """
    y_preds = np.zeros(y.shape)

    # We will essentially do a leave-one-out cross validation
    # to predict the value of each cell at each timepoint.
    for i, t_instance in enumerate(timepoints):
        X_without_i = np.delete(X, i, axis=0)
        y_without_i = np.delete(y, i)
        timepoints_without_i = np.delete(timepoints, i)

        y_preds[i] = _predict_one_point_with_rolling_linear_model(
            X_to_pred=X[i],
            t_to_pred=t_instance,
            X_train=X_without_i,
            y_train=y_without_i,
            t_train=timepoints_without_i,
            epsilon=epsilon,
            min_points_per_regression=min_points_per_regression,
            use_bias=use_bias,
        )

    return y_preds
