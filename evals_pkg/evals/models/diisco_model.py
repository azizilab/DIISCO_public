from numpy import ndarray
from jaxtyping import Float, Bool
import torch

import diisco.names as names
import diisco.diisco as external_diisco
from evals.models import Model
from evals.models.base import register_model

# This is needed at the moment for numerical stability
# but ideally it should be fixed in future versions of the model.
_DEFAULT_TENSOR_TYPE = torch.DoubleTensor
_DEFAULT_DTYPE = torch.float64

_DEFAULT_GUIDE = "DiagonalNormal"
_SUBSAMPLE_PROPORTION = 1


@register_model
class DiiscoModel(Model):
    """
    Defines the abstract interface for a model  for the task of learning latent
    interaction matrix W(t) from a sequence of observed cell-counts Y(t).
    """

    def __init__(
        self,
        w_length_scale: float = 1.0,
        w_length_scale_range: float = 0.2,
        w_variance: float = 3.0,
        y_length_scale: float = 3,
        y_variance: float = 3.0,
        y_sigma: float = 0.2,
        lr: float = 0.0001,
        verbose: bool = True,
        verbose_freq: bool = 100,
        patience: int = 5000,
        n_iter: int = 100_000,
    ):
        """
        The conventions for the hyper-parameters are according to best practices.

        Parameters
        ----------
        w_length_scale : float
            Specifies the mean of the lengthscale of the kernel used to model the
            interaction matrix W(t). The lengthscale determines how quickly the
            correlation between two cells decays as the distance between them
            increases. A larger lengthscale means that the correlation decays more
            slowly. A smaller lengthscale means that the correlation decays more
            quickly. The lengthscale should be positive.
        w_length_scale_range : float
            Specifies the range of the lengthscale of the kernel used to model the
            The lengthscale is drawn from a hyperprior such that 90% of the time
            the lengthscale is within the range [w_length_scale - w_length_scale_range,
            w_length_scale + w_length_scale_range]. The lengthscale should be positive.
        w_variance : float
            Specifies the variance of the kernel used to model the interaction matrix W(t).
            The variance determines how much the values of the interaction matrix can vary
            from the mean. The variance should be positive. A variance of 1.0 means that
            63% of the values of the interaction matrix are within the range [-1, 1].
        y_length_scale : float
            Specifies the lengthscale of the kernel used to model the observed values of the
            the cells Y(t). The lengthscale determines how quickly the correlation between
            two cells decays as the distance between them increases. A larger lengthscale
            means that the correlation decays more slowly. A smaller lengthscale means that
            the correlation decays more quickly. The lengthscale should be positive.
        y_sigma : float
            Specifies the standard deviation of the noise in the observed values of the cells
            Y(t). The noise is assumed to be Gaussian with a standard deviation of y_sigma.
            The noise should be positive.
        y_variance : float
            Specifies the variance of the kernel used to model the observed values of the cells
            Y(t). The variance determines how much the values of the observed values of the cells
            can vary from the mean. The variance should be positive. A variance of 1.0 means that
            63% of the values of the observed values of the cells are within the range [-1, 1].
        lr : float
            Learning rate for opimization.
        verbose : bool
            Whether to print out information during optimization.
        verbose_freq : int
            How often to print out information during optimization.
        patience : int
            How many iterations to wait before stopping optimization if the loss
            has not improved.
        n_iter : int
            The maximum number of iterations to run optimization for.
        """

        # DIISCO HYPER-PARAMETERS
        # All Diisco hyper-parameters. The hyper-parameters
        # If a hyper-parameter has a reaonable value that value is
        # used here and a small explanation of why that value is
        # reasonable is provided
        self.w_length_scale = w_length_scale
        self.w_length_scale_range = w_length_scale_range
        self.w_variance = w_variance
        # We can set this value to be very small because the
        # snr should be very high
        self.w_sigma = w_variance / 100
        self.y_sigma = y_sigma
        self.f_sigma = y_sigma / 1
        self.f_length_scale = y_length_scale
        self.f_variance = y_variance  # (f and y are the same essentially)

        # OPTIMIZATION HYPER-PARAMETERS
        # All hyper-parameters used for optimization
        self.lr = lr
        self.verbose = verbose
        self.verbose_freq = verbose_freq
        self.patience = patience
        self.n_iter = n_iter

        # CLASS VARIABLES
        # All variables used by this class
        self._model: external_diisco.DIISCO = None
        self._is_fitted: bool = False
        self._n_cells_train: int = None
        self._n_timepoints_train: int = None
        self._t_train: Float[ndarray, " n_timepoints"] = None
        self._y_train: Float[ndarray, "n_timepoints n_cells"] = None
        # Variable for caching the means of the interactions
        self._train_means: dict = None

    def fit(
        self,
        t: Float[ndarray, " n_timepoints"],
        y: Float[ndarray, "n_timepoints n_cells"],
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
        n_timepoints, n_cells = y.shape
        assert is_active.shape == (n_cells, n_cells)
        assert t.shape == (n_timepoints,)

        self._n_cells_train = n_cells
        self._n_timepoints_train = n_timepoints
        self._t_train = t
        self._y_train = y

        hyperparameters = {
            names.LENGTHSCALE_W: self.w_length_scale,
            names.LENGTHSCALE_W_RANGE: self.w_length_scale_range,
            names.VARIANCE_W: self.w_variance,
            names.VARIANCE_F: self.f_variance,
            names.SIGMA_Y: self.y_sigma,
            names.SIGMA_W: self.w_sigma,
            names.SIGMA_F: self.f_sigma,
        }
        # Set the default tensor to be a float 64 tensor
        # This is the default tensor type for the model
        # TODO: Refactor this to do it internally in diisco
        torch.set_default_tensor_type(_DEFAULT_TENSOR_TYPE)

        # Convert the data to torch tensors and ensure it has the correct
        # shape for the model
        t_tensor = torch.tensor(t, dtype=_DEFAULT_DTYPE).reshape(-1, 1)
        y_tensor = torch.tensor(y, dtype=_DEFAULT_DTYPE)
        is_active_tensor = torch.tensor(is_active, dtype=_DEFAULT_DTYPE)

        self._model = external_diisco.DIISCO(
            lambda_matrix=is_active_tensor,
            hypers_init_vals=hyperparameters,
            verbose=self.verbose,
            verbose_freq=self.verbose_freq,
        )

        self._model.fit(
            timepoints=t_tensor,
            proportions=y_tensor,
            n_iter=self.n_iter,
            lr=self.lr,
            patience=self.patience,
            guide=_DEFAULT_GUIDE,
            subsample_size=int(_SUBSAMPLE_PROPORTION * t.shape[0]),
        )

        self._is_fitted = True

    def predict_interactions(
        self,
        t: Float[ndarray, " n_timepoints"],
        y: Float[ndarray, "n_timepoints n_cells"] = None,
    ) -> Float[ndarray, "n_timepoints n_cells n_cells"]:
        """
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
            Interaction matrix at time t.
            interaction[t, i, j] is the mean weight from cell j to cell i at time t.
            where the "weight" can be any reaonsable and consistent measure quantifying
            the strength of the interaction between cell j and cell i.
        """
        raise NotImplementedError("This method has not been implemented yet.")

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
        if not self._is_fitted:
            raise ValueError("The model has not been fitted yet.")

        means = self._get_train_means_alternative()
        W_mean = means[names.W]
        assert W_mean.shape == (
            self._n_timepoints_train,
            self._n_cells_train,
            self._n_cells_train,
        )
        return W_mean

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

        means = self._get_train_means_alternative()
        Y_mean = means[names.Y]
        assert Y_mean.shape == (self._n_timepoints_train, self._n_cells_train)
        return Y_mean

    def _get_train_means(self):
        """
        Returns the means of the interactions at the timepoints used during the
        fitting phase.
        """
        if not self._is_fitted:
            raise ValueError("The model has not been fitted yet.")

        if self._train_means is None:
            t_tensor = torch.tensor(self._t_train, dtype=_DEFAULT_DTYPE).reshape(-1, 1)
            self._train_means: dict = self._model.get_means(timepoints=t_tensor)
            self._train_means = {
                key: val
                for key, val in self._train_means.items()
                if key in [names.W, names.Y]
            }
            # convert to numpy
            self._train_means = {
                key: val.detach().numpy() for key, val in self._train_means.items()
            }

        return self._train_means


    def _get_train_means_alternative(self):
        if not self._is_fitted:
            raise ValueError("The model has not been fitted yet.")

        if self._train_means is None:
            W = self._model.get_W_mean(n_samples=100)
            f = self._model.get_f_mean(n_samples=100)
            assert W.shape == (self._n_cells_train, self._n_cells_train, self._n_timepoints_train)
            assert f.shape == (self._n_timepoints_train, self._n_cells_train)

            W  = W.permute(2, 0, 1)
            Y = torch.matmul(W, f.unsqueeze(-1)).squeeze(-1)
            self._train_means = {names.W: W.detach().numpy(), names.Y: Y.detach().numpy()}

        return self._train_means



