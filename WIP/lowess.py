# Brian Blakely is not the author, but I forgot who was

# Cell
import pandas as pd
import numpy as np
import psutil

from functools import partial
from multiprocessing import Pool, cpu_count

from collections.abc import Iterable
from sklearn import linear_model

from sklearn.base import BaseEstimator, RegressorMixin
from scipy.optimize import minimize
from scipy import linalg

from tqdm import tqdm

# Cell
cpu_count = cpu_count() 
if cpu_count > 1: cpu_count -= 1 # Let's save 1 cpu for the user

get_dist = lambda X, x: np.abs(X - x)

# Cell
def get_dist_threshold(dist, frac=0.4):
    """Identifies the minimum distance that contains the desired data fraction"""
    frac_idx = int(np.ceil(len(dist)*frac))
    dist_threshold = sorted(dist)[frac_idx]

    return dist_threshold

# Cell
dist_to_weights = lambda dist, dist_threshold=1: (1 - ((np.abs(dist)/dist_threshold).clip(0, 1) ** 3)) ** 3


# Cell
vector_to_dist_matrix = lambda x: np.abs(x.reshape(-1, 1) - x.reshape(1, -1))

# Cell
get_frac_idx = lambda x, frac: int(np.ceil(len(x) * frac)) - 1

# Cell
get_dist_thresholds = lambda x, frac_idx, dist_matrix: np.sort(dist_matrix)[:, frac_idx]

# Cell
def clean_weights(weights):
    """Normalises each models weightings and removes non-finite values"""
    with np.errstate(divide='ignore', invalid='ignore'):
        weights = weights/weights.sum(axis=0) # We'll then normalise the weights so that for each model they sum to 1 for a single data point

    weights = np.where(~np.isfinite(weights), 0, weights) # And remove any non-finite values

    return weights

def dist_2_weights_matrix(dist_matrix, dist_thresholds):
    """Converts distance matrix and thresholds to weightings"""
    weights = dist_to_weights(dist_matrix, dist_thresholds.reshape(-1, 1))
    weights = clean_weights(weights)

    return weights

# Cell
def get_full_dataset_weights_matrix(x, frac=0.4):
    """Wrapper for calculating weights from the raw data and LOWESS fraction"""
    frac_idx = get_frac_idx(x, frac)

    dist_matrix = vector_to_dist_matrix(x)
    dist_thresholds = get_dist_thresholds(x, frac_idx, dist_matrix)

    weights = dist_2_weights_matrix(dist_matrix, dist_thresholds)

    return weights

# Cell
num_fits_2_reg_anchors = lambda x, num_fits: np.linspace(x.min(), x.max(), num=num_fits)

def get_weighting_locs(x, reg_anchors=None, num_fits=None):
    """Identifies the weighting locations for the provided dataset"""
    num_type_2_dist_rows = {
        type(None) : lambda x, num_fits: x.reshape(-1, 1),
        int : lambda x, num_fits: num_fits_2_reg_anchors(x, num_fits).reshape(-1, 1),
    }

    if reg_anchors is None:
        weighting_locs = num_type_2_dist_rows[type(num_fits)](x, num_fits)
    else:
        weighting_locs = reg_anchors.reshape(-1, 1)

    return weighting_locs

def create_dist_matrix(x, reg_anchors=None, num_fits=None):
    """Constructs the distance matrix for the desired weighting locations"""
    weighting_locs = get_weighting_locs(x, reg_anchors=reg_anchors, num_fits=num_fits)
    dist_matrix = np.abs(weighting_locs - x.reshape(1, -1))

    return dist_matrix

# Cell
def get_weights_matrix(x, frac=0.4, weighting_locs=None, reg_anchors=None, num_fits=None):
    """Wrapper for calculating weights from the raw data and LOWESS fraction"""
    frac_idx = get_frac_idx(x, frac)

    if weighting_locs is not None:
        dist_matrix = np.abs(weighting_locs - x.reshape(1, -1))
    else:
        dist_matrix = create_dist_matrix(x, reg_anchors=reg_anchors, num_fits=num_fits)

    dist_thresholds = get_dist_thresholds(x, frac_idx, dist_matrix)
    weights = dist_2_weights_matrix(dist_matrix, dist_thresholds)

    return weights

# Cell
def calc_lin_reg_betas(x, y, weights=None):
    """Calculates the intercept and gradient for the specified local regressions"""
    if weights is None:
        weights = np.ones(len(x))

    b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
    A = np.array([[np.sum(weights), np.sum(weights * x)],
                  [np.sum(weights * x), np.sum(weights * x * x)]])

    betas = np.linalg.lstsq(A, b, rcond=None)[0]

    return betas

# Cell
check_array = lambda array, x: np.ones(len(x)) if array is None else array

def fit_regressions(x, y, weights=None, reg_func=calc_lin_reg_betas, num_coef=2, **reg_params):
    """Calculates the design matrix for the specified local regressions"""
    if weights is None:
        weights = np.ones(len(x))
    n = weights.shape[0]

    y_pred = np.zeros(n)
    design_matrix = np.zeros((n, num_coef))

    for i in range(n):
        design_matrix[i, :] = reg_func(x, y, weights=weights[i, :], **reg_params)
    
    # Only slows things down
    # partial_function = partial(reg_func,x,y,**reg_params)
    # with Pool(4) as pool:
    #     design_matrix = pool.map(partial_function,weights)
    # design_matrix = np.vstack(design_matrix)
    
    return design_matrix

# Cell
def lowess_fit_and_predict(x, y, frac=0.4, reg_anchors=None, num_fits=None, x_pred=None):
    """Fits and predicts smoothed local regressions at the specified locations"""
    weighting_locs = get_weighting_locs(x, reg_anchors=reg_anchors, num_fits=num_fits)
    weights = get_weights_matrix(x, frac=frac, weighting_locs=weighting_locs)
    design_matrix = fit_regressions(x, y, weights)

    if x_pred is None:
        x_pred = x

    point_evals = design_matrix[:, 0] + np.dot(x_pred.reshape(-1, 1), design_matrix[:, 1].reshape(1, -1))
    pred_weights = get_weights_matrix(x_pred, frac=frac, reg_anchors=weighting_locs)

    y_pred = np.multiply(pred_weights, point_evals.T).sum(axis=0)

    return y_pred

# Cell
def calc_robust_weights(y, y_pred, max_std_dev=6):
    """Calculates robustifying weightings that penalise outliers"""
    residuals = y - y_pred
    std_dev = np.quantile(np.abs(residuals), 0.682)

    cleaned_residuals = np.clip(residuals / (max_std_dev * std_dev), -1, 1)
    robust_weights = (1 - cleaned_residuals ** 2) ** 2

    return robust_weights

# Cell
def robust_lowess_fit_and_predict(x, y, frac=0.4, reg_anchors=None, num_fits=None, x_pred=None, robust_weights=None, robust_iters=3):
    """Fits and predicts robust smoothed local regressions at the specified locations"""
    # Identifying the initial loading weights
    weighting_locs = get_weighting_locs(x, reg_anchors=reg_anchors, num_fits=num_fits)
    loading_weights = get_weights_matrix(x, frac=frac, weighting_locs=weighting_locs)

    # Robustifying the weights (to reduce outlier influence)
    if robust_weights is None:
        robust_loading_weights = loading_weights
    else:
        robust_loading_weights = np.multiply(robust_weights, loading_weights)

        with np.errstate(divide='ignore', invalid='ignore'):
            robust_loading_weights = robust_loading_weights/robust_loading_weights.sum(axis=0)

        robust_loading_weights = np.where(~np.isfinite(robust_loading_weights), 0, robust_loading_weights)

    # Fitting the model and making predictions
    design_matrix = fit_regressions(x, y, robust_loading_weights)

    if x_pred is None:
        x_pred = x

    point_evals = design_matrix[:, 0] + np.dot(x_pred.reshape(-1, 1), design_matrix[:, 1].reshape(1, -1))
    pred_weights = get_weights_matrix(x_pred, frac=frac, reg_anchors=weighting_locs)

    y_pred = np.multiply(pred_weights, point_evals.T).sum(axis=0)

    # Recursive robust regression
    robust_weights = calc_robust_weights(y, y_pred)

    if robust_iters > 1:
        robust_iters -= 1
        y_pred = robust_lowess_fit_and_predict(x, y, frac=frac, reg_anchors=reg_anchors, num_fits=num_fits, x_pred=x_pred, robust_weights=robust_weights, robust_iters=robust_iters)

    return y_pred

# Cell
class Lowess(BaseEstimator, RegressorMixin):
    """
    This class provides a Scikit-Learn compatible model for Locally Weighted
    Scatterplot Smoothing, including robustifying procedures against outliers.

    For more information on the underlying algorithm please refer to
    * William S. Cleveland: "Robust locally weighted regression and smoothing
      scatterplots", Journal of the American Statistical Association, December 1979,
      volume 74, number 368, pp. 829-836.
    * William S. Cleveland and Susan J. Devlin: "Locally weighted regression: An
      approach to regression analysis by local fitting", Journal of the American
      Statistical Association, September 1988, volume 83, number 403, pp. 596-610.

    Example Usage:
    ```
    x = np.linspace(0, 5, num=150)
    y = np.sin(x)
    y_noisy = y + (np.random.normal(size=len(y)))/10

    lowess = Lowess()
    lowess.fit(x, y_noisy, frac=0.2)

    x_pred = np.linspace(0, 5, 26)
    y_pred = lowess.predict(x_pred)
    ```

    Initialisation Parameters:
        reg_func: function that accepts the x and y values then returns the intercepts and gradients

    Attributes:
        reg_func: function that accepts the x and y values then returns the intercepts and gradients
        fitted: Boolean flag indicating whether the model has been fitted
        frac: Fraction of the dataset to use in each local regression
        weighting_locs: Locations of the local regression centers
        loading_weights: Weights of each data-point across the localalised models
        design_matrix: Regression coefficients for each of the localised models
    """

    def __init__(self, reg_func=calc_lin_reg_betas):
        self.reg_func = reg_func
        self.fitted = False
        return


    def calculate_loading_weights(self, x, reg_anchors=None, num_fits=None, external_weights=None, robust_weights=None):
        """
        Calculates the loading weights for each data-point across the localised models

        Parameters:
            x: values for the independent variable
            reg_anchors: Locations at which to center the local regressions
            num_fits: Number of locations at which to carry out a local regression
            external_weights: Further weighting for the specific regression
            robust_weights: Robustifying weights to remove the influence of outliers
        """

        # Calculating the initial loading weights
        weighting_locs = get_weighting_locs(x, reg_anchors=reg_anchors, num_fits=num_fits)
        loading_weights = get_weights_matrix(x, frac=self.frac, weighting_locs=weighting_locs)

        # Applying weight adjustments
        if external_weights is None:
            external_weights = np.ones(x.shape[0])

        if robust_weights is None:
            robust_weights = np.ones(x.shape[0])

        weight_adj = np.multiply(external_weights, robust_weights)
        loading_weights = np.multiply(weight_adj, loading_weights)

        # Post-processing weights
        with np.errstate(divide='ignore', invalid='ignore'):
            loading_weights = loading_weights/loading_weights.sum(axis=0) # normalising

        loading_weights = np.where(~np.isfinite(loading_weights), 0, loading_weights) # removing non-finite values

        self.weighting_locs = weighting_locs
        self.loading_weights = loading_weights

        return


    def fit(self, x, y, frac=0.4, reg_anchors=None,
            num_fits=None, external_weights=None,
            robust_weights=None, robust_iters=3, **reg_params):
        """
        Calculation of the local regression coefficients for
        a LOWESS model across the dataset provided. This method
        will reassign the `frac`, `weighting_locs`, `loading_weights`,
        and `design_matrix` attributes of the `Lowess` object.

        Parameters:
            x: values for the independent variable
            y: values for the dependent variable
            frac: LOWESS bandwidth for local regression as a fraction
            reg_anchors: Locations at which to center the local regressions
            num_fits: Number of locations at which to carry out a local regression
            external_weights: Further weighting for the specific regression
            robust_weights: Robustifying weights to remove the influence of outliers
            robust_iters: Number of robustifying iterations to carry out
        """

        self.frac = frac

        # Solving for the design matrix
        self.calculate_loading_weights(x, reg_anchors=reg_anchors, num_fits=num_fits, external_weights=external_weights, robust_weights=robust_weights)
        self.design_matrix = fit_regressions(x, y, weights=self.loading_weights, reg_func=self.reg_func, **reg_params)

        # Recursive robust regression
        if robust_iters > 1:
            y_pred = self.predict(x,return_pairs=False)
            robust_weights = calc_robust_weights(y, y_pred)

            robust_iters -= 1
            y_pred = self.fit(x, y, frac=self.frac, reg_anchors=reg_anchors, num_fits=num_fits, external_weights=external_weights, robust_weights=robust_weights, robust_iters=robust_iters, **reg_params)

            return y_pred

        self.fitted = True

        return


    def predict(self, x_pred, return_pairs = True):
        """
        Inference using the design matrix from the LOWESS fit

        Parameters:
            x_pred: Locations for the LOWESS inference

        Returns:
            y_pred: Estimated values using the LOWESS fit
        """
        
        point_evals = self.design_matrix[:, 0] + np.dot(x_pred.reshape(-1, 1), self.design_matrix[:, 1].reshape(1, -1))
        pred_weights = get_weights_matrix(x_pred, frac=self.frac, reg_anchors=self.weighting_locs)
        
        y_pred = np.multiply(pred_weights, point_evals.T).sum(axis=0)
        if return_pairs:
            return np.array([x_pred,y_pred]).T
        return y_pred
    
    
    def predict_fast(self, x_pred, chunk_n = None):
        # Utilize all cores
        if chunk_n is None:
            # Try to size our chunks to best distribute our data
            chunk_n = len(x_pred) // cpu_count
            # See if we have enough memory to support that chunkation, if not, reduce size by half until we can
            while x_pred[:chunk_n].nbytes * cpu_count >= psutil.virtual_memory()[1]:
                chunk_n //= 2
        # Set some arbitrary limit where its faster to serialize the operation due to parrellization overhead
        if chunk_n < 25000:
            return self.predict(x_pred,return_pairs=False)
        
        chunks = [x_pred[i:i+chunk_n] for i in range(0,x_pred.shape[0],chunk_n)]
        
        del x_pred
        
        with Pool(cpu_count) as pool:
            result = pool.map(self.predict,chunks)
        
        return result
    
def loess_predict(x_pred,design_matrix,frac,weighting_locs,return_pairs = True):
    point_evals = design_matrix[:, 0] + np.dot(x_pred.reshape(-1, 1), design_matrix[:, 1].reshape(1, -1))
    pred_weights = get_weights_matrix(x_pred, frac=frac, reg_anchors=weighting_locs)
    
    y_pred = np.multiply(pred_weights, point_evals.T).sum(axis=0)
    if return_pairs:
        return np.array([x_pred,y_pred]).T
    return y_pred

