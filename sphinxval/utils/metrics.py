import numpy as np
import pandas as pd
from sklearn.utils.validation import check_consistent_length
from sklearn.utils.validation import check_array
from sklearn.metrics import brier_score_loss
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.metrics import brier_score_loss
import math
import statistics
import sklearn.metrics as skl
import sys
import logging
# from contingency_space.contingency_space import ContingencySpace
# from contingency_space.confusion_matrix import ConfusionMatrix
import matplotlib.pyplot as plt
from sphinxval.utils.tau import Tau, ContingencySpace, ConfusionMatrix




__author__ = "Phil Quinn"
__maintainer__ = "Kathryn Whitman"
__email__ = "kathryn.whitman@nasa.gov"

# Changes in 0.4: Added calc_contingency_bool which generates scores from
#   boolean all clear predictions.
#   Added Brier Skill Score.
#2022-02-07, Changes in 0.5: Added RMSE and RMSLE
#2023-03-21, Changes in 0.6: Fixed HSS calculation in two places
#2023-05-08, Changes in 0.7: Added in a try and except in calc_pearson and changed
#   syntax in calc_contingency_bool as found was crashing on another system.
#2024-02-28, Changes: added Brier skill score, SEDS, and ROC curve. Added
#   comments defining metrics and their various names

'''Contains functions for calculating various metrics used
    for comparing modeled forecast results to observations.
'''

#Create logger
logger = logging.getLogger(__name__)

def switch_error_func(metric, y_true, y_pred):
    """
    Switch statement for ease of use

    Parameters
    ----------
    metric : string
        Desired metric to be calculated
        Can only be a metric from the list below

    y_true : array-like
        Observed (true) values

    y_pred : array-like
        Forecasted (estimated) values

    Returns
    -------
    error : array-like
        Error between forecast and observation
    """
    if len(y_true) < 1: return None
    
    func = {
        'E': calc_E,                        # Error
        'Ratio': calc_ratio,                # Ratio
        'AE': calc_AE,                      # Absolute Error
        'LE': calc_LE,                      # Log Error
        'ALE': calc_ALE,                    # Absolute Log Error
        'SE': calc_SE,                      # Squared Error
        'SLE': calc_SLE,                    # Squared Log Error
        'RMSE': calc_RMSE,                  # Root Mean Squared Error
        'RMSLE': calc_RMSLE,                # Root Mean Squared Log Error
        'PE': calc_PE,                      # Percent Error
        'APE': calc_APE,                    # Absolute Percent Error, absolute percentage deviation
#        'PLE': calc_PLE,                   # Percent Log Error
#        'APLE': calc_APLE,                 # Absolute Percent Log Error
        'SPE': calc_SPE,                    # Symmetric Percent Error
        'SAPE': calc_SAPE,                  # Symmetric Absolute Percent Error
#        'SPLE': calc_SPLE,                 # Symmetric Percent Log Error
 #       'SAPLE': calc_SAPLE,               # Symmetric Absolute Percent Log Error
        'r': calc_pearson,                  # Pearson, linear correlation coefficient
        'MAR': calc_MAR,                    # Mean Accuracy Ratio
        'MdSA': calc_MdSA,                  # Median Symmetric  Accuracy
        'spearman': calc_spearman,          # Spearman, rank order correlation coefficient
        # 'brier': calc_brier   #Probably shouldn't be here since is a probability metric (its also just SE)
        }.get(metric)

    if not callable(func):
        logger.error(str(metric) + " is an invalid metric.")
        raise ValueError(str(metric) + " is an invalid metric.")

    error = func(y_true, y_pred)

    return error


def calc_mean(metric):
    """
    Calculates the mean of any error type
    Ignores NANs and +/-infinity

    Parameters
    ----------
    metric : array-like
        Values of the metric for each data point

    Returns
    -------
    mean : float
        Mean error between forecast and observation
    """

    metric = np.asarray(metric)

    return np.nanmean(metric[np.isfinite(metric)])


def calc_ratio(y_true, y_pred):
    """
    Calculates a variant of standard error

    Best value is 0.0
    Range is (-inf,inf)
    No asymptotes

    Note: Defined in the non-standard way as error = y_pred - y_true
        so that a negative error means the model is underforecasting

    Parameters
    ----------
    y_true : array-like
        Observed (true) values

    y_pred : array-like
        Forecasted (estimated) values

    Returns
    -------
    error : array-like
        Error between forecast and observation

    Formula
    -------
    E = y_pred - y_true
    """

    check_consistent_length(y_true, y_pred)

    y_true = check_array(y_true, force_all_finite=True, ensure_2d=False)
    y_pred = check_array(y_pred, force_all_finite=True, ensure_2d=False)
    
    if (y_true < 0).any():
        logger.error("Logarithmic Error cannot be used when "
                         "targets contain negative values.")
        raise ValueError("Logarithmic Error cannot be used when "
                         "targets contain negative values.")

    return y_pred/y_true




def calc_E(y_true, y_pred):
    """
    Calculates a variant of standard error

    Best value is 0.0
    Range is (-inf,inf)
    No asymptotes

    Note: Defined in the non-standard way as error = y_pred - y_true
        so that a negative error means the model is underforecasting

    Parameters
    ----------
    y_true : array-like
        Observed (true) values

    y_pred : array-like
        Forecasted (estimated) values

    Returns
    -------
    error : array-like
        Error between forecast and observation

    Formula
    -------
    E = y_pred - y_true
    """

    check_consistent_length(y_true, y_pred)

    y_true = check_array(y_true, force_all_finite=True, ensure_2d=False)
    y_pred = check_array(y_pred, force_all_finite=True, ensure_2d=False)

    return y_pred - y_true


def calc_AE(y_true, y_pred):
    """
    Calculates absolute error

    Best value is 0.0
    Range is [0.0,inf)
    No asymptotes

    Note: Defined in the non-standard way as error = y_pred - y_true
        so that a negative error means the model is underforecasting

    Parameters
    ----------
    y_true : array-like
        Observed (true) values

    y_pred : array-like
        Forecasted (estimated) values

    Returns
    -------
    error : array-like
        Error between forecast and observation

    Formula
    -------
    AE = |y_pred - y_true|
    """

    check_consistent_length(y_true, y_pred)

    y_true = check_array(y_true, force_all_finite=True, ensure_2d=False)
    y_pred = check_array(y_pred, force_all_finite=True, ensure_2d=False)

    return np.abs(y_pred - y_true)


def calc_LE(y_true, y_pred):
    """
    Calculates log (base 10) error

    Best value is 0.0
    Range is (-inf,inf)
    No asymptotes

    Note: Defined in the non-standard way as error = y_pred - y_true
        so that a negative error means the model is underforecasting

    Parameters
    ----------
    y_true : array-like
        Observed (true) values

    y_pred : array-like
        Forecasted (estimated) values

    Returns
    -------
    error : array-like
        Error between forecast and observation

    Formula
    -------
    LE = log10(y_pred) - log10(y_true)
    """

    check_consistent_length(y_true, y_pred)

    y_true = check_array(y_true, force_all_finite=True, ensure_2d=False)
    y_pred = check_array(y_pred, force_all_finite=True, ensure_2d=False)

    if (y_true < 0).any() or (y_pred < 0).any():
        logger.error("Logarithmic Error cannot be used when "
                         "targets contain negative values.")
        raise ValueError("Logarithmic Error cannot be used when "
                         "targets contain negative values.")

    return np.log10(y_pred) - np.log10(y_true)


def calc_ALE(y_true, y_pred):
    """
    Calculates absolute log (base 10) error

    Best value is 0.0
    Range is [0.0,inf)
    No asymptotes

    Note: Defined in the non-standard way as error = y_pred - y_true
        so that a negative error means the model is underforecasting

    Parameters
    ----------
    y_true : array-like
        Observed (true) values

    y_pred : array-like
        Forecasted (estimated) values

    Returns
    -------
    error : array-like
        Error between forecast and observation

    Formula
    -------
    ALE = |log10(y_pred) - log10(y_true)|
    """

    check_consistent_length(y_true, y_pred)

    y_true = check_array(y_true, force_all_finite=True, ensure_2d=False)
    y_pred = check_array(y_pred, force_all_finite=True, ensure_2d=False)

    if (y_true < 0).any() or (y_pred < 0).any():
        logger.error("Absolute Logarithmic Error cannot be used when "
                         "targets contain negative values.")
        raise ValueError("Absolute Logarithmic Error cannot be used when "
                         "targets contain negative values.")

    return np.abs(np.log10(y_pred) - np.log10(y_true))


def calc_SE(y_true, y_pred):
    """
    Calculates squared error

    Best value is 0.0
    Range is [0.0,inf)
    No asymptotes

    Note: Defined in the non-standard way as error = y_pred - y_true
        so that a negative error means the model is underforecasting

    Parameters
    ----------
    y_true : array-like
        Observed (true) values

    y_pred : array-like
        Forecasted (estimated) values

    Returns
    -------
    error : array-like
        Error between forecast and observation

    Formula
    -------
    SE = (y_pred - y_true)**2
    """

    check_consistent_length(y_true, y_pred)

    y_true = check_array(y_true, force_all_finite=True, ensure_2d=False)
    y_pred = check_array(y_pred, force_all_finite=True, ensure_2d=False)

    return (y_pred - y_true)**2


def calc_SLE(y_true, y_pred):
    """
    Calculates squared log (base 10) error

    Best value is 0.0
    Range is [0.0,inf)
    No asymptotes

    Note: Defined in the non-standard way as error = y_pred - y_true
        so that a negative error means the model is underforecasting

    Parameters
    ----------
    y_true : array-like
        Observed (true) values

    y_pred : array-like
        Forecasted (estimated) values

    Returns
    -------
    error : array-like
        Error between forecast and observation

    Formula
    -------
    SLE = (log10(y_pred) - log10(y_true))**2
    """

    check_consistent_length(y_true, y_pred)

    y_true = check_array(y_true, force_all_finite=True, ensure_2d=False)
    y_pred = check_array(y_pred, force_all_finite=True, ensure_2d=False)

    if (y_true < 0).any() or (y_pred < 0).any():
        logger.error("Squared Logarithmic Error cannot be used when "
                         "targets contain negative values.")
        raise ValueError("Squared Logarithmic Error cannot be used when "
                         "targets contain negative values.")

    return (np.log10(y_pred) - np.log10(y_true))**2
    
    
#KW
def calc_RMSE(y_true, y_pred):
    """
    Calculates root mean squared error

    Best value is 0.0
    Range is [0.0,inf)
    No asymptotes

    Note: Defined in the non-standard way as error = y_pred - y_true
        so that a negative error means the model is underforecasting

    Parameters
    ----------
    y_true : array-like
        Observed (true) values

    y_pred : array-like
        Forecasted (estimated) values

    Returns
    -------
    error : float
        Error between forecast and observation

    Formula
    -------
    RMSE = sqrt(sum((y_pred - y_true)**2) / N)
    where N is the number of prediction-observation pairs
    """

    check_consistent_length(y_true, y_pred)

    y_true = check_array(y_true, force_all_finite=True, ensure_2d=False)
    y_pred = check_array(y_pred, force_all_finite=True, ensure_2d=False)
    
    error = (y_pred - y_true)**2
    rmse = math.sqrt(sum(error)/len(error))

    return rmse

#KW
def calc_RMSLE(y_true, y_pred):
    """
    Calculates root mean squared log (base 10) error

    Best value is 0.0
    Range is [0.0,inf)
    No asymptotes

    Note: Defined in the non-standard way as error = y_pred - y_true
        so that a negative error means the model is underforecasting

    Parameters
    ----------
    y_true : array-like
        Observed (true) values

    y_pred : array-like
        Forecasted (estimated) values

    Returns
    -------
    error : float
        Error between forecast and observation

    Formula
    -------

    RMSLE = sqrt(sum((log10(y_pred) - log10(y_true))**2) / N)
    where N is the number of prediction-observation pairs
    """

    check_consistent_length(y_true, y_pred)

    y_true = check_array(y_true, force_all_finite=True, ensure_2d=False)
    y_pred = check_array(y_pred, force_all_finite=True, ensure_2d=False)

    if (y_true < 0).any() or (y_pred < 0).any():
        logger.error("Squared Logarithmic Error cannot be used when "
                         "targets contain negative values.")
        raise ValueError("Squared Logarithmic Error cannot be used when "
                         "targets contain negative values.")

    error = (np.log10(y_pred) - np.log10(y_true))**2
    rmsle = math.sqrt(sum(error)/len(error))

    return rmsle



def calc_PE(y_true, y_pred):
    """
    Calculates percent error

    Best value is 0.0
    Range is (-inf,inf)
    Asymptote at y_true = 0

    Note: Defined in the non-standard way as error = y_pred - y_true
        so that a negative error means the model is underforecasting

    Parameters
    ----------
    y_true : array-like
        Observed (true) values

    y_pred : array-like
        Forecasted (estimated) values

    Returns
    -------
    error : array-like
        Error between forecast and observation

    Formula
    -------
    PE = (y_pred - y_true) / y_true
    """

    check_consistent_length(y_true, y_pred)

    y_true = check_array(y_true, force_all_finite=True, ensure_2d=False)
    y_pred = check_array(y_pred, force_all_finite=True, ensure_2d=False)

    if (y_true == 0).any():
        logger.error("Percent Error cannot be used when "
                         "targets contain values of zero.")
        raise ValueError("Percent Error cannot be used when "
                         "targets contain values of zero.")

    return (y_pred - y_true) / y_true


def calc_APE(y_true, y_pred):
    """
    Calculates absolute percent error

    Best value is 0.0
    Range is [0.0,inf)
    Asymptote at y_true = 0

    Note: Defined in the non-standard way as error = y_pred - y_true
        so that a negative error means the model is underforecasting

    Parameters
    ----------
    y_true : array-like
        Observed (true) values

    y_pred : array-like
        Forecasted (estimated) values

    Returns
    -------
    error : array-like
        Error between forecast and observation

    Formula
    -------
    APE = |(y_pred - y_true) / y_true|
    """

    check_consistent_length(y_true, y_pred)

    y_true = check_array(y_true, force_all_finite=True, ensure_2d=False)
    y_pred = check_array(y_pred, force_all_finite=True, ensure_2d=False)

    if (y_true == 0).any():
        logger.error("Absolute Percent Error cannot be used when "
                         "targets contain values of zero.")
        raise ValueError("Absolute Percent Error cannot be used when "
                         "targets contain values of zero.")

    return np.abs((y_pred - y_true) / y_true)


#def calc_PLE(y_true, y_pred):
#    """
#    Calculates percent log (base 10) error
#
#    Best value is 0.0
#    Range is (-inf,inf)
#    Asymptote at y_true = 1
#
#    Note: Defined in the non-standard way as error = y_pred - y_true
#        so that a negative error means the model is underforecasting
#
#    Parameters
#    ----------
#    y_true : array-like
#        Observed (true) values
#
#    y_pred : array-like
#        Forecasted (estimated) values
#
#    Returns
#    -------
#    error : array-like
#        Error between forecast and observation
#    """
#
#    check_consistent_length(y_true, y_pred)
#
#    y_true = check_array(y_true, force_all_finite=True, ensure_2d=False)
#    y_pred = check_array(y_pred, force_all_finite=True, ensure_2d=False)
#
#    if (y_true < 0).any() or (y_pred < 0).any():
#        raise ValueError("Percent Logarithmic Error cannot be used when "
#                         "targets contain negative values.")
#
#    if (y_true == 1).any():
#        raise ValueError("Percent Logarithmic Error cannot be used when "
#                         "targets contain values of 1.")
#
#    return (np.log10(y_pred) - np.log10(y_true)) / np.log10(y_true)
#
#
#def calc_APLE(y_true, y_pred):
#    """
#    Calculates absolute percent log (base 10) error
#
#    Best value is 0.0
#    Range is [0.0,inf)
#    Asymptote at y_true = 1
#
#    Note: Defined in the non-standard way as error = y_pred - y_true
#        so that a negative error means the model is underforecasting
#
#    Parameters
#    ----------
#    y_true : array-like
#        Observed (true) values
#
#    y_pred : array-like
#        Forecasted (estimated) values
#
#    Returns
#    -------
#    error : array-like
#        Error between forecast and observation
#    """
#
#    check_consistent_length(y_true, y_pred)
#
#    y_true = check_array(y_true, force_all_finite=True, ensure_2d=False)
#    y_pred = check_array(y_pred, force_all_finite=True, ensure_2d=False)
#
#    if (y_true < 0).any() or (y_pred < 0).any():
#        raise ValueError("Absolute Percent Logarithmic Error cannot be used when "
#                         "targets contain negative values.")
#
#    if (y_true == 1).any():
#        raise ValueError("Absolute Percent Logarithmic Error cannot be used when "
#                         "targets contain values of 1.")
#
#    return np.abs(np.log10(y_pred) - np.log10(y_true)) / np.abs(np.log10(y_true))


def calc_SPE(y_true, y_pred):
    """
    Calculates symmetric percent error

    Best value is 0.0
    Range is [-2.0,2.0]
    Asymptote at y_pred + y_true = 0

    Note: Defined in the non-standard way as error = y_pred - y_true
        so that a negative error means the model is underforecasting

    Parameters
    ----------
    y_true : array-like
        Observed (true) values

    y_pred : array-like
        Forecasted (estimated) values

    Returns
    -------
    error : array-like
        Error between forecast and observation

    Formula
    -------
    SPE = 2.0 * (y_pred - y_true) / (y_pred + y_true)
    """

    check_consistent_length(y_true, y_pred)

    y_true = check_array(y_true, force_all_finite=True, ensure_2d=False)
    y_pred = check_array(y_pred, force_all_finite=True, ensure_2d=False)

    if (y_true + y_pred == 0).any():
        logger.error("Symmetric Percent Error cannot be used when "
                         "predicted targets and true targets sum to zero.")
        raise ValueError("Symmetric Percent Error cannot be used when "
                         "predicted targets and true targets sum to zero.")

    return 2.0 * (y_pred - y_true) / (y_pred + y_true)


def calc_SAPE(y_true, y_pred):
    """
    Calculates symmetric absolute percent error

    Best value is 0.0
    Range is [0.0,2.0]
    Asymptote at |y_pred| + |y_true| = 0

    Note: Defined in the non-standard way as error = y_pred - y_true
        so that a negative error means the model is underforecasting

    Parameters
    ----------
    y_true : array-like
        Observed (true) values

    y_pred : array-like
        Forecasted (estimated) values

    Returns
    -------
    error : array-like
        Error between forecast and observation

    Formula
    -------
    SAPE = 2.0 * |y_pred - y_true| / (|y_pred| + |y_true|)
    """

    check_consistent_length(y_true, y_pred)

    y_true = check_array(y_true, force_all_finite=True, ensure_2d=False)
    y_pred = check_array(y_pred, force_all_finite=True, ensure_2d=False)

    if (np.abs(y_true) + np.abs(y_pred) == 0).any():
        logger.error("Symmetric Absolute Percent Error cannot be used when "
                         "predicted targets and true targets sum to zero.")
        raise ValueError("Symmetric Absolute Percent Error cannot be used when "
                         "predicted targets and true targets sum to zero.")

    return 2.0 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))

###Commenting for now because I don't think SPLE and SAPLE are mathematically
###correct expressions.
#def calc_SPLE(y_true, y_pred):
#    """
#    Calculates symmetric percent log (base 10) error
#
#    Best value is 0.0
#    Range is (-inf,inf)
#    Asymptote at log(y_pred) + log(y_true) = 0
#
#    Note: Defined in the non-standard way as error = y_pred - y_true
#        so that a negative error means the model is underforecasting
#
#    Parameters
#    ----------
#    y_true : array-like
#        Observed (true) values
#
#    y_pred : array-like
#        Forecasted (estimated) values
#
#    Returns
#    -------
#    error : array-like
#        error between forecast and observation
#    """
#
#    check_consistent_length(y_true, y_pred)
#
#    y_true = check_array(y_true, force_all_finite=True, ensure_2d=False)
#    y_pred = check_array(y_pred, force_all_finite=True, ensure_2d=False)
#
#    if (y_true < 0).any() or (y_pred < 0).any():
#        raise ValueError("Symmetric Percent Logarithmic Error cannot be used when "
#                         "targets contain negative values.")
#
#    if (np.log10(y_true) + np.log10(y_pred) == 0).any():
#        raise ValueError("Symmetric Percent Logarithmic Error cannot be used when "
#                         "predicted targets and true targets sum to zero.")
#
#    return 2.0 * (np.log10(y_pred) - np.log10(y_true)) / (np.log10(y_pred) + np.log10(y_true))
#
#
#def calc_SAPLE(y_true, y_pred):
#    """
#    Calculates symmetric absolute percent log (base 10) error
#
#    Best value is 0.0
#    Range is [0.0,inf)
#    Asymptote at |log(y_pred)| + |log(y_true)| = 0
#
#    Note: Defined in the non-standard way as error = y_pred - y_true
#        so that a negative error means the model is underforecasting
#
#    Parameters
#    ----------
#    y_true : array-like
#        Observed (true) values
#
#    y_pred : array-like
#        Forecasted (estimated) values
#
#    Returns
#    -------
#    error : array-like
#        error between forecast and observation
#    """
#
#    check_consistent_length(y_true, y_pred)
#
#    y_true = check_array(y_true, force_all_finite=True, ensure_2d=False)
#    y_pred = check_array(y_pred, force_all_finite=True, ensure_2d=False)
#
#    if (y_true < 0).any() or (y_pred < 0).any():
#        raise ValueError("Symmetric Absolute Percent Logarithmic Error cannot be used when "
#                         "targets contain negative values.")
#
#    if (np.abs(np.log10(y_true)) + np.abs(np.log10(y_pred)) == 0).any():
#        raise ValueError("Symmetric Absolute Percent Logarithmic Error cannot be used when "
#                         "predicted targets and true targets sum to zero.")
#
#    return 2.0 * np.abs(np.log10(y_pred) - np.log10(y_true)) / (np.abs(np.log10(y_pred)) + np.abs(np.log10(y_true)))


def calc_pearson(y_true, y_pred, type=""):
    """
    Calculates the pearson coefficient while considering the scale

    Best value is 1.0

    Parameters
    ----------
    y_true : array-like
        Observed (true) values

    y_pred : array-like
        Forecasted (estimated) values
        
    type : string
        may be log, linear, or empty. Empty indicates will attempt
        to calculate both log and linear.
        log - calculate only log coeff
        linear - calculate only linear coeff

    Returns
    -------
    r_lin : float
        Pearson coefficient using linear scale

    r_log : float
        Pearson coefficient using log scale
    """

    check_consistent_length(y_true, y_pred)

    y_true = check_array(y_true, force_all_finite=True, ensure_2d=False)
    y_pred = check_array(y_pred, force_all_finite=True, ensure_2d=False)
    
    r_log = np.nan
    r_lin = np.nan

    
    # calculating r on log scale if possible
    if type == "log" or type == "":
        try:
            r_log = pearsonr(np.log10(y_true), np.log10(y_pred))[0]
        except:
            r_log = np.nan

    if type == "linear" or type == "":
        try:
            r_lin = pearsonr(y_true, y_pred)[0]
        except:
            r_lin = np.nan

    return r_lin, r_log



def calc_brier(y_true, y_pred):
    """
    Calculates the Brier Score from probability predictions.

    Mean squared difference between the predicted probability
    and actual outcome. It can be composed into the sum of
    refinement loss and calibration loss.

    Sensitive to climatological frequency of the event: the 
    more rare an event, the easier it is to get a good BS without 
    having any real skill. 

    Brier score and Brier Skill Score are different metrics,
    the skill score requires a climatological/reference forecast to 
    compare to - this is the subroutine for the Brier Score.

    Best value is 0.0

    Parameters
    ----------
    y_true : array-like
        Observed (true) values (1, 0)

    y_pred : array-like
        Forecasted probabilities

    Returns
    -------
    score : float
        Brier Score

    Formula
    -------

    BS = ((y_pred - y_true)**2) / N
    where N is the number of prediction-observation pairs.
    """

    check_consistent_length(y_true, y_pred)

    # If None exists in the model or
    # in the observations, remove the model-obs pair
    # from the arrays.
    for i in range(len(y_true)-1,-1,-1):
        if y_true[i] == None or y_pred[i] == None:
            y_true.pop(i)
            y_pred.pop(i)

    y_true = check_array(y_true, force_all_finite=True, ensure_2d=False)
    y_pred = check_array(y_pred, force_all_finite=True, ensure_2d=False)

    score = brier_score_loss(y_true, y_pred)

    return score



#CA
def calc_MAR(y_true, y_pred):
    """
    Calculates Mean Accuracy Ratio

    Best value is 1.0
    Range is [0.0, inf)

    Note: 

    Parameters
    ----------
    y_true : array-like
        Observed (true) values

    y_pred : array-like
        Forecasted (estimated) values

    Returns
    -------
    MAR: float
    
    Formula
    -------
    MAR = mean(y_pred / y_true)
    """

    check_consistent_length(y_true, y_pred)

    y_true = check_array(y_true, force_all_finite=True, ensure_2d=False)
    y_pred = check_array(y_pred, force_all_finite=True, ensure_2d=False)

    if (y_true < 0).any() or (y_pred < 0).any():
        logger.error("Mean Accuracy Ratio cannot be used when "
                         "targets contain negative values.")
        raise ValueError("Mean Accuracy Ratio cannot be used when "
                         "targets contain negative values.")


    return np.mean(y_pred / y_true)



# CA
def calc_MdSA(y_true, y_pred):
    """
    Calculates median symmetric accuracy based on eqn 11 in:
    Morley, S. K., Brito, T. V., & Welling, D. T. (2018). 
    Measures of model performance based on the log accuracy ratio.
    Space Weather, 16, 69–88. https://doi.org/10.1002/2017SW001669

    Best value is 1
    Range is [0.0, inf)

    Note: 

    Parameters
    ----------
    y_true : array-like
        Observed (true) values

    y_pred : array-like
        Forecasted (estimated) values

    Returns
    -------
    MSA: float

    Formula
    -------
    MdSA = exp(median(|ln(y_true/y_pred)|)) - 1
    """

    check_consistent_length(y_true, y_pred)

    y_true = check_array(y_true, force_all_finite=True, ensure_2d=False)
    y_pred = check_array(y_pred, force_all_finite=True, ensure_2d=False)

    if (y_true < 0).any() or (y_pred < 0).any():
        logger.error("Median symmetric accuracy cannot be used when "
                         "targets contain negative values.")
        raise ValueError("Median symmetric accuracy cannot be used when "
                         "targets contain negative values.")

    #Using the natural log as in the definition of this metric in eqn 11 of the Morley paper
    #Small note - order of y_true and y_pred (in numerator or denominator) does not matter
    #due to property of natural logs and absolute value. So if you compare to MAR the order is
    #flipped but this does not affect the outcome - CA 
    #Removing 100% from definition - RE, KW
    return (np.exp(np.median(np.abs(np.log(y_true / y_pred)))) - 1.0)


#CA
def calc_spearman(y_true, y_pred):
    """
    Calculates the spearman coefficient while considering the scale

    Best value is 1.0

    Parameters
    ----------
    y_true : array-like
        Observed (true) values

    y_pred : array-like
        Forecasted (estimated) values

    Returns
    -------
    SciPy's spearmanr function returns an object with two attributes:
    statistic: float or ndarray
        Spearman correlation matrix or correlation coefficient
        (if only 2 variables are given as parameters). Correlation 
        matrix is square with length equal to total number of 
        variables (columns or rows) in a and b combined.
    pvalue: float
        The p-value for a hypothesis test whose null hypothesis 
        is that two samples have no ordinal correlation.

    I only return the statistic attribute back to the rest of the code

    Spearman coefficients can range from [-1,1] with 0 representing no
    correlation. 
    """

    check_consistent_length(y_true, y_pred)

    y_true = check_array(y_true, force_all_finite=True, ensure_2d=False)
    y_pred = check_array(y_pred, force_all_finite=True, ensure_2d=False)

    s_lin = np.nan

    # calculating the spearman correlation coefficient
    try:
        s_lin = spearmanr(y_true, y_pred).correlation
    except:
        s_lin = np.nan

    return s_lin


#CA
def calc_brier_skill(y_true, y_pred):
    """
    Calculates the Brier Skill score using Hazel Bain's
    climatology probability from SC24 which is 0.033.
    This climatology is not adequate for all solar cycles
    since each solar cycle will be different but there has
    not been a value calculated for SC25 as of yet.

    Best value is 1.0
    Formula
    -------
    BSS = 1 - Brier_score / Brier_score_climatology

    Brier_score_climatology is just using the same formula as
    the model's Brier score but using the climatology probability
    from above.
    """
    check_consistent_length(y_true, y_pred)
    clim = np.ndarray(np.size(y_pred))
    hb_clim = 0.033 #Hazel's climatology (Bain et al. 2021)
    clim.fill(hb_clim)
 
    score = brier_score_loss(y_true, y_pred)
    clim_score = brier_score_loss(y_true, clim)
    BSS = 1 - (score / clim_score)
    return BSS



def check_GSS(h, f, m, n):
    """
    Calculates the Gilbert Skill Score using the contigency table

    Best value is 1.0

    Formula
    -------
    GSS = (h- ((h+f)*(h+m)/n) ) / (h + f + m - ((h+f)*(h+m)/n) )
    
    """
    chk = check_div((h+m),n)
    if math.isinf(chk) or math.isnan(chk): # Only way to hit this is to have 0 forecasts somehow...
        return chk
    else:
       return check_div((h-(h+f)*(h+m)/n), (h+f+m-(h+f)*(h+m)/n))


def check_SEDS(h, f, m, n):
    """
    check zero values and division
        h - hits
        m - misses
        f - false alarms
        c - correct negatives
        n = h + m + f + c
 
    From Liemohn et al. 2021, RSME is not enough, JASTP Sec 3.2.5 "Extremes"
        SEDS = [ln( (a+b) / N ) + ln( (a+c) / N)] / ln( a/N ) -1
    where a = Hits, b = False Alarms, c = Misses, d = Correct Negatives,
    N = a + b + c + d

    Best value is 1.0
     
    Formula
    -------
    SEDS = ( (ln((h+f)/n)+ln((h+m)/n)) / ln(h/n) ) - 1
    """
    #Zero values in numerator or denominator that will cause mathematical errors
    if h+f == 0 or h+m == 0 or h == 0 or n == 0:
        return np.nan

    chk = check_div(np.log((h+f)/n)+np.log((h+m)/n),np.log(h/n))
    if math.isinf(chk) or math.isnan(chk):
        return chk
    else:
        seds = chk - 1
        return seds

# F_beta Score
def calc_fbeta(h, m, f, beta):
    """
    Inputs
    h = hits
    m = misses
    f = false alarms
    beta = weighting factor between precision and recall

    Formula
    -------
    F_beta = ((1+beta^2)*h)/((1+beta^2)*h+f+m*beta^2)
    beta = 0.5, 1, 2 (typically)

    Notes
    -------
    F1 Score (Beta = 1) is the harmonic mean of precision and recall
    When Beta = 2 the recall is weighted higher than precision and when Beta = 0.5 precision is weighted higher.
    """
    
    score = ((1+beta**2)*h) / ((1+beta**2)*h+f+m*beta**2)


    return score

def calc_PT(h, m, f, c):
    """
    Prevalence Threshold 
    Presented as a percentage?
    """
    score = check_div(np.sqrt(check_div(h, h+m)*check_div(f, f+c))- check_div(f, f+c), check_div(h, h+m)-check_div(f, f+c))
    return score

def calc_MCC(h, m, f, c):
    """
    Matthew's Correlation Coefficnet
    Phi Coefficient
    For binary classification ranges from -1 to 1, perfect score is 1

    Formula
    -------
    MCC = (h*c-f*m)/Sqrt((h+f)*(h+m)*(c+f)*(c+m))

    Notes
    -------
    The Pearson correlation reduces to the Phi Coefficient for a 2x2 confusion (contigency) matrix
    """
    score = check_div(h*c-f*m, np.sqrt((h+f)*(h+m)*(c+f)*(c+m)))
    return score


def arr_to_df(arr, keys):
    """ Convert arrays into a dataframe.
    
    Parameters
    ----------
    arr : array of arrays
        each sub-array will be converted into a column in the df
        each sub-array must be the same length
        
    keys : arr of strings
        each string is a column name for the corresponding sub-array in arr
        
    Returns
    -------
    df : pandas dataframe
        sub-arrays as columns named as keys
        
    """

    if len(arr) != len(keys):
        logger.error("Input arrays must be the same length. arr (column values) and keys (column names) must match.")
        sys.exit("metrics.py: arr_to_df: input arrays must be the same length. arr (column values) and keys (column names) must match.")

    dict = {}
    for i in range(len(keys)):
        dict.update({keys[i]: arr[i]})
        
    return pd.DataFrame(dict)



def contingency_scores(h,m,f,c):
    """
    Calculates a variety of metrics from the contigency table

    INPUT:
    
        h : hits
        m : misses
        f : false alarms
        c : correct negatives

    Formulas:
        True Positive (TP) = h
        False Negatives/Misses (FN) = m
        False Positive (FP) = f
        True Negatives (TN) = c
        Percent Correct (PC) = (h+c)/n
        Bias (B)= (h+f)/(h+m)
        Hit Rate (H) = h/(h+m)
        False Alarm Ratio (FAR) = f/(f+h)
        False Alarm Rate (F) = f/(f+c)
        False Negative Rate (FNR) = m/(h+m)
        Frequency of Hits (FOH) = h/(h+f)
        Frequency of Misses (FOM) = m/(h+m)
        Probability of Correct Negatives (POCN) = c/(c+f)
        Detection Failure Ratio (DFR) = m/(m+c)
        Frequency of Correct Negatives (FOCN) = c/(c+m)
        Threat Score (TS) = h/(h+f+m)
        Odds Ratio (OR) = hc/fm
        Gilbert Skill Score (GSS) = (h-(h+f)*(h+m)/n)/(h+f+m-(h+f)*(h+m)/n)
        True Skill Score (TSS) = h/(h+m) - f/(f+c)
        Heidke Skill Score (HSS) = 2(hc-fm)/((h+m)(m+c)+(h+f)(f+c))
        Odds Ratio Skill Score (ORSS) = (hc-mf)/(hc+mf)
        Symmetric Extreme Dependency Score (SEDS) = ((ln((h+f)/n)+ln((h+m)/n))/ln(h/n)) - 1
        F-Scores (Beta = 0.5, 1, 2) = ((1+ Beta^2)* h) / ((1+Beta^2)*h + Beta^2 * m + f)
        Prevalence = (h+m)/n
        Matthew Correlation Coefficient = (h*c-f*m)/Sqrt((h+f)*(h+m)*(c+f)*(c+m))
        Informedness = h/(h+m) + c/(f+c) - 1
        Markedness = h/(h+f) + c/(m+c) - 1
        Prevalence Threshold = (Sqrt(h/(h+m)*f/(f+c))-(f/(f+c))) / (h/(h+m)-f/(f+c))
        Balanced Accuracy = (1/2)*(h/(h+m)+c/(f+c)
        Fowlkes-Mallows Index = Sqrt((h/(h+f))*(h/(h+m)))
            https://en.wikipedia.org/wiki/Fowlkes%E2%80%93Mallows_index
        
        Some sources of these formuli are found here: https://en.wikipedia.org/wiki/Confusion_matrix
     
    """
    n = h + m + f + c

    # all scores while checking for dividing by zero
    scores = {
    'TP': h,
    'FN': m,
    'FP': f,
    'TN': c,
    'PC': check_div(h+c, n),                           # Accuracy, Percent Correct
    'B': check_div(h+f, h+m),                          # Bias, Precision
    'H': check_div(h, h+m),                            # Hit Rate, Probability of Detection, Recall, Sensitivity
    'FAR': check_div(f, h+f),                          # False Alarm Ratio
    'F': check_div(f, f+c),                            # False Alarm Rate
    'FNR': check_div(m,h+m),                           # False Negative Rate, Miss Rate
    'FOH': check_div(h, h+f),                          # Frequency of Hits
    'FOM': check_div(m, h+m),                          # Frequency of Misses
    'POCN': check_div(c, f+c),                         # Probability of Correct Negatives, Specificity, Selectivity
    'DFR': check_div(m, m+c),                          # Detection Failure Ratio
    'FOCN': check_div(c, m+c),                         # Frequency of Correct Negatives
    'TS': check_div(h, h+f+m),                         # Threat Score, Critical success index
    'OR': check_div(h*c, f*m),                         # Odds Ratio
    'GSS': check_GSS(h, f, m, n),                      # Gilbert Skill score
    'TSS': check_div(h, h+m) - check_div(f, f+c),      # True Skill Score
    'HSS': check_div(2.0 * (h*c - f*m), ((h+m) * (m+c) + (h+f) * (f+c))), # Heidke Skill Score
    'ORSS': check_div((h*c - m*f), (h*c + m*f)),       # Odds Ratio Skill Score
    'SEDS': check_SEDS(h, f, m, n),                    # Symmetric Extreme Dependency Score
    'FONE': calc_fbeta(h, m, f, 1),                    # F_beta (1)
    'FTWO': calc_fbeta(h, m, f, 2),                    # F_beta (2)
    'FHALF': calc_fbeta(h, m, f, 0.5),                 # F_beta (0.5)
    'PREV': check_div(h+m, n),                         # Prevalence 
    'MCC': calc_MCC(h, m, f, c),                       # Matthew Correlation Coefficient, Phi Coefficient
    'INFORM': check_div(h, h+m) + check_div(c, f+c) - 1, # Informedness
    'MARK': check_div(h, h+f) + check_div(c, m+c) - 1,  # Markedness
    'PT': calc_PT(h, m, f ,c),                         # Prevalence Threshold
    'BA': check_div(check_div(h, h+m)+check_div(c, f+c), 2), # Balanced Accuracy
    'FM': np.sqrt(check_div(h, h+f)*check_div(h, h+m)),  # Fowlkes-Mallows Index (Geometric mean of precision and recall)
    'FAER' : check_div(f, h+m),                          # False Alarm Event Ratio
    'Tau': None
    }
    #### Just doing some testing here with likelihood ratios, keep commented out for now
    # print(df[obs_key], df[pred_key])
    # clr_pos = check_div(h, h+m) / (1-check_div(c, c+f))
    # clr_neg = (1-check_div(h, h+m)) / check_div(c, c+f)
    # print(clr_pos, clr_neg)
    # input()
    
    return scores



def calc_contingency(y_true, y_pred, thresh):
    """
    Calculates a contingency table and relevant
    ratios and skill scores based on a given threshold

    Parameters
    ----------
    y_true : array-like
        Observed (true) values

    y_pred : array-like
        Forecasted (estimated) values

    thresh : float
        Threshold between a "yes" event and "no" event

    Returns
    -------
    scores : dictionary
        Ratios and skill scores
    """

    y_true, y_pred = remove_none(y_true, y_pred)
    
    check_consistent_length(y_true, y_pred)

    y_true = check_array(y_true, force_all_finite=True, ensure_2d=False)
    y_pred = check_array(y_pred, force_all_finite=True, ensure_2d=False)

    #Convert to boolean values to match All Clear state
    #True = All Clear (no event), False = Not Clear (event)
    y_true = [not x>=thresh for x in y_true]
    y_pred = [not x>=thresh for x in y_pred]
    
    keys = ["Observed Peak Flux All Clear", "Predicted Peak Flux All Clear"]
    df = arr_to_df([y_true, y_pred], keys)

    scores = calc_contingency_all_clear(df, keys[0], keys[1])
    
    return scores



def calc_contingency_all_clear(df, obs_key, pred_key):
    """
    Calculates a contingency table and relevant
    ratios and skill scores based on booleans that indicate
    all clear. Note that all clear = False indicates that it
    is NOT clear and that a threshold has been crossed.
    
    False = threshold crossed (event)
    True = threshold not crossed (no event)

    Parameters
    ----------
    df : pandas dataframe
        contains observed and predicted all clear values
        
    obs_key : string
        column name containing observed boolean values

    pred_key : string
        column name containing forecasted boolean values


    Returns
    -------
    scores : dictionary
        Ratios and skill scores
        
    """
    
    #HITS: obs = False, pred = False
    result = (df[obs_key] == False) & (df[pred_key] == False)
    h = result.sum(axis=0)
    
    #MISSES: obs = False, pred = True
    result = (df[obs_key] == False) & (df[pred_key] == True)
    m = result.sum(axis=0)
    
    #FALSE POSITIVE: obs = True, pred = False
    result = (df[obs_key] == True) & (df[pred_key] == False)
    f = result.sum(axis=0)

    #TRUE NEGATIVES: obs = True, pred = True
    result = (df[obs_key] == True) & (df[pred_key] == True)
    c = result.sum(axis=0)
    
    scores = contingency_scores(h,m,f,c)

    tau = calc_tau(h, m, f, c, df.iloc[0, 0], df.iloc[0, 1])
    scores['Tau'] = tau
    logger.info(str(scores))
    return scores



def check_div(n, d):
    """
    Checks if dividing by zero
    Needed since contingency table might have 0 as entries

    Parameters
    ----------
    n : float
        Numerator

    d : float
        Denominator

    Returns
    -------
    ret : float
        Returns NAN or +/-inf based on division
    """

    if d == 0:
        if n > 0:
            return np.inf
        elif n < 0:
            return -np.inf
        else:
            return np.nan
    else:
        return n/d


def remove_none(obs, model):
    '''Remove None values from corresponding observations and model lists.
        Only values that are real in both lists are kept.
        obs is a single list of observations
        model is a single list of model forecasts
    '''
    #Error checking
    if len(obs) != len(model):
        logger.error('Both input arrays must be the same length! Exiting.')
        sys.exit('remove_none: Both input arrays must be the same length! '
                'Exiting.')
    #Clean None values from observations and remove correponding entries in
    #the model
    bad_index = [bad for bad, value in enumerate(obs) if value == None]
    obs_clean = list(obs)
    model_clean = list(model)
    for bad in sorted(bad_index, reverse=True):
        del obs_clean[bad]
        del model_clean[bad]

    #Clean None values from the model and remove correponding entries in
    #the observations
    bad_index = [bad for bad, value in enumerate(model_clean) if value == None]
    for bad in sorted(bad_index, reverse=True):
        del obs_clean[bad]
        del model_clean[bad]

    return obs_clean, model_clean
    


def receiver_operator_characteristic(obs, pred, model_name):
    """
    Subroutine that does the calculations for the Reciever
    Operator Characteristic (ROC) using the scikitlearn package.

    Parameters
    ----------
    obs : array-like
        Observed (true) values

    pred : array-like
        Forecasted (estimated) values

    model_name : string
        String containing the model name,
        used as part of the plotting for the
        RO

    Returns
    -------
    roc_auc : float
        Area under the ROC curve
        Ideal values are closer to 1 (random guess will be 1/2)

    roc_curve_plt : sklearn.metrics._plot.roc_curve.RocCurveDisplay
        Figure generated by the scikitlearn package to create
        the ROC plot - which is returned and added to as part of 
        validation.py

    Details
    -------
    ROC is calculated by comparing the predicted probabilities to
    various probabality thresholds defining event/no event and using that
    to calculate false positive rate and true positive rate at each threshold.

    The area under the curve is calculated by integrating the resulting line
    that is generated by the false positive rate and true positive rate for 
    each threshold. A value closer to 1 is ideal as it represents no increase in 
    false positives until the true positive rate is at 1.

    In validation.py, a diagonal line is added to the ROC plot to represent
    a random guess, and the model's result can be compared to this guess. Models
    above the diagonal line represent having some level of skill, and below/on the 
    diagonal having no skill. 
    """
    fpr, tpr, thresholds = skl.roc_curve(obs, pred) # fpr: false positive rate, tpr: true positive rate, thresholds: Decreasing thresholds on the decision function used to compute fpr and tpr. 
    roc_auc = skl.auc(fpr, tpr) # Area under the curve 
    roc_curve_plt = skl.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,estimator_name=model_name)
    return roc_auc, roc_curve_plt



def remove_zero(obs, model):
    '''Remove None values from corresponding observations and model lists.
        Only values that are real in both lists are kept.
        obs is a single list of observations
        model is a single list of model forecasts
    '''
    #Error checking
    if len(obs) != len(model):
        logger.error('Both input arrays must be the same length! Exiting.')
        sys.exit('remove_zero: Both input arrays must be the same length! '
                'Exiting.')
    #Clean None values from observations and remove correponding entries in
    #the model
    bad_index = [bad for bad, value in enumerate(obs) if value == 0.]
    obs_clean = list(obs)
    model_clean = list(model)
    for bad in sorted(bad_index, reverse=True):
        del obs_clean[bad]
        del model_clean[bad]

    #Clean None values from the model and remove correponding entries in
    #the observations
    bad_index = [bad for bad, value in enumerate(model_clean) if value == 0.]
    for bad in sorted(bad_index, reverse=True):
        del obs_clean[bad]
        del model_clean[bad]

    return obs_clean, model_clean

def calc_tau(h, m, f, c, model_name, energy_channel, visualize = False):
    
        
        logger.info(str(model_name))
        logger.info(str(energy_channel))
        
        # input()
        if visualize:
            try:
                matrix = {'t': [h, m], 'f': [f, c]}
            
                fig = plt.figure(figsize=(20, 16))
                ax = fig.add_subplot(111)#, projection='2d')
                cs = ContingencySpace([ConfusionMatrix(matrix)])
                cs.visualize(metric=Tau(cm = ConfusionMatrix(matrix), do_normalize = False), labels = model_name, projection='2d', title='Tau', ax = ax, step_size = 20, lines = False)
                fig.savefig('./output/plots/tau_2d_' + model_name + '_' + energy_channel + '.png', dpi=600, bbox_inches='tight')
                plt.close()
            except:
                logger.info('Welp I tried but it failed, next model')

        tau  = 1 - (np.sqrt((f/(c + f))**2 + (m/(h + m))**2)/np.sqrt(2))
        return tau

    