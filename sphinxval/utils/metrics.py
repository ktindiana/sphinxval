import numpy as np
import pandas as pd
from sklearn.utils.validation import check_consistent_length
from sklearn.utils.validation import check_array
from sklearn.metrics import brier_score_loss
from scipy.stats import pearsonr
import math

__version__ = "0.7"
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

'''Contains functions for calculating various metrics used
    for comparing modeled forecast results to observations.
    Written on 2020-07-17.
    Updated 2022-02-07.
'''


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

    func = {
        'E': calc_E,
        'AE': calc_AE,
        'LE': calc_LE,
        'ALE': calc_ALE,
        'SE': calc_SE,
        'SLE': calc_SLE,
        'RMSE': calc_RMSE,
        'RMSLE': calc_RMSLE,
        'PE': calc_PE,
        'APE': calc_APE,
        'PLE': calc_PLE,
        'APLE': calc_APLE,
        'SPE': calc_SPE,
        'SAPE': calc_SAPE,
        'SPLE': calc_SPLE,
        'SAPLE': calc_SAPLE,
        'r': calc_pearson,
        }.get(metric)

    if not callable(func):
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
    """

    check_consistent_length(y_true, y_pred)

    y_true = check_array(y_true, force_all_finite=True, ensure_2d=False)
    y_pred = check_array(y_pred, force_all_finite=True, ensure_2d=False)

    if (y_true < 0).any() or (y_pred < 0).any():
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
    """

    check_consistent_length(y_true, y_pred)

    y_true = check_array(y_true, force_all_finite=True, ensure_2d=False)
    y_pred = check_array(y_pred, force_all_finite=True, ensure_2d=False)

    if (y_true < 0).any() or (y_pred < 0).any():
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
    """

    check_consistent_length(y_true, y_pred)

    y_true = check_array(y_true, force_all_finite=True, ensure_2d=False)
    y_pred = check_array(y_pred, force_all_finite=True, ensure_2d=False)

    if (y_true < 0).any() or (y_pred < 0).any():
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
    """

    check_consistent_length(y_true, y_pred)

    y_true = check_array(y_true, force_all_finite=True, ensure_2d=False)
    y_pred = check_array(y_pred, force_all_finite=True, ensure_2d=False)

    if (y_true < 0).any() or (y_pred < 0).any():
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
    """

    check_consistent_length(y_true, y_pred)

    y_true = check_array(y_true, force_all_finite=True, ensure_2d=False)
    y_pred = check_array(y_pred, force_all_finite=True, ensure_2d=False)

    if (y_true == 0).any():
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
    """

    check_consistent_length(y_true, y_pred)

    y_true = check_array(y_true, force_all_finite=True, ensure_2d=False)
    y_pred = check_array(y_pred, force_all_finite=True, ensure_2d=False)

    if (y_true == 0).any():
        raise ValueError("Absolute Percent Error cannot be used when "
                         "targets contain values of zero.")

    return np.abs(y_pred - y_true) / y_true


def calc_PLE(y_true, y_pred):
    """
    Calculates percent log (base 10) error

    Best value is 0.0
    Range is (-inf,inf)
    Asymptote at y_true = 1

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
    """

    check_consistent_length(y_true, y_pred)

    y_true = check_array(y_true, force_all_finite=True, ensure_2d=False)
    y_pred = check_array(y_pred, force_all_finite=True, ensure_2d=False)

    if (y_true < 0).any() or (y_pred < 0).any():
        raise ValueError("Percent Logarithmic Error cannot be used when "
                         "targets contain negative values.")

    if (y_true == 1).any():
        raise ValueError("Percent Logarithmic Error cannot be used when "
                         "targets contain values of 1.")

    return (np.log10(y_pred) - np.log10(y_true)) / np.log10(y_true)


def calc_APLE(y_true, y_pred):
    """
    Calculates absolute percent log (base 10) error

    Best value is 0.0
    Range is [0.0,inf)
    Asymptote at y_true = 1

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
    """

    check_consistent_length(y_true, y_pred)

    y_true = check_array(y_true, force_all_finite=True, ensure_2d=False)
    y_pred = check_array(y_pred, force_all_finite=True, ensure_2d=False)

    if (y_true < 0).any() or (y_pred < 0).any():
        raise ValueError("Absolute Percent Logarithmic Error cannot be used when "
                         "targets contain negative values.")

    if (y_true == 1).any():
        raise ValueError("Absolute Percent Logarithmic Error cannot be used when "
                         "targets contain values of 1.")

    return np.abs(np.log10(y_pred) - np.log10(y_true)) / np.abs(np.log10(y_true))


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
    """

    check_consistent_length(y_true, y_pred)

    y_true = check_array(y_true, force_all_finite=True, ensure_2d=False)
    y_pred = check_array(y_pred, force_all_finite=True, ensure_2d=False)

    if (y_true + y_pred == 0).any():
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
    """

    check_consistent_length(y_true, y_pred)

    y_true = check_array(y_true, force_all_finite=True, ensure_2d=False)
    y_pred = check_array(y_pred, force_all_finite=True, ensure_2d=False)

    if (np.abs(y_true) + np.abs(y_pred) == 0).any():
        raise ValueError("Symmetric Absolute Percent Error cannot be used when "
                         "predicted targets and true targets sum to zero.")

    return 2.0 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))


def calc_SPLE(y_true, y_pred):
    """
    Calculates symmetric percent log (base 10) error

    Best value is 0.0
    Range is (-inf,inf)
    Asymptote at log(y_pred) + log(y_true) = 0

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
        error between forecast and observation
    """

    check_consistent_length(y_true, y_pred)

    y_true = check_array(y_true, force_all_finite=True, ensure_2d=False)
    y_pred = check_array(y_pred, force_all_finite=True, ensure_2d=False)

    if (y_true < 0).any() or (y_pred < 0).any():
        raise ValueError("Symmetric Percent Logarithmic Error cannot be used when "
                         "targets contain negative values.")

    if (np.log10(y_true) + np.log10(y_pred) == 0).any():
        raise ValueError("Symmetric Percent Logarithmic Error cannot be used when "
                         "predicted targets and true targets sum to zero.")

    return 2.0 * (np.log10(y_pred) - np.log10(y_true)) / (np.log10(y_pred) + np.log10(y_true))


def calc_SAPLE(y_true, y_pred):
    """
    Calculates symmetric absolute percent log (base 10) error

    Best value is 0.0
    Range is [0.0,inf)
    Asymptote at |log(y_pred)| + |log(y_true)| = 0

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
        error between forecast and observation
    """

    check_consistent_length(y_true, y_pred)

    y_true = check_array(y_true, force_all_finite=True, ensure_2d=False)
    y_pred = check_array(y_pred, force_all_finite=True, ensure_2d=False)

    if (y_true < 0).any() or (y_pred < 0).any():
        raise ValueError("Symmetric Absolute Percent Logarithmic Error cannot be used when "
                         "targets contain negative values.")

    if (np.abs(np.log10(y_true)) + np.abs(np.log10(y_pred)) == 0).any():
        raise ValueError("Symmetric Absolute Percent Logarithmic Error cannot be used when "
                         "predicted targets and true targets sum to zero.")

    return 2.0 * np.abs(np.log10(y_pred) - np.log10(y_true)) / (np.abs(np.log10(y_pred)) + np.abs(np.log10(y_true)))


def calc_pearson(y_true, y_pred):
    """
    Calculates the pearson coefficient while considering the scale

    Parameters
    ----------
    y_true : array-like
        Observed (true) values

    y_pred : array-like
        Forecasted (estimated) values

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

    # calculating r on log scale if possible
    try:
        r_log = pearsonr(np.log10(y_true), np.log10(y_pred))[0]
    except:
        r_log = np.nan

    try:
        r_lin = pearsonr(y_true, y_pred)[0]
    except:
        r_lin = np.nan

    return r_lin, r_log



def calc_brier(y_true, y_pred):
    """
    Calculates the Brier Skill Score from probability predictions.
    Mean squared difference between the predicted probability
    and actual outcome. It can be composed into the sum of
    refinement loss and calibration loss.

    Parameters
    ----------
    y_true : array-like
        Observed (true) values (1, 0)

    y_pred : array-like
        Forecasted probabilities

    Returns
    -------
    Brier Skill Score
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


def check_GSS(h, f, m, n):
    """check h+m/n first"""
    chk = check_div((h+m),n)
    if math.isinf(chk) or math.isnan(chk):
        return chk
    else:
       return check_div((h-(h+f)*(h+m)/n), (h+f+m-(h+f)*(h+m)/n))


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

    # checking for None types in the predictions
    # found when the model made no prediction
    # replacing the Nones with 10% (abritrary) of the threshold
    # such that it always counts as a miss
    # operational_sep_quantities.py will output the maximum flux in a time
    # period even if no thresholds are crossed.
    y_true = [0.1*thresh if x==None else x for x in y_true]
    y_pred = [0.1*thresh if x==None else x for x in y_pred]

    check_consistent_length(y_true, y_pred)

    y_true = check_array(y_true, force_all_finite=True, ensure_2d=False)
    y_pred = check_array(y_pred, force_all_finite=True, ensure_2d=False)

    matrix = pd.crosstab(y_true>=thresh, y_pred>=thresh)

    # if any of the table items are empty, make 0 instead
    try:
        h = matrix[1][1]
    except:
        h = 0
    try:
        m = matrix[0][1]
    except:
        m = 0
    try:
        f = matrix[1][0]
    except:
        f = 0
    try:
        c = matrix[0][0]
    except:
        c = 0

    n = h+m+f+c

    # all scores while checking for dividing by zero
    scores = {
    'TP': h,
    'FN': m,
    'FP': f,
    'TN': c,
    'PC': check_div(h+c, n),
    'B': check_div(h+f, h+m),
    'H': check_div(h, h+m),
    'FAR': check_div(f, h+f),
    'F': check_div(f, f+c),
    'FOH': check_div(h, h+f),
    'FOM': check_div(m, h+m),
    'POCN': check_div(c, f+c),
    'DFR': check_div(m, m+c),
    'FOCN': check_div(c, m+c),
    'TS': check_div(h, h+f+m),
    'OR': check_div(h*c, f*m),
    'GSS': check_GSS(h, f, m, n), #check_div((h-(h+f)*(h+m)/n), (h+f+m-(h+f)*(h+m)/n)),
    'TSS': check_div(h, h+m) - check_div(f, f+c),
    'HSS': check_div(2.0 * (h*c + f*m), ((h+m) * (m+c) + (h+f) * (f+c))),
    'ORSS': check_div((h*c - m*f), (h*c + m*f))
    }
    return scores



def calc_contingency_bool(y_true, y_pred):
    """
    Calculates a contingency table and relevant
    ratios and skill scores based on booleans
    True = threshold crossed (event)
    False = threshold not crossed (no event)

    Parameters
    ----------
    y_true : array-like
        Observed boolean values

    y_pred : array-like
        Forecasted boolean values


    Returns
    -------
    scores : dictionary
        Ratios and skill scores
    """
    # The pandas crosstab predicts booleans as follows:
    #   True = event
    #   False = no event
    # ALL CLEAR booleans are as follows:
    #   True = no event
    #   False = event
    # Prior to inputting all clear predictions into this code, need to
    #   switch the booleans to match how event/no event are interpreted here.


    # Although it shouldn't happen, if None exists
    # in the observations, remove the model-obs pair
    # from the arrays.
    for i in range(len(y_true)-1,-1,-1):
        if y_true[i] == None:
            y_true.pop(i)
            y_pred.pop(i)

    check_consistent_length(y_true, y_pred)

    y_true = check_array(y_true, force_all_finite=True, ensure_2d=False)
    y_pred = check_array(y_pred, force_all_finite=True, ensure_2d=False)

    matrix = pd.crosstab(y_true, y_pred)

    # if any of the table items are empty, make 0 instead
    #On one computer system, matrix[1][1] did not work and resulted in all values 0.
    #Had to modify syntax to get correct values.
    try:
        h = matrix[True][1] #h = matrix[1][1]
    except:
        h = 0
    try:
        m = matrix[False][1] #m = matrix[0][1]
    except:
        m = 0
    try:
        f = matrix[True][0] #f = matrix[1][0]
    except:
        f = 0
    try:
        c = matrix[False][0] #c = matrix[0][0]
    except:
        c = 0

    n = h+m+f+c

    # all scores while checking for dividing by zero
    scores = {
    'TP': h,
    'FN': m,
    'FP': f,
    'TN': c,
    'PC': check_div(h+c, n),
    'B': check_div(h+f, h+m),
    'H': check_div(h, h+m),
    'FAR': check_div(f, h+f), #False Alarm Ratio
    'F': check_div(f, f+c), #False Alarm Rate
    'FOH': check_div(h, h+f),
    'FOM': check_div(m, h+m),
    'POCN': check_div(c, f+c),
    'DFR': check_div(m, m+c),
    'FOCN': check_div(c, m+c),
    'TS': check_div(h, h+f+m),
    'OR': check_div(h*c, f*m),
    'GSS': check_GSS(h, f, m, n), #check_div((h-(h+f)*(h+m)/n), (h+f+m-(h+f)*(h+m)/n)),
    'TSS': check_div(h, h+m) - check_div(f, f+c),
    'HSS': check_div(2.0 * (h*c + f*m), ((h+m) * (m+c) + (h+f) * (f+c))),
    'ORSS': check_div((h*c - m*f), (h*c + m*f))
    }
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
