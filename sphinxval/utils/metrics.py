import numpy as np
import pandas as pd
from sklearn.utils.validation import check_consistent_length
from sklearn.utils.validation import check_array
from sklearn.metrics import brier_score_loss
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.metrics import brier_score_loss
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
    if len(y_true) < 1: return None
    
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
#        'PLE': calc_PLE,
#        'APLE': calc_APLE,
        'SPE': calc_SPE,
        'SAPE': calc_SAPE,
#        'SPLE': calc_SPLE,
 #       'SAPLE': calc_SAPLE,
        'r': calc_pearson,
        'MAR': calc_MAR,
        'MdSA': calc_MdSA,
        'spearman': calc_spearman,
        'brier': calc_brier
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
    Calculates the Brier Score from probability predictions.
    Mean squared difference between the predicted probability
    and actual outcome. It can be composed into the sum of
    refinement loss and calibration loss.
    Brier score and Brier Skill Score are different metrics,
    the skill score requires a climatological/reference forecast to 
    compare to.

    Parameters
    ----------
    y_true : array-like
        Observed (true) values (1, 0)

    y_pred : array-like
        Forecasted probabilities

    Returns
    -------
    Brier Score
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

    Best value is 0 (probably)
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
    
    """

    check_consistent_length(y_true, y_pred)

    y_true = check_array(y_true, force_all_finite=True, ensure_2d=False)
    y_pred = check_array(y_pred, force_all_finite=True, ensure_2d=False)

    if (y_true < 0).any() or (y_pred < 0).any():
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
    
    """

    check_consistent_length(y_true, y_pred)

    y_true = check_array(y_true, force_all_finite=True, ensure_2d=False)
    y_pred = check_array(y_pred, force_all_finite=True, ensure_2d=False)

    if (y_true < 0).any() or (y_pred < 0).any():
        raise ValueError("Median symmetric accuracy cannot be used when "
                         "targets contain negative values.")

    #Using the natural log as in the definition of this metric in eqn 11 of the Morley paper
    return 100*(np.exp(np.median(np.abs(np.log(y_true / y_pred)))) - 1.0) 


#CA
def calc_spearman(y_true, y_pred):
    """
    Calculates the spearman coefficient while considering the scale

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

    # calculating the spearman correlation coefficient
    try:
        s_log = spearmanr(np.log10(y_true), np.log10(y_pred)).correlation
    except:
        s_log = np.nan

    try:
        s_lin = spearmanr(y_true, y_pred).correlation
    except:
        s_lin = np.nan
    # s_p = spearmanr(y_true, y_pred).pvalue

    return s_lin, s_log


#CA
def calc_brier_skill(y_true, y_pred):
    """
    Calculates the Brier Skill score using Hazel Bain's
    climatology probability from SC24 which is 0.033
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
    """check h+m/n first"""
    chk = check_div((h+m),n)
    if math.isinf(chk) or math.isnan(chk):
        return chk
    else:
       return check_div((h-(h+f)*(h+m)/n), (h+f+m-(h+f)*(h+m)/n))


def check_SEDS(h, f, m, n):
    """check zero values and division
        h - hits
        m - misses
        f - false alarms
        c - correct negatives
        n = h + m + f + c
 
        From Liemohn et al. 2021, RSME is not enough, JASTP Sec 3.2.5 "Extremes"
        SEDS = [ln( (a+b) / N ) + ln( (a+c) / N)] / ln( a/N ) -1
        where a = Hits, b = False Alarms, c = Misses, d = Correct Negatives,
        N = a + b + c + d
 
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
#    y_true = [0.1*thresh if x==None else x for x in y_true]
#    y_pred = [0.1*thresh if x==None else x for x in y_pred]

    #Applying logic that does not count None forecasts as a miss.
    y_true, y_pred = remove_none(y_true, y_pred)
    
    check_consistent_length(y_true, y_pred)

    y_true = check_array(y_true, force_all_finite=True, ensure_2d=False)
    y_pred = check_array(y_pred, force_all_finite=True, ensure_2d=False)

    #Convert to boolean values to include in calc_contingency_bool
    #True = event, False = no event
    y_true = [x>=thresh for x in y_true]
    y_pred = [x>=thresh for x in y_pred]

    scores = calc_contingency_bool(y_true, y_pred)
    
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


    # Remove None values from the observation and forecast pairs
    # None forecasts are not penalized, simply not counted
    y_true, y_pred = remove_none(y_true, y_pred)

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
    'HSS': check_div(2.0 * (h*c - f*m), ((h+m) * (m+c) + (h+f) * (f+c))),
    'ORSS': check_div((h*c - m*f), (h*c + m*f)),
    'SEDS': check_SEDS(h, f, m, n)#((np.log((h+f)/n)+np.log((h+m)/n))/np.log(h/n))-1
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


def remove_none(obs, model):
    '''Remove None values from corresponding observations and model lists.
        Only values that are real in both lists are kept.
        obs is a single list of observations
        model is a single list of model forecasts
    '''
    #Error checking
    if len(obs) != len(model):
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
    
    
def remove_zero(obs, model):
    '''Remove None values from corresponding observations and model lists.
        Only values that are real in both lists are kept.
        obs is a single list of observations
        model is a single list of model forecasts
    '''
    #Error checking
    if len(obs) != len(model):
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
