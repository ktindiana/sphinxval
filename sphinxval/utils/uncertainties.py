import numpy as np
import pandas as pd
import statistics
from datetime import datetime
from scipy import stats
import sklearn.metrics as skl
from . import metrics
from . import metrics_dicts
import logging


"""
Independent module intended to determine metric uncertainties within
SPHINX with extensions for VIVID. 

Plan/Outline:
    For SPHINX - 
        After calculating metrics and putting them into dataframe 
            validation -> calculate_intuitive_metrics
        Grab _selections files for each model/predicted quantity subset
        Use Scipy.stats bootstrap to resample and calculate the standard error
        Add row/column (need to decide this) to metrics file for the metric uncertainties

    VIVID -
        Add wrapper from VIVID feeder to generate the same subset that 
            is given to the _selections files but doesn't need to actually
            read in the file
        Do the same bootstrapping
        Give out metric uncertainties (need to talk to Phil about VIVID inputs)
"""
#Create logger
logger = logging.getLogger(__name__)
mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)

def feeder_from_sphinx(df, dict, label):
    # take the dataframe passed from sphinx and insert into uncertainty workflow
    # needs to be a feeder function to make sure that the actual workflow is 
    # independent and modular
  
    uncertainty_workflow(df, label, dict)
    # dict.append(uncertainty_dict)
    return


def feeder_from_vivid():


    return


def test_feeder():
    filename = './junk/peak_intensity_max_selections_SEPSTER2D CME_min.10.0.max.-1.0.units.MeV_threshold_10.0.csv'
    # filename = './junk/peak_intensity_max_time_selections_ZEUS+iPATH_CME_min.10.0.max.-1.0.units.MeV_threshold_10.0.csv'
    df = pd.read_csv(filename, index_col = 0)
    label = 'peak_intensity_max'
    # df = time_uncertainties(df, label)
    df = flux_uncertainties(df, label)
    print(df)




def uncertainty_workflow(df, label, dict):
    
    # Splitting function that will correctly funnel to the correct uncertainty calculation
    flux_filter = ['point_intensity', 'peak_intensity', 'peak_intensity_max', 'max_flux_in_pred_win', 'fluence']
    time_filter = ['start_time', 'end_time', 'peak_intensity_time', 'peak_intensity_max_time', 'threshold_crossing', 'duration', 'last_data_to_issue_time']
    
    if label in time_filter:
        time_uncertainties(df, label, dict)
    elif label in flux_filter:
        flux_uncertainties(df, label, dict)
    else:
        if label == 'probability':
            probability_uncertainties(df, label, dict)
        elif label == 'all_clear':
            all_clear_uncertainties(df, label, dict)
        # what should be here?  all_clear, awt
        # I.e. the hardest ones 




    return 


def flux_uncertainties(df, label, dict):
    metrics_func = {
        'E': metrics.calc_E,                        # Error
        'Ratio': metrics.calc_ratio,                # Ratio
        'AE': metrics.calc_AE,                      # Absolute Error
        'LE': metrics.calc_LE,                      # Log Error
        'ALE': metrics.calc_ALE,                    # Absolute Log Error
        'SE': metrics.calc_SE,                      # Squared Error
        'SLE': metrics.calc_SLE,                    # Squared Log Error
        'RMSE': metrics.calc_RMSE,                  # Root Mean Squared Error
        'RMSLE': metrics.calc_RMSLE,                # Root Mean Squared Log Error
        'PE': metrics.calc_PE,                      # Percent Error
        'APE': metrics.calc_APE,                    # Absolute Percent Error, absolute percentage deviation
        'SPE': metrics.calc_SPE,                    # Symmetric Percent Error
        'SAPE': metrics.calc_SAPE,                  # Symmetric Absolute Percent Error
        'MAR': metrics.calc_MAR,                    # Mean Accuracy Ratio
        'MdSA': metrics.calc_MdSA,                  # Median Symmetric  Accuracy
        'spearman': metrics.calc_spearman,          # Spearman, rank order correlation coefficient
        }
    


    mapped_label = flux_label_mapping(label)
    logger.info(str(label) + ' ' + str(mapped_label))
    logger.info(str(df.columns))
    dict['Model'].append(df['Model'].iloc[0] + ' Uncertainty')
    dict['Energy Channel'].append(df['Energy Channel Key'].iloc[0])
    dict['Threshold'].append(df['Threshold Key'].iloc[0])
    dict['Prediction Energy Channel'].append(df['Prediction Energy Channel Key'].iloc[0])
    dict['Prediction Threshold'].append(df['Prediction Threshold Key'].iloc[0])
    dict['Scatter Plot'].append(None)
    dict['Linear Regression Slope'].append(None)
    dict['Linear Regression y-intercept'].append(None)
    
    if 'Max Flux in Prediction Window' in mapped_label:
        pred_label = 'Predicted SEP Peak Intensity Max (Max Flux)'
        obs_label = 'Observed Max Flux in Prediction Window'
    else:
        pred_label = 'Predicted ' + mapped_label
        obs_label = 'Observed ' + mapped_label
    if 'SEPSTER' in df['Model'].iloc[0] and label != 'fluence':
        pred_label = 'Predicted SEP Peak Intensity (Onset Peak)'
    try:
        obs = df[obs_label].to_list()
        pred = df[pred_label].to_list()
    except:
        pred_label = 'Predicted SEP Peak Intensity (Onset Peak)'
        obs = df[obs_label].to_list()
        pred = df[pred_label].to_list()

    mean_metrics_list = ['E', 'Ratio', 'AE', 'LE', 'ALE', 'PE', 'APE', 'SPE', 'SAPE']
    # metrics_list = ['LE']
    for met in mean_metrics_list:
        
        func = metrics_func[met]
        uncertainty = mean_call_bootstrapper(obs, pred, func)
        error = uncertainty.standard_error
        dict[flux_metric_mapping(met)].append(error)
        print(uncertainty, met)
    median_metrics_list = ['E', 'Ratio', 'AE', 'LE', 'ALE']
    for met in median_metrics_list:
        func = metrics_func[met]
        metric_label = met
        uncertainty = median_call_bootstrapper(obs, pred, func)
        error = uncertainty.standard_error
        metric_label = "Med" + metric_label
        dict[flux_metric_mapping(metric_label)].append(error)
        print(uncertainty, metric_label)

    #other metrics 
    other = ['MAR', 'RMSE', 'RMSLE', 'MdSA', 'spearman', 'r']
    for met in other:

        if met != 'r':
            func = metrics_func[met]
            uncertainty = stats.bootstrap((obs, pred), statistic=func, method='basic', vectorized = False, paired = True)
            error = uncertainty.standard_error
            dict[flux_metric_mapping(met)].append(error)
            print(uncertainty, met)
        else:
            pearson_array = ['r_lin', 'r_log']
            for rs in pearson_array:
                uncertainty = pearson_call_bootstrapper(obs, pred, rs)
                error = uncertainty.standard_error
                dict[flux_metric_mapping(rs)].append(error)

                
        


    factors = ['fac10', 'fac2']
    for fac in factors:
        if fac == 'fac10':
            thresh = 1
        else:
            thresh = np.log10(2)
        uncertainty = factor_call_bootstrapper(obs, pred, thresh)
        error = uncertainty.standard_error
        dict[flux_metric_mapping(fac)].append(error)
        print(uncertainty, fac)

    return


def mean_call_bootstrapper(obs, pred, func):
    def wrapper(obs, pred):
        return statistics.mean(func(obs, pred))
    return stats.bootstrap((obs, pred), statistic=wrapper, method='basic', vectorized = False, paired = True)

def median_call_bootstrapper(obs, pred, func):
    def wrapper(obs, pred):
        return statistics.median(func(obs, pred))
    return stats.bootstrap((obs, pred), statistic=wrapper, method='basic', vectorized = False, paired = True)

def factor_call_bootstrapper(obs, pred, thresh):
    def wrapper(obs, pred):
        temp = metrics.switch_error_func('LE', obs, pred)
        count = sum(1 for x in temp if x <= thresh and x >= -thresh)
        # print(count)
        return count / len(temp)
    return stats.bootstrap((obs, pred), statistic=wrapper, method='basic', vectorized = False, paired = True)
    
def pearson_call_bootstrapper(obs, pred, label):
    def wrapper(obs, pred):
        if label == 'r_lin':
            metric, _ = metrics.calc_pearson(obs, pred)
        else:
            _, metric = metrics.calc_pearson(obs, pred)
        return metric
    return stats.bootstrap((obs, pred), statistic=wrapper, method='basic', vectorized = False, paired = True)




def flux_label_mapping(label):
    map_dict = {
        'point_intensity': 'SEP Point Intensity',
        'peak_intensity_max': 'SEP Peak Intensity Max (Max Flux)',
        'peak_intensity': 'SEP Peak Intensity (Onset Peak)',
        'time_profile': 'Time Profile',
        'max_flux_in_pred_win': 'Max Flux in Prediction Window',
        'fluence': 'SEP Fluence'
    }

    mapped_label = map_dict[label]
    return mapped_label


def time_label_mapping(label):
    map_dict = {
        'peak_intensity_max_time': 'SEP Peak Intensity Max (Max Flux) Time',
        'peak_intensity_time': 'SEP Peak Intensity (Onset Peak) Time',
        'start_time': 'SEP Start Time',
        'threshold_crossing': 'SEP Threshold Crossing Time',
        'end_time': 'SEP End Time',
        'duration': 'SEP Duration'
    }
    mapped_label = map_dict[label]
    return mapped_label


def time_uncertainties(df, label, dict):

    mapped_label = time_label_mapping(label)
    pred_label = 'Predicted ' + mapped_label
    obs_label = 'Observed ' + mapped_label

    obs = df[obs_label]
    pred = df[pred_label]
    dict['Model'].append(df['Model'].iloc[0] + ' Uncertainty')
    dict['Energy Channel'].append(df['Energy Channel Key'].iloc[0])
    dict['Threshold'].append(df['Threshold Key'].iloc[0])
    dict['Prediction Energy Channel'].append(df['Prediction Energy Channel Key'].iloc[0])
    dict['Prediction Threshold'].append(df['Prediction Threshold Key'].iloc[0])

    if type(obs.iloc[0]) == str:
        obs = [datetime.strptime(x, "%Y-%m-%d %H:%M:%S") for x in obs]
    if type(pred.iloc[0]) == str:
        pred = [datetime.strptime(x, "%Y-%m-%d %H:%M:%S") for x in pred]
    mean_metrics_list = ['E', 'AE']
    # metrics_list = ['LE']
    for met in mean_metrics_list:
        uncertainty = mean_time_call_bootstrapper(obs, pred, met)
        error = uncertainty.standard_error
        dict[time_metric_mapping(met)].append(error)
        print(uncertainty, met)

    for met in mean_metrics_list:
        metric_label = met
        uncertainty = median_time_call_bootstrapper(obs, pred, met)
        error = uncertainty.standard_error
        metric_label = "Med" + metric_label
        dict[time_metric_mapping(metric_label)].append(error)
        print(uncertainty, metric_label)

    return



def mean_time_call_bootstrapper(obs, pred, metric_label):
    def wrapper(obs, pred):
        # print(type(obs[0]), type(pred[0]))
        td = (pred - obs)
        try:
            td = [x.total_seconds()/(60*60) for x in td]
        except:
            # print(td, td[0], type(td[0]))
            td = [pd.Timedelta(x).total_seconds()/(60*60) for x in td]
        if metric_label == 'AE':
            td = [np.abs(x) for x in td]
        return statistics.mean(td)
    return stats.bootstrap((obs, pred), statistic=wrapper, method='basic', vectorized = False, paired = True)

def median_time_call_bootstrapper(obs, pred, metric_label):
    def wrapper(obs, pred):
        td = (pred - obs)
        try:
            td = [x.total_seconds()/(60*60) for x in td]
        except:
            # print(td, td[0], type(td[0]))
            td = [pd.Timedelta(x).total_seconds()/(60*60) for x in td]
        if metric_label == 'medAE':
            td = [np.abs(x) for x in td]
        return statistics.median(td)
    return stats.bootstrap((obs, pred), statistic=wrapper, method='basic', vectorized = False, paired = True)


def flux_metric_mapping(metric_label):
    metrics = {
        'E': 'Mean Error (ME)',     
        'MedE': 'Median Error (MedE)',
        'Ratio': 'Mean Ratio',             
        'MedRatio': 'Median Ratio',
        'AE': 'Mean Absolute Error (MAE)',
        'MedAE': 'Median Absolute Error (MedAE)',     
        'LE': 'Mean Log Error (MLE)',
        'MedLE': 'Median Log Error (MedLE)',
        'ALE': 'Mean Absolute Log Error (MALE)',
        'MedALE': 'Median Absolute Log Error (MedALE)',
        'RMSE': 'Root Mean Square Error (RMSE)',
        'RMSLE': 'Root Mean Square Log Error (RMSLE)',  
        'PE': 'Mean Percent Error (MPE)',     
        'APE': 'Mean Absolute Percent Error (MAPE)',
        'SPE': 'Mean Symmetric Percent Error (MSPE)',                  
        'SAPE': 'Mean Symmetric Absolute Percent Error (SMAPE)',                
        'r_lin':  "Pearson Correlation Coefficient (Linear)",
        'r_log': "Pearson Correlation Coefficient (Log)",
        'MAR': 'Mean Accuracy Ratio (MAR)',
        'MdSA': "Median Symmetric Accuracy (MdSA)",
        'spearman': 'Spearman Correlation Coefficient (Linear)',
        'fac2': 'Percentage within a factor of 2 (%)',
        'fac10': 'Percentage within an Order of Magnitude (%)'
    }
    return metrics[metric_label]


def time_metric_mapping(metric_label):
    metrics = {
        'E': 'Mean Error (pred - obs)',
        'MedE': 'Median Error (pred - obs)',
        'AE': 'Mean Absolute Error (|pred - obs|)',
        'MedAE': 'Median Absolute Error (|pred - obs|)'
    }

    return metrics[metric_label]



def probability_uncertainties(df, label, dict):
    metrics_dict ={
        'brier_score': metrics.calc_brier,
        'brier_skill': metrics.calc_brier_skill,
        'spearman': metrics.calc_spearman,
        'roc_auc': metrics.receiver_operator_characteristic
    }
    mapped_label = 'SEP Probability'
    pred_label = 'Predicted ' + mapped_label
    obs_label = 'Observed ' + mapped_label

    obs = df[obs_label]
    pred = df[pred_label]
    print(obs, pred)
    dict['Model'].append(df['Model'].iloc[0] + ' Uncertainty')
    dict['Energy Channel'].append(df['Energy Channel Key'].iloc[0])
    dict['Threshold'].append(df['Threshold Key'].iloc[0])
    dict['Prediction Energy Channel'].append(df['Prediction Energy Channel Key'].iloc[0])
    dict['Prediction Threshold'].append(df['Prediction Threshold Key'].iloc[0])
    dict['ROC Curve Plot'].append(None)

   
    mean_metrics_list = ['brier_score', 'brier_skill', 'spearman']
    # metrics_list = ['LE']
    for met in mean_metrics_list:
        func = metrics_dict[met]
        uncertainty = probability_call_bootstrapper(obs, pred, func)
        error = uncertainty.standard_error
        dict[prob_metric_mapping(met)].append(error)
        print(uncertainty, met)
    
    uncertainty = roc_call_bootstrapper(obs, pred)
    error = uncertainty.standard_error
    dict[prob_metric_mapping('roc_auc')].append(error)
    print(uncertainty, 'roc_auc')


    return

def probability_call_bootstrapper(obs, pred, func):
    return stats.bootstrap((obs, pred), statistic=func, method='basic', vectorized = False, paired = True)

def roc_call_bootstrapper(obs, pred):
    # I dont want to create the plots for ROC so not doing the call to the metrics.py instead I am 
    # creating the metric here
    def wrapper(obs, pred):
        fpr, tpr, thresholds = skl.roc_curve(obs, pred)
        roc_auc = skl.auc(fpr, tpr)
        return roc_auc
    
    return stats.bootstrap((obs, pred), statistic=wrapper, method='basic', vectorized = False, paired = True)


def prob_metric_mapping(metric_label):
    metrics = {
        'brier_score': 'Brier Score',
        'brier_skill': 'Brier Skill Score',
        'spearman': 'Spearman Correlation Coefficient',
        'roc_auc': 'Area Under ROC Curve'
    }

    return metrics[metric_label]



def all_clear_uncertainties(df, label, dict):
    """ 
    This is the most different from the rest of the uncertainies
    as it won't use the scipy library for its bootstrapping. I will
    write the algorithm for resampling by splitting the Observed All-Clear
    False and True into seperate arrays and take 80% of each array for
    the resampling. This should preserve the biased dataset and ensure
    any all clear metric that is influenced by dataset bias will retain 
    that bias (looking at you HSS). After resampling, calculating the 
    metrics, then the standard devaition will be found for each metric
    which will be the uncertainty.

    """
    pred_label = 'Predicted SEP All Clear'
    obs_label = 'Observed SEP All Clear'
    df = df.loc[(df['All Clear Match Status'] != 'Ongoing SEP Event')]
    df = df.dropna(subset='All Clear Match Status')
    all_clear_true =  df[df['Observed SEP All Clear'] == True]
    all_clear_false = df[df['Observed SEP All Clear'] == False]
    # print(all_clear_true)
    # print(len(all_clear_true), len(all_clear_false))
    # input()
    n_samples = 1000
    scores = scores_dict()
    for n in range(n_samples):
        sub_true = all_clear_true.sample(frac=0.75, replace = True)
        sub_false = all_clear_false.sample(frac=0.75, replace = True)

        sub_current = pd.concat([sub_true, sub_false])
        current_scores = metrics.calc_contingency_all_clear(sub_current, 'Observed SEP All Clear',
                'Predicted SEP All Clear')
        for met in current_scores:
            scores[met].append(current_scores[met])
    
    dict['Model'].append(df['Model'].iloc[0] + ' Uncertainty')
    dict['Energy Channel'].append(df['Energy Channel Key'].iloc[0])
    dict['Threshold'].append(df['Threshold Key'].iloc[0])
    dict['Prediction Energy Channel'].append(df['Prediction Energy Channel Key'].iloc[0])
    dict['Prediction Threshold'].append(df['Prediction Threshold Key'].iloc[0])
    dict["All Clear 'True Positives' (Hits)"].append(np.std(scores['TP'])) #Hits
    dict["All Clear 'False Positives' (False Alarms)"].append(np.std(scores['FP'])) #False Alarms
    dict["All Clear 'True Negatives' (Correct Negatives)"].append(np.std(scores['TN']))  #Correct negatives
    dict["All Clear 'False Negatives' (Misses)"].append(np.std(scores['FN'])) #Misses
    dict["N (Total Number of Forecasts)"].append(np.std(scores['TP'] + scores['FP'] + scores['TN'] + scores['FN']))
    dict["Percent Correct"].append(np.std(scores['PC']))
    dict["Bias"].append(np.std(scores['B']))
    dict["Hit Rate"].append(np.std(scores['H']))
    dict["False Alarm Rate"].append(np.std(scores['F']))
    dict['False Negative Rate'].append(np.std(scores['FNR']))
    dict["Frequency of Misses"].append(np.std(scores['FOM']))
    dict["Frequency of Hits"].append(np.std(scores['FOH']))
    dict["Probability of Correct Negatives"].append(np.std(scores['POCN']))
    dict["Frequency of Correct Negatives"].append(np.std(scores['FOCN']))
    dict["False Alarm Ratio"].append(np.std(scores['FAR']))
    dict["Detection Failure Ratio"].append(np.std(scores['DFR']))
    dict["Threat Score"].append(np.std(scores['TS'])) #Critical Success Index
    dict["Odds Ratio"].append(np.std(scores['OR']))
    dict["Gilbert Skill Score"].append(np.std(scores['GSS'])) #Equitable Threat Score
    dict["True Skill Statistic"].append(np.std(scores['TSS']))
    dict["Heidke Skill Score"].append(np.std(scores['HSS']))
    dict["Odds Ratio Skill Score"].append(np.std(scores['ORSS']))
    dict["Symmetric Extreme Dependency Score"].append(np.std(scores['SEDS']))
    dict["F1 Score"].append(np.std(scores['FONE']))
    dict["F2 Score"].append(np.std(scores['FTWO']))
    dict["Fhalf Score"].append(np.std(scores['FHALF']))
    dict['Prevalence'].append(np.std(scores['PREV']))
    dict['Matthew Correlation Coefficient'].append(np.std(scores['MCC']))
    dict['Informedness'].append(np.std(scores['INFORM']))
    dict['Markedness'].append(np.std(scores['MARK']))
    dict['Prevalence Threshold'].append(np.std(scores['PT']))
    dict['Balanced Accuracy'].append(np.std(scores['BA']))
    dict['Fowlkes-Mallows Index'].append(np.std(scores['FM']))
    dict["Number SEP Events Correctly Predicted"].append(None)
    dict["Number SEP Events Missed"].append(None)
    dict["Predicted SEP Events"].append(None)
    dict["Missed SEP Events"].append(None)


def scores_dict():
    scores = {
        'TP': [],
        'FN': [],
        'FP': [],
        'TN': [],
        'PC': [],                           # Accuracy, Percent Correct
        'B': [],                          # Bias, Precision
        'H': [],                            # Hit Rate, Probability of Detection, Recall, Sensitivity
        'FAR': [],                          # False Alarm Ratio
        'F': [],                            # False Alarm Rate
        'FNR': [],                           # False Negative Rate, Miss Rate
        'FOH': [],                          # Frequency of Hits
        'FOM': [],                          # Frequency of Misses
        'POCN': [],                         # Probability of Correct Negatives, Specificity, Selectivity
        'DFR': [],                          # Detection Failure Ratio
        'FOCN': [],                         # Frequency of Correct Negatives
        'TS': [],                         # Threat Score, Critical success index
        'OR': [],                         # Odds Ratio
        'GSS': [],                      # Gilbert Skill score
        'TSS': [],      # True Skill Score
        'HSS': [], # Heidke Skill Score
        'ORSS': [],       # Odds Ratio Skill Score
        'SEDS': [],                    # Symmetric Extreme Dependency Score
        'FONE': [],                    # F_beta (1)
        'FTWO': [],                    # F_beta (2)
        'FHALF': [],                 # F_beta (0.5)
        'PREV': [],                         # Prevalence 
        'MCC': [],                       # Matthew Correlation Coefficient, Phi Coefficient
        'INFORM': [], # Informedness
        'MARK': [],  # Markedness
        'PT': [],                         # Prevalence Threshold
        'BA': [], # Balanced Accuracy
        'FM': []  # Fowlkes-Mallows Index (Geometric mean of precision and recall)
    }
    return scores