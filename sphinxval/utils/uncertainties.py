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
        After calculating metrics and putting them into dictionaries 
            validation -> calculate_intuitive_metrics -> after fill_*_dict
        Grab df (equivalent to _selections files) for each
            model/predicted quantity subset and dictionary containing metrics
        Use Scipy.stats bootstrap to resample and calculate the standard error
        Add row to metric dictionaries containing metric uncertainties

    VIVID -
        Add wrapper from VIVID feeder to generate the same subset that 
            is given to the _selections files but doesn't need to actually
            read in the file (take sphinx_dataframe and loop over energy/threshold/model?)
        Do the same bootstrapping
        Give out metric uncertainties (need to talk to Phil about VIVID inputs)
"""
#Create logger
logger = logging.getLogger(__name__)
mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)

def feeder_from_sphinx(df, dict, label, uncert_boolean):
    # take the dataframe passed from sphinx and insert into uncertainty workflow
    # needs to be a feeder function to make sure that the actual workflow is 
    # independent and modular
   
    uncertainty_workflow(df, label, dict, uncert_boolean)
    
    return


def feeder_from_vivid():



    # uncertainty_workflow(df, label, dict)

    return


def test_feeder():
    filename = './junk/peak_intensity_max_selections_SEPSTER2D CME_min.10.0.max.-1.0.units.MeV_threshold_10.0.csv'
    # filename = './junk/peak_intensity_max_time_selections_ZEUS+iPATH_CME_min.10.0.max.-1.0.units.MeV_threshold_10.0.csv'
    df = pd.read_csv(filename, index_col = 0)
    label = 'peak_intensity_max'
    # df = time_uncertainties(df, label)
    df = flux_uncertainties(df, label)




def uncertainty_workflow(df, label, dict, uncert_boolean):
    if uncert_boolean:
        # Splitting function that will correctly funnel to the correct uncertainty calculation
        flux_filter = ['point_intensity', 'peak_intensity', 'peak_intensity_max', 'max_flux_in_pred_win', 'fluence']
        time_filter = ['start_time', 'end_time', 'peak_intensity_time', 'peak_intensity_max_time', 'threshold_crossing', 'duration', 'last_data_to_issue_time']
        if label in time_filter:
            time_uncert_feeder(df, label, dict)
        elif label in flux_filter:
            flux_uncert_feeder(df, label, dict)
        else:
            if label == 'probability':
                probability_uncert_feeder(df, label, dict)
            elif label == 'all_clear':
                all_clear_uncert_feeder(df, label, dict)
    else:
        headers = dict.keys()
        for he in headers:
            if 'Uncertainty' in he:
                dict[he].append(np.nan)




    return 


def flux_uncert_feeder(df, label, dict):
    
    
    


    mapped_label = flux_label_mapping(label)

 
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



    if len(pred) < 2:
        headers = dict.keys()
        for he in headers:
            if 'Uncertainty' in he:
                dict[he].append(np.nan)
            else:
                pass
        return
    

    uncert_dict = calc_flux_uncertainties(obs, pred)
    headers = dict.keys()
    for he in headers:
        if 'Uncertainty' in he:
            dict[he].append(uncert_dict[he])
        else:
            pass

    return

def calc_flux_uncertainties(obs, pred):
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
    flux_dict = metrics_dicts.initialize_flux_dict()
    mean_metrics_list = ['E', 'Ratio', 'AE', 'LE', 'ALE', 'PE', 'APE', 'SPE', 'SAPE']
    for met in mean_metrics_list:
        
        func = metrics_func[met]
        uncertainty = mean_call_bootstrapper(obs, pred, func)
        error = uncertainty.standard_error
        flux_dict[flux_metric_mapping(met) + ' Uncertainty'].append(error)
        

        
       
    median_metrics_list = ['E', 'Ratio', 'AE', 'LE', 'ALE']
    for met in median_metrics_list:
        func = metrics_func[met]
        metric_label = met
        uncertainty = median_call_bootstrapper(obs, pred, func)
        error = uncertainty.standard_error
        metric_label = "Med" + metric_label
        flux_dict[flux_metric_mapping(metric_label) + ' Uncertainty'].append(error)
        
    #other metrics 
    other = ['MAR', 'RMSE', 'RMSLE', 'MdSA', 'spearman', 'r']
    for met in other:

        if met != 'r':
            func = metrics_func[met]
            uncertainty = stats.bootstrap((obs, pred), statistic=func, method='basic', vectorized = False, paired = True)
            error = uncertainty.standard_error
            flux_dict[flux_metric_mapping(met) + ' Uncertainty'].append(error)
            
        else:
            pearson_array = ['r_lin', 'r_log']
            for rs in pearson_array:
                uncertainty = pearson_call_bootstrapper(obs, pred, rs)
                error = uncertainty.standard_error
                flux_dict[flux_metric_mapping(rs) + ' Uncertainty'].append(error)

                
        


    factors = ['fac10', 'fac2']
    for fac in factors:
        if fac == 'fac10':
            thresh = 1
        else:
            thresh = np.log10(2)
        uncertainty = factor_call_bootstrapper(obs, pred, thresh)
        error = uncertainty.standard_error
        flux_dict[flux_metric_mapping(fac) + ' Uncertainty'].append(error)


    corr_plot_metrics = ['slope', 'yint']
    for met in corr_plot_metrics:
        uncertainty = corr_call_bootstrapper(obs, pred, met)
        error = uncertainty.standard_error
        logger.info(str(flux_metric_mapping(met) + ' Uncertainty'))
        flux_dict[flux_metric_mapping(met) + ' Uncertainty'].append(error)
    return flux_dict


def mean_call_bootstrapper(obs, pred, func):
    def wrapper(obs, pred):
        return np.nanmean(func(obs, pred))
    return stats.bootstrap((obs, pred), statistic=wrapper, method='basic', vectorized = False, paired = True, n_resamples = 1000)

def median_call_bootstrapper(obs, pred, func):
    def wrapper(obs, pred):
        return statistics.median(func(obs, pred))
    return stats.bootstrap((obs, pred), statistic=wrapper, method='basic', vectorized = False, paired = True, n_resamples = 1000)

def factor_call_bootstrapper(obs, pred, thresh):
    def wrapper(obs, pred):
        temp = metrics.switch_error_func('LE', obs, pred)
        count = sum(1 for x in temp if x <= thresh and x >= -thresh)
        return count / len(temp)
    return stats.bootstrap((obs, pred), statistic=wrapper, method='basic', vectorized = False, paired = True, n_resamples = 1000)
    
def pearson_call_bootstrapper(obs, pred, label):
    def wrapper(obs, pred):
        if label == 'r_lin':
            metric, _ = metrics.calc_pearson(obs, pred)
        else:
            _, metric = metrics.calc_pearson(obs, pred)
        return metric
    return stats.bootstrap((obs, pred), statistic=wrapper, method='basic', vectorized = False, paired = True, n_resamples = 1000)


def corr_call_bootstrapper(obs, pred, label):
    def wrapper(obs, pred):
        if label == 'slope':
            metric, _ = np.polyfit(obs, pred, 1)
        else:
            _, metric = np.polyfit(obs, pred, 1)
        return metric
    return stats.bootstrap((np.log10(obs), np.log10(pred)), statistic=wrapper, method='basic', vectorized = False, paired = True, n_resamples = 1000)



def flux_label_mapping(label):
    map_dict = {
        'point_intensity': 'SEP Point Intensity',
        'peak_intensity_max': 'SEP Peak Intensity Max (Max Flux)',
        'peak_intensity': 'SEP Peak Intensity (Onset Peak)',
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
        'duration': 'SEP Duration',
        'last_data_to_issue_time': "Last Data Time to Issue Time"
    }
    mapped_label = map_dict[label]
    return mapped_label


def time_uncert_feeder(df, label, dict):

    mapped_label = time_label_mapping(label)
    pred_label = 'Predicted ' + mapped_label
    obs_label = 'Observed ' + mapped_label
    
    obs = df[obs_label]
    pred = df[pred_label]
    

    if type(obs.iloc[0]) == str:
        obs = [datetime.strptime(x, "%Y-%m-%d %H:%M:%S") for x in obs]
    if type(pred.iloc[0]) == str:
        pred = [datetime.strptime(x, "%Y-%m-%d %H:%M:%S") for x in pred]

    if len(pred) < 2:
        mean_metrics_list = ['E', 'AE']
        for met in mean_metrics_list:
            dict[time_metric_mapping(met) + ' Uncertainty'].append(np.nan)

        for met in mean_metrics_list:
            metric_label = met
            metric_label = "Med" + metric_label
            dict[time_metric_mapping(metric_label) + ' Uncertainty'].append(np.nan)
        return

    uncert_dict = calc_time_uncertainties(obs, pred)
    headers = dict.keys()
    for he in headers:
        if 'Uncertainty' in he:
            dict[he].append(uncert_dict[he])
        else:
            pass
    return

def calc_time_uncertainties(obs, pred):
    time_dict = metrics_dicts.initialize_time_dict()
    mean_metrics_list = ['E', 'AE']
    for met in mean_metrics_list:
        uncertainty = mean_time_call_bootstrapper(obs, pred, met)
        error = uncertainty.standard_error
        time_dict[time_metric_mapping(met) + ' Uncertainty'].append(error)

    for met in mean_metrics_list:
        metric_label = met
        uncertainty = median_time_call_bootstrapper(obs, pred, met)
        error = uncertainty.standard_error
        metric_label = "Med" + metric_label
        time_dict[time_metric_mapping(metric_label) + ' Uncertainty'].append(error)
    
    return time_dict



def mean_time_call_bootstrapper(obs, pred, metric_label):
    def wrapper(obs, pred):
        
        td = (pred - obs)
        try:
            td = [x.total_seconds()/(60*60) for x in td]
        except:
            td = [pd.Timedelta(x).total_seconds()/(60*60) for x in td]
        if metric_label == 'AE':
            td = [np.abs(x) for x in td]
        return np.nanmean(td)
    return stats.bootstrap((obs, pred), statistic=wrapper, method='basic', vectorized = False, paired = True, n_resamples = 1000)

def median_time_call_bootstrapper(obs, pred, metric_label):
    def wrapper(obs, pred):
        td = (pred - obs)
        try:
            td = [x.total_seconds()/(60*60) for x in td]
        except:
            td = [pd.Timedelta(x).total_seconds()/(60*60) for x in td]
        if metric_label == 'medAE':
            td = [np.abs(x) for x in td]
        return statistics.median(td)
    return stats.bootstrap((obs, pred), statistic=wrapper, method='basic', vectorized = False, paired = True, n_resamples = 1000)


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
        'fac10': 'Percentage within an Order of Magnitude (%)',
        'slope': 'Linear Regression Slope',
        'yint': 'Linear Regression y-intercept'
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



def probability_uncert_feeder(df, label, dict):
    
    
    # prob_dict_mapping = prob_metric_mappin()
    
    mapped_label = 'SEP Probability'
    pred_label = 'Predicted ' + mapped_label
    obs_label = 'Observed ' + mapped_label
    all_clear_true =  df[df['Observed SEP Probability'] == 0.0]
    all_clear_false = df[df['Observed SEP Probability'] == 1.0]


    
    n_resamples = 1000
    # scores = prob_dict()
    for n in range(n_resamples):
        sub_true = all_clear_true.sample(frac=0.75, replace = True)
        sub_false = all_clear_false.sample(frac=0.75, replace = True)
        obs = pd.concat([sub_true[obs_label], sub_false[obs_label]], ignore_index = True)
        pred = pd.concat([sub_true[pred_label], sub_false[pred_label]], ignore_index = True)
        prob_dict = calc_prob_uncertainty(obs, pred)
        
    
    for met in prob_dict.keys():
        dict[prob_metric_mapping(met)+ ' Uncertainty'].append(np.nanstd(prob_dict[met]))


    


    return

def calc_prob_uncertainty(obs, pred):
    metrics_dict ={
        'brier_score': metrics.calc_brier,
        'brier_skill': metrics.calc_brier_skill,
        'spearman': metrics.calc_spearman,
        'roc_auc': metrics.receiver_operator_characteristic
    }

    prob_dict = {
        'brier_score': [],
        'brier_skill': [],
        'spearman': [],
        'roc_auc': []
    }
    prob_metrics =  ['brier_score', 'brier_skill', 'spearman', 'roc_auc']
    for met in prob_metrics:
            current_scores = probability_call_bootstrapper(obs, pred, metrics_dict[met], met)
        
            prob_dict[met].append(current_scores)


    return prob_dict

def probability_call_bootstrapper(obs, pred, func, metric):
    def roc_wrapper(obs, pred):  
        fpr, tpr, thresholds = skl.roc_curve(obs, pred)
        roc_auc = skl.auc(fpr, tpr)
        return roc_auc
    if metric == 'roc_auc':
        score = roc_wrapper(obs, pred)
    else:
        score = func(obs, pred)
    return score



# def probability_call_bootstrapper(obs, pred, func):
#     return stats.bootstrap((obs, pred), statistic=func, method='basic', vectorized = False, paired = True)

def roc_call_bootstrapper(obs, pred):
    # I dont want to create the plots for ROC so not doing the call to the metrics.py instead I am 
    # creating the metric here
    def wrapper(obs, pred):
        fpr, tpr, thresholds = skl.roc_curve(obs, pred)
        roc_auc = skl.auc(fpr, tpr)
        return roc_auc
    
    return stats.bootstrap((obs, pred), statistic=wrapper, method='basic', vectorized = False, paired = True, n_resamples = 1000)


def prob_metric_mapping(metric_label):
    metrics = {
        'brier_score': 'Brier Score',
        'brier_skill': 'Brier Skill Score',
        'spearman': 'Spearman Correlation Coefficient',
        'roc_auc': 'Area Under ROC Curve'
    }

    return metrics[metric_label]



def all_clear_uncert_feeder(df, label, dict):
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
    print(len(all_clear_false), len(all_clear_true))
    print(all_clear_false)
    print(all_clear_true)
    # input()
    if len(all_clear_false) < 2 or len(all_clear_true) < 2:
        headers = dict.keys()
        for he in headers:
            if 'Uncertainty' in he:
                dict[he].append(np.nan)
            else:
                pass
        return
    n_resamples = 1000
    scores = scores_dict()
    for n in range(n_resamples):
        sub_true = all_clear_true.sample(frac=0.75, replace = True)
        sub_false = all_clear_false.sample(frac=0.75, replace = True)

        sub_current = pd.concat([sub_true, sub_false])
        obs = sub_current['Observed SEP All Clear']
        pred = sub_current['Predicted SEP All Clear']
        
        current_scores = contingency_uncertainty(obs, pred)
        for met in current_scores:
            scores[met].append(current_scores[met])
    
    dict["All Clear 'True Positives' (Hits) Uncertainty"].append(np.nanstd(scores['TP'])) #Hits
    dict["All Clear 'False Positives' (False Alarms) Uncertainty"].append(np.nanstd(scores['FP'])) #False Alarms
    dict["All Clear 'True Negatives' (Correct Negatives) Uncertainty"].append(np.nanstd(scores['TN']))  #Correct negatives
    dict["All Clear 'False Negatives' (Misses) Uncertainty"].append(np.nanstd(scores['FN'])) #Misses
    dict["Percent Correct Uncertainty"].append(np.nanstd(scores['PC']))
    dict["Bias Uncertainty"].append(np.nanstd(scores['B']))
    dict["Hit Rate Uncertainty"].append(np.nanstd(scores['H']))
    dict["False Alarm Rate Uncertainty"].append(np.nanstd(scores['F']))
    dict['False Negative Rate Uncertainty'].append(np.nanstd(scores['FNR']))
    dict["Frequency of Misses Uncertainty"].append(np.nanstd(scores['FOM']))
    dict["Frequency of Hits Uncertainty"].append(np.nanstd(scores['FOH']))
    dict["Probability of Correct Negatives Uncertainty"].append(np.nanstd(scores['POCN']))
    dict["Frequency of Correct Negatives Uncertainty"].append(np.nanstd(scores['FOCN']))
    dict["False Alarm Ratio Uncertainty"].append(np.nanstd(scores['FAR']))
    dict["False Alarm Event Ratio Uncertainty"].append(np.nanstd(scores['FAER']))
    dict["Tau Uncertainty"].append(np.nanstd(scores['Tau']))
    dict["Detection Failure Ratio Uncertainty"].append(np.nanstd(scores['DFR']))
    dict["Threat Score Uncertainty"].append(np.nanstd(scores['TS']))
    dict["Odds Ratio Uncertainty"].append(np.nanstd(scores['OR']))
    dict["Gilbert Skill Score Uncertainty"].append(np.nanstd(scores['GSS']))
    dict["True Skill Statistic Uncertainty"].append(np.nanstd(scores['TSS']))
    dict["Heidke Skill Score Uncertainty"].append(np.nanstd(scores['HSS']))
    dict["Odds Ratio Skill Score Uncertainty"].append(np.nanstd(scores['ORSS']))
    dict["Symmetric Extreme Dependency Score Uncertainty"].append(np.nanstd(scores['SEDS']))
    dict["F1 Score Uncertainty"].append(np.nanstd(scores['FONE']))
    dict["F2 Score Uncertainty"].append(np.nanstd(scores['FTWO']))
    dict["Fhalf Score Uncertainty"].append(np.nanstd(scores['FHALF']))
    dict['Prevalence Uncertainty'].append(np.nanstd(scores['PREV']))
    dict['Matthew Correlation Coefficient Uncertainty'].append(np.nanstd(scores['MCC']))
    dict['Informedness Uncertainty'].append(np.nanstd(scores['INFORM']))
    dict['Markedness Uncertainty'].append(np.nanstd(scores['MARK']))
    dict['Prevalence Threshold Uncertainty'].append(np.nanstd(scores['PT']))
    dict['Balanced Accuracy Uncertainty'].append(np.nanstd(scores['BA']))
    dict['Fowlkes-Mallows Index Uncertainty'].append(np.nanstd(scores['FM']))
    return


def build_contingency_table(obs, pred):
    result = (obs == False) & (pred == False)
    h = result.sum(axis=0)
    
    #MISSES: obs = False, pred = True
    result = (obs == False) & (pred == True)
    m = result.sum(axis=0)
    
    #FALSE POSITIVE: obs = True, pred = False
    result = (obs == True) & (pred == False)
    f = result.sum(axis=0)

    #TRUE NEGATIVES: obs = True, pred = True
    result = (obs == True) & (pred == True)
    c = result.sum(axis=0)



    return h, m, f, c


def contingency_uncertainty(obs, pred):
    h, m, f, c = build_contingency_table(obs, pred)
    scores = metrics.contingency_scores(h,m,f,c)

    tau = 1 - (np.sqrt((f/(c + f))**2 + (m/(h + m))**2)/np.sqrt(2))
    scores['Tau'] = tau
    return scores


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
        'FAER': [],
        'Tau': [],
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
        'FM': [],  # Fowlkes-Mallows Index (Geometric mean of precision and recall)
    }
    return scores



def time_profile_uncertainties(error_dict, dict):

    n_resamples = 1000
    for key in error_dict.keys():    
        current_metrics = []
        for n in range(n_resamples):
            
            current_metrics.append(pd.Series(error_dict[key]).sample(frac=0.75, replace = True))
        # logger.info(key)
        # logger.info(np.mean(error_dict[key]))
        # logger.info(np.nanstd(current_metrics))
        dict[key + ' Uncertainty'].append(np.nanstd(current_metrics))


    return