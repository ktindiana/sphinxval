#Subroutines related to validation
from . import object_handler as objh
from . import metrics
from . import plotting_tools as plt_tools
from . import config
from scipy.stats import pearsonr
import statistics
import numpy as np
import sys
import os.path
import pandas as pd


__version__ = "0.1"
__author__ = "Katie Whitman"
__maintainer__ = "Katie Whitman"
__email__ = "kathryn.whitman@nasa.gov"

""" utils/validation.py contains subroutines to validate forecasts after
    they have been matched to observations.
    
"""

######### DATAFRAMES CONTAINING OBSERVATIONS AND PREDICTIONS ######

def initialize_dict():
    """ Set up a pandas df to hold each possible quantity,
        each observed energy channel, and predicted and
        observed values.
        
    """
    #Convert to Pandas dataframe
    #Include triggers with as much flattened info
    #If need multiple dimension, then could be used as tooltip info
    #Last CME, N CMEs, Last speed, last location, Timestamps array of all CMEs used
    

    dict = {"Model": [],
            "Energy Channel Key": [],
            "Threshold Key": [],
            "Forecast Source": [],
            "Forecast Issue Time":[],
            "Prediction Window Start": [],
            "Prediction Window End": [],
            "Number of CMEs": [],
            "CME Start Time": [], #Timestamp of 1st
                #coronagraph image CME is visible in
            "CME Liftoff Time": [], #Timestamp of coronagraph
                #image with 1st indication of CME liftoff (used by
                #CACTUS)
            "CME Latitude": [],
            "CME Longitude": [],
            "CME Speed": [],
            "CME Half Width": [],
            "CME PA": [],
            "CME Catalog": [],
            "Number of Flares": [],
            "Flare Latitude": [],
            "Flare Longitude": [],
            "Flare Start Time": [],
            "Flare Peak Time": [],
            "Flare End Time": [],
            "Flare Last Data Time": [],
            "Flare Intensity": [],
            "Flare Integrated Intensity": [],
            "Flare NOAA AR": [],
            "Observatory": [],
            "Observed Time Profiles": [], #string of comma
                                          #separated filenames
            "Observed SEP All Clear": [],
            "Observed SEP Probability": [],
            "Observed SEP Threshold Crossing Time": [],
            "Observed SEP Start Time":[],
            "Observed SEP End Time": [],
            "Observed SEP Peak Intensity (Onset Peak)": [],
            "Observed SEP Peak Intensity (Onset Peak) Units": [],
            "Observed SEP Peak Intensity (Onset Peak) Time": [],
            "Observed SEP Peak Intensity Max (Max Flux)": [],
            "Observed SEP Peak Intensity Max (Max Flux) Units": [],
            "Observed SEP Peak Intensity Max (Max Flux) Time": [],
            "Observed SEP Fluence": [],
            "Observed SEP Fluence Units": [],
            "Observed SEP Fluence Spectrum": [],
            "Observed SEP Fluence Spectrum Units": [],
            "Predicted SEP All Clear": [],
            "All Clear Match Status": [],
            "Predicted SEP Probability": [],
            "Probability Match Status": [],
            "Predicted SEP Threshold Crossing Time": [],
            "Threshold Crossing Time Match Status": [],
            "Predicted SEP Start Time":[],
            "Start Time Match Status": [],
            "Predicted SEP End Time": [],
            "End Time Match Status": [],
            "Predicted SEP Peak Intensity (Onset Peak)": [],
            "Predicted SEP Peak Intensity (Onset Peak) Units": [],
            "Predicted SEP Peak Intensity (Onset Peak) Time": [],
            "Peak Intensity Match Status": [],
            "Predicted SEP Peak Intensity Max (Max Flux)": [],
            "Predicted SEP Peak Intensity Max (Max Flux) Units": [],
            "Predicted SEP Peak Intensity Max (Max Flux) Time": [],
            "Peak Intensity Max Match Status": [],
            "Predicted SEP Fluence": [],
            "Predicted SEP Fluence Units": [],
            "Fluence Match Status": [],
            "Predicted SEP Fluence Spectrum": [],
            "Predicted SEP Fluence Spectrum Units": [],
            "Fluence Spectrum Match Status": [],
            "Time Profile Match Status": []}

    return dict



def fill_dict_row(sphinx, dict, energy_key, thresh_key):
    """ Add a row to a dataframe with all of the supporting information
        for the forecast and observations that needs to be passed to
        SPHINX-Web.
        
    Input:
    
        :sphinx: (SPHINX object) contains all prediction and matched observation
            information
        :predicted: The predicted value for one specific type of quantity (e.g.
            peak_intensity, all_clear, start_time)
        :observed: The matched up observed value of the same quantity
        :df: (pandas DataFrame) contains all matched and observed values for
            a specific quantity along with supporting information
        
    Output:
    
        :updated_df: (pandas DataFrame) The dataframe is updated another
            another row
        
    """

    ncme = len(sphinx.prediction.cmes)
    if ncme > 0:
        cme_start = sphinx.prediction.cmes[-1].start_time
        cme_liftoff = sphinx.prediction.cmes[-1].liftoff_time
        cme_lat = sphinx.prediction.cmes[-1].lat
        cme_lon = sphinx.prediction.cmes[-1].lon
        cme_pa = sphinx.prediction.cmes[-1].pa
        cme_half_width = sphinx.prediction.cmes[-1].half_width
        cme_speed = sphinx.prediction.cmes[-1].speed
        cme_catalog = sphinx.prediction.cmes[-1].catalog
    else:
        cme_start = None
        cme_liftoff = None
        cme_lat = None
        cme_lon = None
        cme_pa = None
        cme_half_width = None
        cme_speed = None
        cme_catalog = None
        
    nfl = len(sphinx.prediction.flares)
    if nfl > 0:
        fl_lat = sphinx.prediction.flares[-1].lat
        fl_lon = sphinx.prediction.flares[-1].lon
        fl_last_data_time = sphinx.prediction.flares[-1].last_data_time
        fl_start_time = sphinx.prediction.flares[-1].start_time
        fl_peak_time = sphinx.prediction.flares[-1].peak_time
        fl_end_time = sphinx.prediction.flares[-1].end_time
        fl_intensity = sphinx.prediction.flares[-1].intensity
        fl_integrated_intensity = sphinx.prediction.flares[-1].integrated_intensity
        fl_AR = sphinx.prediction.flares[-1].noaa_region
    else:
        fl_lat = None
        fl_lon = None
        fl_last_data_time = None
        fl_start_time = None
        fl_peak_time = None
        fl_end_time = None
        fl_intensity = None
        fl_integrated_intensity = None
        fl_AR = None

    observatory = ""
    obs_time_prof = ""
    for i in range(len(sphinx.prediction_observation_windows_overlap)):
        if i == 0:
            observatory = sphinx.prediction_observation_windows_overlap[i].short_name
            obs_time_prof = sphinx.observed_sep_profiles[i]
        else:
            observatory += "," + sphinx.prediction_observation_windows_overlap[i].short_name
            obs_time_prof += "," + sphinx.observed_sep_profiles[i]
    

    ####PREDICTED VALUES
    pred_all_clear, ac_match_status = sphinx.return_predicted_all_clear()
    pred_prob, prob_match_status = sphinx.return_predicted_probability(thresh_key)
    pred_thresh_cross, tc_match_status =\
        sphinx.return_predicted_threshold_crossing_time(thresh_key)
    pred_start_time, st_match_status =\
        sphinx.return_predicted_start_time(thresh_key)
    pred_end_time, et_match_status =\
        sphinx.return_predicted_end_time(thresh_key)
    pred_fluence, pred_fl_units, fl_match_status =\
        sphinx.return_predicted_fluence(thresh_key)
    pred_fl_spec, pred_flsp_units, flsp_match_status =\
        sphinx.return_predicted_fluence_spectrum(thresh_key)
    pred_peak_intensity, pred_pi_units, pred_pi_time, pi_match_status =\
        sphinx.return_predicted_peak_intensity()
    pred_peak_intensity_max, pred_pimax_units, pred_pimax_time,\
        pimax_match_status = sphinx.return_predicted_peak_intensity_max()


    dict["Model"].append(sphinx.prediction.short_name)
    dict["Energy Channel Key"].append(energy_key)
    dict["Threshold Key"].append(thresh_key)
    dict["Forecast Source"].append(sphinx.prediction.source)
    dict["Forecast Issue Time"].append(sphinx.prediction.issue_time)
    dict["Prediction Window Start"].append(sphinx.prediction.prediction_window_start)
    dict["Prediction Window End"].append(sphinx.prediction.prediction_window_end)
    dict["Number of CMEs"].append(ncme)
    dict["CME Start Time"].append(cme_start) #Timestamp of 1st
            #coronagraph image CME is visible in
    dict["CME Liftoff Time"].append(cme_liftoff) #Timestamp of coronagraph
            #image with 1st indication of CME liftoff (used by
            #CACTUS)
    dict["CME Latitude"].append(cme_lat)
    dict["CME Longitude"].append(cme_lon)
    dict["CME Speed"].append(cme_speed)
    dict["CME Half Width"].append(cme_half_width)
    dict["CME PA"].append(cme_pa)
    dict["CME Catalog"].append(cme_catalog)
    dict["Number of Flares"].append(nfl)
    dict["Flare Latitude"].append(fl_lat)
    dict["Flare Longitude"].append(fl_lon)
    dict["Flare Start Time"].append(fl_start_time)
    dict["Flare Peak Time"].append(fl_peak_time)
    dict["Flare End Time"].append(fl_end_time)
    dict["Flare Last Data Time"].append(fl_last_data_time)
    dict["Flare Intensity"].append(fl_intensity)
    dict["Flare Integrated Intensity"].append(fl_integrated_intensity)
    dict["Flare NOAA AR"].append(fl_AR)
    dict["Observatory"].append(observatory)
    dict["Observed Time Profiles"].append(obs_time_prof) #string of comma
                              #separated filenames
    dict["Observed SEP All Clear"].append(sphinx.observed_all_clear.all_clear_boolean)
    
    try:
        dict["Observed SEP Probability"].append(sphinx.observed_probability[thresh_key].probability_value)
    except:
        dict["Observed SEP Probability"].append(None)

    try:
        dict["Observed SEP Threshold Crossing Time"].append(sphinx.observed_threshold_crossing[thresh_key].crossing_time)
    except:
        dict["Observed SEP Threshold Crossing Time"].append(None)

    try:
        dict["Observed SEP Start Time"].append(sphinx.observed_start_time[thresh_key])
    except:
        dict["Observed SEP Start Time"].append(None)

    try:
        dict["Observed SEP End Time"].append(sphinx.observed_end_time[thresh_key])
    except:
        dict["Observed SEP End Time"].append(None)

    dict["Observed SEP Peak Intensity (Onset Peak)"].append(sphinx.observed_peak_intensity.intensity)
    dict["Observed SEP Peak Intensity (Onset Peak) Units"].append(sphinx.observed_peak_intensity.units)
    dict["Observed SEP Peak Intensity (Onset Peak) Time"].append(sphinx.observed_peak_intensity.time)
    dict["Observed SEP Peak Intensity Max (Max Flux)"].append(sphinx.observed_peak_intensity_max.intensity)
    dict["Observed SEP Peak Intensity Max (Max Flux) Units"].append(sphinx.observed_peak_intensity_max.units)
    dict["Observed SEP Peak Intensity Max (Max Flux) Time"].append(sphinx.observed_peak_intensity_max.time)
    
    try:
        dict["Observed SEP Fluence"].append(sphinx.observed_fluence[thresh_key].fluence)
    except:
        dict["Observed SEP Fluence"].append(None)

    try:
        dict["Observed SEP Fluence Units"].append(sphinx.observed_fluence[thresh_key].units)
    except:
        dict["Observed SEP Fluence Units"].append(None)


    try:
        dict["Observed SEP Fluence Spectrum"].append(sphinx.observed_fluence_spectrum[thresh_key].fluence_spectrum)
    except:
        dict["Observed SEP Fluence Spectrum"].append(None)

    try:
        dict["Observed SEP Fluence Spectrum Units"].append(sphinx.observed_fluence_spectrum[thresh_key].fluence_units)
    except:
        dict["Observed SEP Fluence Spectrum Units"].append(None)


    dict["Predicted SEP All Clear"].append(pred_all_clear)
    dict["All Clear Match Status"].append(ac_match_status)
    dict["Predicted SEP Probability"].append(pred_prob)
    dict["Probability Match Status"].append(prob_match_status)
    dict["Predicted SEP Threshold Crossing Time"].append(pred_thresh_cross)
    dict["Threshold Crossing Time Match Status"].append(tc_match_status)
    dict["Predicted SEP Start Time"].append(pred_start_time)
    dict["Start Time Match Status"].append(st_match_status)
    dict["Predicted SEP End Time"].append(pred_end_time)
    dict["End Time Match Status"].append(et_match_status)
    dict["Predicted SEP Peak Intensity (Onset Peak)"].append(pred_peak_intensity)
    dict["Predicted SEP Peak Intensity (Onset Peak) Units"].append(pred_pi_units)
    dict["Predicted SEP Peak Intensity (Onset Peak) Time"].append(pred_pi_time)
    dict["Peak Intensity Match Status"].append(pi_match_status)
    dict["Predicted SEP Peak Intensity Max (Max Flux)"].append(pred_peak_intensity_max)
    dict["Predicted SEP Peak Intensity Max (Max Flux) Units"].append(pred_pimax_units)
    dict["Predicted SEP Peak Intensity Max (Max Flux) Time"].append(pred_pimax_time)
    dict["Peak Intensity Max Match Status"].append(pimax_match_status)
    dict["Predicted SEP Fluence"].append(pred_fluence)
    dict["Predicted SEP Fluence Units"].append(pred_fl_units)
    dict["Fluence Match Status"].append(fl_match_status)
    dict["Predicted SEP Fluence Spectrum"].append(pred_fl_spec)
    dict["Predicted SEP Fluence Spectrum Units"].append(pred_flsp_units)
    dict["Fluence Spectrum Match Status"].append(flsp_match_status)
    dict["Time Profile Match Status"].append(None)


def prepare_outdirs():
    for datafmt in ('pkl', 'csv', 'json', 'md'):
        outdir = os.path.join(config.outpath, datafmt)
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
    
def write_df(df, name, log=True):
    """Writes a pandas dataframe to the standard location in multiple formats
    """
    dataformats = (('pkl',  getattr(df, 'to_pickle'), {}),
                   ('csv',  getattr(df, 'to_csv'), {}),
                   ('json', getattr(df, 'to_json'), {'default_handler':str}),
                   ('md',   getattr(df, 'to_markdown'), {}))
    for ext, write_func, kwargs in dataformats:
        filepath = os.path.join(config.outpath, ext, name + '.' + ext)
        write_func(filepath, **kwargs)
        if log:
            print('Wrote', filepath)

def fill_df(matched_sphinx, model_names, all_energy_channels,
    all_obs_thresholds):
    """ Fill in a dictionary with the all clear predictions and observations
        organized by model and energy channel.
    """
    #sorted by model, quantity, energy channel, threshold
    dict = initialize_dict()

    #Loop through the forecasts for each model and fill in quantity_dict
    #as appropriate
    for model in model_names:
        for channel in all_energy_channels:
            ek = objh.energy_channel_to_key(channel)

            print("---Model: " + model + ", Energy Channel: " + ek)
            for sphinx in matched_sphinx[model][ek]:
                for thresh in all_obs_thresholds[ek]:
                    tk = objh.threshold_to_key(thresh)
                    fill_dict_row(sphinx, dict, ek, tk)
                
    
    df = pd.DataFrame(dict)
    write_df(df, "SPHINX_dataframe")
    return df



##################### METRICS #####################
def initialize_flux_dict():
    """ Metrics used for fluxes.
    
    """
    dict = {"Model": [],
            "Energy Channel": [],
            "Threshold": [],
            "Scatter Plot": [],
            "Linear Regression Slope": [],
            "Linear Regression y-intercept": [],
            "Pearson Correlation Coefficient (Linear)": [],
            "Pearson Correlation Coefficient (Log)": [],
            "Spearman Correlation Coefficient (Linear)": [],
            "Spearman Correlation Coefficient (Log)": [],
            "Mean Error (ME)": [],
            "Median Error (MedE)": [],
            "Mean Log Error (MLE)": [],
            "Median Log Error (MedLE)": [],
            "Mean Absolute Error (MAE)": [],
            "Median Absolute Error (MedAE)": [],
            "Mean Absolute Log Error (MALE)": [],
            "Median Absolute Log Error (MedALE)": [],
            "Mean Absolute Percentage Error (MAPE)": [],
            "Mean Accuracy Ratio": [],
            "Root Mean Square Error (RMSE)": [],
            "Root Mean Square Log Error (RMSLE)": [],
            "Median Symmetric Accuracy (MdSA)": []
            }
    
    return dict


def initialize_time_dict():
    """ Metrics for predictions related to time.
    
    """
    dict = {"Model": [],
            "Energy Channel": [],
            "Threshold": [],
            "Mean Error (pred - obs)": [],
            "Median Error (pred - obs)": [],
            "Mean Absolute Error (|pred - obs|)": [],
            "Median Absolute Error (|pred - obs|)": [],
            }
            
    return dict


def initialize_all_clear_dict():
    """ Metrics for all clear predictions.
    
    """
    dict = {"Model": [],
            "Energy Channel": [],
            "Threshold": [],
            "All Clear True Positives": [], #Hits
            "All Clear False Positives": [], #False Alarms
            "All Clear True Negatives": [],  #Correct negatives
            "All Clear False Negatives": [], #Misses
            "Percent Correct": [],
            "Bias": [],
            "Hit Rate": [],
            "False Alarm Rate": [],
            "Frequency of Misses": [],
            "Frequency of Hits": [],
            "Probability of Correct Negatives": [],
            "Frequency of Correct Negatives": [],
            "False Alarm Ratio": [],
            "Detection Failure Ratio": [],
            "Threat Score": [],
            "Odds Ratio": [],
            "Gilbert Skill Score": [],
            "True Skill Statistic": [],
            "Heidke Skill Score": [],
            "Odds Ratio Skill Score": [],
#            "Mean Percentage Error": [],
#            "Mean Absolute Percentage Error": []
            }
            
    return dict

            
def initialize_probability_dict():
    """ Metrics for probability predictions.
    
    """
    dict = {"Model": [],
            "Energy Channel": [],
            "Threshold": [],
            "Brier Score": [],
            "Brier Skill Score": [],
            "Linear Correlation Coefficient": [],
            "Rank Order Correlation Coefficient": [],
            }
            
    return dict




def fill_flux_metrics_dict(dict, model, energy_key, thresh_key, figname,
    slope, yint, r_lin, r_log, s_lin, s_log, ME, MedE, MLE, MedLE, MAE,
    MedAE, MALE, MedALE, MAPE, MAR, RMSE, RMSLE, MdSA):
    """ Put flux-related metrics into metrics dictionary.
    
    """
    dict["Model"].append(model)
    dict["Energy Channel"].append(energy_key)
    dict["Threshold"].append(thresh_key)
    dict["Scatter Plot"].append(figname)
    dict["Linear Regression Slope"].append(slope)
    dict["Linear Regression y-intercept"].append(yint)
    dict["Pearson Correlation Coefficient (Linear)"].append(r_lin)
    dict["Pearson Correlation Coefficient (Log)"].append(r_log)
    dict["Spearman Correlation Coefficient (Linear)"].append(s_lin)
    dict["Spearman Correlation Coefficient (Log)"].append(s_log)
    dict["Mean Error (ME)"].append(ME)
    dict["Median Error (MedE)"].append(MedE)
    dict["Mean Log Error (MLE)"].append(MLE)
    dict["Median Log Error (MedLE)"].append(MedLE)
    dict["Mean Absolute Error (MAE)"].append(MAE)
    dict["Median Absolute Error (MedAE)"].append(MedAE)
    dict["Mean Absolute Log Error (MALE)"].append(MALE)
    dict["Median Absolute Log Error (MedALE)"].append(MedALE)
    dict["Mean Absolute Percentage Error (MAPE)"].append(MAPE)
    dict["Mean Accuracy Ratio"].append(MAR)
    dict["Root Mean Square Error (RMSE)"].append(RMSE)
    dict["Root Mean Square Log Error (RMSLE)"].append(RMSLE)
    dict["Median Symmetric Accuracy (MdSA)"].append(MdSA)
    


def fill_time_metrics_dict(dict, model, energy_key, thresh_key,
    ME, MedE, MAE, MedAE):
    """ Fill in metrics for time
    """
    dict["Model"].append(model)
    dict["Energy Channel"].append(energy_key)
    dict["Threshold"].append(thresh_key)
    dict["Mean Error (pred - obs)"].append(ME)
    dict["Median Error (pred - obs)"].append(MedE)
    dict["Mean Absolute Error (|pred - obs|)"].append(MAE)
    dict["Median Absolute Error (|pred - obs|)"].append(MedAE)



def fill_all_clear_dict(dict, model, energy_key, thresh_key, scores):
    """ Fill the all clear metrics dictionary with metrics for each model.
    
    """
    dict["Model"].append(model)
    dict["Energy Channel"].append(energy_key)
    dict["Threshold"].append(thresh_key)
    dict["All Clear True Positives"].append(scores['TP']) #Hits
    dict["All Clear False Positives"].append(scores['FP']) #False Alarms
    dict["All Clear True Negatives"].append(scores['TN'])  #Correct negatives
    dict["All Clear False Negatives"].append(scores['FN']) #Misses
    dict["Percent Correct"].append(scores['PC'])
    dict["Bias"].append(scores['B'])
    dict["Hit Rate"].append(scores['H'])
    dict["False Alarm Rate"].append(scores['F'])
    dict["Frequency of Misses"].append(scores['FOM'])
    dict["Frequency of Hits"].append(scores['FOH'])
    dict["Probability of Correct Negatives"].append(scores['POCN'])
    dict["Frequency of Correct Negatives"].append(scores['FOCN'])
    dict["False Alarm Ratio"].append(scores['FAR'])
    dict["Detection Failure Ratio"].append(scores['DFR'])
    dict["Threat Score"].append(scores['TS']) #Critical Success Index
    dict["Odds Ratio"].append(scores['OR'])
    dict["Gilbert Skill Score"].append(scores['GSS']) #Equitable Threat Score
    dict["True Skill Statistic"].append(scores['TSS']) #Hanssen and Kuipers
            #discriminant (true skill statistic, Peirce's skill score)
    dict["Heidke Skill Score"].append(scores['HSS'])
    dict["Odds Ratio Skill Score"].append(scores['ORSS'])
#    dict["Mean Percentage Error"].append(scores[])
#    dict["Mean Absolute Percentage Error"].append(scores[])



def all_clear_intuitive_metrics(df, dict, model, energy_key, thresh_key):
    """ Extract the appropriate predictions and calculate metrics
        All Clear

    """
    #Select rows to calculate metrics
    sub = df.loc[(df['Model'] == model) & (df['Energy Channel Key'] ==
        energy_key) & (df['Threshold Key'] == thresh_key)]

    sub = sub[['Model','Energy Channel Key', 'Threshold Key', 'Forecast Source',
            'Prediction Window Start', 'Prediction Window End',
            'Observed SEP All Clear', 'Predicted SEP All Clear',
            'All Clear Match Status']]
    sub = sub.loc[(sub['All Clear Match Status'] != 'Ongoing SEP Event')]
    sub = sub.dropna() #drop rows containing None
      
    if sub.empty:
        return
    thr = thresh_key.strip().split(".")
    thresh_fnm = thr[0] + "_" + thr[1]
    write_df(sub, "all_clear_selections_" + model + "_" + energy_key.strip() + "_" + thresh_fnm)

    obs = sub['Observed SEP All Clear'].to_list()
    pred = sub['Predicted SEP All Clear'].to_list()

    print(obs)
    print(pred)

    #The metrics.py/calc_contingency_bool() routine needs the opposite boolean
    # In calc_contingency_bool, the pandas crosstab predicts booleans as:
    #   True = event
    #   False = no event
    # ALL CLEAR booleans are as follows:
    #   True = no event
    #   False = event
    # Prior to inputting all clear predictions into this code, need to
    #   switch the booleans
    opposite_obs = [not x for x in obs]
    opposite_pred = [not x for x in pred]
    
    scores = metrics.calc_contingency_bool(opposite_obs, opposite_pred)
    fill_all_clear_dict(dict, model, energy_key, thresh_key, scores)



def probabilty_intuitive_metrics(df, dict, model, energy_key, thresh_key):
    """ Extract the appropriate predictions and calculate metrics
        Probability

    dict = {"Model": [],
            "Energy Channel": [],
            "Threshold": [],
            "Brier Score": [],
            "Brier Skill Score": [], #Need a reference to calculate
            "Linear Correlation Coefficient": [],
            "Rank Order Correlation Coefficient": [],
            }

    """
    #Select rows to calculate metrics
    sub = df.loc[(df['Model'] == model) & (df['Energy Channel Key'] ==
        energy_key) & (df['Threshold Key'] == thresh_key)]

    sub = sub[['Model','Energy Channel Key', 'Threshold Key', 'Forecast Source',
            'Prediction Window Start', 'Prediction Window End',
            'Observed SEP Probability',
            'Predicted SEP Probability', 'Probability Match Status']]
    sub = sub.loc[(sub['Probability Match Status'] != 'Ongoing SEP Event')]
    sub = sub.dropna() #drop rows containing None
      
    if sub.empty:
        return
    thr = thresh_key.strip().split(".")
    thresh_fnm = thr[0] + "_" + thr[1]
    write_df(sub,"probability_selections_" + model + "_" + energy_key.strip() + "_" + thresh_fnm)

    obs = sub['Observed SEP Probability'].to_list()
    pred = sub['Predicted SEP Probability'].to_list()

    print(obs)
    print(pred)

    #Calculate metrics
    brier_score = metrics.calc_brier(obs, pred)
    brier_skill = None
    lin_corr_coeff = None
    rank_corr_coeff = None
    
    #Save to dict (ultimately dataframe)
    dict['Model'].append(model)
    dict['Energy Channel'].append(energy_key)
    dict['Threshold'].append(thresh_key)
    dict['Brier Score'].append(brier_score)
    dict['Brier Skill Score'].append(brier_skill)
    dict['Linear Correlation Coefficient'].append(lin_corr_coeff)
    dict['Rank Order Correlation Coefficient'].append(rank_corr_coeff)

    

def peak_intensity_intuitive_metrics(df, dict, model, energy_key, thresh_key):
    """ Extract the appropriate predictions and calculate metrics
        Peak intensity

    """
    #Select rows to calculate metrics
    sub = df.loc[(df['Model'] == model) & (df['Energy Channel Key'] ==
        energy_key) & (df['Threshold Key'] == thresh_key)]

    sub = sub[['Model','Energy Channel Key', 'Threshold Key', 'Forecast Source',
            'Prediction Window Start', 'Prediction Window End',
            'Observed SEP Threshold Crossing Time',
            'Observed SEP Peak Intensity (Onset Peak)',
            'Observed SEP Peak Intensity (Onset Peak) Units',
            'Predicted SEP Peak Intensity (Onset Peak)',
            'Predicted SEP Peak Intensity (Onset Peak) Units',
            'Peak Intensity Match Status']]
    sub = sub.loc[(sub['Peak Intensity Match Status'] == 'SEP Event')]
    sub = sub.dropna() #drop rows containing None
      
    if sub.empty:
        return

    thr = thresh_key.strip().split(".")
    thresh_fnm = thr[0] + "_" + thr[1]
    write_df(sub, "peak_intensity_selections_" + model + "_" + energy_key.strip() + "_" + thresh_fnm)

    obs = sub['Observed SEP Peak Intensity (Onset Peak)'].to_list()
    pred = sub['Predicted SEP Peak Intensity (Onset Peak)'].to_list()
    units = sub.iloc[0]['Observed SEP Peak Intensity (Onset Peak) Units']
    print(obs)
    print(pred)

    if len(obs) > 1:
        #PEARSON CORRELATION
        r_lin, r_log = metrics.switch_error_func('r',obs,pred)
        s_lin = None
        s_log = None
        
        #LINEAR REGRESSION
        obs_np = np.array(obs)
        pred_np = np.array(pred)
        slope, yint = np.polyfit(obs_np, pred_np, 1)

        #Correlation Plot
        corr_plot = plt_tools.correlation_plot(obs, pred,
        "Peak Intensity Correlation", xlabel="Observations",
        ylabel=("Model Predictions (" + str(units) + ")"), use_log = True)

        figname = config.plotpath + '/Correlation_peak_intensity_' + model + "_" \
            + energy_key.strip() + "_" + thresh_fnm + ".pdf"
        corr_plot.savefig(figname, dpi=300, bbox_inches='tight')
    else:
        r_lin = None
        r_log = None
        s_lin = None
        s_log = None
        slope = None
        yint = None
        figname = ""


    ME = statistics.mean(metrics.switch_error_func('E',obs,pred))
    MedE = statistics.median(metrics.switch_error_func('E',obs,pred))
    MAE = statistics.mean(metrics.switch_error_func('AE',obs,pred))
    MedAE = statistics.median(metrics.switch_error_func('AE',obs,pred))
    MLE = statistics.mean(metrics.switch_error_func('LE',obs,pred))
    MedLE = statistics.median(metrics.switch_error_func('LE',obs,pred))
    MALE = statistics.mean(metrics.switch_error_func('LE',obs,pred))
    MedALE = statistics.median(metrics.switch_error_func('LE',obs,pred))
    MAPE = statistics.mean(metrics.switch_error_func('APE',obs,pred))
    MAR = None #Mean Accuracy Ratio
    RMSE = metrics.switch_error_func('RMSE',obs,pred)
    RMSLE = metrics.switch_error_func('RMSLE',obs,pred)
    MdSA = None

    ####METRICS
    fill_flux_metrics_dict(dict, model, energy_key, thresh_key, figname,
    slope, yint, r_lin, r_log, s_lin, s_log, ME, MedE, MLE, MedLE, MAE,
    MedAE, MALE, MedALE, MAPE, MAR, RMSE, RMSLE, MdSA)




def peak_intensity_max_intuitive_metrics(df, dict, model, energy_key,
    thresh_key):
    """ Extract the appropriate predictions and calculate metrics
        Peak intensity

    """
    #Select rows to calculate metrics
    sub = df.loc[(df['Model'] == model) & (df['Energy Channel Key'] ==
        energy_key) & (df['Threshold Key'] == thresh_key)]

    sub = sub[['Model','Energy Channel Key', 'Threshold Key', 'Forecast Source',
            'Prediction Window Start', 'Prediction Window End',
            'Observed SEP Threshold Crossing Time',
            'Observed SEP Peak Intensity Max (Max Flux)',
            'Observed SEP Peak Intensity Max (Max Flux) Units',
            'Predicted SEP Peak Intensity Max (Max Flux)',
            'Predicted SEP Peak Intensity Max (Max Flux) Units',
            'Peak Intensity Max Match Status']]
    sub = sub.loc[(sub['Peak Intensity Max Match Status'] == 'SEP Event')]
    sub = sub.dropna() #drop rows containing None
      
    if sub.empty:
        return
    thr = thresh_key.strip().split(".")
    thresh_fnm = thr[0] + "_" + thr[1]
    write_df(sub, "peak_intensity_max_selections_" + model + "_" + energy_key.strip() + "_" + thresh_fnm)

    obs = sub['Observed SEP Peak Intensity Max (Max Flux)'].to_list()
    pred = sub['Predicted SEP Peak Intensity Max (Max Flux)'].to_list()
    units = sub.iloc[0]['Observed SEP Peak Intensity Max (Max Flux) Units']
    print(obs)
    print(pred)

    if len(obs) > 1:
        #PEARSON CORRELATION
        r_lin, r_log = metrics.switch_error_func('r',obs,pred)
        s_lin = None
        s_log = None
        
        #LINEAR REGRESSION
        obs_np = np.array(obs)
        pred_np = np.array(pred)
        slope, yint = np.polyfit(obs_np, pred_np, 1)

        #Correlation Plot
        corr_plot = plt_tools.correlation_plot(obs, pred,
        "Peak Intensity Max (Max Flux) Correlation", xlabel="Observations",
        ylabel=("Model Predictions (" + str(units) + ")"),
        value="Peak Intensity Max (Max Flux)", use_log = True)

        figname = config.plotpath + '/Correlation_peak_intensity_max' + model + "_" \
                + energy_key.strip() + "_" + thresh_fnm + ".pdf"
        corr_plot.savefig(figname, dpi=300, bbox_inches='tight')
    else:
        r_lin = None
        r_log = None
        s_lin = None
        s_log = None
        slope = None
        yint = None
        figname = ""


    ME = statistics.mean(metrics.switch_error_func('E',obs,pred))
    MedE = statistics.median(metrics.switch_error_func('E',obs,pred))
    MAE = statistics.mean(metrics.switch_error_func('AE',obs,pred))
    MedAE = statistics.median(metrics.switch_error_func('AE',obs,pred))
    MLE = statistics.mean(metrics.switch_error_func('LE',obs,pred))
    MedLE = statistics.median(metrics.switch_error_func('LE',obs,pred))
    MALE = statistics.mean(metrics.switch_error_func('LE',obs,pred))
    MedALE = statistics.median(metrics.switch_error_func('LE',obs,pred))
    MAPE = statistics.mean(metrics.switch_error_func('APE',obs,pred))
    MAR = None #Mean Accuracy Ratio
    RMSE = metrics.switch_error_func('RMSE',obs,pred)
    RMSLE = metrics.switch_error_func('RMSLE',obs,pred)
    MdSA = None

    ####METRICS
    fill_flux_metrics_dict(dict, model, energy_key, thresh_key, figname,
    slope, yint, r_lin, r_log, s_lin, s_log, ME, MedE, MLE, MedLE, MAE,
    MedAE, MALE, MedALE, MAPE, MAR, RMSE, RMSLE, MdSA)




def fluence_intuitive_metrics(df, dict, model, energy_key,
    thresh_key):
    """ Extract the appropriate predictions and calculate metrics
        Fluence

    """
    #Select rows to calculate metrics
    sub = df.loc[(df['Model'] == model) & (df['Energy Channel Key'] ==
        energy_key) & (df['Threshold Key'] == thresh_key)]

    sub = sub[['Model','Energy Channel Key', 'Threshold Key', 'Forecast Source',
            'Prediction Window Start', 'Prediction Window End',
            'Observed SEP Threshold Crossing Time',
            'Observed SEP Fluence',
            'Observed SEP Fluence Units',
            'Predicted SEP Fluence',
            'Predicted SEP Fluence Units',
            'Fluence Match Status']]
    sub = sub.loc[(sub['Fluence Match Status'] == 'SEP Event')]
    sub = sub.dropna() #drop rows containing None
      
    if sub.empty:
        return
    thr = thresh_key.strip().split(".")
    thresh_fnm = thr[0] + "_" + thr[1]
    write_df(sub, "fluence_selections_" + model + "_" + energy_key.strip() + "_" + thresh_fnm)

    obs = sub['Observed SEP Fluence'].to_list()
    pred = sub['Predicted SEP Fluence'].to_list()
    units = sub.iloc[0]['Observed SEP Fluence Units']
    print(obs)
    print(pred)

    if len(obs) > 1:
        #PEARSON CORRELATION
        r_lin, r_log = metrics.switch_error_func('r',obs,pred)
        s_lin = None
        s_log = None
        
        #LINEAR REGRESSION
        obs_np = np.array(obs)
        pred_np = np.array(pred)
        slope, yint = np.polyfit(obs_np, pred_np, 1)

        #Correlation Plot
        corr_plot = plt_tools.correlation_plot(obs, pred,
        "Fluence Correlation", xlabel="Observations",
        ylabel=("Model Predictions (" + str(units) + ")"),
        use_log = True)

        figname = config.plotpath + '/Correlation_fluence_' + model + "_" \
                + energy_key.strip() + "_" + thresh_fnm + ".pdf"
        corr_plot.savefig(figname, dpi=300, bbox_inches='tight')
    else:
        r_lin = None
        r_log = None
        s_lin = None
        s_log = None
        slope = None
        yint = None
        figname = ""


    ME = statistics.mean(metrics.switch_error_func('E',obs,pred))
    MedE = statistics.median(metrics.switch_error_func('E',obs,pred))
    MAE = statistics.mean(metrics.switch_error_func('AE',obs,pred))
    MedAE = statistics.median(metrics.switch_error_func('AE',obs,pred))
    MLE = statistics.mean(metrics.switch_error_func('LE',obs,pred))
    MedLE = statistics.median(metrics.switch_error_func('LE',obs,pred))
    MALE = statistics.mean(metrics.switch_error_func('LE',obs,pred))
    MedALE = statistics.median(metrics.switch_error_func('LE',obs,pred))
    MAPE = statistics.mean(metrics.switch_error_func('APE',obs,pred))
    MAR = None #Mean Accuracy Ratio
    RMSE = metrics.switch_error_func('RMSE',obs,pred)
    RMSLE = metrics.switch_error_func('RMSLE',obs,pred)
    MdSA = None

    ####METRICS
    fill_flux_metrics_dict(dict, model, energy_key, thresh_key, figname,
    slope, yint, r_lin, r_log, s_lin, s_log, ME, MedE, MLE, MedLE, MAE,
    MedAE, MALE, MedALE, MAPE, MAR, RMSE, RMSLE, MdSA)



def threshold_crossing_intuitive_metrics(df, dict, model, energy_key,
    thresh_key):
    """ Extract the appropriate predictions and calculate metrics
        Threshold Crossing

    """
    #Select rows to calculate metrics
    sub = df.loc[(df['Model'] == model) & (df['Energy Channel Key'] ==
        energy_key) & (df['Threshold Key'] == thresh_key)]

    sub = sub[['Model','Energy Channel Key', 'Threshold Key', 'Forecast Source',
            'Prediction Window Start', 'Prediction Window End',
            'Observed SEP Threshold Crossing Time',
            'Predicted SEP Threshold Crossing Time',
            'Threshold Crossing Time Match Status']]
    sub = sub.loc[(sub['Threshold Crossing Time Match Status'] == 'SEP Event')]
    sub = sub.dropna() #drop rows containing None
      
    if sub.empty:
        return
    thr = thresh_key.strip().split(".")
    thresh_fnm = thr[0] + "_" + thr[1]
    write_df(sub, "threshold_crossing_time_selections_" + model + "_" + energy_key.strip() + "_" + thresh_fnm)

    obs = sub['Observed SEP Threshold Crossing Time'].to_list()
    pred = sub['Predicted SEP Threshold Crossing Time'].to_list()
    td = (sub['Predicted SEP Threshold Crossing Time'] - sub['Observed SEP Threshold Crossing Time']) #.to_list()
    print(obs)
    print(pred)
    
    td = td.dt.total_seconds()/(60*60) #convert to hours
    td = td.to_list()
    abs_td = [abs(x) for x in td]
    print(td)

    ME = statistics.mean(td)
    MedE = statistics.median(td)
    MAE = statistics.mean(abs_td)
    MedAE = statistics.median(abs_td)
    
    fill_time_metrics_dict(dict, model, energy_key, thresh_key,
    ME, MedE, MAE, MedAE)
    

def start_time_intuitive_metrics(df, dict, model, energy_key, thresh_key):
    """ Extract the appropriate predictions and calculate metrics
        Start Time

    """
    #Select rows to calculate metrics
    sub = df.loc[(df['Model'] == model) & (df['Energy Channel Key'] ==
        energy_key) & (df['Threshold Key'] == thresh_key)]

    sub = sub[['Model','Energy Channel Key', 'Threshold Key', 'Forecast Source',
            'Prediction Window Start', 'Prediction Window End',
            'Observed SEP Start Time',
            'Predicted SEP Start Time',
            'Start Time Match Status']]
    sub = sub.loc[(sub['Start Time Match Status'] == 'SEP Event')]
    sub = sub.dropna() #drop rows containing None
      
    if sub.empty:
        return
    thr = thresh_key.strip().split(".")
    thresh_fnm = thr[0] + "_" + thr[1]
    write_df(sub, "start_time_selections_" + model + "_" + energy_key.strip() + "_" + thresh_fnm)

    obs = sub['Observed SEP Start Time'].to_list()
    pred = sub['Predicted SEP Start Time'].to_list()
    td = (sub['Predicted SEP Start Time'] - sub['Observed SEP Start Time']).to_list()
    print(obs)
    print(pred)
    
    td = td.dt.total_seconds()/(60*60) #convert to hours
    td = td.to_list()
    abs_td = [abs(x) for x in td]
    print(td)

    ME = statistics.mean(td)
    MedE = statistics.median(td)
    MAE = statistics.mean(abs_td)
    MedAE = statistics.median(abs_td)
    
    fill_time_metrics_dict(dict, model, energy_key, thresh_key,
    ME, MedE, MAE, MedAE)


def end_time_intuitive_metrics(df, dict, model, energy_key,
    thresh_key):
    """ Extract the appropriate predictions and calculate metrics
        End Time

    """
    #Select rows to calculate metrics
    sub = df.loc[(df['Model'] == model) & (df['Energy Channel Key'] ==
        energy_key) & (df['Threshold Key'] == thresh_key)]

    sub = sub[['Model','Energy Channel Key', 'Threshold Key', 'Forecast Source',
            'Prediction Window Start', 'Prediction Window End',
            'Observed SEP End Time',
            'Predicted SEP End Time',
            'End Time Match Status']]
    sub = sub.loc[(sub['End Time Match Status'] == 'SEP Event')]
    sub = sub.dropna() #drop rows containing None
      
    if sub.empty:
        return
    thr = thresh_key.strip().split(".")
    thresh_fnm = thr[0] + "_" + thr[1]
    write_df(sub, "end_time_selections_" + model + "_" + energy_key.strip() + "_" + thresh_fnm)

    obs = sub['Observed SEP End Time'].to_list()
    pred = sub['Predicted End Time'].to_list()
    td = (sub['Predicted End Time'] - sub['Observed End Time']).to_list()
    print(obs)
    print(pred)
    
    td = td.dt.total_seconds()/(60*60) #convert to hours
    td = td.to_list()
    abs_td = [abs(x) for x in td]
    print(td)

    ME = statistics.mean(td)
    MedE = statistics.median(td)
    MAE = statistics.mean(abs_td)
    MedAE = statistics.median(abs_td)
    
    fill_time_metrics_dict(dict, model, energy_key, thresh_key,
    ME, MedE, MAE, MedAE)
 


def duration_intuitive_metrics(df, dict, model, energy_key, thresh_key):
    """ Extract the appropriate predictions and calculate metrics
        Start Time

    """
    #Select rows to calculate metrics
    sub = df.loc[(df['Model'] == model) & (df['Energy Channel Key'] ==
        energy_key) & (df['Threshold Key'] == thresh_key)]

    sub = sub[['Model','Energy Channel Key', 'Threshold Key', 'Forecast Source',
            'Prediction Window Start', 'Prediction Window End',
            'Observed SEP Start Time',
            'Observed SEP End Time',
            'Predicted SEP Start Time',
            'Predicted SEP End Time',
            'Start Time Match Status']]
    sub = sub.loc[(sub['Start Time Match Status'] == 'SEP Event')]
    sub = sub.dropna() #drop rows containing None
      
    if sub.empty:
        return
    thr = thresh_key.strip().split(".")
    thresh_fnm = thr[0] + "_" + thr[1]
    write_df(sub, "start_time_selections_" + model + "_" + energy_key.strip() + "_" + thresh_fnm)

    obs = (sub['Observed SEP End Time'] - sub['Observed SEP Start Time'])
    pred = (sub['Predicted SEP End Time'] - sub['Predicted SEP Start Time'])
    print(obs)
    print(pred)
    
    obs = obs.dt.total_seconds()/(60*60) #convert to hours
    pred = pred.dt.total_seconds()/(60*60)

    td = pred - obs #shorter duration is negative
    td = td.to_list()
    abs_td = [abs(x) for x in td]
    print(td)

    ME = statistics.mean(td)
    MedE = statistics.median(td)
    MAE = statistics.mean(abs_td)
    MedAE = statistics.median(abs_td)
    
    fill_time_metrics_dict(dict, model, energy_key, thresh_key,
    ME, MedE, MAE, MedAE)




def peak_intensity_time_intuitive_metrics(df, dict, model, energy_key,
    thresh_key):
    """ Extract the appropriate predictions and calculate metrics
        Peak Intensity Time

    """
    #Select rows to calculate metrics
    sub = df.loc[(df['Model'] == model) & (df['Energy Channel Key'] ==
        energy_key) & (df['Threshold Key'] == thresh_key)]

    sub = sub[['Model','Energy Channel Key', 'Threshold Key', 'Forecast Source',
            'Prediction Window Start', 'Prediction Window End',
            'Observed SEP Threshold Crossing Time',
            'Observed SEP Peak Intensity (Onset Peak) Time',
            'Predicted SEP Peak Intensity (Onset Peak) Time',
            'Peak Intensity Match Status']]
    sub = sub.loc[(sub['Peak Intensity Match Status'] == 'SEP Event')]
    sub = sub.dropna() #drop rows containing None
      
    if sub.empty:
        return
    thr = thresh_key.strip().split(".")
    thresh_fnm = thr[0] + "_" + thr[1]
    write_df(sub, "peak_intensity_time_selections_" + model + "_" + energy_key.strip() + "_" + thresh_fnm)

    obs = sub['Observed SEP Peak Intensity (Onset Peak) Time'].to_list()
    pred = sub['Predicted SEP Peak Intensity (Onset Peak) Time'].to_list()
    td = (sub['Predicted SEP Peak Intensity (Onset Peak) Time'] - sub['Observed SEP Peak Intensity (Onset Peak) Time'])#.to_list()
    print(obs)
    print(pred)
    
    td = td.dt.total_seconds()/(60*60) #convert to hours
    td = td.to_list()
    abs_td = [abs(x) for x in td]
    print(td)

    ME = statistics.mean(td)
    MedE = statistics.median(td)
    MAE = statistics.mean(abs_td)
    MedAE = statistics.median(abs_td)
    
    fill_time_metrics_dict(dict, model, energy_key, thresh_key,
    ME, MedE, MAE, MedAE)



def peak_intensity_max_time_intuitive_metrics(df, dict, model, energy_key,
    thresh_key):
    """ Extract the appropriate predictions and calculate metrics
        Peak Intensity Max Time

    """
    #Select rows to calculate metrics
    sub = df.loc[(df['Model'] == model) & (df['Energy Channel Key'] ==
        energy_key) & (df['Threshold Key'] == thresh_key)]

    sub = sub[['Model','Energy Channel Key', 'Threshold Key', 'Forecast Source',
            'Prediction Window Start', 'Prediction Window End',
            'Observed SEP Threshold Crossing Time',
            'Observed SEP Peak Intensity Max (Max Flux) Time',
            'Predicted SEP Peak Intensity Max (Max Flux) Time',
            'Peak Intensity Max Match Status']]
    sub = sub.loc[(sub['Peak Intensity Max Match Status'] == 'SEP Event')]
    sub = sub.dropna() #drop rows containing None
      
    if sub.empty:
        return
    thr = thresh_key.strip().split(".")
    thresh_fnm = thr[0] + "_" + thr[1]
    write_df(sub, "peak_intensity_max_time_selections_" + model + "_" + energy_key.strip() + "_" + thresh_fnm)

    obs = sub['Observed SEP Peak Intensity Max (Max Flux) Time'].to_list()
    pred = sub['Predicted SEP Peak Intensity Max (Max Flux) Time'].to_list()
    td = (sub['Predicted SEP Peak Intensity Max (Max Flux) Time'] - sub['Observed SEP Peak Intensity Max (Max Flux) Time']).to_list()
    print(obs)
    print(pred)
    
    td = td.dt.total_seconds()/(60*60) #convert to hours
    td = td.to_list()
    abs_td = [abs(x) for x in td]
    print(td)

    ME = statistics.mean(td)
    MedE = statistics.median(td)
    MAE = statistics.mean(abs_td)
    MedAE = statistics.median(abs_td)
    
    fill_time_metrics_dict(dict, model, energy_key, thresh_key,
    ME, MedE, MAE, MedAE)


def AWT_metrics(df, model_names, all_energy_channels,
    all_observed_thresholds):
    """ Metrics for Advanced Warning Time.
    
    """






def calculate_intuitive_metrics(df, model_names, all_energy_channels,
    all_observed_thresholds):
    """ Calculate metrics appropriate to each quantity and
        store in dataframes.
            
    Input:
    
        :df: (pandas DataFrame) containes matched observations and predictions
        :model_names: (array of strings) all models read into code
        
    Output:
    
        Metrics pandas dataframes
    
    """
    
    all_clear_dict = initialize_all_clear_dict ()
    probability_dict = initialize_probability_dict()
    peak_intensity_dict = initialize_flux_dict()
    peak_intensity_max_dict = initialize_flux_dict()
    fluence_dict = initialize_flux_dict()
    profile_dict = initialize_flux_dict()
    thresh_cross_dict = initialize_time_dict()
    start_time_dict = initialize_time_dict()
    end_time_dict = initialize_time_dict()
    duration_dict = initialize_time_dict()
    peak_intensity_time_dict = initialize_time_dict()
    peak_intensity_max_time_dict = initialize_time_dict()

    
    for model in model_names:
        for channel in all_energy_channels:
            ek = objh.energy_channel_to_key(channel)
            for thresh in all_observed_thresholds[ek]:
                tk = objh.threshold_to_key(thresh)

                #Probability
                probabilty_intuitive_metrics(df, probability_dict, model,
                    ek, tk)
                peak_intensity_intuitive_metrics(df, peak_intensity_dict, model,
                    ek, tk)
                peak_intensity_max_intuitive_metrics(df,
                    peak_intensity_max_dict, model, ek, tk)
                fluence_intuitive_metrics(df,fluence_dict, model, ek, tk)
                threshold_crossing_intuitive_metrics(df, thresh_cross_dict, model, ek, tk)
                start_time_intuitive_metrics(df, start_time_dict, model, ek, tk)
                end_time_intuitive_metrics(df, end_time_dict, model, ek, tk)
                duration_intuitive_metrics(df, duration_dict, model, ek, tk)
                peak_intensity_time_intuitive_metrics(df,
                    peak_intensity_time_dict, model, ek, tk)
                peak_intensity_max_time_intuitive_metrics(df,
                    peak_intensity_max_time_dict, model, ek, tk)
                all_clear_intuitive_metrics(df, all_clear_dict, model, ek, tk)

    prob_metrics_df = pd.DataFrame(probability_dict)
    peak_intensity_metrics_df = pd.DataFrame(peak_intensity_dict)
    peak_intensity_max_metrics_df = pd.DataFrame(peak_intensity_max_dict)
    fluence_metrics_df = pd.DataFrame(fluence_dict)
    thresh_cross_metrics_df = pd.DataFrame(thresh_cross_dict)
    start_time_metrics_df = pd.DataFrame(start_time_dict)
    end_time_metrics_df = pd.DataFrame(end_time_dict)
    duration_metrics_df = pd.DataFrame(duration_dict)
    peak_intensity_time_metrics_df = pd.DataFrame(peak_intensity_time_dict)
    peak_intensity_max_time_metrics_df = pd.DataFrame(peak_intensity_max_time_dict)
    all_clear_metrics_df = pd.DataFrame(all_clear_dict)

    if not prob_metrics_df.empty:
        write_df(prob_metrics_df, "probability_metrics")
    if not peak_intensity_metrics_df.empty:
        write_df(peak_intensity_metrics_df, "peak_intensity_metrics")
    if not peak_intensity_max_metrics_df.empty:
        write_df(peak_intensity_max_metrics_df, "peak_intensity_max_metrics")
    if not fluence_metrics_df.empty:
        write_df(fluence_metrics_df, "fluence_metrics")
    if not thresh_cross_metrics_df.empty:
        write_df(thresh_cross_metrics_df, "threshold_crossing_metrics")
    if not start_time_metrics_df.empty:
        write_df(start_time_metrics_df, "start_time_metrics")
    if not end_time_metrics_df.empty:
        write_df(end_time_metrics_df, "end_time_metrics")
    if not duration_metrics_df.empty:
        write_df(duration_metrics_df, "duration_metrics")
    if not peak_intensity_time_metrics_df.empty:
        write_df(peak_intensity_time_metrics_df, "peak_intensity_time_metrics")
    if not peak_intensity_max_time_metrics_df.empty:
        write_df(peak_intensity_max_time_metrics_df, "peak_intensity_max_time_metrics")
    if not all_clear_metrics_df.empty:
        write_df(all_clear_metrics_df, "all_clear_metrics")


def intuitive_validation(matched_sphinx, model_names, all_energy_channels,
    all_observed_thresholds, observed_sep_events):
    """ In the intuitive_validation subroutine, forecasts are validated in a
        way similar to which people would interpret forecasts.
    
        Forecasts are assessed (or useful to end users) up until the observed
        phenomenon happens. For example, only forecasts of peak flux are
        useful up until the observed peak happens. After that, a human would
        mentally filter out any additional forecasts for peak coming in from
        a model. Or, if the model's prediction window is large enough,
        continued peak flux forecasts could/would be interpreted for the
        next possible SEP event.
        
        In match.py, observed values have been matched to predicted values
        only if the last trigger or input time for the prediction was before
        the observed phenomenon.
        
        If a forecast was issued after the observed phenomenon, that forecast
        is ignored or, if the prediction window is large and extends past the
        current SEP event, is considered as a forecast for a next SEP event.
        
        This subroutine compared the predicted values to the matched
        observed values
        
        
    Input:
    
        :matched_sphinx: (SPHINX object) contains a Forecast object,
            Observation objects that are inside the forecast prediction
            window, and the observed values that are appropriately matched up
            to the forecast given the timing of the triggers/inputs and
            observed phenomena
        :model_names: (str array) array of the models whose predictions were
            read into the code
        :all_observed_thresholds: (dict) dictionary organized by energy
            channel and thresholds that were applied to observations (only
            predictions corresponding to thresholds that were applied to the
            observations can be validated)
        :observed_sep_events: (dict) dictionary organized by model name,
            energy channel, and threshold containing all unique observed SEP
            events that fell inside a forecast prediction window
    
    Output:
    
    
    
    """
    # Make sure the output directories exist
    prepare_outdirs()
    
    #For each model and predicted quantity, create arrays of paired up values
    #so can calculate metrics
    df = fill_df(matched_sphinx, model_names,
            all_energy_channels, all_observed_thresholds)


    calculate_intuitive_metrics(df, model_names, all_energy_channels,
            all_observed_thresholds)
