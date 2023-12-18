#Subroutines related to validation
from . import object_handler as objh
from . import metrics
from . import plotting_tools as plt_tools
from . import config
from . import time_profile as profile
from . import resume
import matplotlib.pylab as plt
from scipy.stats import pearsonr
import statistics
import numpy as np
import sys
import os.path
import pandas as pd
import datetime

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
            "Mismatch Allowed": [],
            "Prediction Energy Channel Key": [],
            "Prediction Threshold Key": [],
            "Forecast Source": [],
            "Forecast Path": [],
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
            "Observed Time Profile": [], #string of comma
                                          #separated filenames
            "Observed SEP All Clear": [],
            "Observed SEP Probability": [],
            "Observed SEP Threshold Crossing Time": [],
            "Observed SEP Start Time":[],
            "Observed SEP End Time": [],
            "Observed SEP Duration": [],
            "Observed SEP Fluence": [],
            "Observed SEP Fluence Units": [],
            "Observed SEP Fluence Spectrum": [],
            "Observed SEP Fluence Spectrum Units": [],
            "Observed SEP Peak Intensity (Onset Peak)": [],
            "Observed SEP Peak Intensity (Onset Peak) Units": [],
            "Observed SEP Peak Intensity (Onset Peak) Time": [],
            "Observed SEP Peak Intensity Max (Max Flux)": [],
            "Observed SEP Peak Intensity Max (Max Flux) Units": [],
            "Observed SEP Peak Intensity Max (Max Flux) Time": [],

            "Observed Point Intensity": [],
            "Observed Point Intensity Units": [],
            "Observed Point Intensity Time": [],
            "Observed Max Flux in Prediction Window": [],
            "Observed Max Flux in Prediction Window Units": [],
            "Observed Max Flux in Prediction Window Time": [],

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
            "Predicted SEP Duration": [],
            "Duration Match Status": [],
            "Predicted SEP Fluence": [],
            "Predicted SEP Fluence Units": [],
            "Fluence Match Status": [],
            "Predicted SEP Fluence Spectrum": [],
            "Predicted SEP Fluence Spectrum Units": [],
            "Fluence Spectrum Match Status": [],
            "Predicted SEP Peak Intensity (Onset Peak)": [],
            "Predicted SEP Peak Intensity (Onset Peak) Units": [],
            "Predicted SEP Peak Intensity (Onset Peak) Time": [],
            "Peak Intensity Match Status": [],
            "Predicted SEP Peak Intensity Max (Max Flux)": [],
            "Predicted SEP Peak Intensity Max (Max Flux) Units": [],
            "Predicted SEP Peak Intensity Max (Max Flux) Time": [],
            "Peak Intensity Max Match Status": [],
            
            "Predicted Point Intensity": [],
            "Predicted Point Intensity Units": [],
            "Predicted Point Intensity Time": [],

            "Predicted Time Profile": [],
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
    pred_energy_key = energy_key
    pred_thresh_key = thresh_key
    
    #If mismatch allowed for this prediction
    mismatch = sphinx.mismatch
    if mismatch:
        pred_energy_key = objh.energy_channel_to_key(config.mm_pred_energy_channel)
        pred_thresh_key = objh.threshold_to_key(config.mm_pred_threshold)
    
    pred_all_clear, ac_match_status = sphinx.return_predicted_all_clear()
    pred_prob, prob_match_status = sphinx.return_predicted_probability(thresh_key)
    pred_thresh_cross, tc_match_status =\
        sphinx.return_predicted_threshold_crossing_time(thresh_key)
    pred_start_time, st_match_status =\
        sphinx.return_predicted_start_time(thresh_key)
    pred_duration, duration_match_status =\
        sphinx.return_predicted_duration(thresh_key)
    pred_end_time, et_match_status =\
        sphinx.return_predicted_end_time(thresh_key)
    pred_fluence, pred_fl_units, fl_match_status =\
        sphinx.return_predicted_fluence(thresh_key)
    pred_fl_spec, pred_flsp_units, flsp_match_status =\
        sphinx.return_predicted_fluence_spectrum(thresh_key)
    pred_point_intensity, pred_pti_units, pred_pti_time =\
        sphinx.return_predicted_point_intensity()
    pred_peak_intensity, pred_pi_units, pred_pi_time, pi_match_status =\
        sphinx.return_predicted_peak_intensity()
    pred_peak_intensity_max, pred_pimax_units, pred_pimax_time,\
        pimax_match_status = sphinx.return_predicted_peak_intensity_max()
    pred_time_profile = sphinx.prediction.sep_profile
    tp_match_status = et_match_status
        

    dict["Model"].append(sphinx.prediction.short_name)
    dict["Energy Channel Key"].append(energy_key)
    dict["Threshold Key"].append(thresh_key)
    dict["Mismatch Allowed"].append(mismatch)
    dict["Prediction Energy Channel Key"].append(pred_energy_key)
    dict["Prediction Threshold Key"].append(pred_thresh_key)
    dict["Forecast Source"].append(sphinx.prediction.source)
    dict["Forecast Path"].append(sphinx.prediction.path)
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
    dict["Observed Time Profile"].append(obs_time_prof) #string of comma
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

    try:
        dict["Observed SEP Duration"].append(sphinx.observed_duration[thresh_key])
    except:
        dict["Observed SEP Duration"].append(None)


    dict["Observed SEP Peak Intensity (Onset Peak)"].append(sphinx.observed_peak_intensity.intensity)
    dict["Observed SEP Peak Intensity (Onset Peak) Units"].append(sphinx.observed_peak_intensity.units)
    dict["Observed SEP Peak Intensity (Onset Peak) Time"].append(sphinx.observed_peak_intensity.time)
    dict["Observed SEP Peak Intensity Max (Max Flux)"].append(sphinx.observed_peak_intensity_max.intensity)
    dict["Observed SEP Peak Intensity Max (Max Flux) Units"].append(sphinx.observed_peak_intensity_max.units)
    dict["Observed SEP Peak Intensity Max (Max Flux) Time"].append(sphinx.observed_peak_intensity_max.time)

    dict["Observed Point Intensity"].append(sphinx.observed_point_intensity.intensity)
    dict["Observed Point Intensity Units"].append(sphinx.observed_point_intensity.units)
    dict["Observed Point Intensity Time"].append(sphinx.observed_point_intensity.time)

    dict["Observed Max Flux in Prediction Window"].append(sphinx.observed_max_flux_in_prediction_window.intensity)
    dict["Observed Max Flux in Prediction Window Units"].append(sphinx.observed_max_flux_in_prediction_window.units)
    dict["Observed Max Flux in Prediction Window Time"].append(sphinx.observed_max_flux_in_prediction_window.time)

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
    dict["Predicted Point Intensity"].append(pred_point_intensity)
    dict["Predicted Point Intensity Units"].append(pred_pti_units)
    dict["Predicted Point Intensity Time"].append(pred_pti_time)
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
    dict["Predicted Time Profile"].append(pred_time_profile)
    dict["Time Profile Match Status"].append(tp_match_status)
    
    dict["Duration Match Status"].append(et_match_status)
    try:
        pred_duration = (pred_end_time - \
            pred_start_time).total_seconds()/(60.*60.)
        dict["Predicted SEP Duration"].append(pred_duration)
    except:
        dict["Predicted SEP Duration"].append(None)
        


def prepare_outdirs():
    for datafmt in ('pkl', 'csv', 'plots'):
        outdir = os.path.join(config.outpath, datafmt)
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
    
    if not os.path.isdir(config.reportpath):
        os.mkdir(config.reportpath)



def write_df(df, name, log=True):
    """Writes a pandas dataframe to the standard location in multiple formats
    """
    dataformats = (('pkl',  getattr(df, 'to_pickle'), {}),
                   ('csv',  getattr(df, 'to_csv'), {}))
    for ext, write_func, kwargs in dataformats:
        filepath = os.path.join(config.outpath, ext, name + '.' + ext)
        write_func(filepath, **kwargs)
        if log:
            print('Wrote', filepath)



def fill_df(matched_sphinx, model_names, all_energy_channels,
    all_obs_thresholds, DoResume):
    """ Fill in a dictionary with the all clear predictions and observations
        organized by model and energy channel.
    """
    #sorted by model, quantity, energy channel, threshold
    dict = initialize_dict()

    #Loop through the forecasts for each model and fill in quantity_dict
    #as appropriate
    for model in model_names:
        for ek in all_energy_channels:
            print("---Model: " + model + ", Energy Channel: " + ek)
            for sphinx in matched_sphinx[model][ek]:
                for tk in all_obs_thresholds[ek]:
                    fill_dict_row(sphinx, dict, ek, tk)
                
    
    df = pd.DataFrame(dict)
    if not DoResume: write_df(df, "SPHINX_dataframe")
    return df


##################### METRICS #####################
def initialize_flux_dict():
    """ Metrics used for fluxes.
    
    """
    dict = {"Model": [],
            "Energy Channel": [],
            "Threshold": [],
            "Prediction Energy Channel": [],
            "Prediction Threshold": [],
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
            "Prediction Energy Channel": [],
            "Prediction Threshold": [],
            "Mean Error (pred - obs)": [],
            "Median Error (pred - obs)": [],
            "Mean Absolute Error (|pred - obs|)": [],
            "Median Absolute Error (|pred - obs|)": [],
            }
            
    return dict
    
    
def initialize_awt_dict():
    """ Metrics for Adanced Warning Time to SEP start, SEP peak, SEP end.
        The "Forecasted Value" field indicates which forecasted quantity
        was used to calculate the AWT.
    """
    dict = {"Model": [],
            "Energy Channel": [],
            "Threshold": [],
            "Prediction Energy Channel": [],
            "Prediction Threshold": [],
            
            #All Clear Forecasts
            "Mean AWT for Predicted SEP All Clear to Observed SEP Threshold Crossing Time": [],
            "Median AWT for Predicted SEP All Clear to Observed SEP Threshold Crossing Time": [],
            "Mean AWT for Predicted SEP All Clear to Observed SEP Start Time": [],
            "Median AWT for Predicted SEP All Clear to Observed SEP Start Time": [],

#            #Probability Forecasts - cannot without an explicit threshold
#            "Mean AWT for Probability to Observed Threshold Crossing Time": [],
#            "Median AWT for Probability to Observed Threshold Crossing Time": [],
#            "Mean AWT for Probability to Observed Start Time": [],
#            "Median AWT for Probability to Observed Start Time": [],

            #Threshold Crossing Time Forecasts
            "Mean AWT for Predicted SEP Threshold Crossing Time to Observed SEP Threshold Crossing Time": [],
            "Median AWT for Predicted SEP Threshold Crossing Time to Observed SEP Threshold Crossing Time": [],
            "Mean AWT for Predicted SEP Threshold Crossing Time to Observed SEP Start Time": [],
            "Median AWT for Predicted SEP Threshold Crossing Time to Observed SEP Start Time": [],

            #Start Time Forecasts
            "Mean AWT for Predicted SEP Start Time to Observed SEP Threshold Crossing Time": [],
            "Median AWT for Predicted SEP Start Time to Observed SEP Threshold Crossing Time": [],
            "Mean AWT for Predicted SEP Start Time to Observed SEP Start Time": [],
            "Median AWT for Predicted SEP Start Time to Observed SEP Start Time": [],
 
             #Point Intensity Forecasts
            "Mean AWT for Predicted Point Intensity to Observed SEP Threshold Crossing Time": [],
            "Median AWT for Predicted Point Intensity to Observed SEP Threshold Crossing Time": [],
            "Mean AWT for Predicted Point Intensity to Observed SEP Start Time": [],
            "Median AWT for Predicted Point Intensity to Observed SEP Start Time": [],
 
 
            #Peak Intensity Forecasts
            "Mean AWT for Predicted SEP Peak Intensity (Onset Peak) to Observed SEP Threshold Crossing Time": [],
            "Median AWT for Predicted SEP Peak Intensity (Onset Peak) to Observed SEP Threshold Crossing Time": [],
            "Mean AWT for Predicted SEP Peak Intensity (Onset Peak) to Observed SEP Start Time": [],
            "Median AWT for Predicted SEP Peak Intensity (Onset Peak) to Observed SEP Start Time": [],
            "Mean AWT for Predicted SEP Peak Intensity (Onset Peak) to Observed SEP Peak Intensity (Onset Peak) Time": [],
            "Median AWT for Predicted SEP Peak Intensity (Onset Peak) to Observed SEP Peak Intensity (Onset Peak) Time": [],

            #Peak Intensity Max Forecasts
            "Mean AWT for Predicted SEP Peak Intensity Max (Max Flux) to Observed SEP Threshold Crossing Time": [],
            "Median AWT for Predicted SEP Peak Intensity Max (Max Flux) to Observed SEP Threshold Crossing Time": [],
            "Mean AWT for Predicted SEP Peak Intensity Max (Max Flux) to Observed SEP Start Time": [],
            "Median AWT for Predicted SEP Peak Intensity Max (Max Flux) to Observed SEP Start Time": [],
            "Mean AWT for Predicted SEP Peak Intensity Max (Max Flux) to Observed SEP Peak Intensity Max (Max Flux) Time": [],
            "Median AWT for Predicted SEP Peak Intensity Max (Max Flux) to Observed SEP Peak Intensity Max (Max Flux) Time": [],

            #End Time Forecasts
            "Mean AWT for Predicted SEP End Time to Observed SEP Threshold Crossing Time": [],
            "Median AWT for Predicted SEP End Time to Observed SEP Threshold Crossing Time": [],
            "Mean AWT for Predicted SEP End Time to Observed SEP Start Time": [],
            "Median AWT for Predicted SEP End Time to Observed SEP Start Time": [],
            "Mean AWT for Predicted SEP End Time to Observed SEP End Time": [],
            "Median AWT for Predicted SEP End Time to Observed SEP End Time": []
            }
            
    return dict


def initialize_all_clear_dict():
    """ Metrics for all clear predictions.
    
    """
    dict = {"Model": [],
            "Energy Channel": [],
            "Threshold": [],
            "Prediction Energy Channel": [],
            "Prediction Threshold": [],
            "All Clear 'True Positives' (Hits)": [], #Hits
            "All Clear 'False Positives' (False Alarms)": [], #False Alarms
            "All Clear 'True Negatives' (Correct Negatives)": [],  #Correct negatives
            "All Clear 'False Negatives' (Misses)": [], #Misses
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
            "Prediction Energy Channel": [],
            "Prediction Threshold": [],
            "Brier Score": [],
            "Brier Skill Score": [],
            "Linear Correlation Coefficient": [],
            "Rank Order Correlation Coefficient": [],
            }
            
    return dict




def fill_flux_metrics_dict(dict, model, energy_key, thresh_key,
    pred_energy_key, pred_thresh_key, figname,
    slope, yint, r_lin, r_log, s_lin, s_log, ME, MedE, MLE, MedLE, MAE,
    MedAE, MALE, MedALE, MAPE, MAR, RMSE, RMSLE, MdSA,timeprofplot=None):
    """ Put flux-related metrics into metrics dictionary.
    
    """
    dict["Model"].append(model)
    dict["Energy Channel"].append(energy_key)
    dict["Threshold"].append(thresh_key)
    dict["Prediction Energy Channel"].append(pred_energy_key)
    dict["Prediction Threshold"].append(pred_thresh_key)
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
        
    if timeprofplot != None:
        if "Time Profile Selection Plot" not in dict.keys():
            dict.update({"Time Profile Selection Plot": [timeprofplot]})
        else:
            dict["Time Profile Selection Plot"].append(timeprofplot)



def fill_time_metrics_dict(dict, model, energy_key, thresh_key, pred_energy_key,
    pred_thresh_key, ME, MedE, MAE, MedAE):
    """ Fill in metrics for time
    """
    dict["Model"].append(model)
    dict["Energy Channel"].append(energy_key)
    dict["Threshold"].append(thresh_key)
    dict["Prediction Energy Channel"].append(pred_energy_key)
    dict["Prediction Threshold"].append(pred_thresh_key)
    dict["Mean Error (pred - obs)"].append(ME)
    dict["Median Error (pred - obs)"].append(MedE)
    dict["Mean Absolute Error (|pred - obs|)"].append(MAE)
    dict["Median Absolute Error (|pred - obs|)"].append(MedAE)



def fill_all_clear_dict(dict, model, energy_key, thresh_key, pred_energy_key,
    pred_thresh_key, scores):
    """ Fill the all clear metrics dictionary with metrics for each model.
    
    """
    dict["Model"].append(model)
    dict["Energy Channel"].append(energy_key)
    dict["Threshold"].append(thresh_key)
    dict["Prediction Energy Channel"].append(pred_energy_key)
    dict["Prediction Threshold"].append(pred_thresh_key)
    dict["All Clear 'True Positives' (Hits)"].append(scores['TP']) #Hits
    dict["All Clear 'False Positives' (False Alarms)"].append(scores['FP']) #False Alarms
    dict["All Clear 'True Negatives' (Correct Negatives)"].append(scores['TN'])  #Correct negatives
    dict["All Clear 'False Negatives' (Misses)"].append(scores['FN']) #Misses
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


def make_thresh_fname(thresh_key):
    """ Make threshold string for filenames.
    
    """
    thr = thresh_key.strip().split(".units") #threshold.10.0.units.1 / (cm2 s sr)
    thr = thr[0] #threshold.10.0
    thr = thr.strip().split("threshold.")
    thresh_fnm = "threshold_" + thr[1]
    return thresh_fnm



def all_clear_intuitive_metrics(df, dict, model, energy_key, thresh_key):
    """ Extract the appropriate predictions and calculate metrics
        All Clear

        If mismatch = True, will extract only the predictions where
        the Mismatch Allowed field is True.
        
    """
    #Select rows to calculate metrics
    sub = df.loc[(df['Model'] == model) & (df['Energy Channel Key'] ==
        energy_key) & (df['Threshold Key'] == thresh_key)]

    sub = sub[['Model','Energy Channel Key', 'Threshold Key',
            'Mismatch Allowed',
            'Prediction Energy Channel Key', 'Prediction Threshold Key',
            'Forecast Source',
            'Prediction Window Start', 'Prediction Window End',
            'Observed SEP All Clear', 'Predicted SEP All Clear',
            'All Clear Match Status']]
    sub = sub.loc[(sub['All Clear Match Status'] != 'Ongoing SEP Event')]
    sub = sub.dropna() #drop rows containing None
      
    if sub.empty:
        return
    mismatch = bool(sub.iloc[0]['Mismatch Allowed'])
    pred_energy_key = str(sub.iloc[0]['Prediction Energy Channel Key'])
    pred_thresh_key = str(sub.iloc[0]['Prediction Threshold Key'])
    
    thresh_fnm = make_thresh_fname(thresh_key)
    fnm = "all_clear_selections_" + model + "_" + energy_key.strip() + "_" +\
            thresh_fnm
    if mismatch:
        fnm = fnm + "_mm"
    write_df(sub, fnm)

    obs = sub['Observed SEP All Clear'].to_list()
    pred = sub['Predicted SEP All Clear'].to_list()

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
    fill_all_clear_dict(dict, model, energy_key, thresh_key, pred_energy_key,
        pred_thresh_key, scores)

    return sub


def probability_intuitive_metrics(df, dict, model, energy_key, thresh_key):
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

    sub = sub[['Model','Energy Channel Key', 'Threshold Key',
            'Mismatch Allowed',
            'Prediction Energy Channel Key', 'Prediction Threshold Key',
            'Forecast Source',
            'Prediction Window Start', 'Prediction Window End',
            'Observed SEP Probability',
            'Predicted SEP Probability', 'Probability Match Status']]
    sub = sub.loc[(sub['Probability Match Status'] != 'Ongoing SEP Event')]
    sub = sub.dropna() #drop rows containing None
      
    if sub.empty:
        return

    mismatch = bool(sub.iloc[0]['Mismatch Allowed'])
    pred_energy_key = str(sub.iloc[0]['Prediction Energy Channel Key'])
    pred_thresh_key = str(sub.iloc[0]['Prediction Threshold Key'])


    thresh_fnm = make_thresh_fname(thresh_key)
    fnm = "probability_selections_" + model + "_" + energy_key.strip() + "_" +\
        thresh_fnm
    if mismatch:
        fnm = fnm + "_mm"
    write_df(sub,fnm)

    obs = sub['Observed SEP Probability'].to_list()
    pred = sub['Predicted SEP Probability'].to_list()

    #Calculate metrics
    brier_score = metrics.calc_brier(obs, pred)
    brier_skill = None
    lin_corr_coeff = None
    rank_corr_coeff = None
    
    #Save to dict (ultimately dataframe)
    dict['Model'].append(model)
    dict['Energy Channel'].append(energy_key)
    dict['Threshold'].append(thresh_key)
    dict['Prediction Energy Channel'].append(pred_energy_key)
    dict['Prediction Threshold'].append(pred_thresh_key)
    dict['Brier Score'].append(brier_score)
    dict['Brier Skill Score'].append(brier_skill)
    dict['Linear Correlation Coefficient'].append(lin_corr_coeff)
    dict['Rank Order Correlation Coefficient'].append(rank_corr_coeff)


def calc_all_flux_metrics(obs, pred):
    """ Calculate the metrics used for assessing fluxes.
    """
    ME = None
    MedE = None
    MAE = None
    MedAE = None
    MLE = None
    MedLE = None
    MALE = None
    MedALE = None
    MAPE = None
    MAR = None #Mean Accuracy Ratio
    RMSE = None
    RMSLE = None
    MdSA = None

    if len(obs) >= 1:
        ME = statistics.mean(metrics.switch_error_func('E',obs,pred))
        MedE = statistics.median(metrics.switch_error_func('E',obs,pred))
        MAE = statistics.mean(metrics.switch_error_func('AE',obs,pred))
        MedAE = statistics.median(metrics.switch_error_func('AE',obs,pred))
        MLE = statistics.mean(metrics.switch_error_func('LE',obs,pred))
        MedLE = statistics.median(metrics.switch_error_func('LE',obs,pred))
        MALE = statistics.mean(metrics.switch_error_func('ALE',obs,pred))
        MedALE = statistics.median(metrics.switch_error_func('ALE',obs,pred))
        MAPE = statistics.mean(metrics.switch_error_func('APE',obs,pred))
        MAR = None #Mean Accuracy Ratio
        RMSE = metrics.switch_error_func('RMSE',obs,pred)
        RMSLE = metrics.switch_error_func('RMSLE',obs,pred)
        MdSA = None

    return ME, MedE, MAE, MedAE, MLE, MedLE, MALE, MedALE, MAPE, MAR, RMSE,\
        RMSLE, MdSA




def point_intensity_intuitive_metrics(df, dict, model, energy_key, thresh_key, flux_threshold=0):
    """ Extract the appropriate predictions and calculate metrics
        Point intensity

        All observed point fluxes below flux_threshold will be excluded.
        Makes sense to set above a detector background level or to
        a warning threshold.
        
    """
    #Select rows to calculate metrics
    sub = df.loc[(df['Model'] == model) & (df['Energy Channel Key'] ==
        energy_key) & (df['Threshold Key'] == thresh_key)]

    sub = sub[['Model','Energy Channel Key', 'Threshold Key',
            'Mismatch Allowed',
            'Prediction Energy Channel Key', 'Prediction Threshold Key',
            'Forecast Source',
            'Prediction Window Start', 'Prediction Window End',
            'Observed Point Intensity',
            'Observed Point Intensity Time',
            'Observed Point Intensity Units',
            'Predicted Point Intensity',
            'Predicted Point Intensity Time',
            'Predicted Point Intensity Units']]
    
    sub = sub.loc[(sub['Observed Point Intensity'] >= flux_threshold)]

    sub = sub.dropna() #drop rows containing None
    if sub.empty:
        return

    mismatch = bool(sub.iloc[0]['Mismatch Allowed'])
    pred_energy_key = str(sub.iloc[0]['Prediction Energy Channel Key'])
    pred_thresh_key = str(sub.iloc[0]['Prediction Threshold Key'])
    thresh_fnm = make_thresh_fname(thresh_key)
    fnm = "point_intensity_selections_" + model + "_" + energy_key.strip() \
            + "_" + thresh_fnm
    if mismatch:
        fnm = fnm + "_mm"
    write_df(sub, fnm)



    #Calculate observed values via interpolation in the time profiles
    point_times = sub['Predicted Point Intensity Time'].to_list()
    pred = sub['Predicted Point Intensity'].to_list()
    units = sub.iloc[0]['Predicted Point Intensity Units']
    obs = sub['Observed Point Intensity'].to_list()

    if len(obs) > 1:
        #SAVE TIME PROFILE PLOTS FOR EVERY INTERVALS OF TIME
        first = min(point_times)
        last = max(point_times)
        interval = datetime.timedelta(days=15)
        nintervals = int((last-first).total_seconds()/interval.total_seconds()) + 1
        tp_plotnames = ""
        thresh_fnm = make_thresh_fname(thresh_key)
        
        for kk in range(nintervals):
            st_plot = first + kk*interval
            end_plot = st_plot + interval
            trim_times = []
            trim_pred = []
            trim_obs = []
            for ll in range(len(point_times)):
                if point_times[ll] >= st_plot and point_times[ll] < end_plot:
                    trim_times.append(point_times[ll])
                    trim_pred.append(pred[ll])
                    trim_obs.append(obs[ll])
            
            if not trim_times:
                continue
            
            str_date = date_to_string(st_plot)
            labels = [model, "Observations"]
            title = model + ", " + energy_key + " Point Intensity Time Profile"
            tpfigname = config.outpath + "/plots/Point_Intensity_Time_Profile_" + model \
                + "_" + energy_key + "_" + thresh_fnm  + "_" + str_date
            if mismatch:
                tpfigame = tpfigname + "_mm"
 
            if tp_plotnames == "":
                tp_plotnames = tpfigname + ".pdf"
            else:
                tp_plotnames += ";" + tpfigname + ".pdf"
 
 
            plt_tools.plot_time_profile([trim_times, trim_times], [trim_pred,trim_obs],
            labels, title=title, x_label="Date", y_min=1e-5, y_max=1e4,
            y_label="Particle Intensity", uselog_x = False, uselog_y = True,
            date_format="none", showplot=False,
            closeplot=True, saveplot=True, figname = tpfigname + ".pdf")
    
    
    
        #PEARSON CORRELATION
        r_lin, r_log = metrics.switch_error_func('r',obs,pred)
        s_lin = None
        s_log = None
        
        #LINEAR REGRESSION
        obs_np = np.array(obs)
        pred_np = np.array(pred)
        slope, yint = np.polyfit(obs_np, pred_np, 1)

        #Correlation Plot
        title = "Point Intensity Correlation (" + model + ", flux >= " + str(flux_threshold) + ")"
        corr_plot = plt_tools.correlation_plot(obs, pred, title,
            xlabel="Observations",
            ylabel=("Model Predictions (" + str(units) + ")"), use_log = True)

        figname = config.outpath + '/plots/Correlation_point_intensity_' \
            + model + "_" + energy_key.strip() + "_" + thresh_fnm
        if mismatch:
            figname = figname + "_mm"
        corr_plot.savefig(figname + ".pdf", dpi=300, bbox_inches='tight')
        corr_plot.close()
   
    else:
        r_lin = None
        r_log = None
        s_lin = None
        s_log = None
        slope = None
        yint = None
        figname = ""


    obs, pred = profile.remove_none(obs,pred)
    obs, pred = profile.remove_zero(obs, pred)
    if obs == [] or pred == []: return
    
    ME, MedE, MAE, MedAE, MLE, MedLE, MALE, MedALE, MAPE, MAR, RMSE,\
        RMSLE, MdSA = calc_all_flux_metrics(obs, pred)

    ####METRICS
    fill_flux_metrics_dict(dict, model, energy_key, thresh_key,
    pred_energy_key, pred_thresh_key, figname,
    slope, yint, r_lin, r_log, s_lin, s_log, ME, MedE, MLE, MedLE, MAE,
    MedAE, MALE, MedALE, MAPE, MAR, RMSE, RMSLE, MdSA, tp_plotnames)




def peak_intensity_intuitive_metrics(df, dict, model, energy_key, thresh_key):
    """ Extract the appropriate predictions and calculate metrics
        Peak intensity

    """
    #Select rows to calculate metrics
    sub = df.loc[(df['Model'] == model) & (df['Energy Channel Key'] ==
        energy_key) & (df['Threshold Key'] == thresh_key)]

    sub = sub[['Model','Energy Channel Key', 'Threshold Key',
            'Mismatch Allowed',
            'Prediction Energy Channel Key', 'Prediction Threshold Key',
            'Forecast Source',
            'Prediction Window Start', 'Prediction Window End',
            'Observed SEP Threshold Crossing Time',
            'Observed SEP Peak Intensity (Onset Peak)',
            'Observed SEP Peak Intensity (Onset Peak) Units',
            'Predicted SEP Peak Intensity (Onset Peak)',
            'Predicted SEP Peak Intensity (Onset Peak) Units',
            'Peak Intensity Match Status']]
    sub = sub.loc[(sub['Peak Intensity Match Status'] == 'SEP Event')]
    
    #Find predicted None values
    noneval = pd.isna(sub['Predicted SEP Peak Intensity (Onset Peak)'])
    #Extract only indices for Nones
    #True indicates that peak intensity was a None value
    noneval = noneval.loc[noneval == True]
    noneval = noneval.index.to_list()
    if len(noneval) > 0:
        for ix in noneval:
            sub = sub.drop(index=ix)
      
    if sub.empty:
        return

    mismatch = bool(sub.iloc[0]['Mismatch Allowed'])
    pred_energy_key = str(sub.iloc[0]['Prediction Energy Channel Key'])
    pred_thresh_key = str(sub.iloc[0]['Prediction Threshold Key'])
    
    thresh_fnm = make_thresh_fname(thresh_key)
    fnm = "peak_intensity_selections_" + model + "_" + energy_key.strip() \
            + "_" + thresh_fnm
    if mismatch:
        fnm = fnm + "_mm"
    write_df(sub, fnm)

    obs = sub['Observed SEP Peak Intensity (Onset Peak)'].to_list()
    pred = sub['Predicted SEP Peak Intensity (Onset Peak)'].to_list()
    units = sub.iloc[0]['Observed SEP Peak Intensity (Onset Peak) Units']

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

        figname = config.outpath + '/plots/Correlation_peak_intensity_' \
            + model + "_" + energy_key.strip() + "_" + thresh_fnm
        if mismatch:
            figname = figname + "_mm"
        corr_plot.savefig(figname + ".pdf", dpi=300, bbox_inches='tight')
        corr_plot.close()
    else:
        r_lin = None
        r_log = None
        s_lin = None
        s_log = None
        slope = None
        yint = None
        figname = ""


    ME, MedE, MAE, MedAE, MLE, MedLE, MALE, MedALE, MAPE, MAR, RMSE,\
        RMSLE, MdSA = calc_all_flux_metrics(obs, pred)

    ####METRICS
    fill_flux_metrics_dict(dict, model, energy_key, thresh_key,
    pred_energy_key, pred_thresh_key, figname,
    slope, yint, r_lin, r_log, s_lin, s_log, ME, MedE, MLE, MedLE, MAE,
    MedAE, MALE, MedALE, MAPE, MAR, RMSE, RMSLE, MdSA)




def peak_intensity_max_intuitive_metrics(df, dict, model, energy_key,
    thresh_key):
    """ Extract the appropriate predictions and calculate metrics
        Peak intensity

    """
    peak_key = 'Predicted SEP Peak Intensity Max (Max Flux)'

    #Select rows to calculate metrics
    sub = df.loc[(df['Model'] == model) & (df['Energy Channel Key'] ==
        energy_key) & (df['Threshold Key'] == thresh_key)]

    sub = sub[['Model','Energy Channel Key', 'Threshold Key',
            'Mismatch Allowed',
            'Prediction Energy Channel Key', 'Prediction Threshold Key',
            'Forecast Source',
            'Prediction Window Start', 'Prediction Window End',
            'Observed SEP Threshold Crossing Time',
            'Observed SEP Peak Intensity Max (Max Flux)',
            'Observed SEP Peak Intensity Max (Max Flux) Units',
            'Predicted SEP Peak Intensity Max (Max Flux)',
            'Predicted SEP Peak Intensity Max (Max Flux) Units',
            'Peak Intensity Max Match Status']]
    sub = sub.loc[(sub['Peak Intensity Max Match Status'] == 'SEP Event')]

    #Find predicted None values
    noneval = pd.isna(sub['Predicted SEP Peak Intensity Max (Max Flux)'])
    #Extract only indices for Nones
    #True indicates that peak intensity was a None value
    noneval = noneval.loc[noneval == True]
    noneval = noneval.index.to_list()
    if len(noneval) > 0:
        for ix in noneval:
            sub = sub.drop(index=ix)

    if not sub.empty:
        #Find predicted None values
        noneval = pd.isna(sub['Observed SEP Peak Intensity Max (Max Flux)'])
        #Extract only indices for Nones
        #True indicates that peak intensity was a None value
        noneval = noneval.loc[noneval == True]
        noneval = noneval.index.to_list()
        if len(noneval) > 0:
            for ix in noneval:
                sub = sub.drop(index=ix)
      
    #Models may fill only the Peak Intensity field. It can be ambiguous whether
    #the prediction is intended as onset peak or max flux. If no max flux field
    #found, then compare Peak Intensity to observed Max Flux.
    if sub.empty:
        sub = df.loc[(df['Model'] == model) & (df['Energy Channel Key'] ==
            energy_key) & (df['Threshold Key'] == thresh_key)]
        sub = sub[['Model','Energy Channel Key', 'Threshold Key',
            'Mismatch Allowed',
            'Prediction Energy Channel Key', 'Prediction Threshold Key',
            'Forecast Source',
            'Prediction Window Start', 'Prediction Window End',
            'Observed SEP Threshold Crossing Time',
            'Observed SEP Peak Intensity Max (Max Flux)',
            'Observed SEP Peak Intensity Max (Max Flux) Units',
            'Predicted SEP Peak Intensity (Onset Peak)',
            'Predicted SEP Peak Intensity (Onset Peak) Units',
            'Peak Intensity Match Status']]
        sub = sub.loc[(sub['Peak Intensity Match Status'] == 'SEP Event')]

        #Find predicted None values
        noneval = pd.isna(sub['Predicted SEP Peak Intensity (Onset Peak)'])
        #Extract only indices for Nones
        #True indicates that peak intensity was a None value
        noneval = noneval.loc[noneval == True]
        noneval = noneval.index.to_list()
        if len(noneval) > 0:
            for ix in noneval:
                sub = sub.drop(index=ix)

        #Find predicted None values
        noneval = pd.isna(sub['Observed SEP Peak Intensity Max (Max Flux)'])
        #Extract only indices for Nones
        #True indicates that peak intensity was a None value
        noneval = noneval.loc[noneval == True]
        noneval = noneval.index.to_list()
        if len(noneval) > 0:
            for ix in noneval:
                sub = sub.drop(index=ix)

        if sub.empty:
            return
        peak_key = 'Predicted SEP Peak Intensity (Onset Peak)'
        print("peak_intensity_max_intuitive_metrics: Model " + model +
                " did not explicitly "
                "include a peak_intensity_max field. Comparing "
                "peak_intensity to observed max flux.")

    mismatch = bool(sub.iloc[0]['Mismatch Allowed'])
    pred_energy_key = str(sub.iloc[0]['Prediction Energy Channel Key'])
    pred_thresh_key = str(sub.iloc[0]['Prediction Threshold Key'])

    thresh_fnm = make_thresh_fname(thresh_key)
    fnm = "peak_intensity_max_selections_" + model + "_" + energy_key.strip()\
        + "_" + thresh_fnm
    if mismatch:
        fnm = fnm + "_mm"
    write_df(sub, fnm)

    obs = sub['Observed SEP Peak Intensity Max (Max Flux)'].to_list()
    units = sub.iloc[0]['Observed SEP Peak Intensity Max (Max Flux) Units']
    pred = sub[peak_key].to_list()
 
    print("peak_intensity_max obs, pred")
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

        figname = config.outpath + '/plots/Correlation_peak_intensity_max_' \
                + model + "_" + energy_key.strip() + "_" + thresh_fnm
        print(figname)
        if mismatch:
            figname = figname + "_mm"
        corr_plot.savefig(figname+ ".pdf", dpi=300, bbox_inches='tight')
        corr_plot.close()
    else:
        r_lin = None
        r_log = None
        s_lin = None
        s_log = None
        slope = None
        yint = None
        figname = ""


    ME, MedE, MAE, MedAE, MLE, MedLE, MALE, MedALE, MAPE, MAR, RMSE,\
        RMSLE, MdSA = calc_all_flux_metrics(obs, pred)

    ####METRICS
    fill_flux_metrics_dict(dict, model, energy_key, thresh_key,
    pred_energy_key, pred_thresh_key, figname,
    slope, yint, r_lin, r_log, s_lin, s_log, ME, MedE, MLE, MedLE, MAE,
    MedAE, MALE, MedALE, MAPE, MAR, RMSE, RMSLE, MdSA)





def max_flux_in_pred_win_metrics(df, dict, model, energy_key,
    thresh_key):
    """ Extract the appropriate predictions and calculate metrics
        Compare predicted max or onset peak flux to the max observed
        flux in the model's prediction window.

        If a model provides peak_intensity_max, then will compare with
        observed max flux in prediction window. If that field isn't present in
        the prediction, then will compare the peak_intensity.

    """
    peak_key = 'Predicted SEP Peak Intensity Max (Max Flux)'
    
    #Select rows to calculate metrics
    sub = df.loc[(df['Model'] == model) & (df['Energy Channel Key'] ==
        energy_key) & (df['Threshold Key'] == thresh_key)]

    sub = sub[['Model','Energy Channel Key', 'Threshold Key',
            'Mismatch Allowed',
            'Prediction Energy Channel Key', 'Prediction Threshold Key',
            'Forecast Source',
            'Prediction Window Start', 'Prediction Window End',
            'Observed Max Flux in Prediction Window',
            'Observed Max Flux in Prediction Window Time',
            'Observed Max Flux in Prediction Window Units',
            'Predicted SEP Peak Intensity Max (Max Flux)',
            'Predicted SEP Peak Intensity Max (Max Flux) Units']]
    sub_test = sub['Predicted SEP Peak Intensity Max (Max Flux)'].dropna()
    #drop rows containing None
      
      
    #Models may fill only the Peak Intensity field. It can be ambiguous whether
    #the prediction is intended as onset peak or max flux. If no max flux field
    #found, then compare Peak Intensity to observed Max Flux.
    if sub_test.empty:
        sub = df.loc[(df['Model'] == model) & (df['Energy Channel Key'] ==
            energy_key) & (df['Threshold Key'] == thresh_key)]

        sub = sub[['Model','Energy Channel Key', 'Threshold Key',
            'Mismatch Allowed',
            'Prediction Energy Channel Key', 'Prediction Threshold Key',
            'Forecast Source',
            'Prediction Window Start', 'Prediction Window End',
            'Observed Max Flux in Prediction Window',
            'Observed Max Flux in Prediction Window Time',
            'Observed Max Flux in Prediction Window Units',
            'Predicted SEP Peak Intensity (Onset Peak)',
            'Predicted SEP Peak Intensity (Onset Peak) Units']]
        sub_test = sub['Predicted SEP Peak Intensity (Onset Peak)'].dropna()
        #drop rows containing None
        if sub_test.empty:
            return
 
        peak_key = 'Predicted SEP Peak Intensity (Onset Peak)'

        print("max_flux_in_pred_win: Model " + model + " did not explicitly "
            "include a peak_intensity_max field. Comparing peak_intensity to "
            "observed max flux in the prediction window.")



    mismatch = bool(sub_test.iloc[0]['Mismatch Allowed'])
    pred_energy_key = str(sub_test.iloc[0]['Prediction Energy Channel Key'])
    pred_thresh_key = str(sub_test.iloc[0]['Prediction Threshold Key'])

    thresh_fnm = make_thresh_fname(thresh_key)
    fnm = "max_flux_in_pred_win_selections_" + model + "_" + energy_key.strip() + "_" + thresh_fnm
    if mismatch:
        fnm = fnm + "_mm"
    write_df(sub_test, fnm)

    obs = sub_test['Observed Max Flux in Prediction Window']
    pred = sub_test[peak_key]
 
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
        "Max Flux in Prediction Window Correlation", xlabel="Observations",
        ylabel=("Model Predictions (" + str(units) + ")"),
        value="Max Flux in Prediction Window", use_log = True)

        figname = config.outpath + '/plots/Correlation_max_flux_in_pred_win_' \
                + model + "_" + energy_key.strip() + "_" + thresh_fnm
        if mismatch:
            figname = figname + "_mm"
        corr_plot.savefig(figname+ ".pdf", dpi=300, bbox_inches='tight')
        corr_plot.close()
    else:
        r_lin = None
        r_log = None
        s_lin = None
        s_log = None
        slope = None
        yint = None
        figname = ""


    ME, MedE, MAE, MedAE, MLE, MedLE, MALE, MedALE, MAPE, MAR, RMSE,\
        RMSLE, MdSA = calc_all_flux_metrics(obs, pred)

    ####METRICS
    fill_flux_metrics_dict(dict, model, energy_key, thresh_key,
    pred_energy_key, pred_thresh_key, figname,
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
    
    sub = sub[['Model','Energy Channel Key', 'Threshold Key',
            'Mismatch Allowed',
            'Prediction Energy Channel Key', 'Prediction Threshold Key',
            'Forecast Source',
            'Prediction Window Start', 'Prediction Window End',
            'Observed SEP Threshold Crossing Time',
            'Observed SEP Fluence',
            'Observed SEP Fluence Units',
            'Predicted SEP Fluence',
            'Predicted SEP Fluence Units',
            'Fluence Match Status']]
    sub = sub.loc[(sub['Fluence Match Status'] == 'SEP Event')]

    #Find predicted None values
    noneval = pd.isna(sub['Predicted SEP Fluence'])
    #Extract only indices for Nones
    #True indicates that peak intensity was a None value
    noneval = noneval.loc[noneval == True]
    noneval = noneval.index.to_list()
    if len(noneval) > 0:
        for ix in noneval:
            sub = sub.drop(index=ix)
    if sub.empty:
        return

    mismatch = bool(sub.iloc[0]['Mismatch Allowed'])
    pred_energy_key = str(sub.iloc[0]['Prediction Energy Channel Key'])
    pred_thresh_key = str(sub.iloc[0]['Prediction Threshold Key'])

    thresh_fnm = make_thresh_fname(thresh_key)
    fnm = "fluence_selections_" + model + "_" + energy_key.strip() + "_" \
            + thresh_fnm
    if mismatch:
        fnm = fnm + "_mm"
    write_df(sub, fnm)

    obs = sub['Observed SEP Fluence'].to_list()
    pred = sub['Predicted SEP Fluence'].to_list()
    units = sub.iloc[0]['Observed SEP Fluence Units']

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

        figname = config.outpath + '/plots/Correlation_fluence_' + model + "_" \
                + energy_key.strip() + "_" + thresh_fnm
        if mismatch:
            figname = figname + "_mm"
        corr_plot.savefig(figname+ ".pdf", dpi=300, bbox_inches='tight')
        corr_plot.close()
    else:
        r_lin = None
        r_log = None
        s_lin = None
        s_log = None
        slope = None
        yint = None
        figname = ""


    ME, MedE, MAE, MedAE, MLE, MedLE, MALE, MedALE, MAPE, MAR, RMSE,\
        RMSLE, MdSA = calc_all_flux_metrics(obs, pred)

    ####METRICS
    fill_flux_metrics_dict(dict, model, energy_key, thresh_key,
    pred_energy_key, pred_thresh_key, figname,
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

    sub = sub[['Model','Energy Channel Key', 'Threshold Key',
            'Mismatch Allowed',
            'Prediction Energy Channel Key', 'Prediction Threshold Key',
            'Forecast Source',
            'Prediction Window Start', 'Prediction Window End',
            'Observed SEP Threshold Crossing Time',
            'Predicted SEP Threshold Crossing Time',
            'Threshold Crossing Time Match Status']]
    sub = sub.loc[(sub['Threshold Crossing Time Match Status'] == 'SEP Event')]
    sub = sub.dropna() #drop rows containing None
      
    if sub.empty:
        return

    mismatch = bool(sub.iloc[0]['Mismatch Allowed'])
    pred_energy_key = str(sub.iloc[0]['Prediction Energy Channel Key'])
    pred_thresh_key = str(sub.iloc[0]['Prediction Threshold Key'])

    thresh_fnm = make_thresh_fname(thresh_key)
    fnm = "threshold_crossing_time_selections_" + model + "_" \
            + energy_key.strip() + "_" + thresh_fnm
    if mismatch:
        fnm = fnm + "_mm"
    write_df(sub, fnm)

    obs = sub['Observed SEP Threshold Crossing Time'].to_list()
    pred = sub['Predicted SEP Threshold Crossing Time'].to_list()
    td = (sub['Predicted SEP Threshold Crossing Time'] - sub['Observed SEP Threshold Crossing Time'])

    td = td.dt.total_seconds()/(60*60) #convert to hours
    td = td.to_list()
    abs_td = [abs(x) for x in td]

    ME = statistics.mean(td)
    MedE = statistics.median(td)
    MAE = statistics.mean(abs_td)
    MedAE = statistics.median(abs_td)
    
    fill_time_metrics_dict(dict, model, energy_key, thresh_key,
    pred_energy_key, pred_thresh_key, ME, MedE, MAE, MedAE)
    

def start_time_intuitive_metrics(df, dict, model, energy_key, thresh_key):
    """ Extract the appropriate predictions and calculate metrics
        Start Time

    """
    #Select rows to calculate metrics
    sub = df.loc[(df['Model'] == model) & (df['Energy Channel Key'] ==
        energy_key) & (df['Threshold Key'] == thresh_key)]

    sub = sub[['Model','Energy Channel Key', 'Threshold Key',
            'Mismatch Allowed',
            'Prediction Energy Channel Key', 'Prediction Threshold Key',
            'Forecast Source',
            'Prediction Window Start', 'Prediction Window End',
            'Observed SEP Start Time',
            'Predicted SEP Start Time',
            'Start Time Match Status']]
    sub = sub.loc[(sub['Start Time Match Status'] == 'SEP Event')]
    sub = sub.dropna() #drop rows containing None
      
    if sub.empty:
        return

    mismatch = bool(sub.iloc[0]['Mismatch Allowed'])
    pred_energy_key = str(sub.iloc[0]['Prediction Energy Channel Key'])
    pred_thresh_key = str(sub.iloc[0]['Prediction Threshold Key'])

    thresh_fnm = make_thresh_fname(thresh_key)
    fnm = "start_time_selections_" + model + "_" + energy_key.strip() \
            + "_" + thresh_fnm
    if mismatch:
        fnm = fnm + "_mm"
    write_df(sub, fnm)

    obs = sub['Observed SEP Start Time'].to_list()
    pred = sub['Predicted SEP Start Time'].to_list()
    td = (sub['Predicted SEP Start Time'] - sub['Observed SEP Start Time'])
    
    td = td.dt.total_seconds()/(60*60) #convert to hours
    td = td.to_list()
    abs_td = [abs(x) for x in td]

    ME = statistics.mean(td)
    MedE = statistics.median(td)
    MAE = statistics.mean(abs_td)
    MedAE = statistics.median(abs_td)
    
    fill_time_metrics_dict(dict, model, energy_key, thresh_key,
    pred_energy_key, pred_thresh_key, ME, MedE, MAE, MedAE)


def end_time_intuitive_metrics(df, dict, model, energy_key,
    thresh_key):
    """ Extract the appropriate predictions and calculate metrics
        End Time

    """
    #Select rows to calculate metrics
    sub = df.loc[(df['Model'] == model) & (df['Energy Channel Key'] ==
        energy_key) & (df['Threshold Key'] == thresh_key)]

    sub = sub[['Model','Energy Channel Key', 'Threshold Key',
            'Mismatch Allowed',
            'Prediction Energy Channel Key', 'Prediction Threshold Key',
            'Forecast Source',
            'Prediction Window Start', 'Prediction Window End',
            'Observed SEP End Time',
            'Predicted SEP End Time',
            'End Time Match Status']]
    sub = sub.loc[(sub['End Time Match Status'] == 'SEP Event')]
    sub = sub.dropna() #drop rows containing None
      
    if sub.empty:
        return

    mismatch = bool(sub.iloc[0]['Mismatch Allowed'])
    pred_energy_key = str(sub.iloc[0]['Prediction Energy Channel Key'])
    pred_thresh_key = str(sub.iloc[0]['Prediction Threshold Key'])

    thresh_fnm = make_thresh_fname(thresh_key)
    fnm = "end_time_selections_" + model + "_" + energy_key.strip() + "_" + thresh_fnm
    if mismatch:
        fnm = fnm + "_mm"
    write_df(sub, fnm)

    obs = sub['Observed SEP End Time'].to_list()
    pred = sub['Predicted SEP End Time'].to_list()
    td = (sub['Predicted SEP End Time'] - sub['Observed SEP End Time'])
    
    td = td.dt.total_seconds()/(60*60) #convert to hours
    td = td.to_list()
    abs_td = [abs(x) for x in td]

    ME = statistics.mean(td)
    MedE = statistics.median(td)
    MAE = statistics.mean(abs_td)
    MedAE = statistics.median(abs_td)
    
    fill_time_metrics_dict(dict, model, energy_key, thresh_key,
    pred_energy_key, pred_thresh_key, ME, MedE, MAE, MedAE)
 


def duration_intuitive_metrics(df, dict, model, energy_key, thresh_key):
    """ Extract the appropriate predictions and calculate metrics
        Start Time

    """
    #Select rows to calculate metrics
    sub = df.loc[(df['Model'] == model) & (df['Energy Channel Key'] ==
        energy_key) & (df['Threshold Key'] == thresh_key)]

    sub = sub[['Model','Energy Channel Key', 'Threshold Key',
            'Mismatch Allowed',
            'Prediction Energy Channel Key', 'Prediction Threshold Key',
            'Forecast Source',
            'Prediction Window Start', 'Prediction Window End',
            'Observed SEP Duration',
            'Predicted SEP Duration',
            'Duration Match Status']]
    sub = sub.loc[(sub['Duration Match Status'] == 'SEP Event')]
    sub = sub.dropna() #drop rows containing None
      
    if sub.empty:
        return

    mismatch = bool(sub.iloc[0]['Mismatch Allowed'])
    pred_energy_key = str(sub.iloc[0]['Prediction Energy Channel Key'])
    pred_thresh_key = str(sub.iloc[0]['Prediction Threshold Key'])

    thresh_fnm = make_thresh_fname(thresh_key)
    fnm = "duration_selections_" + model + "_" + energy_key.strip() \
            + "_" + thresh_fnm
    if mismatch:
        fnm = fnm + "_mm"
    write_df(sub, fnm)

    obs = sub['Observed SEP Duration']
    pred = sub['Predicted SEP Duration']

    td = pred - obs #shorter duration is negative
    td = td.to_list()
    abs_td = [abs(x) for x in td]

    ME = statistics.mean(td)
    MedE = statistics.median(td)
    MAE = statistics.mean(abs_td)
    MedAE = statistics.median(abs_td)
    
    fill_time_metrics_dict(dict, model, energy_key, thresh_key,
    pred_energy_key, pred_thresh_key, ME, MedE, MAE, MedAE)




def peak_intensity_time_intuitive_metrics(df, dict, model, energy_key,
    thresh_key):
    """ Extract the appropriate predictions and calculate metrics
        Peak Intensity Time

    """
    #Select rows to calculate metrics
    sub = df.loc[(df['Model'] == model) & (df['Energy Channel Key'] ==
        energy_key) & (df['Threshold Key'] == thresh_key)]

    sub = sub[['Model','Energy Channel Key', 'Threshold Key',
            'Mismatch Allowed',
            'Prediction Energy Channel Key', 'Prediction Threshold Key',
            'Forecast Source',
            'Prediction Window Start', 'Prediction Window End',
            'Observed SEP Threshold Crossing Time',
            'Observed SEP Peak Intensity (Onset Peak) Time',
            'Predicted SEP Peak Intensity (Onset Peak) Time',
            'Peak Intensity Match Status']]
    sub = sub.loc[(sub['Peak Intensity Match Status'] == 'SEP Event')]

    #Find predicted None values
    noneval = pd.isna(sub['Predicted SEP Peak Intensity (Onset Peak) Time'])
    #Extract only indices for Nones
    #True indicates that peak intensity was a None value
    noneval = noneval.loc[noneval == True]
    noneval = noneval.index.to_list()
    if len(noneval) > 0:
        for ix in noneval:
            sub = sub.drop(index=ix)
    if sub.empty:
        return

    mismatch = bool(sub.iloc[0]['Mismatch Allowed'])
    pred_energy_key = str(sub.iloc[0]['Prediction Energy Channel Key'])
    pred_thresh_key = str(sub.iloc[0]['Prediction Threshold Key'])

    thresh_fnm = make_thresh_fname(thresh_key)
    fnm = "peak_intensity_time_selections_" + model + "_" \
            + energy_key.strip() + "_" + thresh_fnm
    if mismatch:
        fnm = fnm + "_mm"
    write_df(sub, fnm)

    obs = sub['Observed SEP Peak Intensity (Onset Peak) Time'].to_list()
    pred = sub['Predicted SEP Peak Intensity (Onset Peak) Time'].to_list()
    td = (sub['Predicted SEP Peak Intensity (Onset Peak) Time'] - sub['Observed SEP Peak Intensity (Onset Peak) Time'])
    
    td = td.dt.total_seconds()/(60*60) #convert to hours
    td = td.to_list()
    abs_td = [abs(x) for x in td]

    ME = statistics.mean(td)
    MedE = statistics.median(td)
    MAE = statistics.mean(abs_td)
    MedAE = statistics.median(abs_td)
    
    fill_time_metrics_dict(dict, model, energy_key, thresh_key,
    pred_energy_key, pred_thresh_key, ME, MedE, MAE, MedAE)


def date_to_string(date):
    """ Turn datetime into appropriate strings for filenames.
    
    """
    year = date.year
    month = date.month
    day = date.day
    hour = date.hour
    min = date.minute
    sec = date.second
    
    date_str = '{:d}{:02d}{:02d}_{:02d}hr{:02d}min{:02d}sec'.format(year, month, day, hour, min, sec)
    
    return date_str



def peak_intensity_max_time_intuitive_metrics(df, dict, model, energy_key,
    thresh_key):
    """ Extract the appropriate predictions and calculate metrics
        Peak Intensity Max Time

    """
    #Select rows to calculate metrics
    sub = df.loc[(df['Model'] == model) & (df['Energy Channel Key'] ==
        energy_key) & (df['Threshold Key'] == thresh_key)]

    sub = sub[['Model','Energy Channel Key', 'Threshold Key',
            'Mismatch Allowed',
            'Prediction Energy Channel Key', 'Prediction Threshold Key',
            'Forecast Source',
            'Prediction Window Start', 'Prediction Window End',
            'Observed SEP Threshold Crossing Time',
            'Observed SEP Peak Intensity Max (Max Flux) Time',
            'Predicted SEP Peak Intensity Max (Max Flux) Time',
            'Peak Intensity Max Match Status']]
    sub = sub.loc[(sub['Peak Intensity Max Match Status'] == 'SEP Event')]
    #Find predicted None values
    noneval = pd.isna(sub['Predicted SEP Peak Intensity Max (Max Flux) Time'])
    #Extract only indices for Nones
    #True indicates that peak intensity was a None value
    noneval = noneval.loc[noneval == True]
    noneval = noneval.index.to_list()
    if len(noneval) > 0:
        for ix in noneval:
            sub = sub.drop(index=ix)
    if sub.empty:
        return

    mismatch = bool(sub.iloc[0]['Mismatch Allowed'])
    pred_energy_key = str(sub.iloc[0]['Prediction Energy Channel Key'])
    pred_thresh_key = str(sub.iloc[0]['Prediction Threshold Key'])

    thresh_fnm = make_thresh_fname(thresh_key)
    fnm = "peak_intensity_max_time_selections_" + model + "_" \
            + energy_key.strip() + "_" + thresh_fnm
    if mismatch:
        fnm = fnm + "_mm"
    write_df(sub, fnm)

    obs = sub['Observed SEP Peak Intensity Max (Max Flux) Time'].to_list()
    pred = sub['Predicted SEP Peak Intensity Max (Max Flux) Time'].to_list()
    td = (sub['Predicted SEP Peak Intensity Max (Max Flux) Time'] - sub['Observed SEP Peak Intensity Max (Max Flux) Time'])
    
    td = td.dt.total_seconds()/(60*60) #convert to hours
    td = td.to_list()
    abs_td = [abs(x) for x in td]

    ME = statistics.mean(td)
    MedE = statistics.median(td)
    MAE = statistics.mean(abs_td)
    MedAE = statistics.median(abs_td)
    
    fill_time_metrics_dict(dict, model, energy_key, thresh_key,
    pred_energy_key, pred_thresh_key, ME, MedE, MAE, MedAE)


def time_profile_intuitive_metrics(df, dict, model, energy_key,
    thresh_key):
    """ Extract the appropriate predictions and calculate metrics
        Time Profile

    """
    #Select rows to calculate metrics
    sub = df.loc[(df['Model'] == model) & (df['Energy Channel Key'] ==
        energy_key) & (df['Threshold Key'] == thresh_key)]

    sub = sub[['Model','Energy Channel Key', 'Threshold Key',
            'Mismatch Allowed',
            'Prediction Energy Channel Key', 'Prediction Threshold Key',
            'Forecast Source',
            'Forecast Path',
            'Prediction Window Start', 'Prediction Window End',
            'Observed SEP Start Time',
            'Observed SEP End Time',
            'Predicted SEP Start Time',
            'Predicted SEP End Time',
            'Observed Time Profile',
            'Predicted Time Profile',
            'Time Profile Match Status']]
    sub = sub.loc[(sub['Time Profile Match Status'] == 'SEP Event')]

    #Find predicted None values
    noneval = pd.isna(sub['Predicted Time Profile'])
    #Extract only indices for Nones
    #True indicates that peak intensity was a None value
    noneval = noneval.loc[noneval == True]
    noneval = noneval.index.to_list()
    if len(noneval) > 0:
        for ix in noneval:
            sub = sub.drop(index=ix)
    if sub.empty:
        return
        
    
    mismatch = bool(sub.iloc[0]['Mismatch Allowed'])
    pred_energy_key = str(sub.iloc[0]['Prediction Energy Channel Key'])
    pred_thresh_key = str(sub.iloc[0]['Prediction Threshold Key'])

    thresh_fnm = make_thresh_fname(thresh_key)
    fnm = "time_profile_selections_" + model + "_" + energy_key.strip() \
            + "_" + thresh_fnm
    if mismatch:
        fnm = fnm + "_mm"
    write_df(sub, fnm)

    obs_st = sub['Observed SEP Start Time'].to_list()
    obs_et = sub['Observed SEP End Time'].to_list()
    obs_profs = sub['Observed Time Profile'].to_list()
    pred_profs = sub['Predicted Time Profile'].to_list()
    pred_paths = sub['Forecast Path'].to_list()
    model_names = sub['Model'].to_list()
    energy_chan = sub['Energy Channel Key'].to_list()
    
    sepE = []
    sepAE = []
    sepLE = []
    sepALE = []
    sepAPE = []
#    sepMAR = None #Mean Accuracy Ratio
    sepRMSE = []
    sepRMSLE = []
#    sepMdSA = None

    tp_plotnames = ""
    for i in range(len(obs_profs)):
        print("Time profile of " + pred_profs[i] + " compared to observations.")
        all_obs_dates = []
        all_obs_flux = []
        #Read in and combine time profiles of observations inside
        #prediction window
        print("======NAMES OF OBSERVED TIME PROFILE++++")
        obs_fnames = obs_profs[i].strip().split(",")
        for j in range(len(obs_fnames)):
            dt, flx = profile.read_single_time_profile(obs_fnames[j])
            all_obs_dates.append(dt)
            all_obs_flux.append(flx)
        
        obs_dates, obs_flux = profile.combine_time_profiles(all_obs_dates,
            all_obs_flux)
        pred_dates, pred_flux = profile.read_single_time_profile(pred_paths[i]
            + pred_profs[i])
        if pred_flux == []:
            return
        
        #Remove zeros
        obs_flux, obs_dates = zip(*filter(lambda x:x[0]>0.0, zip(obs_flux,
            obs_dates)))
        pred_flux, pred_dates = zip(*filter(lambda x:x[0]>0.0, zip(pred_flux, pred_dates)))
        
        #If predicted time profile is all zeros
        if pred_flux == []:
            continue
        
        #Interpolate observed time profile onto predicted timestamps
        obs_flux_interp = profile.interp_timeseries(obs_dates, obs_flux, "log",
            pred_dates)
        
        #Trim the time profiles to the observed start and end time
        trim_pred_dates, trim_pred_flux = profile.trim_profile(obs_st[i],
                obs_et[i], pred_dates, pred_flux)
        trim_obs_dates, trim_obs_flux = profile.trim_profile(obs_st[i],
                obs_et[i], pred_dates,  obs_flux_interp)
        
        
        #PLOT TIME PROFILE TO CHECK
        date = [obs_dates, trim_obs_dates, pred_dates, trim_pred_dates]
        values = [obs_flux, trim_obs_flux, pred_flux, trim_pred_flux]
        labels=["Observations", "Interp Trimmed Obs", "Model", "Trimmed Model"]
        str_date = date_to_string(pred_dates[0])
        title = model_names[i] + ", " + energy_chan[i] + " Time Profile"
        tpfigname = config.outpath + "/plots/Time_Profile_" + model_names[i] \
            + "_" + energy_chan[i] + "_" + thresh_fnm  + "_" + str_date
        if mismatch:
            tpfigame = tpfigname + "_mm"
        
        if tp_plotnames == "":
            tp_plotnames = tpfigname + ".pdf"
        else:
            tp_plotnames += ";" + tpfigname + ".pdf"
            
            
        plt_tools.plot_time_profile(date, values, labels,
        title=title, x_label="Date", y_min=1e-5, y_max=1e4,
        y_label="Particle Intensity", uselog_x = False, uselog_y = True,
        date_format="year", showplot=False,
        closeplot=True, saveplot=True, figname = tpfigname + ".pdf")
        
        #Check for None and Zero values and remove
        if trim_pred_flux == [] or trim_obs_flux == []: continue
        obs, pred = profile.remove_none(trim_obs_flux,trim_pred_flux)
        obs, pred = profile.remove_zero(obs, pred)
        if obs == [] or pred == []: continue
        
        #Calculate a mean metric across an individual time profile
        if len(obs) >= 1 and len(pred) >= 1:
            E1 = statistics.mean(metrics.switch_error_func('E',obs,pred))
            AE1 = statistics.mean(metrics.switch_error_func('AE',obs,pred))
            LE1 = statistics.mean(metrics.switch_error_func('LE',obs,pred))
            ALE1 = statistics.mean(metrics.switch_error_func('ALE',obs,pred))
            APE1 = statistics.mean(metrics.switch_error_func('APE',obs,pred))
    #        MAR1 = None #Mean Accuracy Ratio
            RMSE1 = metrics.switch_error_func('RMSE',obs,pred)
            RMSLE1 = metrics.switch_error_func('RMSLE',obs,pred)
    #        MdSA1 = None

            sepE.append(E1)
            sepAE.append(AE1)
            sepLE.append(LE1)
            sepALE.append(ALE1)
            sepAPE.append(APE1)
        #    sepMAR = None #Mean Accuracy Ratio
            sepRMSE.append(RMSE1)
            sepRMSLE.append(RMSLE1)
        #    sepMdSA = None

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
                "Time Profile Correlation", xlabel="Observations",
                ylabel=("Model Predictions"),
                use_log = True)

                figname = config.outpath + '/plots/Correlation_time_profile_'\
                    + model + "_" + energy_key.strip() + "_" + thresh_fnm \
                    + "_" + str_date
                if mismatch:
                    figname = figname + "_mm"
                corr_plot.savefig(figname + ".pdf", dpi=300, bbox_inches='tight')
                corr_plot.close()

    #Calculate mean of metrics for all time profiles
    if len(sepE) > 1:
        ME = statistics.mean(sepE)
        MedE = statistics.median(sepE)
        MAE = statistics.mean(sepAE)
        MedAE = statistics.median(sepAE)
        MLE = statistics.mean(sepLE)
        MedLE = statistics.median(sepLE)
        MALE = statistics.mean(sepALE)
        MedALE = statistics.median(sepALE)
        MAPE = statistics.mean(sepAPE)
        MAR = None #Mean Accuracy Ratio
        RMSE = statistics.mean(sepRMSE)
        RMSLE = statistics.mean(sepRMSLE)
        MdSA = None
        
    elif len(sepE) == 1:
        ME = sepE[0]
        MedE = sepE[0]
        MAE = sepAE[0]
        MedAE = sepAE[0]
        MLE = sepLE[0]
        MedLE = sepLE[0]
        MALE = sepALE[0]
        MedALE = sepALE[0]
        MAPE = sepAPE[0]
        MAR = None #Mean Accuracy Ratio
        RMSE = sepRMSE[0]
        RMSLE = sepRMSLE[0]
        MdSA = None
    else:
        ME = None
        MedE = None
        MAE = None
        MedAE = None
        MLE = None
        MedLE = None
        MALE = None
        MedALE = None
        MAPE = None
        MAR = None #Mean Accuracy Ratio
        RMSE = None
        RMSLE = None
        MdSA = None

    r_lin = None
    r_log = None
    s_lin = None
    s_log = None
    slope = None
    yint = None

    ####METRICS
    fill_flux_metrics_dict(dict, model, energy_key, thresh_key,
    pred_energy_key, pred_thresh_key, figname,
    slope, yint, r_lin, r_log, s_lin, s_log, ME, MedE, MLE, MedLE, MAE,
    MedAE, MALE, MedALE, MAPE, MAR, RMSE, RMSLE, MdSA, tp_plotnames)



def identify_first_all_clear_forecast(df):
    """ Finds the first forecast associated with an SEP event.
        In the case of consecutive forecasts leading up to an SEP event,
        the first forecast will be selected for a series of forecasts that ALL
        predict an SEP event will occur.
        
        INPUT:
        
            :df: (pandas dataframe) forecasts for a single SEP event for a single
                model, energy channel, and threshold

        OUTPUT:
        
            :idx: index indicating row of df associated with first forecast
        
    """
    all_clear = df['Predicted SEP All Clear'].to_list()
    ac_idx = None
    #Search in reverse order checking of forecast is False All Clear
    #As soon as hit a True All Clear, exit
    for i in range(len(all_clear)-1,-1,-1):
        if all_clear[i] == False:
            ac_idx = i
        if all_clear[i] == True:
            break
    
    return ac_idx


def identify_first_time_forecast(df, pred_key):
    """ Finds the first forecast associated with an SEP event.
        In the case of consecutive forecasts leading up to an SEP event,
        the first forecast will be selected for a series of forecasts that ALL
        predict an SEP event will occur.
        
        INPUT:
        
            :df: (pandas dataframe) forecasts for a single SEP event for a single
                model, energy channel, and threshold
            :pred_key: (string) indicates predicted time, e.g.
                'Predicted SEP Threshold Crossing Time'
                'Predicted SEP Start Time'
                'Predicted SEP End Time'

        OUTPUT:
        
            :idx: index indicating row of df associated with first forecast
        
    """
    times = df[pred_key].to_list()
    idx = None
    for i in range(len(times)-1,-1,-1):
        if times[i] != pd.NaT:
            idx = i
        else:
            break

    return idx


def identify_first_flux_forecast(df, pred_key, thresh_key):
    """ Finds the first forecast associated with an SEP event.
        In the case of consecutive forecasts leading up to an SEP event,
        the first forecast will be selected for a series of forecasts that ALL
        predict an SEP event will occur.
        
        INPUT:
        
            :df: (pandas dataframe) forecasts for a single SEP event for a single
                model, energy channel, and threshold
            :pred_key: (string) indicates predicted time, e.g.
                'Predicted SEP Peak Intensity (Onset Peak)'
                'Predicted SEP Peak Intensity Max (Max Flux)'
            :thresh_key: (string) threshold applied to flux to define SEP event

        OUTPUT:
        
            :idx: index indicating row of df associated with first forecast
        
    """
    threshold = objh.key_to_threshold(thresh_key)
    thresh = threshold['threshold']
    
    fluxes = df[pred_key].to_list()
    idx = None
    for i in range(len(fluxes)-1,-1,-1):
        if fluxes[i] >= thresh:
            idx = i
        else:
            break

    return idx




def extract_awt_sub(df, model, energy_key, thresh_key, pred_key, match_key, obs_key=''):
    """ Extracts the forecast to be used to calculate AWT. Compares issue
        time to Observed SEP Threshold Crossing Time, Observed SEP Start Time.
        
        INPUT:
        
            :df: (pandas DataFrame) dataframe containing observed and
                forecasted values
            :pred_key: (string) indicating which forecasted value to use to identify
                forecasts: 'Predicted SEP All Clear', 'Predicted SEP Start Time', etc
            :match_key: (string) match status identifier associated with pred_ref
                'All Clear Match Status', 'Start Time Match Status', etc
            :obs_key: (string, optional) Additional reference time to calculate AWT
                'Observed Peak Intensity (Onset Peak) Time'
                
        OUTPUT:
            
            :sub: (pandas DataFrame) dataframe containing

            
    """
    sub = df.loc[(df['Model'] == model) & (df['Energy Channel Key'] == energy_key)
                & (df['Threshold Key'] == thresh_key)]

    if obs_key != '':
        sub = sub[['Model','Energy Channel Key', 'Threshold Key',
                'Mismatch Allowed',
                'Prediction Energy Channel Key', 'Prediction Threshold Key',
                'Forecast Source', 'Forecast Issue Time',
                'Prediction Window Start', 'Prediction Window End',
                'Observed SEP Threshold Crossing Time',
                'Observed SEP Start Time',
                obs_key, pred_key, match_key]]
    else:
        sub = sub[['Model','Energy Channel Key', 'Threshold Key',
                'Mismatch Allowed',
                'Prediction Energy Channel Key', 'Prediction Threshold Key',
                'Forecast Source', 'Forecast Issue Time',
                'Prediction Window Start', 'Prediction Window End',
                'Observed SEP Threshold Crossing Time',
                'Observed SEP Start Time',
                pred_key, match_key]]

    sub = sub.loc[sub[match_key] == "SEP Event"]

    return sub


def awt_metrics(df, dict, model, energy_key, thresh_key):
    """ Metrics for Advanced Warning Time.
        Find the first forecast ahead of SEP events and calculate AWT
        for a given model, energy channel, and threshold.
        
        AWT is calculated by:
        Observed Time - Issue Time
        
        Positive AWT indicates the issue time was BEFORE the observed time.
        Negative AWT indicates that the issue time was AFTER the observed time.
    
        AWT with respect to following times depending on predicted quantity:
            'Observed SEP Threshold Crossing Time'
            'Observed SEP Start Time' (same as above)
            'Observed SEP Peak Intensity (Onset Peak) Time'
            'Observed SEP Peak Intensity Max (Max Flux) Time'
            'Observed SEP End Time'
    
        Any type of forecast can be used to derive AWT except fluence only
        (because no threshold to associate with) and probability only (because
        no probability threshold to determine if probability considered an
        event or not - may add user input probability threshold in the future).
        As long as a quantity was forecasted indicating that there would be an
        SEP event, that forecast qualifies to be used to calculate AWT.
        
        For models that forecast multiple times leading up to an SEP event,
        the AWT will be calculated using the first forecasting in a CONTINUOUS
        series of forecasts indicating that an SEP event will occur.
    
        The AWT will be calculated from forecasts for the various quantities:
            All Clear
            Threshold Crossing Time
            Start Time
            Peak Intensity (Onset Peak)
            Peak Intensity Max (Max Flux)
            End Time
    
    """

    sub = df.loc[(df['Model'] == model) & (df['Energy Channel Key'] == energy_key)
                & (df['Threshold Key'] == thresh_key)]
    if sub.empty:
        return

    #Fill in dictionary
    dict['Model'].append(model)
    dict['Energy Channel'].append(energy_key)
    dict['Threshold'].append(thresh_key)
    dict['Prediction Energy Channel'].append(sub.iloc[0]['Prediction Energy Channel Key'])
    dict['Prediction Threshold'].append(sub.iloc[0]['Prediction Threshold Key'])


    #AWT is always compared to Observed SEP Threshold Crossing Time and
    #Observed SEP Start Time. obs_ref allows for the calculation of AWT wrt
    #another observation time.
    forecasts = [{'pred_key': 'Predicted SEP All Clear',
                    'match_key': 'All Clear Match Status',
                    'obs_key': ''},
                 {'pred_key': 'Predicted SEP Threshold Crossing Time',
                    'match_key': 'Threshold Crossing Time Match Status',
                    'obs_key': ''},
                 {'pred_key': 'Predicted SEP Start Time',
                    'match_key': 'Start Time Match Status',
                    'obs_key': ''},
                 {'pred_key': 'Predicted SEP Peak Intensity (Onset Peak)',
                    'match_key': 'Peak Intensity Match Status',
                    'obs_key': 'Observed SEP Peak Intensity (Onset Peak) Time'},
                 {'pred_key': 'Predicted SEP Peak Intensity Max (Max Flux)',
                    'match_key': 'Peak Intensity Max Match Status',
                    'obs_key': 'Observed SEP Peak Intensity Max (Max Flux) Time'},
                {'pred_key': 'Predicted Point Intensity',
                    'match_key': 'All Clear Match Status',
                    'obs_key': ''},
                 {'pred_key': 'Predicted SEP End Time',
                    'match_key': 'End Time Match Status',
                    'obs_key': 'Observed SEP End Time'}]

    #Extract values in df relevant to each forecast
    for ftype in forecasts:
        #Extract relevant fields only for forecasts associated with SEP Events
        sub = extract_awt_sub(df, model, energy_key, thresh_key, ftype['pred_key'],
            ftype['match_key'], ftype['obs_key'])
       
        #Create an empty dataframe with the same columns plus AWT info
        cols = sub.columns.to_list()
        cols.append('AWT to Observed SEP Threshold Crossing Time')
        cols.append('AWT to Observed SEP Start Time')
        if ftype['obs_key'] != '':
            cols.append('AWT to ' + ftype['obs_key'])
        sel_df = pd.DataFrame(columns=cols) #Selected forecasts and AWT results
       
       
        #Make a list of the unique SEP events in the df
        sep_events = resume.identify_unique(sub, 'Observed SEP Threshold Crossing Time')
        
        #For each SEP event, identify the first forecast for that SEP event.
        for sep in sep_events:
            sep_sub = sub.loc[sub['Observed SEP Threshold Crossing Time'] == sep]
            
            idx = None
            if 'All Clear' in ftype['pred_key']:
                idx = identify_first_all_clear_forecast(sep_sub)
        
            if 'Time' in ftype['pred_key']:
                idx = identify_first_time_forecast(sep_sub, ftype['pred_key'])
                
            if 'Peak' in ftype['pred_key'] or 'Point' in ftype['pred_key']:
                idx = identify_first_flux_forecast(sep_sub, ftype['pred_key'],
                        thresh_key)
            if idx == None:
                continue
            
            row = sep_sub.iloc[idx].to_list()
            
            #Calculate AWT
            issue_time = sep_sub.iloc[idx]['Forecast Issue Time']
            if issue_time == pd.NaT or issue_time == None:
                continue
            
            tct = sep_sub.iloc[idx]['Observed SEP Threshold Crossing Time']
            tc_awt = (tct - issue_time).total_seconds()/(60.*60.) #hours
            #If issue time is more than 7 days later than the threshold crossing time,
            #the assume this is not a realistic issue time and ignore
            if tc_awt < -7.*24.:
                tc_awt = None
            row.append(tc_awt)
            
            st = sep_sub.iloc[idx]['Observed SEP Start Time']
            st_awt = (st - issue_time).total_seconds()/(60.*60.)
            #If issue time is more than 7 days later than the start time,
            #the assume this is not a realistic issue time and ignore
            if st_awt < -7.*24.:
                st_awt = None
            row.append(st_awt)

            if ftype['obs_key'] != '':
                tm = sep_sub.iloc[idx][ftype['obs_key']]
                obs_awt = (tm - issue_time).total_seconds()/(60.*60.)
                #If issue time is more than 7 days later than the observed time,
                #the assume this is not a realistic issue time and ignore
                if obs_awt < -7.*24.:
                    obs_awt = None
                row.append(obs_awt)

            #Insert value into dataframe to save AWT calculations for each SEP
            sel_df.loc[len(sel_df)] = row

        if sel_df.empty:
            continue

        #Have AWT values for all SEPs for a given model, energy channel, and
        #threshold
        #Write to file
        thresh_fnm = make_thresh_fname(thresh_key)
        fnm = "awt_selections_" + model + "_" + energy_key.strip() + "_" +\
                thresh_fnm + "_" + ftype['pred_key']
        if bool(sel_df.iloc[0]['Mismatch Allowed']):
            fnm = fnm + "_mm"
        write_df(sel_df, fnm)


        #Calculate metrics for AWT to different times
        time_keys = ['Observed SEP Threshold Crossing Time','Observed SEP Start Time']
        if ftype['obs_key'] != '':
            time_keys.append(ftype['obs_key'])
        
        for key in time_keys:
            awts = sel_df["AWT to " + key].to_list()
            awts = [x for x in awts if x != None]
            if len(awts) >= 1:
                mean_awt = statistics.mean(awts)
                median_awt = statistics.median(awts)
            else:
                mean_awt = None
                median_awt = None
            
            mean_key = "Mean AWT for " + ftype['pred_key'] + " to " + key
            median_key = "Median AWT for " + ftype['pred_key'] + " to " + key
            dict[mean_key].append(mean_awt)
            dict[median_key].append(median_awt)

    return


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
    point_intensity_dict = initialize_flux_dict()
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
    max_dict = initialize_flux_dict() #max in prediction window
    awt_dict = initialize_awt_dict() #Advanced Warning Time
    
    for model in model_names:
        for ek in all_energy_channels:
            for tk in all_observed_thresholds[ek]:
                probability_intuitive_metrics(df, probability_dict,model,ek,tk)
                point_intensity_intuitive_metrics(df, point_intensity_dict, model, ek, tk)
                peak_intensity_intuitive_metrics(df, peak_intensity_dict,
                    model,ek,tk)
                peak_intensity_max_intuitive_metrics(df,
                    peak_intensity_max_dict,model,ek,tk)
                fluence_intuitive_metrics(df,fluence_dict, model,ek,tk)
                threshold_crossing_intuitive_metrics(df, thresh_cross_dict,
                    model,ek,tk)
                start_time_intuitive_metrics(df, start_time_dict,
                    model,ek,tk)
                end_time_intuitive_metrics(df, end_time_dict,model,ek,tk)
                duration_intuitive_metrics(df, duration_dict,model,ek,tk)
                peak_intensity_time_intuitive_metrics(df,
                    peak_intensity_time_dict,model,ek,tk)
                peak_intensity_max_time_intuitive_metrics(df,
                    peak_intensity_max_time_dict,model,ek,tk)
                all_clear_intuitive_metrics(df, all_clear_dict,model,ek,tk)
                time_profile_intuitive_metrics(df, profile_dict,model,ek,tk)
                max_flux_in_pred_win_metrics(df, max_dict, model,ek,tk)
                awt_metrics(df, awt_dict, model, ek, tk)


    print("calculate_intuitive_validation: Completed calculating all metrics: " + str(datetime.datetime.now()))

    prob_metrics_df = pd.DataFrame(probability_dict)
    point_intensity_metrics_df = pd.DataFrame(point_intensity_dict)
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
    time_profile_metrics_df = pd.DataFrame(profile_dict)
    max_metrics_df = pd.DataFrame(max_dict)

    #It is possible for awt_dict to have some empty fields. Remove them
    #for writing to file.
    awt_keys = list(awt_dict.keys())
    for jj in range(len(awt_keys)-1,-1,-1):
        if not awt_dict[awt_keys[jj]]:
            del awt_dict[awt_keys[jj]]
    awt_metrics_df = pd.DataFrame(awt_dict)

    if not prob_metrics_df.empty:
        write_df(prob_metrics_df, "probability_metrics")
    if not point_intensity_metrics_df.empty:
        write_df(point_intensity_metrics_df, "point_intensity_metrics")
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
    if not time_profile_metrics_df.empty:
        write_df(time_profile_metrics_df, "time_profile_metrics")
    if not max_metrics_df.empty:
        write_df(max_metrics_df, "max_flux_in_pred_win_metrics")
    if not awt_metrics_df.empty:
        write_df(awt_metrics_df, "awt_metrics")

    print("calculate_intuitive_validation: Wrote out all metrics to file: " + str(datetime.datetime.now()))




def intuitive_validation(matched_sphinx, model_names, all_energy_channels,
    all_observed_thresholds, observed_sep_events, DoResume=False, r_df=None):
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
        :DoResume: (bool) boolean to indicate whether the user wants to resume
            building on a previous run of SPHINX
        :r_df: (pandas dataframe) dataframe created from a previous run of
            SPHINX. Newly input predictions will be appended.
    
    Output:
    
    
    
    """
    print("intuitive_validation: Beginning validation process: " + str(datetime.datetime.now()))
    # Make sure the output directories exist
    prepare_outdirs()
    
    #For each model and predicted quantity, create arrays of paired up values
    #so can calculate metrics
    print("intuitive_validation: Filling dataframe with information from matched sphinx objects: " + str(datetime.datetime.now()))

    df = fill_df(matched_sphinx, model_names,
            all_energy_channels, all_observed_thresholds, DoResume)
 
    print("intuitive_validation: Completed filling dataframe (and possibly writing): " + str(datetime.datetime.now()))

    ###RESUME WILL APPEND DF TO PREVIOUS DF
    print("Resume boolean is " + str(DoResume))
    if DoResume:
        print("intuitive_validation: Resuming from a previous run. Concatenating new dataframe and previous dataframe: " + str(datetime.datetime.now()))
 
        df = pd.concat([r_df, df], ignore_index=True)

        print("intuitive_validation: Completed concatenation. Writing out to file: " + str(datetime.datetime.now()))

        write_df(df, "SPHINX_dataframe")

        print("intuitive_validation: Completed writing SPHINX_dataframe to file: " + str(datetime.datetime.now()))

        model_names = resume.identify_unique(df, 'Model')
        all_energy_channels = resume.identify_unique(df, 'Energy Channel Key')
        all_observed_thresholds = resume.identify_thresholds_per_energy_channel(df)

    print("intuitive_validation: Calculating metrics from dataframe: " + str(datetime.datetime.now()))
 
    calculate_intuitive_metrics(df, model_names, all_energy_channels,
            all_observed_thresholds)
 
    print("intuitive_validation: Validation process complete: " + str(datetime.datetime.now()))
