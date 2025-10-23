#Subroutines related to validation
from . import object_handler as objh
from . import metrics
from . import plotting_tools as plt_tools
from . import config
from . import time_profile as profile
from . import resume
from . import duplicates
import matplotlib.pylab as plt
from scipy.stats import pearsonr
import statistics
import numpy as np
import sys
import os.path
import pandas as pd
import datetime
import scipy
from pandas.api.types import is_datetime64_any_dtype as is_datetime
import sklearn.metrics as skl
import os.path
import logging
import pickle
import json

__author__ = "Katie Whitman"
__maintainer__ = "Katie Whitman"
__email__ = "kathryn.whitman@nasa.gov"

""" utils/validation.py contains subroutines to validate forecasts after
    they have been matched to observations.
    
"""

#Create logger
logger = logging.getLogger(__name__)
mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)

######### DATAFRAMES CONTAINING OBSERVATIONS AND PREDICTIONS ######

def initialize_sphinx_dict():
    """ Set up a dictionary for a pandas df to hold each possible
        quantity, each observed energy channel, and predicted and
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
            "Evaluation Status": [],
            "Forecast Issue Time":[],
            "Prediction Window Start": [],
            "Prediction Window End": [],
            
            #OBSERVATIONS
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
            
            #PREDICTIONS
            "Predicted SEP All Clear": [],
            "Predicted SEP All Clear Probability Threshold": [],
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
            "Time Profile Match Status": [],
            
            "Last Data Time to Issue Time": [],
            
            #MATCHING INFORMATION
            "All Thresholds in Prediction": [],
            "Last Eruption Time": [], #Last time for flare/CME
            "Last Trigger Time": [],
            "Last Input Time": [],
            "Threshold Crossed in Prediction Window": [],
            "All Threshold Crossing Times": [],
            "Eruption before Threshold Crossed": [],
            "Time Difference between Eruption and Threshold Crossing": [],
            "Eruption in Range": [],
            "Triggers before Threshold Crossing": [],
            "Inputs before Threshold Crossing": [],
            "Triggers before Peak Intensity": [],
            "Time Difference between Triggers and Peak Intensity": [],
            "Inputs before Peak Intensity": [],
            "Time Difference between Inputs and Peak Intensity": [],
            "Triggers before Peak Intensity Max": [],
            "Time Difference between Triggers and Peak Intensity Max": [],
            "Inputs before Peak Intensity Max": [],
            "Time Difference between Inputs and Peak Intensity Max": [],
            "Triggers before SEP End": [],
            "Time Difference between Triggers and SEP End": [],
            "Inputs before SEP End": [],
            "Time Difference between Inputs and SEP End": [],
            "Prediction Window Overlap with Observed SEP Event": [],
            "Ongoing SEP Event": [],
            "Trigger Advance Time": [],
            "Original Model Short Name": []
            
            }

    return dict



def fill_sphinx_dict_row(sphinx, dict, energy_key, thresh_key, profname_dict):
    """ Add a row to a dataframe with all of the supporting information
        for the forecast and observations that needs to be passed to
        VIVID and contains traceability for the matching process and outcomes.
        
    Input:
    
        :sphinx: (SPHINX object) contains all prediction and matched observation
            information
        :dict: (dictionary) dictionary initialized with initialize_sphinx_dict()
        :energy_key: (string) energy channel key
        :thresh_key: (string) threshold key
        :profname_dict: (dictionary) dictionary that might have been created if
            TopDirectory was specified at run time. Contains location of all txt
            files in the subdirectories of interest, including the locations of
            time profiles specified in the sep_profile field.
    
    Output:
    
        None; dict filled by reference
        
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
        cme_start = pd.NaT
        cme_liftoff = pd.NaT
        cme_lat = np.nan
        cme_lon = np.nan
        cme_pa = np.nan
        cme_half_width = np.nan
        cme_speed = np.nan
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
        fl_lat = np.nan
        fl_lon = np.nan
        fl_last_data_time = pd.NaT
        fl_start_time = pd.NaT
        fl_peak_time = pd.NaT
        fl_end_time = pd.NaT
        fl_intensity = np.nan
        fl_integrated_intensity = np.nan
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
   
   #Extract predicted values associated only with the desired energy channel
   #and threshold combination
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
    #Add path .txt files
    if pred_time_profile is not None and pred_time_profile != '':
        #First check if time profile is in the same directory as the json
        if os.path.isfile(os.path.join(sphinx.prediction.path, pred_time_profile)):
            pred_time_profile = os.path.join(sphinx.prediction.path, pred_time_profile)
        else:
            #From dictionary created by searching subdirectories
            if profname_dict is not None:
                try:
                    pred_time_profile = profname_dict[pred_time_profile]
                except:
                    logger.warning('Cannot locate time profile file ' + pred_time_profile)
                    pred_time_profile = None
                

    tp_match_status = et_match_status
        

    dict["Model"].append(sphinx.prediction.short_name)
    dict["Energy Channel Key"].append(energy_key)
    dict["Threshold Key"].append(thresh_key)
    dict["Mismatch Allowed"].append(mismatch)
    dict["Prediction Energy Channel Key"].append(pred_energy_key)
    dict["Prediction Threshold Key"].append(pred_thresh_key)
    dict["Forecast Source"].append(sphinx.prediction.source)
    dict["Forecast Path"].append(sphinx.prediction.path)
     #FORECAST EVALUATED? Explanatory status
    dict["Evaluation Status"].append(sphinx.not_evaluated)
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
        dict["Observed SEP Probability"].append(np.nan)

    try:
        dict["Observed SEP Threshold Crossing Time"].append(sphinx.observed_threshold_crossing[thresh_key].crossing_time)
    except:
        dict["Observed SEP Threshold Crossing Time"].append(pd.NaT)

    try:
        dict["Observed SEP Start Time"].append(sphinx.observed_start_time[thresh_key])
    except:
        dict["Observed SEP Start Time"].append(pd.NaT)

    try:
        dict["Observed SEP End Time"].append(sphinx.observed_end_time[thresh_key])
    except:
        dict["Observed SEP End Time"].append(pd.NaT)

    try:
        dict["Observed SEP Duration"].append(sphinx.observed_duration[thresh_key])
    except:
        dict["Observed SEP Duration"].append(np.nan)


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
        dict["Observed SEP Fluence"].append(np.nan)

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


    #PREDICTION INFORMATION
    dict["Predicted SEP All Clear"].append(pred_all_clear)
    dict["Predicted SEP All Clear Probability Threshold"].append(sphinx.prediction.all_clear.probability_threshold)
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
        dict["Predicted SEP Duration"].append(np.nan)
    
    dict["Last Data Time to Issue Time"].append(sphinx.prediction.last_data_time_to_issue_time())


    #MATCHING INFORMATION - cast all matching info to strings to avoid problems
    #with read/write. Kept mainly for human reference and traceability. Not used
    #in the validation process.
    dict["All Thresholds in Prediction"].append(str(sphinx.prediction.all_thresholds))
    dict["Last Eruption Time"].append(str(sphinx.last_eruption_time))
    dict["Last Trigger Time"].append(str(sphinx.last_trigger_time))
    dict["Last Input Time"].append(str(sphinx.last_input_time))
    
    try:
        dict["Threshold Crossed in Prediction Window"].append(str(sphinx.threshold_crossed_in_pred_win[thresh_key]))
        tc = [str(x) for x in sphinx.all_threshold_crossing_times[thresh_key]]
        dict["All Threshold Crossing Times"].append(str(tc))
    except:
        dict["Threshold Crossed in Prediction Window"].append(None)
        dict["All Threshold Crossing Times"].append(None)
        
    try:
        dict["Eruption before Threshold Crossed"].append(str(sphinx.eruptions_before_threshold_crossing[thresh_key]))
        dict["Time Difference between Eruption and Threshold Crossing"].append(str(sphinx.time_difference_eruptions_threshold_crossing[thresh_key]))
        dict["Eruption in Range"].append(str(sphinx.is_eruption_in_range[thresh_key]))
    except:
        dict["Eruption before Threshold Crossed"].append(None)
        dict["Time Difference between Eruption and Threshold Crossing"].append(None)
        dict["Eruption in Range"].append(None)
    

    
    try:
        dict["Triggers before Threshold Crossing"].append(str(sphinx.triggers_before_threshold_crossing[thresh_key]))
    except:
        dict["Triggers before Threshold Crossing"].append(None)
    
    
    try:
        dict["Inputs before Threshold Crossing"].append(str(sphinx.inputs_before_threshold_crossing[thresh_key]))
    except:
        dict["Inputs before Threshold Crossing"].append(None)


    dict["Triggers before Peak Intensity"].append(str(sphinx.triggers_before_peak_intensity))
    dict["Time Difference between Triggers and Peak Intensity"].append(str(sphinx.time_difference_triggers_peak_intensity))
    dict["Inputs before Peak Intensity"].append(str(sphinx.inputs_before_peak_intensity))
    dict["Time Difference between Inputs and Peak Intensity"].append(str(sphinx.time_difference_inputs_peak_intensity))
    dict["Triggers before Peak Intensity Max"].append(str(sphinx.triggers_before_peak_intensity_max))
    dict["Time Difference between Triggers and Peak Intensity Max"].append(str(sphinx.time_difference_triggers_peak_intensity_max))
    dict["Inputs before Peak Intensity Max"].append(str(sphinx.inputs_before_peak_intensity_max))
    dict["Time Difference between Inputs and Peak Intensity Max"].append(str(sphinx.time_difference_inputs_peak_intensity_max))

    try:
        dict["Triggers before SEP End"].append(str(sphinx.triggers_before_sep_end[thresh_key]))
        dict["Time Difference between Triggers and SEP End"].append(str(sphinx.time_difference_triggers_sep_end[thresh_key]))
    except:
        dict["Triggers before SEP End"].append(None)
        dict["Time Difference between Triggers and SEP End"].append(None)
    
    try:
        dict["Inputs before SEP End"].append(str(sphinx.inputs_before_sep_end[thresh_key]))
        dict["Time Difference between Inputs and SEP End"].append(str(sphinx.time_difference_inputs_sep_end[thresh_key]))
    except:
        dict["Inputs before SEP End"].append(None)
        dict["Time Difference between Inputs and SEP End"].append(None)
        
    try:
        dict["Prediction Window Overlap with Observed SEP Event"].append(str(sphinx.prediction_window_sep_overlap[thresh_key]))
    except:
        dict["Prediction Window Overlap with Observed SEP Event"].append(None)
    
    try:
        dict["Ongoing SEP Event"].append(str(sphinx.observed_ongoing_events[thresh_key]))
    except:
        dict["Ongoing SEP Event"].append(None)
    
    try:    
        time_diff = sphinx.observed_threshold_crossing[thresh_key].crossing_time - sphinx.prediction.last_data_time_to_issue_time()
        total_seconds = time_diff.total_seconds()
        time_diff = total_seconds / 3600
        dict["Trigger Advance Time"].append(time_diff)
    except:
        dict["Trigger Advance Time"].append('NaT')

    dict["Original Model Short Name"].append(sphinx.prediction.original_short_name)




def prepare_outdirs():
    if not os.path.isdir(config.outpath):
        os.mkdir(config.outpath)
    for datafmt in ('pkl', 'csv', 'plots', 'json'):
        outdir = os.path.join(config.outpath, datafmt)
        if not os.path.isdir(outdir):
            os.mkdir(outdir) 

    if not os.path.isdir(config.reportpath):
        os.mkdir(config.reportpath)



def write_df(df, name, verbose=True):
    """Writes a pandas dataframe to the standard location in multiple formats
    """
    dataformats = (('pkl' , getattr(df, 'to_pickle'), {}),
                   ('csv',  getattr(df, 'to_csv'), {}))
    for ext, write_func, kwargs in dataformats:
        filepath = os.path.join(config.outpath, ext, name + '.' + ext)
        write_func(filepath, **kwargs)
        if verbose:
            logger.debug('Wrote ' + filepath)


def fill_sphinx_df(evaluated_sphinx, all_obs_thresholds, profname_dict):
    """ Fill in a dictionary with the all clear predictions and observations
        organized by model and energy channel.
    """
    #sorted by model, quantity, energy channel, threshold
    dict = initialize_sphinx_dict()

    #Loop through the forecasts for each model and fill in quantity_dict
    #as appropriate
    for model in evaluated_sphinx.keys():
        internal_energy_channels = [key for key in evaluated_sphinx[model].keys() if 'eruption' not in key]
        for ek in internal_energy_channels:
            logger.debug("---Model: " + model + ", Energy Channel: " + ek)
            for sphinx in evaluated_sphinx[model][ek]:
                logger.debug(sphinx.prediction.source)
                
                try:
                    for tk in all_obs_thresholds[ek]:
                        fill_sphinx_dict_row(sphinx, dict, ek, tk, profname_dict)
                except:
                    #In the case a new energy channel with added to
                    #removed_sphinx
                    fill_sphinx_dict_row(sphinx, dict, ek, None, profname_dict)
                
    
    df = pd.DataFrame(dict)
    #Sort by prediction window start so in time order for AWT, etc
    df = df.sort_values(by=["Model","Energy Channel Key","Threshold Key","Prediction Window Start", "Forecast Issue Time"],ascending=[True, True, True, True, True])
    
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
            "Mean Ratio": [],
            "Median Ratio": [],
            "Mean Error (ME)": [],
            "Median Error (MedE)": [],
            "Mean Log Error (MLE)": [],
            "Median Log Error (MedLE)": [],
            "Mean Absolute Error (MAE)": [],
            "Median Absolute Error (MedAE)": [],
            "Mean Absolute Log Error (MALE)": [],
            "Median Absolute Log Error (MedALE)": [],
            "Mean Percent Error (MPE)": [],
            "Mean Absolute Percent Error (MAPE)": [],
            "Mean Symmetric Percent Error (MSPE)": [],
            "Mean Symmetric Absolute Percent Error (SMAPE)": [],
            "Mean Accuracy Ratio (MAR)": [],
            "Root Mean Square Error (RMSE)": [],
            "Root Mean Square Log Error (RMSLE)": [],
            "Median Symmetric Accuracy (MdSA)": [],
            "Percentage within an Order of Magnitude (%)": [],
            "Percentage within a factor of 2 (%)": []
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
            #Commenting out redundant comparisons to Observed SEP Threshold Crossing Time and
            #observed SEP Start Time. If there is ever a case where those fields are different
            #then should uncomment them.
            "Mean AWT for Predicted SEP All Clear to Observed SEP Threshold Crossing Time": [],
            "Median AWT for Predicted SEP All Clear to Observed SEP Threshold Crossing Time": [],

#            "Mean AWT for Predicted SEP All Clear to Observed SEP Start Time": [],
#            "Median AWT for Predicted SEP All Clear to Observed SEP Start Time": [],
             "Mean AWT Efficiency for Predicted SEP All Clear to Observed SEP Threshold Crossing Time": [],

#            #Probability Forecasts - cannot without an explicit threshold
#            "Mean AWT for Probability to Observed Threshold Crossing Time": [],
#            "Median AWT for Probability to Observed Threshold Crossing Time": [],
#            "Mean AWT for Probability to Observed Start Time": [],
#            "Median AWT for Probability to Observed Start Time": [],

            #Threshold Crossing Time Forecasts
            "Mean AWT for Predicted SEP Threshold Crossing Time to Observed SEP Threshold Crossing Time": [],
            "Median AWT for Predicted SEP Threshold Crossing Time to Observed SEP Threshold Crossing Time": [],
            "Mean AWT Efficiency for Predicted SEP Threshold Crossing Time to Observed SEP Threshold Crossing Time": [], 
            
#            "Mean AWT for Predicted SEP Threshold Crossing Time to Observed SEP Start Time": [],
#            "Median AWT for Predicted SEP Threshold Crossing Time to Observed SEP Start Time": [],


            #Start Time Forecasts
            "Mean AWT for Predicted SEP Start Time to Observed SEP Threshold Crossing Time": [],
            "Median AWT for Predicted SEP Start Time to Observed SEP Threshold Crossing Time": [],
            "Mean AWT Efficiency for Predicted SEP Start Time to Observed SEP Threshold Crossing Time": [],
#            "Mean AWT for Predicted SEP Start Time to Observed SEP Start Time": [],
#            "Median AWT for Predicted SEP Start Time to Observed SEP Start Time": [],
 
#             #Point Intensity Forecasts
#            "Mean AWT for Predicted Point Intensity to Observed SEP Threshold Crossing Time": [],
#            "Median AWT for Predicted Point Intensity to Observed SEP Threshold Crossing Time": [],
#            "Mean AWT for Predicted Point Intensity to Observed SEP Start Time": [],
#            "Median AWT for Predicted Point Intensity to Observed SEP Start Time": [],
 
 
            #Peak Intensity Forecasts
            "Mean AWT for Predicted SEP Peak Intensity (Onset Peak) to Observed SEP Threshold Crossing Time": [],
            "Median AWT for Predicted SEP Peak Intensity (Onset Peak) to Observed SEP Threshold Crossing Time": [],
#            "Mean AWT for Predicted SEP Peak Intensity (Onset Peak) to Observed SEP Start Time": [],
#            "Median AWT for Predicted SEP Peak Intensity (Onset Peak) to Observed SEP Start Time": [],
            "Mean AWT for Predicted SEP Peak Intensity (Onset Peak) to Observed SEP Peak Intensity (Onset Peak) Time": [],
            "Median AWT for Predicted SEP Peak Intensity (Onset Peak) to Observed SEP Peak Intensity (Onset Peak) Time": [],
 #           "Mean AWT for Predicted SEP Peak Intensity (Onset Peak) to Observed SEP Peak Intensity Max (Max Flux) Time": [],
 #           "Median AWT for Predicted SEP Peak Intensity (Onset Peak) to Observed SEP Peak Intensity Max (Max Flux) Time": [],

            #Peak Intensity Max Forecasts
            "Mean AWT for Predicted SEP Peak Intensity Max (Max Flux) to Observed SEP Threshold Crossing Time": [],
            "Median AWT for Predicted SEP Peak Intensity Max (Max Flux) to Observed SEP Threshold Crossing Time": [],
#            "Mean AWT for Predicted SEP Peak Intensity Max (Max Flux) to Observed SEP Start Time": [],
#            "Median AWT for Predicted SEP Peak Intensity Max (Max Flux) to Observed SEP Start Time": [],
            "Mean AWT for Predicted SEP Peak Intensity Max (Max Flux) to Observed SEP Peak Intensity Max (Max Flux) Time": [],
            "Median AWT for Predicted SEP Peak Intensity Max (Max Flux) to Observed SEP Peak Intensity Max (Max Flux) Time": [],

            #End Time Forecasts
            "Mean AWT for Predicted SEP End Time to Observed SEP Threshold Crossing Time": [],
            "Median AWT for Predicted SEP End Time to Observed SEP Threshold Crossing Time": [],
#            "Mean AWT for Predicted SEP End Time to Observed SEP Start Time": [],
#            "Median AWT for Predicted SEP End Time to Observed SEP Start Time": [],
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
            "N (Total Number of Forecasts)": [],
            "Percent Correct": [],
            "Bias": [],
            "Hit Rate": [],
            "False Alarm Rate": [],
            'False Negative Rate': [],
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
            "Symmetric Extreme Dependency Score": [],
            "F1 Score": [],
            "F2 Score": [],
            "Fhalf Score": [],
            'Prevalence': [],
            'Matthew Correlation Coefficient': [],
            'Informedness': [],
            'Markedness': [],
            'Prevalence Threshold': [],
            'Balanced Accuracy': [],
            'Fowlkes-Mallows Index': [],
            "Number SEP Events Correctly Predicted": [],
            "Number SEP Events Missed": [],
            "Predicted SEP Events": [], #date string
            "Missed SEP Events": [] #date string
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
            "ROC Curve Plot": [],
            "Brier Score": [],
            "Brier Skill Score": [],
            "Spearman Correlation Coefficient": [],
            "Area Under ROC Curve": []
            }
            
    return dict




def fill_flux_metrics_dict(dict, model, energy_key, thresh_key,
    pred_energy_key, pred_thresh_key, figname,
    slope, yint, r_lin, r_log, s_lin, MRatio, MedRatio, ME, MedE,
    MLE, MedLE, MAE, MedAE, MALE, MedALE, MPE, MAPE, MSPE, SMAPE,
    MAR, RMSE, RMSLE, MdSA, fact10, fact2, timeprofplot=None):
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
    dict["Mean Ratio"].append(MRatio)
    dict["Median Ratio"].append(MedRatio)
    dict["Mean Error (ME)"].append(ME)
    dict["Median Error (MedE)"].append(MedE)
    dict["Mean Log Error (MLE)"].append(MLE)
    dict["Median Log Error (MedLE)"].append(MedLE)
    dict["Mean Absolute Error (MAE)"].append(MAE)
    dict["Median Absolute Error (MedAE)"].append(MedAE)
    dict["Mean Absolute Log Error (MALE)"].append(MALE)
    dict["Median Absolute Log Error (MedALE)"].append(MedALE)
    dict["Mean Percent Error (MPE)"].append(MPE)
    dict["Mean Absolute Percent Error (MAPE)"].append(MAPE)
    dict["Mean Symmetric Percent Error (MSPE)"].append(MSPE)
    dict["Mean Symmetric Absolute Percent Error (SMAPE)"].append(SMAPE)
    dict["Mean Accuracy Ratio (MAR)"].append(MAR)
    dict["Root Mean Square Error (RMSE)"].append(RMSE)
    dict["Root Mean Square Log Error (RMSLE)"].append(RMSLE)
    dict["Median Symmetric Accuracy (MdSA)"].append(MdSA)
    dict["Percentage within an Order of Magnitude (%)"].append(fact10)
    dict["Percentage within a factor of 2 (%)"].append(fact2)


    if timeprofplot is not None:
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
    pred_thresh_key, scores, n_caught, sep_caught_str, n_miss, sep_miss_str):
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
    dict["N (Total Number of Forecasts)"].append(scores['TP'] + scores['FP'] + scores['TN'] + scores['FN'])
    dict["Percent Correct"].append(scores['PC'])
    dict["Bias"].append(scores['B'])
    dict["Hit Rate"].append(scores['H'])
    dict["False Alarm Rate"].append(scores['F'])
    dict['False Negative Rate'].append(scores['FNR'])
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
    dict["Symmetric Extreme Dependency Score"].append(scores['SEDS'])
    dict["F1 Score"].append(scores['FONE'])
    dict["F2 Score"].append(scores['FTWO'])
    dict["Fhalf Score"].append(scores['FHALF'])
    dict['Prevalence'].append(scores['PREV'])
    dict['Matthew Correlation Coefficient'].append(scores['MCC'])
    dict['Informedness'].append(scores['INFORM'])
    dict['Markedness'].append(scores['MARK'])
    dict['Prevalence Threshold'].append(scores['PT'])
    dict['Balanced Accuracy'].append(scores['BA'])
    dict['Fowlkes-Mallows Index'].append(scores['FM'])
    dict["Number SEP Events Correctly Predicted"].append(n_caught)
    dict["Number SEP Events Missed"].append(n_miss)
    dict["Predicted SEP Events"].append(sep_caught_str)
    dict["Missed SEP Events"].append(sep_miss_str)
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



#####SUBROUTINES TO HELP EXTRACT FIRST, LAST, MAX, MEAN FORECASTS
def identify_not_clear_forecast(df, validation_type):
    """ Identify the row of the appropriate All Clear forecast
        of False for a given SEP event.
        
        Assume that all the forecasts in the df are sorted in ascending
        order of time.
        
        INPUT:
        
            :df: (pandas dataframe) forecasts for a single SEP event for a single
                model, energy channel, and threshold

        OUTPUT:
        
            :row: row of dataframe with all the values associated with the
                desired forecast
        
    """
    all_clear = df['Predicted SEP All Clear'].to_list()

    if validation_type == "Mean" or validation_type == "Max":
        return pd.DataFrame()
    
    if validation_type == "First":
        for i in range(len(all_clear)):
            if all_clear[i] is None:
                continue
            if all_clear[i] == True:
                continue
            if all_clear[i] == False:
                return df.iloc[[i]]
            
    if validation_type == "Last":
        #Search in reverse order checking of forecast is False All Clear
        for i in range(len(all_clear)-1,-1,-1):
            if all_clear[i] is None:
                continue
            if all_clear[i] == True:
                continue
            if all_clear[i] == False:
                return df.iloc[[i]]
    
    #if make it here, then no All Clear = False forecasts were made
    #for this particular SEP event.
    return pd.DataFrame()


#SIMPLIFY EXPECTING ORGANIZED IN TIME ORDER
def identify_flux_forecast(df, thresh_key, pred_key, validation_type):
    """ Identify the row of the appropriate flux forecast
        of False for a given SEP event.
        
        Assume that all the forecasts in the df are sorted in ascending
        order of time.
        
        INPUT:
        
            :df: (pandas dataframe) forecasts for a single SEP event for a single
                model, energy channel, and threshold

        OUTPUT:
        
            :row: row of dataframe with all the values associated with the
                desired forecast
        
    """
    threshold = objh.key_to_threshold(thresh_key)
    thresh = threshold['threshold']

    if validation_type == "Mean" or validation_type == "Max":
        return pd.DataFrame()
    
    if validation_type == "First":
        for i in range(len(df)):
            if pd.isnull(df.iloc[i][pred_key]):
                continue
            if df.iloc[i][pred_key] < thresh:
                continue
            if df.iloc[i][pred_key] >= thresh:
                return df.iloc[[i]]
            
    if validation_type == "Last":
        #Search in reverse order checking of forecast
        for i in range(len(df)-1,-1,-1):
            if pd.isnull(df.iloc[i][pred_key]):
                continue
            if df.iloc[i][pred_key] < thresh:
                continue
            if df.iloc[i][pred_key] >= thresh:
                return df.iloc[[i]]
    
    return pd.DataFrame()


def identify_time_forecast(df, pred_key, validation_type):
    """ Identify the row of the appropriate flux forecast
        of False for a given SEP event.
        
        Assume that all the forecasts in the df are sorted in ascending
        order of time.
        
        INPUT:
        
            :df: (pandas dataframe) forecasts for a single SEP event for a single
                model, energy channel, and threshold

        OUTPUT:
        
            :row: row of dataframe with all the values associated with the
                desired forecast
        
    """

    if validation_type == "Mean" or validation_type == "Max":
        return pd.DataFrame()
    
    if validation_type == "First":
        for i in range(len(df)):
            if pd.isnull(df.iloc[i][pred_key]):
                continue
            else:
                return df.iloc[[i]]
            
    if validation_type == "Last":
        #Search in reverse order checking of forecast is False All Clear
        for i in range(len(df)-1,-1,-1):
            if pd.isnull(df.iloc[i][pred_key]):
                continue
            else:
                return df.iloc[[i]]
    
    return pd.DataFrame()



def identify_max_forecast(df, pred_key):
    """ Calculate the appropriate forecast value and return a row
        of the dataframe.
        
        INPUT:
        
            :df: (pandas dataframe) forecasts for a single SEP event for a single
                model, energy channel, and threshold
            :pred_key: (string) key of the predicted value to identify the max

        OUTPUT:
        
            :row: row of dataframe with all the values associated with the
                desired forecast
        
    """
    if df.empty:
        return pd.DataFrame()

    #If only one entry, no need to calculate max
    if len(df) == 1:
        return df.iloc[[0]]

    maxval = df[pred_key].max()
    if pd.isnull(maxval):
        return pd.DataFrame()
    
    idx = df[pred_key].idxmax() #first instance of max value
            
    return df.loc[[idx]]
    


def calculate_mean_forecast(df, pred_key):
    """ Calculate the appropriate forecast value and return a row
        of the dataframe.
        
        INPUT:
        
            :df: (pandas dataframe) forecasts for a single SEP event for a single
                model, energy channel, and threshold
            :pred_key: (string) key of the predicted value to calculate the mean

        OUTPUT:
        
            :row: row of dataframe with all the values associated with the
                desired forecast
        
    """
    if df.empty:
        return pd.DataFrame()

    #If only one entry, no need to calculate mean
    if len(df) == 1:
        return df.iloc[[0]]

    cols = df.columns.to_list() #Can I do this without going to lists?
    pred_idx = cols.index(pred_key) #index in row where predicted value is stored

    #First drop all the forecasts with None values
    sub = df.dropna(subset=df.columns[pred_idx])
    if sub.empty:
        return pd.DataFrame()

    #Mean value of all forecasts
    meanval = sub[pred_key].mean()

    #Start of earliest prediction window
    pred_st = sub['Prediction Window Start'].min()
    
    #End of latest prediction window
    pred_end = sub['Prediction Window End'].max()
    
    #Record all the files that were used to create the average
    fnames = sub['Forecast Source'].iloc[0]
    for i in range(1,len(sub),1):
        fnames += ";" + sub['Forecast Source'].iloc[i]

    #Use the 0th row of the sub df to replace various values to the ones
    #we want saved
    sub.loc[0,'Forecast Source'] = fnames
    sub.loc[0,'Prediction Window Start'] = pred_st
    sub.loc[0,'Prediction Window End'] = pred_end
    sub.loc[0,pred_key] = meanval

    return sub.iloc[[0]]
    
 
 
def extract_all_clear_forecast_type(df, validation_type):
    """ Extract the correct all clear forecasts depending on the desired
        validation_type.
        
        INPUT:
        
        :df: (pandas DataFrame) contains all the all clear forecasts
            for a given model, energy channel, and threshold
        :validation_type: (string) First, Last, Max, Mean
        
        OUTPUT:
        
        :sub: (pandas DataFrame) probability forecasts relevant to the
            validation_type. Only one forecast per SEP event. ONLY
            forecasts related to observed SEP events.
        
    """
    if validation_type == "All" or validation_type == "":
        return df

    #Create an empty dataframe with the same columns plus AWT info
    sel_df = pd.DataFrame(columns=df.columns) #Selected forecasts
    sel_df = sel_df.astype(dtype=df.dtypes)

    if validation_type == "Max" or validation_type == "Mean":
        return sel_df
    
    #Extract all unique SEP events
    sep_events = resume.identify_unique(df, 'Observed SEP Threshold Crossing Time')
    
    if len(sep_events) == len(df.index):
        return df
   
    #For each SEP event, identify the desired forecast for that SEP event.
    for sep in sep_events:
        sep_sub = df.loc[df['Observed SEP Threshold Crossing Time'] == sep]

        if validation_type == "First" or validation_type == "Last":
            row = identify_not_clear_forecast(sep_sub, validation_type)
        
        if row.empty:
            #In this case, if row is empty, it is because the model
            #didn't issue an All Clear = False forecast for this
            #event. To account for this, save the first row in sep_sub
            row = sep_sub.iloc[[0]]

        sel_df = pd.concat([sel_df,row],ignore_index=True)
        
    return sel_df


def extract_probability_forecast_type(df, validation_type):
    """ Extract the correct probability forecasts depending on the desired
        validation_type.
        
        For probability, the First and Last forecast must depend on the
        All Clear field to indicate whether the probability value
        indicates that an SEP event will occur. If the All Clear field
        is not present, then the First and Last probability forecast
        cannot be calculated.
        
        INPUT:
        
        :df: (pandas DataFrame) contains all the probability forecasts
            for a given model, energy channel, and threshold
        :validation_type: (string) First, Last, Max, Mean
        
        OUTPUT:
        
        :sub: (pandas DataFrame) probability forecasts relevant to the
            validation_type. Only one forecast per SEP event. ONLY
            forecasts related to observed SEP events.
        
    """
    if validation_type == "All" or validation_type == "":
        return df
    
    #Create an empty dataframe with the same columns plus AWT info
    sel_df = pd.DataFrame(columns=df.columns) #Selected forecasts
    sel_df = sel_df.astype(dtype=df.dtypes)
    
    #First and last probabilities will only save the probabilities for
    #events that the model correctly predicted to occur. The resulting
    #metrics will be way overestimated, so don't use these.
    if validation_type == "First" or validation_type == "Last":
        return sel_df
    
    #Extract all unique SEP events
    sep_events = resume.identify_unique(df, 'Observed SEP Threshold Crossing Time')
   
    #For each SEP event, identify the desired forecast for that SEP event.
    for sep in sep_events:
        sep_sub = df.loc[df['Observed SEP Threshold Crossing Time'] == sep]
        if sep_sub.empty: #shouldn't happen
            continue
        
        if validation_type == "Max":
            row = identify_max_forecast(sep_sub, "Predicted SEP Probability")
        
        if validation_type == "Mean":
            row = calculate_mean_forecast(sep_sub, "Predicted SEP Probability")
        
        if row.empty:
            continue

        sel_df = pd.concat([sel_df,row], ignore_index=True)

    return sel_df



def extract_flux_forecast_type(df, thresh_key, pred_key, time_key, validation_type):
    """ Extract the correct flux forecasts depending on the desired
        validation_type.
        
 
        INPUT:
        
        :df: (pandas DataFrame) contains all the probability forecasts
            for a given model, energy channel, and threshold
        :pred_key: (string) specifies predicted value, e.g.
            "Predicted SEP Peak Intensity (Onset Peak)"
        :validation_type: (string) First, Last, Max, Mean
        
        OUTPUT:
        
        :sub: (pandas DataFrame) probability forecasts relevant to the
            validation_type. Only one forecast per SEP event. ONLY
            forecasts related to observed SEP events.
        
        :doType: (bool) True = do First, Last, Mean, Max (more than
                    one forecast per SEP event)
            False = one forecast per SEP event so no First, Last, Mean, Max
        
    """

    if validation_type == "All" or validation_type == "":
        return df, True
    
    #Create an empty dataframe with the same columns plus AWT info
    sel_df = pd.DataFrame(columns=df.columns) #Selected forecasts
    sel_df = sel_df.astype(dtype=df.dtypes)
    
    #Extract all unique SEP events
    sep_events = resume.identify_unique(df, time_key)
   
    #Check if the number of forecasts is equal to the number of SEP
    #events, indicating that there is one forecast per SEP event.
    #If so, then no need to run First, Last, Max, Mean
    if len(sep_events) == len(df.index):
        return df, False
    
    #For each SEP event, identify the desired forecast for that SEP event.
    for sep in sep_events:
        sep_sub = df.loc[df[time_key] == sep]

        if validation_type == "First" or validation_type == "Last":
            row = identify_flux_forecast(sep_sub, thresh_key, pred_key, validation_type)
        
        if validation_type == "Max":
            row = identify_max_forecast(sep_sub, pred_key)
        
        if validation_type == "Mean":
            row = calculate_mean_forecast(sep_sub, pred_key)
        
        if row.empty:
            continue

        sel_df = pd.concat([sel_df,row], ignore_index=True)

    return sel_df, True



def extract_time_forecast_type(df, pred_key, validation_type):
    """ Extract the correct flux forecasts depending on the desired
        validation_type.
        
 
        INPUT:
        
        :df: (pandas DataFrame) contains all the probability forecasts
            for a given model, energy channel, and threshold
        :pred_key: (string) specifies predicted value, e.g.
            "Predicted SEP Peak Intensity (Onset Peak)"
        :validation_type: (string) First, Last, Max, Mean
        
        OUTPUT:
        
        :sub: (pandas DataFrame) forecasts relevant to the
            validation_type. Only one forecast per SEP event. ONLY
            forecasts related to observed SEP events.
            
        :doType: (bool) True = do First, Last, Mean, Max (more than
                    one forecast per SEP event)
            False = one forecast per SEP event so no First, Last, Mean, Max
        
    """
    #Validation type can only be All, First, Last, Mean, Max
    if validation_type == "All" or validation_type == "":
        return df, True
    
    #Create an empty dataframe with the same columns plus AWT info
    sel_df = pd.DataFrame(columns=df.columns) #Selected forecasts
    sel_df = sel_df.astype(dtype=df.dtypes)
    
    if validation_type == "Max" or validation_type == "Mean":
        return sel_df, False
    
    #Extract all unique SEP events
    time_key = pred_key.replace("Predicted", "Observed")
    sep_events = resume.identify_unique(df, time_key)

    
    #If same number of forecasts as SEP events, then only one forecast
    #per SEP event and no need to do First, Last
    if len(sep_events) == len(df.index):
        return df, False
   
    #For each SEP event, identify the desired forecast for that SEP event.
    for sep in sep_events:
        sep_sub = df.loc[df[time_key] == sep]

        if validation_type == "First" or validation_type == "Last":
            row = identify_time_forecast(sep_sub, pred_key, validation_type)
        
            if not row.empty:
                sel_df = pd.concat([sel_df,row], ignore_index=True)
    
        
    return sel_df, True
    
##### END FIRST, LAST, MEAN, MAX #####################




def all_clear_intuitive_metrics(df, dict, model, energy_key, thresh_key,
    validation_type):
    """ Extract the appropriate predictions and calculate metrics
        All Clear

        If mismatch = True, will extract only the predictions where
        the Mismatch Allowed field is True.
        
        The metrics will be calculated for All Clear using all forecasts.
        The "First" forecasts mode will be used within the subroutine to
        determine whether a model "caught" or completely missed an SEP
        event.
        
    """
    val_type = ["", "All"]
    if validation_type not in val_type:
        return
    
    #Select rows to calculate metrics
    sub = df.loc[(df['Model'] == model) & (df['Energy Channel Key'] ==
        energy_key) & (df['Threshold Key'] == thresh_key)]

    sub = sub[['Model','Energy Channel Key', 'Threshold Key',
            'Mismatch Allowed',
            'Prediction Energy Channel Key', 'Prediction Threshold Key',
            'Forecast Source',
            'Prediction Window Start', 'Prediction Window End',
            'Observed SEP Threshold Crossing Time',
            'Observed SEP All Clear', 'Predicted SEP All Clear',
            'All Clear Match Status']]
    sub = sub.loc[(sub['All Clear Match Status'] != 'Ongoing SEP Event')]
    sub = sub.dropna(subset='All Clear Match Status')
      
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
    if validation_type != "" and validation_type != "All":
        fnm = fnm + "_" + validation_type
    write_df(sub, fnm)

    
    scores = metrics.calc_contingency_all_clear(sub, 'Observed SEP All Clear',
                'Predicted SEP All Clear')
    
    #Now extract whether an SEP event was "caught" or missed using the
    #"First" forecast validation type.
    #Contains one line per SEP event and only forecasts associated with SEPs.
    #In the case of a correctly predicted SEP event, contains the first
    #forecast that predicted False all clear (regardless of whether following
    #forecasts switched back to True all clear).
    #In the case of a missed SEP event, contains the first True all clear
    #forecast where the prediction window contained the threshold crossing.
    sub_first = extract_all_clear_forecast_type(sub, "First")
    sub_caught = sub_first.loc[(sub_first['Predicted SEP All Clear'] == False)]
    sep_caught = sub_caught['Observed SEP Threshold Crossing Time'].to_list()
    n_caught = len(sep_caught)
    sep_caught_str = ""
    if n_caught == 0:
        sep_caught_str = "None"
    else:
        sep_caught_str = str(sep_caught[0])
        for jj in range(1,n_caught,1):
            sep_caught_str += ";" + str(sep_caught[jj])
    
    
    sub_miss = sub_first.loc[(sub_first['Predicted SEP All Clear'] == True)]
    sep_miss = sub_miss['Observed SEP Threshold Crossing Time'].to_list()
    n_miss = len(sep_miss)
    sep_miss_str = ""
    if n_miss == 0:
        sep_miss_str = "None"
    else:
        sep_miss_str = str(sep_miss[0])
        for jj in range(1,n_miss,1):
            sep_miss_str += ";" + str(sep_miss[jj])
    
    
    fill_all_clear_dict(dict, model, energy_key, thresh_key, pred_energy_key,
        pred_thresh_key, scores, n_caught, sep_caught_str, n_miss, sep_miss_str)


    return sub



def probability_intuitive_metrics(df, dict, model, energy_key, thresh_key,
    validation_type):
    """ Extract the appropriate predictions and calculate metrics
        Probability

    """
    #Only calculate probability metrics for ALL forecasts
    val_type = ["", "All","Max"]
    if validation_type not in val_type:
        return
    
    #Select rows to calculate metrics
    sub = df.loc[(df['Model'] == model) & (df['Energy Channel Key'] ==
        energy_key) & (df['Threshold Key'] == thresh_key)]

    sub = sub[['Model','Energy Channel Key', 'Threshold Key',
            'Mismatch Allowed',
            'Prediction Energy Channel Key', 'Prediction Threshold Key',
            'Forecast Source',
            'Prediction Window Start', 'Prediction Window End',
            'Observed SEP Threshold Crossing Time',
            'Observed SEP Probability',
            'Predicted SEP All Clear',
            'Predicted SEP Probability', 'Probability Match Status']]
    sub = sub.loc[(sub['Probability Match Status'] != 'Ongoing SEP Event')]
    sub = sub.dropna(subset='Predicted SEP Probability')

    if not sub.empty:
        sub = sub.dropna(subset='Observed SEP Probability')

    if sub.empty:
        return

    #Extract First, Last, Max, Mean, etc if selected
    sub = extract_probability_forecast_type(sub, validation_type)
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
    if validation_type != "" and validation_type != "All":
        fnm = fnm + "_" + validation_type
    write_df(sub,fnm)

    obs = sub['Observed SEP Probability'].to_list()
    pred = sub['Predicted SEP Probability'].to_list()

    #Calculate metrics
    brier_score = metrics.calc_brier(obs, pred)
    brier_skill = metrics.calc_brier_skill(obs, pred)
    rank_corr_coeff = metrics.calc_spearman(obs, pred) 

    roc_auc, roc_curve_plt = metrics.receiver_operator_characteristic(obs, pred, model)
    
    roc_curve_plt.plot()
    skill_line = np.linspace(0.0, 1.0, num=10) # Constructing a diagonal line that represents no skill/random guess
    plt.plot(skill_line, skill_line, '--', label = 'Random Guess')
    figname = config.outpath + '/plots/ROC_curve_' \
            + model + "_" + energy_key.strip() + "_" + thresh_fnm
    if mismatch:
            figname = figname + "_mm"
    if validation_type != "" and validation_type != "All":
            figname = figname + "_" + validation_type
    figname += ".pdf"
    plt.legend(loc="lower right")
    roc_curve_plt.figure_.savefig(figname, dpi=300, bbox_inches='tight')
    plt.close(roc_curve_plt.figure_)
    
    #Save to dict (ultimately dataframe)
    dict['Model'].append(model)
    dict['Energy Channel'].append(energy_key)
    dict['Threshold'].append(thresh_key)
    dict['Prediction Energy Channel'].append(pred_energy_key)
    dict['Prediction Threshold'].append(pred_thresh_key)
    dict['ROC Curve Plot'].append(figname)
    dict['Brier Score'].append(brier_score)
    dict['Brier Skill Score'].append(brier_skill)
    dict['Spearman Correlation Coefficient'].append(rank_corr_coeff)
    dict['Area Under ROC Curve'].append(roc_auc)


def calc_all_flux_metrics(obs, pred):
    """ Calculate the metrics used for assessing fluxes.
    """
    MRatio = None
    MedRatio = None
    ME = None
    MedE = None
    MAE = None
    MedAE = None
    MLE = None
    MedLE = None
    MALE = None
    MedALE = None
    MPE = None
    MAPE = None
    MSPE = None
    SMAPE = None
    MAR = None #Mean Accuracy Ratio
    RMSE = None
    RMSLE = None
    MdSA = None

    sepratio = []
    sepE = []
    sepAE = []
    sepLE = []
    sepALE = []
    sepPE = []
    sepAPE = []

    if len(obs) >= 1:

        #Errors for each observation-forecast pair
        sepratio = metrics.switch_error_func('Ratio',obs,pred)
        sepE = metrics.switch_error_func('E',obs,pred)
        sepAE = metrics.switch_error_func('AE',obs,pred)
        sepLE = metrics.switch_error_func('LE',obs,pred)
        sepALE = metrics.switch_error_func('ALE',obs,pred)
        sepPE = metrics.switch_error_func('PE',obs,pred)
        sepAPE = metrics.switch_error_func('APE',obs,pred)

        #Mean and median errors across all observation-forecast pairs
        MRatio = statistics.mean(metrics.switch_error_func('Ratio',obs,pred))
        MedRatio = statistics.median(metrics.switch_error_func('Ratio',obs,pred))
        ME = statistics.mean(metrics.switch_error_func('E',obs,pred))
        MedE = statistics.median(metrics.switch_error_func('E',obs,pred))
        MAE = statistics.mean(metrics.switch_error_func('AE',obs,pred))
        MedAE = statistics.median(metrics.switch_error_func('AE',obs,pred))
        MLE = statistics.mean(metrics.switch_error_func('LE',obs,pred))
        MedLE = statistics.median(metrics.switch_error_func('LE',obs,pred))
        MALE = statistics.mean(metrics.switch_error_func('ALE',obs,pred))
        MedALE = statistics.median(metrics.switch_error_func('ALE',obs,pred))
        MPE = statistics.mean(metrics.switch_error_func('PE',obs,pred))
        MAPE = statistics.mean(metrics.switch_error_func('APE',obs,pred))
        MSPE = statistics.mean(metrics.switch_error_func('SPE',obs,pred))
        SMAPE = statistics.mean(metrics.switch_error_func('SAPE',obs,pred))
        MAR = metrics.switch_error_func('MAR',obs,pred) #Mean Accuracy Ratio
        RMSE = metrics.switch_error_func('RMSE',obs,pred)
        RMSLE = metrics.switch_error_func('RMSLE',obs,pred)
        MdSA = metrics.switch_error_func('MdSA',obs,pred)


    #Append sub with the metrics for each observed and predicted time profile match
    errors = {'Ratio': sepratio, 'Error': sepE, 'Absolute Error': sepAE, 'Log Error': sepLE, 'Absolute Log Error': sepALE, 'Percent Error': sepPE, 'Absolute Percent Error': sepAPE}

    return MRatio, MedRatio, ME, MedE, MAE, MedAE, MLE, MedLE, MALE, MedALE,\
            MPE, MAPE, MSPE, SMAPE, MAR, RMSE, RMSLE, MdSA, errors




def point_intensity_intuitive_metrics(df, dict, model, energy_key, thresh_key,
    validation_type, flux_threshold=0):
    """ Extract the appropriate predictions and calculate metrics
        Point intensity

        All observed point fluxes below flux_threshold will be excluded.
        Makes sense to set above a detector background level or to
        a warning threshold.
        
    """
    #Only calculate point intensity metrics for All forecasts
    val_type = ["", "All"]
    if validation_type not in val_type:
        return
    
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
            if validation_type != "" and validation_type != "All":
                tpfigname = tpfigname + "_" + validation_type
            tpfigname += ".pdf"

            if tp_plotnames == "":
                tp_plotnames = tpfigname
            else:
                tp_plotnames += ";" + tpfigname
 
 
            plt_tools.plot_time_profile([trim_times, trim_times], [trim_pred,trim_obs],
            labels, title=title, x_label="Date", y_min=1e-7, y_max=1e5,
            y_label="Particle Intensity",
            date_format="none", showplot=False,
            closeplot=True, saveplot=True, figname=tpfigname)
    
    
    
        #PEARSON CORRELATION
        r_lin, r_log = metrics.switch_error_func('r',obs,pred)
        s_lin = metrics.switch_error_func('spearman',obs,pred)
        
        #LINEAR REGRESSION
        obs_np = np.log10(np.array(obs))
        pred_np = np.log10(np.array(pred))
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
        if validation_type != "" and validation_type != "All":
            figname = figname + "_" + validation_type 
        figname += ".pdf"
        corr_plot.savefig(figname, dpi=300, bbox_inches='tight')
        corr_plot.close()
   
    else:
        r_lin = None
        r_log = None
        s_lin = None
        slope = None
        yint = None
        figname = ""

    
    MRatio, MedRatio, ME, MedE, MAE, MedAE, MLE, MedLE, MALE, MedALE, MPE, MAPE,\
    MSPE, SMAPE, MAR, RMSE, RMSLE, MdSA, errors = calc_all_flux_metrics(obs, pred)

    #Add error calculation for individual observation-forecast pairs and
    #write out selections and the metrics associated with each profile match
    if len(errors['Ratio']) != 0:
        sub = sub.assign(**errors)
    fnm = f"point_intensity_selections_{model}_{energy_key.strip()}_{thresh_fnm}"
    if mismatch:
        fnm = fnm + "_mm"
    if validation_type != "" and validation_type != "All":
        fnm = fnm + "_" + validation_type
    write_df(sub, fnm)

    temp = metrics.switch_error_func('LE',obs,pred)
    count = 0
    count_fact_2 = 0
    for i in range(len(temp)):     
        if temp[i] >= -1 and temp[i] <= 1:
            count += 1
        if temp[i] >= -np.log10(2) and temp[i] <= np.log10(2):
            count_fact_2 += 1

    fact10 = count / len(temp)
    fact2 = count_fact_2 / len(temp)
    ####METRICS
    fill_flux_metrics_dict(dict, model, energy_key, thresh_key,
        pred_energy_key, pred_thresh_key, figname,
        slope, yint, r_lin, r_log, s_lin, MRatio, MedRatio, ME, MedE, MLE,
        MedLE, MAE, MedAE, MALE, MedALE, MPE, MAPE, MSPE, SMAPE,
        MAR, RMSE, RMSLE, MdSA, fact10, fact2, tp_plotnames)



def peak_intensity_intuitive_metrics(df, dict, model, energy_key, thresh_key,
    validation_type):
    """ Extract the appropriate predictions and calculate metrics
        Peak intensity

    """
    val_type = ["", "All", "First", "Last", "Max", "Mean"]
    if validation_type not in val_type:
        return
    
    #Select rows to calculate metrics
    sub = df.loc[(df['Model'] == model) & (df['Energy Channel Key'] ==
        energy_key) & (df['Threshold Key'] == thresh_key)]

    sub = sub[['Model','Energy Channel Key', 'Threshold Key',
            'Mismatch Allowed',
            'Prediction Energy Channel Key', 'Prediction Threshold Key',
            'Forecast Source',
            'Prediction Window Start', 'Prediction Window End',
            'Observed SEP Peak Intensity (Onset Peak) Time',
            'Observed SEP Peak Intensity (Onset Peak)',
            'Observed SEP Peak Intensity (Onset Peak) Units',
            'Predicted SEP Peak Intensity (Onset Peak)',
            'Predicted SEP Peak Intensity (Onset Peak) Units',
            'Peak Intensity Match Status']]
    sub = sub.loc[(sub['Peak Intensity Match Status'] == 'SEP Event')]
    sub = sub.dropna(subset=['Predicted SEP Peak Intensity (Onset Peak)', 'Observed SEP Peak Intensity (Onset Peak)'])

    #Drop zero and negative values in predicted or observed peak intensity
    if not sub.empty:
        sub = sub.loc[(sub['Predicted SEP Peak Intensity (Onset Peak)'] > 0) & (sub['Observed SEP Peak Intensity (Onset Peak)'] > 0)]

    if sub.empty:
        return

    sub, doType = extract_flux_forecast_type(sub, thresh_key, 'Predicted SEP Peak Intensity (Onset Peak)', 'Observed SEP Peak Intensity (Onset Peak) Time', validation_type)
    if sub.empty or not doType:
        return
    
    mismatch = bool(sub.iloc[0]['Mismatch Allowed'])
    pred_energy_key = str(sub.iloc[0]['Prediction Energy Channel Key'])
    pred_thresh_key = str(sub.iloc[0]['Prediction Threshold Key'])
    
    thresh_fnm = make_thresh_fname(thresh_key)

    obs = sub['Observed SEP Peak Intensity (Onset Peak)'].to_list()
    pred = sub['Predicted SEP Peak Intensity (Onset Peak)'].to_list()
    units = sub.iloc[0]['Observed SEP Peak Intensity (Onset Peak) Units']

    if len(obs) > 1:
        #PEARSON CORRELATION
        r_lin, r_log = metrics.switch_error_func('r',obs,pred)
        s_lin = metrics.switch_error_func('spearman',obs,pred)

        
        #LINEAR REGRESSION
        obs_np = np.log10(obs)
        pred_np = np.log10(pred)
        slope, yint = np.polyfit(obs_np, pred_np, 1)

        #Correlation Plot
        corr_plot = plt_tools.correlation_plot(obs, pred,
        "Peak Intensity Correlation", xlabel="Observations",
        ylabel=("Model Predictions (" + str(units) + ")"), use_log = True)

        figname = config.outpath + '/plots/Correlation_peak_intensity_' \
            + model + "_" + energy_key.strip() + "_" + thresh_fnm
        if mismatch:
            figname = figname + "_mm"
        if validation_type != "" and validation_type != "All":
            figname = figname + "_" + validation_type 
        figname += ".pdf"
        corr_plot.savefig(figname, dpi=300, bbox_inches='tight')
        corr_plot.close()
    else:
        r_lin = None
        r_log = None
        s_lin = None
        slope = None
        yint = None
        figname = ""


    MRatio, MedRatio, ME, MedE, MAE, MedAE, MLE, MedLE, MALE, MedALE, MPE, MAPE,\
    MSPE, SMAPE, MAR, RMSE, RMSLE, MdSA, errors = calc_all_flux_metrics(obs, pred)


    #Add error calculation for individual observation-forecast pairs and
    #write out selections and the metrics associated with each profile match
    if len(errors['Ratio']) != 0:
        sub = sub.assign(**errors)
    fnm = f"peak_intensity_selections_{model}_{energy_key.strip()}_{thresh_fnm}"
    if mismatch:
        fnm = fnm + "_mm"
    if validation_type != "" and validation_type != "All":
        fnm = fnm + "_" + validation_type
    write_df(sub, fnm)

    temp = metrics.switch_error_func('LE',obs,pred)
    count = 0
    count_fact_2 = 0
    for i in range(len(temp)):     
        if temp[i] >= -1 and temp[i] <= 1:
            count += 1
        if temp[i] >= -np.log10(2) and temp[i] <= np.log10(2):
            count_fact_2 += 1

    fact10 = count / len(temp)
    fact2 = count_fact_2 / len(temp)
    ####METRICS
    fill_flux_metrics_dict(dict, model, energy_key, thresh_key,
        pred_energy_key, pred_thresh_key, figname,
        slope, yint, r_lin, r_log, s_lin, MRatio, MedRatio, ME, MedE, MLE,
        MedLE, MAE, MedAE, MALE, MedALE, MPE, MAPE, MSPE, SMAPE,
        MAR, RMSE, RMSLE, MdSA, fact10, fact2)




def peak_intensity_max_intuitive_metrics(df, dict, model, energy_key,
    thresh_key, validation_type):
    """ Extract the appropriate predictions and calculate metrics
        Peak intensity

    """
    val_type = ["", "All", "First", "Last", "Max", "Mean"]
    if validation_type not in val_type:
        return
    
    peak_key = 'Predicted SEP Peak Intensity Max (Max Flux)'
    time_key = 'Observed SEP Peak Intensity Max (Max Flux) Time'

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
            'Observed SEP Peak Intensity Max (Max Flux)',
            'Observed SEP Peak Intensity Max (Max Flux) Units',
            'Predicted SEP Peak Intensity Max (Max Flux)',
            'Predicted SEP Peak Intensity Max (Max Flux) Units',
            'Peak Intensity Max Match Status']]
    sub = sub.loc[(sub['Peak Intensity Max Match Status'] == 'SEP Event')]
    sub = sub.dropna(subset=['Predicted SEP Peak Intensity Max (Max Flux)', 'Observed SEP Peak Intensity Max (Max Flux)'])
    
    #Drop zero and negative values in predicted or observed peak intensity
    if not sub.empty:
        sub = sub.loc[(sub['Predicted SEP Peak Intensity Max (Max Flux)'] > 0) & (sub['Observed SEP Peak Intensity Max (Max Flux)'] > 0)]

    if not sub.empty:
        sub, doType = extract_flux_forecast_type(sub, thresh_key, peak_key, time_key, validation_type)


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
            'Observed SEP Peak Intensity Max (Max Flux) Time',
            'Observed SEP Peak Intensity Max (Max Flux)',
            'Observed SEP Peak Intensity Max (Max Flux) Units',
            'Predicted SEP Peak Intensity (Onset Peak)',
            'Predicted SEP Peak Intensity (Onset Peak) Units',
            'Peak Intensity Match Status']]
        sub = sub.loc[(sub['Peak Intensity Match Status'] == 'SEP Event')]
        sub = sub.dropna(subset=['Predicted SEP Peak Intensity (Onset Peak)', 'Observed SEP Peak Intensity Max (Max Flux)'])

        #Drop zero and negative flux values in predicted or observed peak intensity
        if not sub.empty:
            sub = sub.loc[(sub['Predicted SEP Peak Intensity (Onset Peak)'] > 0) & (sub['Observed SEP Peak Intensity Max (Max Flux)'] > 0)]

        if sub.empty:
            return

        peak_key = 'Predicted SEP Peak Intensity (Onset Peak)'
        sub, doType = extract_flux_forecast_type(sub, thresh_key, peak_key, time_key, validation_type)
        if sub.empty:
            return
        
        logger.debug("Model " + model + " did not explicitly "
                "include a peak_intensity_max field. Comparing "
                "peak_intensity to observed max flux.")

    if not doType:
        return

    mismatch = bool(sub.iloc[0]['Mismatch Allowed'])
    pred_energy_key = str(sub.iloc[0]['Prediction Energy Channel Key'])
    pred_thresh_key = str(sub.iloc[0]['Prediction Threshold Key'])

    thresh_fnm = make_thresh_fname(thresh_key)

    obs = sub['Observed SEP Peak Intensity Max (Max Flux)'].to_list()
    units = sub.iloc[0]['Observed SEP Peak Intensity Max (Max Flux) Units']
    pred = sub[peak_key].to_list()
 
    if len(obs) > 1:
        #PEARSON CORRELATION
        r_lin, r_log = metrics.switch_error_func('r',obs,pred)
        s_lin= metrics.switch_error_func('spearman',obs,pred)
        
        #LINEAR REGRESSION
        obs_np = np.log10(obs)
        pred_np = np.log10(pred)
        slope, yint = np.polyfit(obs_np, pred_np, 1)

        #Correlation Plot
        corr_plot = plt_tools.correlation_plot(obs,pred,
        "Peak Intensity Max (Max Flux) Correlation", xlabel="Observations",
        ylabel=("Model Predictions (" + str(units) + ")"),
        value="Peak Intensity Max (Max Flux)", use_log = True)

        figname = config.outpath + '/plots/Correlation_peak_intensity_max_' \
                + model + "_" + energy_key.strip() + "_" + thresh_fnm
        if mismatch:
            figname = figname + "_mm"
        if validation_type != "" and validation_type != "All":
            figname = figname + "_" + validation_type
        figname += ".pdf"
        corr_plot.savefig(figname, dpi=300, bbox_inches='tight')
        corr_plot.close()
    else:
        r_lin = None
        r_log = None
        s_lin = None
        slope = None
        yint = None
        figname = ""


    MRatio, MedRatio, ME, MedE, MAE, MedAE, MLE, MedLE, MALE, MedALE, MPE, MAPE,\
    MSPE, SMAPE, MAR, RMSE, RMSLE, MdSA, errors = calc_all_flux_metrics(obs, pred)

    #Add error calculation for individual observation-forecast pairs and
    #write out selections and the metrics associated with each profile match
    if len(errors['Ratio']) != 0:
        sub = sub.assign(**errors)
    fnm = f"peak_intensity_max_selections_{model}_{energy_key.strip()}_{thresh_fnm}"
    if mismatch:
        fnm = fnm + "_mm"
    if validation_type != "" and validation_type != "All":
        fnm = fnm + "_" + validation_type
    write_df(sub, fnm)


    temp = metrics.switch_error_func('LE',obs,pred)
    count = 0
    count_fact_2 = 0
    for i in range(len(temp)):     
        if temp[i] >= -1 and temp[i] <= 1:
            count += 1
        if temp[i] >= -np.log10(2) and temp[i] <= np.log10(2):
            count_fact_2 += 1

    fact10 = count / len(temp)
    fact2 = count_fact_2 / len(temp)
    ####METRICS
    fill_flux_metrics_dict(dict, model, energy_key, thresh_key,
        pred_energy_key, pred_thresh_key, figname,
        slope, yint, r_lin, r_log, s_lin, MRatio, MedRatio, ME, MedE, MLE,
        MedLE, MAE, MedAE, MALE, MedALE, MPE, MAPE, MSPE, SMAPE,
        MAR, RMSE, RMSLE, MdSA, fact10, fact2)



def max_flux_in_pred_win_metrics(df, dict, model, energy_key,
    thresh_key, validation_type):
    """ Extract the appropriate predictions and calculate metrics
        Compare predicted max or onset peak flux to the max observed
        flux in the model's prediction window.

        If a model provides peak_intensity_max, then will compare with
        observed max flux in prediction window. If that field isn't present in
        the prediction, then will compare the peak_intensity.

    """
    val_type = ["", "All"]
    if validation_type not in val_type:
        return

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

    sub = sub.dropna(subset=['Predicted SEP Peak Intensity Max (Max Flux)', 'Observed Max Flux in Prediction Window'])

    #Drop zero and negative values in predicted or observed peak intensity
    if not sub.empty:
        sub = sub.loc[(sub['Predicted SEP Peak Intensity Max (Max Flux)'] > 0) & (sub['Observed Max Flux in Prediction Window'] > 0)]
      
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
            'Observed Max Flux in Prediction Window',
            'Observed Max Flux in Prediction Window Time',
            'Observed Max Flux in Prediction Window Units',
            'Predicted SEP Peak Intensity (Onset Peak)',
            'Predicted SEP Peak Intensity (Onset Peak) Units']]

        sub = sub.dropna(subset=['Predicted SEP Peak Intensity (Onset Peak)', 'Observed Max Flux in Prediction Window'])

        #Drop zero and negative values in predicted or observed peak intensity
        if not sub.empty:
            sub = sub.loc[(sub['Predicted SEP Peak Intensity (Onset Peak)'] > 0) & (sub['Observed Max Flux in Prediction Window'] > 0)]

        if sub.empty:
            return
 
        peak_key = 'Predicted SEP Peak Intensity (Onset Peak)'

        logger.debug("Model " + model + " did not explicitly "
            "include a peak_intensity_max field. Comparing peak_intensity to "
            "observed max flux in the prediction window.")


    mismatch = bool(sub.iloc[0]['Mismatch Allowed'])
    pred_energy_key = str(sub.iloc[0]['Prediction Energy Channel Key'])
    pred_thresh_key = str(sub.iloc[0]['Prediction Threshold Key'])

    thresh_fnm = make_thresh_fname(thresh_key)

    obs = sub['Observed Max Flux in Prediction Window']
    units = sub.iloc[0]['Observed Max Flux in Prediction Window Units']
    pred = sub[peak_key]
 
    if len(obs) > 1:
        #PEARSON CORRELATION
        r_lin, r_log = metrics.switch_error_func('r',obs,pred)
        s_lin = metrics.switch_error_func('spearman',obs,pred)
        
        #LINEAR REGRESSION
        obs_np = np.log10(obs)
        pred_np = np.log10(pred)
        slope, yint = np.polyfit(obs_np, pred_np, 1)

        #Correlation Plot
        corr_plot = plt_tools.correlation_plot(obs,pred,
        "Max Flux in Prediction Window Correlation", xlabel="Observations",
        ylabel=("Model Predictions (" + str(units) + ")"),
        value="Max Flux in Prediction Window", use_log = True)

        figname = config.outpath + '/plots/Correlation_max_flux_in_pred_win_' \
                + model + "_" + energy_key.strip() + "_" + thresh_fnm
        if mismatch:
            figname = figname + "_mm"
        figname += ".pdf"
        corr_plot.savefig(figname, dpi=300, bbox_inches='tight')
        corr_plot.close()
    else:
        r_lin = None
        r_log = None
        s_lin = None
        slope = None
        yint = None
        figname = ""


    MRatio, MedRatio, ME, MedE, MAE, MedAE, MLE, MedLE, MALE, MedALE, MPE, MAPE,\
    MSPE, SMAPE, MAR, RMSE, RMSLE, MdSA, errors = calc_all_flux_metrics(obs, pred)

    #Add error calculation for individual observation-forecast pairs and
    #write out selections and the metrics associated with each profile match
    if len(errors['Ratio']) != 0:
        sub = sub.assign(**errors)
    fnm = f"max_flux_in_pred_win_selections_{model}_{energy_key.strip()}_{thresh_fnm}"
    if mismatch:
        fnm = fnm + "_mm"
    if validation_type != "" and validation_type != "All":
        fnm = fnm + "_" + validation_type
    write_df(sub, fnm)

    temp = metrics.switch_error_func('LE',obs,pred)
    count = 0
    count_fact_2 = 0
    for i in range(len(temp)):     
        if temp[i] >= -1 and temp[i] <= 1:
            count += 1
        if temp[i] >= -np.log10(2) and temp[i] <= np.log10(2):
            count_fact_2 += 1

    fact10 = count / len(temp)
    fact2 = count_fact_2 / len(temp)
    ####METRICS
    fill_flux_metrics_dict(dict, model, energy_key, thresh_key,
        pred_energy_key, pred_thresh_key, figname,
        slope, yint, r_lin, r_log, s_lin, MRatio, MedRatio, ME, MedE, MLE,
        MedLE, MAE, MedAE, MALE, MedALE, MPE, MAPE, MSPE, SMAPE,
        MAR, RMSE, RMSLE, MdSA, fact10, fact2)



def fluence_intuitive_metrics(df, dict, model, energy_key,
    thresh_key, validation_type):
    """ Extract the appropriate predictions and calculate metrics
        Fluence

    """
    val_type = ["", "All", "First", "Last", "Max", "Mean"]
    if validation_type not in val_type:
        return
    
    #Select rows to calculate metrics
    sub = df.loc[(df['Model'] == model) & (df['Energy Channel Key'] ==
        energy_key) & (df['Threshold Key'] == thresh_key)]
    
    sub = sub[['Model','Energy Channel Key', 'Threshold Key',
            'Mismatch Allowed',
            'Prediction Energy Channel Key', 'Prediction Threshold Key',
            'Forecast Source',
            'Prediction Window Start', 'Prediction Window End',
            'Observed SEP Threshold Crossing Time',
            'Observed SEP End Time',
            'Observed SEP Fluence',
            'Observed SEP Fluence Units',
            'Predicted SEP Fluence',
            'Predicted SEP Fluence Units',
            'Fluence Match Status']]
    sub = sub.loc[(sub['Fluence Match Status'] == 'SEP Event')]  
    sub = sub.dropna(subset=['Predicted SEP Fluence','Observed SEP Fluence'])

    #Drop zero and negative values in predicted or observed peak intensity
    if not sub.empty:
        sub = sub.loc[(sub['Predicted SEP Fluence'] > 0) & (sub['Observed SEP Fluence'] > 0)]

    if sub.empty:
        return

    sub, doType = extract_flux_forecast_type(sub, thresh_key, 'Predicted SEP Fluence', 'Observed SEP End Time', validation_type)
    if sub.empty or not doType:
        return

    mismatch = bool(sub.iloc[0]['Mismatch Allowed'])
    pred_energy_key = str(sub.iloc[0]['Prediction Energy Channel Key'])
    pred_thresh_key = str(sub.iloc[0]['Prediction Threshold Key'])

    thresh_fnm = make_thresh_fname(thresh_key)

    obs = sub['Observed SEP Fluence'].to_list()
    pred = sub['Predicted SEP Fluence'].to_list()
    units = sub.iloc[0]['Observed SEP Fluence Units']

    if len(obs) > 1:
        #PEARSON CORRELATION
        r_lin, r_log = metrics.switch_error_func('r',obs,pred)
        s_lin = metrics.switch_error_func('spearman',obs,pred)
        
        #LINEAR REGRESSION
        obs_np = np.log10(obs)
        pred_np = np.log10(pred)
        slope, yint = np.polyfit(obs_np, pred_np, 1)

        #Correlation Plot
        corr_plot = plt_tools.correlation_plot(obs,pred,
        "Fluence Correlation", xlabel="Observations",
        ylabel=("Model Predictions (" + str(units) + ")"),
        use_log = True)

        figname = config.outpath + '/plots/Correlation_fluence_' + model + "_" \
                + energy_key.strip() + "_" + thresh_fnm
        if mismatch:
            figname = figname + "_mm"
        if validation_type != "" and validation_type != "All":
            figname = figname + "_" + validation_type 
        figname += ".pdf"
        corr_plot.savefig(figname, dpi=300, bbox_inches='tight')
        corr_plot.close()
    else:
        r_lin = None
        r_log = None
        s_lin = None
        slope = None
        yint = None
        figname = ""


    MRatio, MedRatio, ME, MedE, MAE, MedAE, MLE, MedLE, MALE, MedALE, MPE, MAPE,\
    MSPE, SMAPE, MAR, RMSE, RMSLE, MdSA, errors = calc_all_flux_metrics(obs, pred)

    #Add error calculation for individual observation-forecast pairs and
    #write out selections and the metrics associated with each profile match
    if len(errors['Ratio']) != 0:
        sub = sub.assign(**errors)
    fnm = f"fluence_selections_{model}_{energy_key.strip()}_{thresh_fnm}"
    if mismatch:
        fnm = fnm + "_mm"
    if validation_type != "" and validation_type != "All":
        fnm = fnm + "_" + validation_type
    write_df(sub, fnm)

    temp = metrics.switch_error_func('LE',obs,pred)
    count = 0
    count_fact_2 = 0
    for i in range(len(temp)):     
        if temp[i] >= -1 and temp[i] <= 1:
            count += 1
        if temp[i] >= -np.log10(2) and temp[i] <= np.log10(2):
            count_fact_2 += 1

    fact10 = count / len(temp)
    fact2 = count_fact_2 / len(temp)
    ####METRICS
    fill_flux_metrics_dict(dict, model, energy_key, thresh_key,
        pred_energy_key, pred_thresh_key, figname,
        slope, yint, r_lin, r_log, s_lin, MRatio, MedRatio, ME, MedE, MLE,
        MedLE, MAE, MedAE, MALE, MedALE, MPE, MAPE, MSPE, SMAPE,
        MAR, RMSE, RMSLE, MdSA, fact10, fact2)




def threshold_crossing_intuitive_metrics(df, dict, model, energy_key,
    thresh_key, validation_type):
    """ Extract the appropriate predictions and calculate metrics
        Threshold Crossing

    """
    val_type = ["", "All", "First", "Last"]
    if validation_type not in val_type:
        return
    
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
    sub = sub.dropna(subset='Predicted SEP Threshold Crossing Time')

    if not sub.empty:
        sub = sub.dropna(subset='Observed SEP Threshold Crossing Time')

    if sub.empty:
        return

    sub, doType = extract_time_forecast_type(sub, 'Predicted SEP Threshold Crossing Time', validation_type)
    if sub.empty or not doType:
        return

    mismatch = bool(sub.iloc[0]['Mismatch Allowed'])
    pred_energy_key = str(sub.iloc[0]['Prediction Energy Channel Key'])
    pred_thresh_key = str(sub.iloc[0]['Prediction Threshold Key'])

    thresh_fnm = make_thresh_fname(thresh_key)
    fnm = "threshold_crossing_time_selections_" + model + "_" \
            + energy_key.strip() + "_" + thresh_fnm
    if mismatch:
        fnm = fnm + "_mm"
    if validation_type != "" and validation_type != "All":
        fnm = fnm + "_" + validation_type
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
    

def start_time_intuitive_metrics(df, dict, model, energy_key, thresh_key,
    validation_type):
    """ Extract the appropriate predictions and calculate metrics
        Start Time

    """
    val_type = ["", "All", "First", "Last"]
    if validation_type not in val_type:
        return
    
    #Select rows to calculate metrics
    sub = df.loc[(df['Model'] == model) & (df['Energy Channel Key'] ==
        energy_key) & (df['Threshold Key'] == thresh_key)]

    sub = sub[['Model','Energy Channel Key', 'Threshold Key',
            'Mismatch Allowed',
            'Prediction Energy Channel Key', 'Prediction Threshold Key',
            'Forecast Source',
            'Prediction Window Start', 'Prediction Window End',
            'Observed SEP Threshold Crossing Time',
            'Observed SEP Start Time',
            'Predicted SEP Start Time',
            'Start Time Match Status']]
    sub = sub.loc[(sub['Start Time Match Status'] == 'SEP Event')]
    sub = sub.dropna(subset='Predicted SEP Start Time')

    if not sub.empty:
        sub = sub.dropna(subset='Observed SEP Start Time')

    if sub.empty:
        return

    sub, doType = extract_time_forecast_type(sub, 'Predicted SEP Start Time', validation_type)
    if sub.empty or not doType:
        return

    mismatch = bool(sub.iloc[0]['Mismatch Allowed'])
    pred_energy_key = str(sub.iloc[0]['Prediction Energy Channel Key'])
    pred_thresh_key = str(sub.iloc[0]['Prediction Threshold Key'])

    thresh_fnm = make_thresh_fname(thresh_key)
    fnm = "start_time_selections_" + model + "_" + energy_key.strip() \
            + "_" + thresh_fnm
    if mismatch:
        fnm = fnm + "_mm"
    if validation_type != "" and validation_type != "All":
        fnm = fnm + "_" + validation_type
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
    thresh_key, validation_type):
    """ Extract the appropriate predictions and calculate metrics
        End Time

    """
    val_type = ["", "All", "First", "Last"]
    if validation_type not in val_type:
        return
    
    #Select rows to calculate metrics
    sub = df.loc[(df['Model'] == model) & (df['Energy Channel Key'] ==
        energy_key) & (df['Threshold Key'] == thresh_key)]

    sub = sub[['Model','Energy Channel Key', 'Threshold Key',
            'Mismatch Allowed',
            'Prediction Energy Channel Key', 'Prediction Threshold Key',
            'Forecast Source',
            'Prediction Window Start', 'Prediction Window End',
            'Observed SEP Threshold Crossing Time',
            'Observed SEP End Time',
            'Predicted SEP End Time',
            'End Time Match Status']]
    sub = sub.loc[(sub['End Time Match Status'] == 'SEP Event')]
    sub = sub.dropna(subset='Predicted SEP End Time')

    if not sub.empty:
        sub = sub.dropna(subset='Observed SEP End Time')

    if sub.empty:
        return

    sub, doType = extract_time_forecast_type(sub, 'Predicted SEP End Time', validation_type)
    if sub.empty or not doType:
        return

    mismatch = bool(sub.iloc[0]['Mismatch Allowed'])
    pred_energy_key = str(sub.iloc[0]['Prediction Energy Channel Key'])
    pred_thresh_key = str(sub.iloc[0]['Prediction Threshold Key'])

    thresh_fnm = make_thresh_fname(thresh_key)
    fnm = "end_time_selections_" + model + "_" + energy_key.strip() + "_" + thresh_fnm
    if mismatch:
        fnm = fnm + "_mm"
    if validation_type != "" and validation_type != "All":
        fnm = fnm + "_" + validation_type
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
 


def duration_intuitive_metrics(df, dict, model, energy_key, thresh_key,
    validation_type):
    """ Extract the appropriate predictions and calculate metrics
        Duration

    """
    val_type = ["", "All", "First", "Last"]
    if validation_type not in val_type:
        return
    
    #Select rows to calculate metrics
    sub = df.loc[(df['Model'] == model) & (df['Energy Channel Key'] ==
        energy_key) & (df['Threshold Key'] == thresh_key)]

    sub = sub[['Model','Energy Channel Key', 'Threshold Key',
            'Mismatch Allowed',
            'Prediction Energy Channel Key', 'Prediction Threshold Key',
            'Forecast Source',
            'Prediction Window Start', 'Prediction Window End',
            'Observed SEP Threshold Crossing Time',
            'Observed SEP End Time',
            'Observed SEP Duration',
            'Predicted SEP Duration',
            'Duration Match Status',
            'Predicted SEP End Time']]
    sub = sub.loc[(sub['Duration Match Status'] == 'SEP Event')]
    sub = sub.dropna(subset='Predicted SEP Duration')

    if not sub.empty:
        sub = sub.dropna(subset='Observed SEP Duration')

    if sub.empty:
        return

    sub, doType = extract_time_forecast_type(sub, 'Predicted SEP End Time', validation_type)
    if sub.empty or not doType:
        return

    mismatch = bool(sub.iloc[0]['Mismatch Allowed'])
    pred_energy_key = str(sub.iloc[0]['Prediction Energy Channel Key'])
    pred_thresh_key = str(sub.iloc[0]['Prediction Threshold Key'])

    thresh_fnm = make_thresh_fname(thresh_key)
    fnm = "duration_selections_" + model + "_" + energy_key.strip() \
            + "_" + thresh_fnm
    if mismatch:
        fnm = fnm + "_mm"
    if validation_type != "" and validation_type != "All":
        fnm = fnm + "_" + validation_type
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
    thresh_key, validation_type):
    """ Extract the appropriate predictions and calculate metrics
        Peak Intensity Time

    """
    val_type = ["", "All", "First", "Last"]
    if validation_type not in val_type:
        return
    
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
            'Peak Intensity Match Status',
            'Predicted SEP Peak Intensity (Onset Peak)']]
    sub = sub.loc[(sub['Peak Intensity Match Status'] == 'SEP Event')]
    sub = sub.dropna(subset='Predicted SEP Peak Intensity (Onset Peak) Time')

    if not sub.empty:
        sub = sub.dropna(subset='Observed SEP Peak Intensity (Onset Peak) Time')

    if sub.empty:
        return


    if validation_type != "Max":
        sub, doType = extract_time_forecast_type(sub, 'Predicted SEP Peak Intensity (Onset Peak) Time', validation_type)
    if validation_type == "Max":
        sub, doType = extract_flux_forecast_type(sub, thresh_key, 'Predicted SEP Peak Intensity (Onset Peak)', 'Observed SEP Peak Intensity (Onset Peak) Time', validation_type)

    if sub.empty or not doType:
        return

    mismatch = bool(sub.iloc[0]['Mismatch Allowed'])
    pred_energy_key = str(sub.iloc[0]['Prediction Energy Channel Key'])
    pred_thresh_key = str(sub.iloc[0]['Prediction Threshold Key'])

    thresh_fnm = make_thresh_fname(thresh_key)
    fnm = "peak_intensity_time_selections_" + model + "_" \
            + energy_key.strip() + "_" + thresh_fnm
    if mismatch:
        fnm = fnm + "_mm"
    if validation_type != "" and validation_type != "All":
        fnm = fnm + "_" + validation_type
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
    
    date_str = '{:d}{:02d}{:02d}T{:02d}{:02d}{:02d}'.format(year, month, day, hour, min, sec)
    
    return date_str



def peak_intensity_max_time_intuitive_metrics(df, dict, model, energy_key,
    thresh_key, validation_type):
    """ Extract the appropriate predictions and calculate metrics
        Peak Intensity Max Time

    """
    val_type = ["", "All", "First", "Last"]
    if validation_type not in val_type:
        return
    
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
            'Peak Intensity Max Match Status',
            'Predicted SEP Peak Intensity Max (Max Flux)']]
    sub = sub.loc[(sub['Peak Intensity Max Match Status'] == 'SEP Event')]
    sub = sub.dropna(subset='Predicted SEP Peak Intensity Max (Max Flux) Time')

    if not sub.empty:
        sub = sub.dropna(subset='Observed SEP Peak Intensity Max (Max Flux) Time')

    if sub.empty:
        return


    if validation_type != "Max":
        sub, doType = extract_time_forecast_type(sub, 'Predicted SEP Peak Intensity Max (Max Flux) Time', validation_type)
    if validation_type == "Max":
        sub, doType = extract_flux_forecast_type(sub, thresh_key, 'Predicted SEP Peak Intensity Max (Max Flux)', 'Observed SEP Peak Intensity Max (Max Flux) Time', validation_type)

    if sub.empty or not doType:
        return

    mismatch = bool(sub.iloc[0]['Mismatch Allowed'])
    pred_energy_key = str(sub.iloc[0]['Prediction Energy Channel Key'])
    pred_thresh_key = str(sub.iloc[0]['Prediction Threshold Key'])

    thresh_fnm = make_thresh_fname(thresh_key)
    fnm = "peak_intensity_max_time_selections_" + model + "_" \
            + energy_key.strip() + "_" + thresh_fnm
    if mismatch:
        fnm = fnm + "_mm"
    if validation_type != "" and validation_type != "All":
        fnm = fnm + "_" + validation_type
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
    thresh_key, validation_type):
    """ Extract the appropriate predictions and calculate metrics
        Time Profile

    """
    val_type = ["", "All"]#, "First", "Last"]
    if validation_type not in val_type: #not implemented in this subroutine
        return
    
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
    sub = sub.dropna(subset=['Predicted Time Profile', 'Observed Time Profile'])

    if sub.empty:
        return
        
    mismatch = bool(sub.iloc[0]['Mismatch Allowed'])
    pred_energy_key = str(sub.iloc[0]['Prediction Energy Channel Key'])
    pred_thresh_key = str(sub.iloc[0]['Prediction Threshold Key'])

    thresh_fnm = make_thresh_fname(thresh_key)

    obs_st = sub['Observed SEP Start Time'].to_list()
    obs_et = sub['Observed SEP End Time'].to_list()
    obs_profs = sub['Observed Time Profile'].to_list()
    pred_st = sub['Predicted SEP Start Time'].to_list()
    pred_et = sub['Predicted SEP End Time'].to_list()
    pred_profs = sub['Predicted Time Profile'].to_list()
    pred_paths = sub['Forecast Path'].to_list()
    model_names = sub['Model'].to_list()
    energy_chan = sub['Energy Channel Key'].to_list()

    logger.debug("Extracted SEP start and end times")
    logger.debug(obs_st)
    logger.debug(obs_et)

    #For metrics calculations
    sepratio = []
    sepmedratio = []
    sepE = []
    sepmedE = []
    sepAE = []
    sepmedAE = []
    sepLE = []
    sepmedLE = []
    sepALE = []
    sepmedALE = []
    sepAPE = []
    sepMAR = [] #Mean Accuracy Ratio
    sepRMSE = []
    sepRMSLE = []
    sepMdSA = []
    sepPE = []
    sepSPE = []
    sepSAPE = []
    sepRlin = []
    sepRlog= []
    sepSlin = []
    sepfact10 = []
    sepfact2 = []

    tp_plotnames = ""
    figname = ""
    tpfigname = ""
    for i in range(len(obs_profs)):
        #Variables to calculate a mean or median metric across an individual time profile
        E1 = np.nan
        medE1 = np.nan
        AE1 = np.nan
        medAE1 = np.nan
        LE1 = np.nan
        medLE1 = np.nan
        ALE1 = np.nan
        medALE1 = np.nan
        MAR1 = np.nan
        RMSE1 = np.nan
        MdSA1 = np.nan
        PE1 = np.nan
        SPE1 = np.nan
        SAPE1 = np.nan
        r_lin = np.nan
        r_log = np.nan
        s_lin = np.nan
        ratio1 = np.nan
        fact10 = np.nan
        fact2 = np.nan
        
        logger.debug("Comparing time profile of " + pred_profs[i] + " to observations.")
        all_obs_dates = []
        all_obs_flux = []
        #Read in and combine time profiles of observations inside
        #prediction window
        obs_fnames = obs_profs[i].strip().split(",")
        logger.debug("Comparing to OBSERVED TIME PROFILES: " + str(obs_fnames))
        for j in range(len(obs_fnames)):
            dt, flx = profile.read_single_time_profile(obs_fnames[j])
            all_obs_dates.append(dt)
            all_obs_flux.append(flx)
        
        obs_dates, obs_flux = profile.combine_time_profiles(all_obs_dates,
            all_obs_flux)
        pred_dates, pred_flux = profile.read_single_time_profile(pred_profs[i])
        if not pred_flux:
            #Remove row for bad time profile from sub
            sub = sub[sub['Predicted Time Profile'] != pred_profs[i]]
            continue
        
        #If all the flux values are zero, then will make the zip lines crash.
        test = [ii for ii in range(len(pred_flux)) if pred_flux[ii] == 0]
        if len(test) == len(pred_flux):
            #Remove row for bad time profile from sub
            sub = sub[sub['Predicted Time Profile'] != pred_profs[i]]
            continue
        
        #Remove zeros
        obs_flux, obs_dates = zip(*filter(lambda x:x[0]>0.0, zip(obs_flux, obs_dates)))
        pred_flux, pred_dates = zip(*filter(lambda x:x[0]>0.0, zip(pred_flux, pred_dates)))
        
        #If predicted time profile is all zeros
        if not pred_flux:
            #Remove row for bad time profile from sub
            sub = sub[sub['Predicted Time Profile'] != pred_profs[i]]
            continue
        
        #Interpolate observed time profile onto predicted timestamps
        obs_flux_interp = profile.interp_timeseries(obs_dates, obs_flux, "log",
            pred_dates)
        
        #Trim the time profiles to the observed start and end time or
        #predicted start and end times - modified after SPHINX TP 2025
        trim_st = obs_st[i]
        trim_et = obs_et[i]
        if not pd.isnull(pred_st[i]):
            trim_st = max(obs_st[i],pred_st[i])
        if not pd.isnull(pred_et[i]):
            time_et = min(obs_et[i], pred_et[i])
        logger.debug("Trimming between " + str(trim_st) + " and " + str(trim_et))
        trim_pred_dates, trim_pred_flux = profile.trim_profile(trim_st,
                trim_et, pred_dates, pred_flux)
        trim_obs_dates, trim_obs_flux = profile.trim_profile(trim_st,
                trim_et, pred_dates,  obs_flux_interp)
        
        
        #PLOT TIME PROFILE TO CHECK
        date = [obs_dates, trim_obs_dates, pred_dates, trim_pred_dates]
        values = [obs_flux, trim_obs_flux, pred_flux, trim_pred_flux]
        labels=["Observations", "Interp Trimmed Obs", model_names[i], "Trimmed " + model_names[i]]
        str_date = date_to_string(pred_dates[0])
        title = model_names[i] + ", " + energy_chan[i] + " Time Profile"
        tpfigname = config.outpath + "/plots/Time_Profile_" + model_names[i] \
            + "_" + energy_chan[i] + "_" + thresh_fnm  + "_" + str_date
        if mismatch:
            tpfigame = tpfigname + "_mm"
        tpfigname += ".pdf" 
        if tp_plotnames == "":
            tp_plotnames = tpfigname
        else:
            tp_plotnames += ";" + tpfigname
            
            
        plt_tools.plot_time_profile(date, values, labels,
        title=title, x_label="Date", y_min=1e-7, y_max=1e5,
        y_label="Particle Intensity",
        date_format="year", showplot=False,
        closeplot=True, saveplot=True, figname=tpfigname)
 
        #Check for None and Zero values and remove
        if not trim_pred_flux or not trim_obs_flux:
            #Remove row for bad time profile from sub
            sub = sub[sub['Predicted Time Profile'] != pred_profs[i]]
            continue

        obs, pred = metrics.remove_none(trim_obs_flux,trim_pred_flux)
        obs, pred = metrics.remove_zero(obs, pred)
        if not obs or not pred:
            #Remove row for bad time profile from sub
            sub = sub[sub['Predicted Time Profile'] != pred_profs[i]]
            continue
        

        if len(obs) >= 1 and len(pred) >= 1:
            ratio1 = statistics.mean(metrics.switch_error_func('Ratio',obs,pred))
            medratio1 = statistics.median(metrics.switch_error_func('Ratio',obs,pred))
            E1 = statistics.mean(metrics.switch_error_func('E',obs,pred))
            medE1 = statistics.median(metrics.switch_error_func('E',obs,pred))
            AE1 = statistics.mean(metrics.switch_error_func('AE',obs,pred))
            medAE1 = statistics.median(metrics.switch_error_func('AE',obs,pred))
            LE1 = statistics.mean(metrics.switch_error_func('LE',obs,pred))
            medLE1 = statistics.median(metrics.switch_error_func('LE',obs,pred))
            ALE1 = statistics.mean(metrics.switch_error_func('ALE',obs,pred))
            medALE1 = statistics.median(metrics.switch_error_func('ALE',obs,pred))
            APE1 = statistics.mean(metrics.switch_error_func('APE',obs,pred))
            MAR1 =  statistics.mean(metrics.switch_error_func('APE',obs,pred))#Mean Accuracy Ratio
            RMSE1 = metrics.switch_error_func('RMSE',obs,pred)
            RMSLE1 = metrics.switch_error_func('RMSLE',obs,pred)
            MdSA1 = metrics.switch_error_func('MdSA',obs,pred)
            PE1 = statistics.mean(metrics.switch_error_func('PE',obs,pred))
            SPE1 = statistics.mean(metrics.switch_error_func('SPE',obs,pred))
            SAPE1 = statistics.mean(metrics.switch_error_func('SAPE',obs,pred))

            #Fluxes within a factor of 10 and 2
            temp = metrics.switch_error_func('LE',obs,pred)
            count = 0
            count_fact_2 = 0
            for j in range(len(temp)):
                if temp[j] >= -1 and temp[j] <= 1:
                    count += 1
                if temp[j] >= -np.log10(2) and temp[j] <= np.log10(2):
                    count_fact_2 += 1

            
            fact10 = count / len(temp)
            fact2 = count_fact_2 / len(temp)

            #In some cases, the predicted time profile can be constant, i.e.
            #all the same value. This will not allow an appropriate calculation
            #of correlation coefficients
            is_const = False
            indices = [k for k, x in enumerate(pred) if x == pred[0]]
            if len(indices) == len(pred):
                is_const = True
            
            if len(obs) > 1 and not is_const:
                #PEARSON CORRELATION
                r_lin, r_log = metrics.switch_error_func('r',obs,pred)
                s_lin = metrics.switch_error_func('spearman',obs,pred)
                
                #LINEAR REGRESSION
                obs_np = np.log10(obs)
                pred_np = np.log10(pred)
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
                figname += ".pdf"
                corr_plot.savefig(figname, dpi=300, bbox_inches='tight')
                corr_plot.close()

        sepratio.append(ratio1)
        sepmedratio.append(medratio1)
        sepE.append(E1)
        sepmedE.append(medE1)
        sepAE.append(AE1)
        sepmedAE.append(medAE1)
        sepLE.append(LE1)
        sepmedLE.append(medLE1)
        sepALE.append(ALE1)
        sepmedALE.append(medALE1)
        sepAPE.append(APE1)
        sepMAR.append(MAR1) #Mean Accuracy Ratio
        sepRMSE.append(RMSE1)
        sepRMSLE.append(RMSLE1)
        sepMdSA.append(MdSA1)
        sepPE.append(PE1)
        sepSPE.append(SPE1)
        sepSAPE.append(SAPE1)
        sepRlin.append(r_lin)
        sepRlog.append(r_log)
        sepSlin.append(s_lin)
        sepfact10.append(fact10)
        sepfact2.append(fact2)

    #Append sub with the metrics for each observed and predicted time profile match
    errors = {'Mean Ratio': sepratio, 'Median Ratio': sepmedratio, 'Mean Error': sepE, 'Median Error': sepmedE,
              'Mean Absolute Error': sepAE, 'Median Absolute Error': sepmedAE,
              'Mean Log Error': sepLE, 'Median Log Error': sepmedLE,
              'Mean Absolute Log Error': sepALE, 'Median Absolute Log Error': sepmedALE,
              'Absolute Percent Error': sepAPE, 'Mean Accuracy Ratio': sepMAR, 'Root Mean Square Error': sepRMSE,
              'Root Mean Square Log Error': sepRMSLE, 'Percent Error': sepPE, 'Symmetric Percent Error': sepSPE, 
              'Symmetric Absolute Perecent Error': sepSAPE,'Pearson Correlation Coefficient (linear)': sepRlin, 
              'Pearson Correlation Coefficient (log)': sepRlog, 'Spearman Correlation Coefficient': sepSlin, 
              'Within a Factor of 10': sepfact10, 'Within a Factor of 2': sepfact2}
    
    if len(errors['Mean Ratio']) != 0:
        sub = sub.assign(**errors)
    
    #Write out selections and the metrics associated with each profile match
    fnm = f"time_profile_selections_{model}_{energy_key.strip()}_{thresh_fnm}"
    if mismatch:
        fnm = fnm + "_mm"
    write_df(sub, fnm)

    #Calculate mean of metrics for all time profiles
    MRatio = None
    MedRatio = None
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
    MPE = None
    MSPE = None
    SMAPE = None
    Rlin = None
    Rlog = None
    Slin = None
    slope = None
    yint = None
    fact10_ = None
    fact2_ = None
    
    if len(sepE) > 1:
        MRatio = statistics.mean(sepratio)
        MedRatio = statistics.median(sepmedratio)
        ME = statistics.mean(sepE)
        MedE = statistics.median(sepmedE)
        MAE = statistics.mean(sepAE)
        MedAE = statistics.median(sepmedAE)
        MLE = statistics.mean(sepLE)
        MedLE = statistics.median(sepmedLE)
        MALE = statistics.mean(sepALE)
        MedALE = statistics.median(sepmedALE)
        MAPE = statistics.mean(sepAPE)
        MAR = statistics.mean(sepMAR) #Mean Accuracy Ratio
        RMSE = statistics.mean(sepRMSE)
        RMSLE = statistics.mean(sepRMSLE)
        MdSA = statistics.mean(sepMdSA)
        MPE = statistics.mean(sepPE)
        MSPE = statistics.mean(sepSPE)
        SMAPE = statistics.mean(sepSAPE)
        fact10_ = statistics.mean(sepfact10)
        fact2_ = statistics.mean(sepfact2)

    if len(sepE) == 1:
        MRatio = sepratio[0]
        MedRatio = sepmedratio[0]
        ME = sepE[0]
        MedE = sepmedE[0]
        MAE = sepAE[0]
        MedAE = sepmedAE[0]
        MLE = sepLE[0]
        MedLE = sepmedLE[0]
        MALE = sepALE[0]
        MedALE = sepmedALE[0]
        MAPE = sepAPE[0]
        MAR = sepMAR[0] #Mean Accuracy Ratio
        RMSE = sepRMSE[0]
        RMSLE = sepRMSLE[0]
        MdSA = sepMdSA[0]
        MPE = sepPE[0]
        MSPE = sepSPE[0]
        SMAPE = sepSAPE[0]
        fact10_ = sepfact10[0]
        fact2_ = sepfact2[0]

    if len(sepRlin) > 1:
        Rlin = statistics.mean(sepRlin)
        Rlog = statistics.mean(sepRlog)
        Slin = statistics.mean(sepSlin)
        
    if len(sepRlin) == 1:
        Rlin = sepRlin[0]
        Rlog = sepRlog[0]
        Slin = sepSlin[0]


   
    
    ####METRICS
    fill_flux_metrics_dict(dict, model, energy_key, thresh_key,
        pred_energy_key, pred_thresh_key, figname,
        slope, yint, Rlin, Rlog, Slin, MRatio, MedRatio, ME, MedE, MLE,
        MedLE, MAE, MedAE, MALE, MedALE, MPE, MAPE, MSPE, SMAPE,
        MAR, RMSE, RMSLE, MdSA, fact10_, fact2_, tp_plotnames)



def identify_first_not_clear_forecast_strict(df):
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
    idx = None
    #Search in reverse order checking of forecast is False All Clear
    #As soon as hit a True All Clear, exit
    for i in range(len(all_clear)-1,-1,-1):
        if all_clear[i] is None:
            break
        if all_clear[i] == False:
            idx = i
        if all_clear[i] == True:
            break
    
    return idx


def identify_first_time_forecast_strict(df, pred_key):
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
        if pd.isnull(times[i]):
            break
        else:
            idx = i

    return idx


def identify_first_flux_forecast_strict(df, pred_key, thresh_key):
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
        if pd.isnull(fluxes[i]):
            break
        elif fluxes[i] >= thresh:
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
                'Trigger Advance Time',
                obs_key, pred_key, match_key]]
    else:
        sub = sub[['Model','Energy Channel Key', 'Threshold Key',
                'Mismatch Allowed',
                'Prediction Energy Channel Key', 'Prediction Threshold Key',
                'Forecast Source', 'Forecast Issue Time',
                'Prediction Window Start', 'Prediction Window End',
                'Observed SEP Threshold Crossing Time',
                'Observed SEP Start Time', 
                'Trigger Advance Time',
                pred_key, match_key]]

    sub = sub.loc[sub[match_key] == "SEP Event"]

    return sub


def awt_metrics(df, dict, model, energy_key, thresh_key, validation_type):
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
    
        Using the "First" mode with AWT will calculate the warning from
        the very first forecast that predicted the SEP event will occur,
        regardless of whether the forecast switched back to "clear" in
        subsequent forecasts. This may be appropriate for some types of
        forecasts (e.g. peak intensity) and misleading for others (e.g.
        probability or all clear). Use with care.
    
    """
    val_type = ["", "All", "First"]
    if validation_type not in val_type:
        return

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

    #AWT is always compared to Observed SEP Threshold Crossing Time.
    #obs_ref allows for the calculation of AWT wrt
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
 #                {'pred_key': 'Predicted SEP Peak Intensity (Onset Peak)',
 #                   'match_key': 'Peak Intensity Match Status',
 #                   'obs_key': 'Observed SEP Peak Intensity Max (Max Flux) Time'},
                 {'pred_key': 'Predicted SEP Peak Intensity Max (Max Flux)',
                    'match_key': 'Peak Intensity Max Match Status',
                    'obs_key': 'Observed SEP Peak Intensity Max (Max Flux) Time'},
#                {'pred_key': 'Predicted Point Intensity',
#                    'match_key': 'All Clear Match Status',
#                    'obs_key': ''},
                 {'pred_key': 'Predicted SEP End Time',
                    'match_key': 'End Time Match Status',
                    'obs_key': 'Observed SEP End Time'}]


    for ftype in forecasts:
        #Extract relevant fields only for forecasts associated with SEP Events
        sub = extract_awt_sub(df, model, energy_key, thresh_key, ftype['pred_key'],
            ftype['match_key'], ftype['obs_key'])
       
        #Create an empty dataframe with the same columns plus AWT info
        cols = sub.columns.to_list()
        cols.append('AWT to Observed SEP Threshold Crossing Time')
#        cols.append('AWT to Observed SEP Start Time')
        if ftype['obs_key'] == '':
            cols.append('AWT Efficiency to Observed SEP Threshold Crossing Time')

        if ftype['obs_key'] != '':
            cols.append('AWT to ' + ftype['obs_key'])
        sel_df = pd.DataFrame(columns=cols) #Selected forecasts and AWT results
       
        #Make a list of the unique SEP events in the df
        sep_events = resume.identify_unique(sub, 'Observed SEP Threshold Crossing Time')
        
        #If only one forecast per SEP event, then no need to do First
        #calculation.
        if validation_type == "First":
            if len(sep_events) == len(sel_df.index):
                #Fill this set of fields in the AWT dict with None
#                time_keys = ['Observed SEP Threshold Crossing Time','Observed SEP Start Time']
                time_keys = ['Observed SEP Threshold Crossing Time']
                if ftype['obs_key'] != '':
                    time_keys.append(ftype['obs_key'])
                for key in time_keys:
                    mean_key = "Mean AWT for " + ftype['pred_key'] + " to " + key
                    median_key = "Median AWT for " + ftype['pred_key'] + " to " + key
                    dict[mean_key].append(None)
                    dict[median_key].append(None)
                if ftype['obs_key'] == '':
                    dict['Mean AWT Efficiency for ' + ftype['pred_key'] + ' to ' + time_keys[0]].append(np.nan)
                
                continue
        
        #For each SEP event, identify the first forecast for that SEP event.
        for sep in sep_events:
            sep_sub = sub.loc[sub['Observed SEP Threshold Crossing Time'] == sep]
            
            idx = None
            if 'All Clear' in ftype['pred_key']:
                if validation_type == "First":
                    sep_sub = extract_all_clear_forecast_type(sep_sub, validation_type)
 
                idx = identify_first_not_clear_forecast_strict(sep_sub)
        
            if 'Time' in ftype['pred_key']:
                if validation_type == "First":
                    sep_sub, doType = extract_time_forecast_type(sep_sub, ftype['pred_key'], validation_type)

                idx = identify_first_time_forecast_strict(sep_sub, ftype['pred_key'])
                
            if 'Peak' in ftype['pred_key'] or 'Point' in ftype['pred_key']:
                if validation_type == "First":
                    sep_sub, doType = extract_flux_forecast_type(sep_sub, thresh_key, ftype['pred_key'], ftype['obs_key'], validation_type)

                idx = identify_first_flux_forecast_strict(sep_sub, ftype['pred_key'],
                        thresh_key)
            
            if idx == None:
                continue
            if sep_sub.empty:
                continue
            
            row = sep_sub.iloc[idx].to_list()
            #Calculate AWT
            issue_time = sep_sub.iloc[idx]['Forecast Issue Time']

            if pd.isnull(issue_time):
                continue
            
            tct = sep_sub.iloc[idx]['Observed SEP Threshold Crossing Time']
            tc_awt = (tct - issue_time).total_seconds()/(60.*60.) #hours
            #If issue time is more than 7 days later than the threshold crossing time,
            #the assume this is not a realistic issue time and ignore
            if tc_awt < -7.*24.:
                tc_awt = np.nan
            row.append(tc_awt)
            
#            st = sep_sub.iloc[idx]['Observed SEP Start Time']
#            st_awt = (st - issue_time).total_seconds()/(60.*60.)
            #If issue time is more than 7 days later than the start time,
            #the assume this is not a realistic issue time and ignore
#            if st_awt < -7.*24.:
#                st_awt = np.nan
#            row.append(st_awt)

            if ftype['obs_key'] != '':
                tm = sep_sub.iloc[idx][ftype['obs_key']]
                obs_awt = (tm - issue_time).total_seconds()/(60.*60.)
                #If issue time is more than 7 days later than the observed time,
                #the assume this is not a realistic issue time and ignore
                if obs_awt < -7.*24.:
                    obs_awt = np.nan
                row.append(obs_awt)

            if ftype['obs_key'] == '': 
                tc_tat = sep_sub.iloc[idx]["Trigger Advance Time"]
                if pd.isnull(tc_awt) or tc_tat == 'NaT':
                    awteff = 0.0
                else:
                    awteff = tc_awt / tc_tat
                
                row.append(awteff) # 
            

                # Only calculating the awteff for things related to SEP Start or Threshcrossing
                # Calculation for AWT Efficiency - which is based upon
                # AWT / (Time between threshold crossing and corresponding trigger time)
                # The denominator (gonna call it ptcl transit time for now) here 
                # needs to account for whichever trigger the 
                # forecasting model includes (eruption/trigger/input) but
                # I don't want to calculate this awteff for each trigger this could be,
                # that feels needless. If multiple of these times are given, find the 
                # latest time and use that. 
            
            #Insert value into dataframe to save AWT calculations for each SEP
            sel_df.loc[len(sel_df)] = row

        if sel_df.empty:
            #No forecasts for this particular quantity
            #Fill this set of fields in the AWT dict with None
#            time_keys = ['Observed SEP Threshold Crossing Time','Observed SEP Start Time']
            time_keys = ['Observed SEP Threshold Crossing Time']
            if ftype['obs_key'] != '':
                time_keys.append(ftype['obs_key'])
            for key in time_keys:
                mean_key = "Mean AWT for " + ftype['pred_key'] + " to " + key
                median_key = "Median AWT for " + ftype['pred_key'] + " to " + key
                dict[mean_key].append(np.nan)
                dict[median_key].append(np.nan)
            if ftype['obs_key'] == '':
                dict['Mean AWT Efficiency for ' + ftype['pred_key'] + ' to ' + time_keys[0]].append(np.nan)

            continue

        #May be the case that none of the forecasts had a valid AWT
        #calculation. If every single AWT value is None, then do not
        #output a file or proceed with AWT calculation.
#        time_keys = ['Observed SEP Threshold Crossing Time','Observed SEP Start Time']
        time_keys = ['Observed SEP Threshold Crossing Time']
        if ftype['obs_key'] != '':
            time_keys.append(ftype['obs_key'])
        chk_awts = []
        for key in time_keys:
            chk = sel_df["AWT to " + key].to_list()
            chk_awts.extend([x for x in chk if x != None])
        
        #Have AWT values for all SEPs for a given model, energy channel, and
        #threshold
        #Write to file if more than None values
        if len(chk_awts) >= 1:
            thresh_fnm = make_thresh_fname(thresh_key)
            fnm = "awt_selections_" + model + "_" + energy_key.strip() + "_" +\
                    thresh_fnm + "_" + ftype['pred_key']
            if bool(sel_df.iloc[0]['Mismatch Allowed']):
                fnm = fnm + "_mm"
            if validation_type != "" and validation_type != "All":
                fnm = fnm + "_" + validation_type
            write_df(sel_df, fnm)


        #Calculate metrics for AWT to different times
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
        if ftype['obs_key'] == '':
            awteffs = sel_df['AWT Efficiency to Observed SEP Threshold Crossing Time'].to_list()
            if len(awts) >= 1:
                mean_awteff = statistics.mean(awteffs)
            else:
                mean_awteff = None
            dict['Mean AWT Efficiency for ' + ftype['pred_key'] + ' to ' + time_keys[0]].append(mean_awteff)

    return



def last_data_to_issue_intuitive_metrics(df, dict, model, energy_key, thresh_key,
    validation_type):
    """ Extract the appropriate values and calculate metrics
        Last data time ingested in the forecast to the forecast issue time

    """
    val_type = ["", "All"]
    if validation_type not in val_type:
        return
    
    #Select rows to calculate metrics
    sub = df.loc[(df['Model'] == model) & (df['Energy Channel Key'] ==
        energy_key) & (df['Threshold Key'] == thresh_key)]

    sub = sub[['Model','Energy Channel Key', 'Threshold Key',
            'Mismatch Allowed',
            'Prediction Energy Channel Key', 'Prediction Threshold Key',
            'Forecast Source',
            'Prediction Window Start', 'Prediction Window End',
            'Observed SEP Threshold Crossing Time',
            'Last Data Time to Issue Time']]

    sub = sub.dropna(subset='Last Data Time to Issue Time')

    if sub.empty:
        return

    mismatch = bool(sub.iloc[0]['Mismatch Allowed'])
    pred_energy_key = str(sub.iloc[0]['Prediction Energy Channel Key'])
    pred_thresh_key = str(sub.iloc[0]['Prediction Threshold Key'])

    thresh_fnm = make_thresh_fname(thresh_key)
    fnm = "last_data_time_to_issue_time_selections_" + model + "_" + energy_key.strip() \
            + "_" + thresh_fnm
    if mismatch:
        fnm = fnm + "_mm"
    if validation_type != "" and validation_type != "All":
        fnm = fnm + "_" + validation_type
    write_df(sub, fnm)

    td = sub['Last Data Time to Issue Time'].to_list()
    abs_td = [abs(x) for x in td]

    ME = statistics.mean(td)
    MedE = statistics.median(td)
    MAE = statistics.mean(abs_td)
    MedAE = statistics.median(abs_td)
    
    fill_time_metrics_dict(dict, model, energy_key, thresh_key,
    pred_energy_key, pred_thresh_key, ME, MedE, MAE, MedAE)



def pretty(d, indent=0):
   for key, value in d.items():
      print('\t' * indent + str(key))
      if isinstance(value, dict):
         pretty(value, indent+1)
      else:
         print('\t' * (indent+1) + str(value))


def profile_output(sphinx_dataframe, resume_obs, resume_model):
    
    # Is there a point to 'resume' for the profiles? 
    u_obs_profs = resume.identify_unique(sphinx_dataframe, 'Observed Time Profile')
    
    u_model_profs = resume.identify_unique(sphinx_dataframe, 'Predicted Time Profile')
    
  

    observed_profs = {}
    model_profs = {}
    for u in u_obs_profs:
        if resume_obs is not None and u in resume_obs:
            continue
        else:
            for u_i in u.rsplit(','):
                obs_dates, obs_profiles = profile.read_single_time_profile(u_i)
                obs_dates = [x.strftime('%Y-%m-%dT%H:%M:%SZ') for x in obs_dates]
                observed_profs[u_i] = {'dates': obs_dates, 'fluxes': obs_profiles}
    for um in u_model_profs:
        if resume_model is not None and um in resume_model:
            continue
        else:
            for um_i in um.rsplit(','):
                model_dates, model_profiles = profile.read_single_time_profile(um_i)
                model_dates = [x.strftime('%Y-%m-%dT%H:%M:%SZ') for x in model_dates]
                model_profs[um_i] = {'dates': model_dates, 'fluxes': model_profiles}


    if resume_obs is not None:
        observed_profs = resume_obs | observed_profs
    if resume_model is not None:
        model_profs = resume_model | model_profs

 
    obs_file_path = os.path.join(config.outpath,os.path.join('json', 'observed_profiles.json'))
    with open(obs_file_path, 'w+') as json_file:
        json.dump(observed_profs, json_file, indent = 4)
    pickle_file_path = os.path.join(config.outpath, os.path.join('pkl', 'observed_profiles.pkl')) # Desired name for your pickle file
    with open(pickle_file_path, 'wb') as f:
        pickle.dump(observed_profs, f)
    

    model_file_path = os.path.join(config.outpath, os.path.join('json', 'model_profiles.json'))
    with open(model_file_path, 'w+') as json_file:
        json.dump(model_profs, json_file, indent = 4)
    pickle_file_path = os.path.join(config.outpath, os.path.join('pkl', 'model_profiles.pkl')) # Desired name for your pickle file
    with open(pickle_file_path, 'wb') as f:
        pickle.dump(model_profs, f)



    return


def calculate_intuitive_metrics(df, model_names, all_energy_channels,
    all_observed_thresholds, validation_type="All"):
    """ Calculate metrics appropriate to each quantity and
        store in dataframes.
            
    Input:
    
        :df: (pandas DataFrame) contains matched observations and predictions
        :model_names: (array of strings) all models read into code
        :all_energy_channels: (array of strings) all energy channel keys associated
            with observations and predictions, passed from match.py
        :all_observed_threshold: (array of strings) all threshold keys associated
            with each energy channel
        :validation_type: (string) indicates whether to use "All", "First",
            "Last", "Max", "Mean" forecast for metrics. The validation types only
            have meaning if multiple forecasts are made for each SEP event.
            
            "All" or "" - all forecasts will be used to calculate all metrics
            "First" - only the first forecast for each SEP event will be used.
                Contingency table won't have any Correct Negatives as they will
                be excluded.
            "Last" - only the last forecast for each SEP event will be used.
                Contingency table won't have any Correct Negatives as they will
                be excluded.
            "Max" - only the maximum valued forecast will be used for each
                SEP event. This selection will only calculate the flux and
                probability-related forecasts (peak, fluence, probability)
            "Mean" - only the mean valued forecast will be used for each
                SEP event. This selection will only calculate the flux and
                probability-related forecasts (peak, fluence, probability)

    Output:
    
        Metrics pandas dataframes
    
    """
    #Check accepted validation types
    #Remove reference to ""
    if validation_type == "All" or validation_type == "all":
        validation_type = ""
    
    all_clear_dict = initialize_all_clear_dict() #All only
    probability_dict = initialize_probability_dict() #All only
    point_intensity_dict = initialize_flux_dict() #All only
    peak_intensity_dict = initialize_flux_dict() #All, First, Last, Max, Mean
    peak_intensity_max_dict = initialize_flux_dict() #All, First, Last, Max, Mean
    fluence_dict = initialize_flux_dict() #All, First, Last, Max, Mean
    profile_dict = initialize_flux_dict() #All, First, Last
    thresh_cross_dict = initialize_time_dict() #All, First, Last
    start_time_dict = initialize_time_dict() #All, First, Last
    end_time_dict = initialize_time_dict() #All, First, Last
    duration_dict = initialize_time_dict() #All, First, Last
    peak_intensity_time_dict = initialize_time_dict() #All, First, Last
    peak_intensity_max_time_dict = initialize_time_dict() #All, First, Last
    max_dict = initialize_flux_dict() #max in prediction window #All only
    awt_dict = initialize_awt_dict() #Advanced Warning Time #All, First
    last_data_to_issue_dict = initialize_time_dict() #All
    
    for model in model_names:
        for ek in all_energy_channels:
            for tk in all_observed_thresholds[ek]:
                logging.info("Calculating metrics for " + model + ", " + ek + ", " + tk)
                
                probability_intuitive_metrics(df, probability_dict,model,ek,tk,
                    validation_type)
                point_intensity_intuitive_metrics(df, point_intensity_dict,
                    model, ek, tk, validation_type)
                peak_intensity_intuitive_metrics(df, peak_intensity_dict,
                    model,ek,tk, validation_type)
                peak_intensity_max_intuitive_metrics(df,
                    peak_intensity_max_dict,model,ek,tk, validation_type)
                fluence_intuitive_metrics(df,fluence_dict, model,ek,tk,
                    validation_type)
                threshold_crossing_intuitive_metrics(df, thresh_cross_dict,
                    model,ek,tk, validation_type)
                start_time_intuitive_metrics(df, start_time_dict,
                    model,ek,tk, validation_type)
                end_time_intuitive_metrics(df, end_time_dict,model,ek,tk,
                    validation_type)
                duration_intuitive_metrics(df, duration_dict,model,ek,tk,
                    validation_type)
                peak_intensity_time_intuitive_metrics(df,
                    peak_intensity_time_dict,model,ek,tk, validation_type)
                peak_intensity_max_time_intuitive_metrics(df,
                    peak_intensity_max_time_dict,model,ek,tk, validation_type)
                all_clear_intuitive_metrics(df, all_clear_dict,model,ek,tk,
                    validation_type)
                time_profile_intuitive_metrics(df, profile_dict,model,ek,tk,
                    validation_type)
                max_flux_in_pred_win_metrics(df, max_dict, model,ek,tk,
                    validation_type)
                awt_metrics(df, awt_dict, model, ek, tk, validation_type)
                last_data_to_issue_intuitive_metrics(df, last_data_to_issue_dict,model,ek,tk,
                    validation_type)

    logger.info("Completed calculating all metrics.")

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
    awt_metrics_df = pd.DataFrame(awt_dict)
    last_data_to_issue_metrics_df = pd.DataFrame(last_data_to_issue_dict)

    valtype = ""
    if validation_type != "" and validation_type != "All":
        valtype = "_" + validation_type
    if not prob_metrics_df.empty:
        write_df(prob_metrics_df, "probability_metrics" + valtype)
    if not point_intensity_metrics_df.empty:
        write_df(point_intensity_metrics_df, "point_intensity_metrics" + valtype)
    if not peak_intensity_metrics_df.empty:
        write_df(peak_intensity_metrics_df, "peak_intensity_metrics" + valtype)
    if not peak_intensity_max_metrics_df.empty:
        write_df(peak_intensity_max_metrics_df, "peak_intensity_max_metrics" + valtype)
    if not fluence_metrics_df.empty:
        write_df(fluence_metrics_df, "fluence_metrics" + valtype)
    if not thresh_cross_metrics_df.empty:
        write_df(thresh_cross_metrics_df, "threshold_crossing_metrics" + valtype)
    if not start_time_metrics_df.empty:
        write_df(start_time_metrics_df, "start_time_metrics" + valtype)
    if not end_time_metrics_df.empty:
        write_df(end_time_metrics_df, "end_time_metrics" + valtype)
    if not duration_metrics_df.empty:
        write_df(duration_metrics_df, "duration_metrics" + valtype)
    if not peak_intensity_time_metrics_df.empty:
        write_df(peak_intensity_time_metrics_df, "peak_intensity_time_metrics" + valtype)
    if not peak_intensity_max_time_metrics_df.empty:
        write_df(peak_intensity_max_time_metrics_df, "peak_intensity_max_time_metrics" + valtype)
    if not all_clear_metrics_df.empty:
        write_df(all_clear_metrics_df, "all_clear_metrics" + valtype)
    if not time_profile_metrics_df.empty:
        write_df(time_profile_metrics_df, "time_profile_metrics" + valtype)
    if not max_metrics_df.empty:
        write_df(max_metrics_df, "max_flux_in_pred_win_metrics" + valtype)
    if not awt_metrics_df.empty:
        write_df(awt_metrics_df, "awt_metrics" + valtype)
    if not last_data_to_issue_metrics_df.empty:
        write_df(last_data_to_issue_metrics_df, "last_data_to_issue_time_metrics" + valtype)

    logger.info("Wrote out all metrics.")


def validation_explanation():
    """ State the selections applied to calculate metrics for each
        quantity. 
        
    """
    logger.info("")
    logger.info("============================")
    logger.info("VALIDATION EXPLANATION")
    logger.info("============================")
    logger.info("Explanation of Match Status:")
    logger.info("============================")
    logger.info("--SEP Event: Forecast is associated with an observed SEP event")
    logger.info("--No SEP Event: Forecast is associated with an observed clear period.")
    logger.info("--Unmatched: The forecast was initially associated to an observed SEP event but a different prediction was found to be a better match. This forecast has been unmatched and is now associated with an observed clear period. Only relevant for forecasts that use flare or CME triggers.")
    logger.info("--Ongoing SEP Event: The observed environment was already enhanced for the forecasted period. The forecast cannot be evaluated.")
    logger.info("--Trigger/Input after Observed Phenomenon: The forecast used input information later than the observed phenomenon, i.e. threshold crossing, peak flux, etc. This cannot be considered a forecast for the particular phenomenon.")
    logger.info("--Eruption Out of Range: The forecast overlaps with an observed SEP event, but used a flare or CME trigger with timing indicating that it is not likely physically connected to that SEP event. This forecast is associated with an observed clear period.")
    logger.info("--No Matching Threshold: The energy channel and threshold combination used in the forecast was not present in the prepared observations. These forecasts are not evaluated.")
    logger.info("")
    logger.info("Selections applied to calculate metrics:")
    logger.info("========================================")
    logger.info("--All Clear: Forecasts with \"Ongoing SEP Event\" match status are not included in All Clear metrics.")
    logger.info("--Probability: Forecasts with \"Ongoing SEP Event\" match status are not included in Probability metrics.")
    logger.info("--Peak Intensity (Onset Peak): Only forecasts with \"SEP Event\" match status are included in Peak Intensity metrics. For models that use the peak_intensity field to indicate SEP onset peak, this results in metrics that are derived from the subset of SEP events that were both predicted and observed.")
    logger.info("--Peak Intensity Max (Max Flux): Only forecasts with \"SEP Event\" match status are included in Peak Intensity Max (Max Flux) metrics. Metrics are derived from all predictions associated with observed SEP events.")
    logger.info("--Fluence: Only forecasts with \"SEP Event\" match status are included in Fluence metrics.")
    logger.info("--Fluence Spectrum: Only forecasts with \"SEP Event\" match status are included in Fluence Spectrum metrics.")
    logger.info("--Start Time: Only forecasts with \"SEP Event\" match status are included in Start Time metrics. Forecasts must predict the occurrence of an SEP event, so the metrics are derived from the subset of SEP events that were both predicted and observed.")
    logger.info("--End Time: Only forecasts with \"SEP Event\" match status are included in End Time metrics. Forecasts must predict the occurrence of an SEP event, so the metrics are derived from the subset of SEP events that were both predicted and observed.")
    logger.info("--SEP Time Profile: Only forecasts with \"SEP Event\" match status are included in SEP Time Profile metrics. Metrics are derived from all predictions associated with observed SEP events. Predicted and observed time profiles are compared during the time period when the observed flux is above threshold.")
    logger.info("--Max Flux in Prediction Window: All forecasts are compared to observed max flux in the forecasted prediction window, regardless of match status. Metrics are derived from all forecasts.")
    logger.info("--Advanced Warning Time: Only forecasts with \"SEP Event\" match status are included in Advanced Warning Time metrics. Forecasts must predict the occurrence of an SEP event to give advanced warning, so the metrics are derived from the subset of SEP events that were both predicted and observed.")
    logger.info("--Last Data Time to Issue Time: Calculated for all forecasts regardless of match status. Metrics are derived from all forecasts.")
    logger.info("============================")
    logger.info("END VALIDATION EXPLANATION")
    logger.info("============================")
    logger.info("")
    

def intuitive_validation(evaluated_sphinx, removed_sphinx, model_names,
    all_energy_channels, all_observed_thresholds, observed_sep_events,
    profname_dict, r_df=None, r_obs= None, r_mod= None):
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
    
        :evaluated_sphinx: (array of SPHINX objects) SHPINX objects which 
            contain a Forecast object and Observation objects that are inside 
            the forecast prediction window, and the observed values that are 
            appropriately matched up to the forecast given the timing of the 
            triggers/inputs and observed phenomena
        :removed_sphinx: (array of SPHINX objects) SPHINX objects that are
            not evaluated by SPHINX
        :model_names: (str array) array of the models whose predictions were
            read into the code
        :all_observed_thresholds: (dict) dictionary organized by energy
            channel and thresholds that were applied to observations (only
            predictions corresponding to thresholds that were applied to the
            observations can be validated)
        :observed_sep_events: (dict) dictionary organized by model name,
            energy channel, and threshold containing all unique observed SEP
            events that fell inside a forecast prediction window
        :profname_dict: (array) Dictionary containing the location of all the .txt files
            in the subdirectories below top.
        :r_df: (pandas dataframe) dataframe created from a previous run of
            SPHINX. Newly input predictions will be appended.
    
    Output:
    
        :df: (pandas dataframe) SPHINX dataframe. 
    
    """
    logger.info("Beginning validation process.")
    # Make sure the output directories exist
    prepare_outdirs()
    
    #For each model and predicted quantity, create dataframe of paired up values
    #so can calculate metrics
    logger.info("Filling SPHINX dataframe with information from matched sphinx objects.")


    #EVALUATED SPHINX OBJECTS: Fill dataframe for evaluated_sphinx
    df = fill_sphinx_df(evaluated_sphinx, all_observed_thresholds, profname_dict)
    
    #Check for duplicated forecasts and remove
    df, duplicate_df = duplicates.remove_sphinx_duplicates(df)
    logger.info("Completed filling evaluated_sphinx dataframe. ")


    logger.info("Filling removed SPHINX dataframe with information from not evaluated sphinx objects.")
    #NOT EVALUATED SPHINX OBJECTS: Fill dataframe for removed_sphinx
    df_not = fill_sphinx_df(removed_sphinx, all_observed_thresholds, profname_dict)
    
    #Add the duplicates discarded from df
    df_not = pd.concat([df_not,duplicate_df])
    logger.info("Completed filling removed_sphinx dataframe. ")


    ### RESUME WILL APPEND DF TO PREVIOUS DF
    if r_df is not None:
        logger.info("RESUME: Resuming from a previous run. Concatenating current and previous forecasts, ensuring that any duplicates are removed. ")
 
        df = pd.concat([r_df, df], ignore_index=True)
        df, duplicate_df = duplicates.remove_sphinx_duplicates(df,"Duplicate in resume dataframe")
        #Add the duplicates discarded from df
        df_not = pd.concat([df_not,duplicate_df])
        logger.debug("RESUME: Completed concatenation and removed any duplicates. Writing SPHINX_evaluated dataframe to file.")

        model_names = resume.identify_unique(df, 'Model')
        all_energy_channels = resume.identify_unique(df, 'Energy Channel Key')
        all_observed_thresholds = resume.identify_thresholds_per_energy_channel(df)
    ### RESUME COMPLETED


    #Write SPHINX dataframe to file
    write_df(df, "SPHINX_evaluated")
    logger.debug("Completed writing SPHINX_evaluated dataframe to file.")

    #Write NOT EVALUATED SPHINX dataframe to file
    write_df(df_not, "SPHINX_removed")
    logger.debug("Completed writing SPHINX_removed dataframe to file.")

    validation_type = ["All", "First", "Last", "Max", "Mean"]
    for type in validation_type:
        logger.info("-----------Starting validation of " + type +" forecasts-------------")
        calculate_intuitive_metrics(df, model_names, all_energy_channels,
                all_observed_thresholds, type)

    #Record explanatory information to the log
    validation_explanation()
    
    

    profile_output(df, r_obs, r_mod)

    logger.info("intuitive_validation: Validation process complete.")

    return df   
 
