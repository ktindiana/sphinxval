# SUBROUTINES RELATED TO WORKFLOW UNIT TESTING
from sphinxval.utils import config
from sphinxval.utils import object_handler as objh
from sphinxval.utils import validation as validate
from sphinxval.utils import metrics
from sphinxval.utils import time_profile as profile
from sphinxval.utils import resume
from sphinxval.utils import plotting_tools as plt_tools
from sphinxval.utils import match
from sphinxval.utils import validation_json_handler as vjson
from sphinxval.utils import classes as cl


from astropy import units
import unittest
import sys
import pprint
import pandas as pd
import os
import csv
import logging
import logging.config
import pathlib
import json

from unittest.mock import patch
import shutil # using this to delete the contents of the output folder each run - since the unittest is based on the existence/creation of certain files each loop

"""
General outline as I start the validation.py workflow unittest
    validation.py (intuitive_validation function) is called after matching,
    where intuitive_validation is handed: 
        matched_sphinx: matched sphinx object 
        model_names
        all_energy_channels
        all_observed_thresholds
        observed_sep_events
        profname_dict
        DoResume
        r_df
    There are three core elements to intuitive_validation:
        Fill the SPHINX dataframe using the obs/pred matching pairs
            fill_df(matched_sphinx, model_names, all_energy_channels,
            all_observed_thresholds, profname_dict, DoResume)
        Checking if we are resuming a SPHINX validation via DoResume
            if DoResume:
        Performing the validation 
            validation_type = ["All", "First", "Last", "Max", "Mean"]
            for type in validation_type:
            calculate_intuitive_metrics(df, model_names, all_energy_channels,
                all_observed_thresholds, type)
    My thinking is that each class I build will go through each of the core
    elements of intuitive_validation but supply different matched_sphinx objects
    of different kinds of forecast types. Each test in the class will then be
    testing the various output of fill_df and the dictionary for each of the
    validation sections that class goes through. 
"""




# HELPER FUNCTIONS
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
 
#             #Point Intensity Forecasts
#            "Mean AWT for Predicted Point Intensity to Observed SEP Threshold Crossing Time": [],
#            "Median AWT for Predicted Point Intensity to Observed SEP Threshold Crossing Time": [],
#            "Mean AWT for Predicted Point Intensity to Observed SEP Start Time": [],
#            "Median AWT for Predicted Point Intensity to Observed SEP Start Time": [],
 
 
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


def utility_setup_logging():
    # Create the tests/logs/ directory if it does not yet exist
    with patch('sphinxval.utils.config.logpath', './tests/logs'):
        if not os.path.exists(config.logpath):
            os.mkdir(config.logpath)

        config_file = pathlib.Path('tests/log_config_tests.json')
        with open(config_file) as f_in:
            config_ = json.load(f_in)
        logging.config.dictConfig(config_)



def utility_load_forecast(filename, energy_channel):
    """
    Utility function for unit testing. Loads forecast object directly from JSON.
       Input:
            :filename: (string) forecast JSON file.
            :energy_channel: (dict) 
       Output:
            :forecast: resultant Forecast() object.
    """
    forecast_dict = vjson.read_in_json(filename)
    forecast, _ = vjson.forecast_object_from_json(forecast_dict, energy_channel)
    forecast.source = filename
    return forecast

def utility_load_observation(filename, energy_channel):
    """
    Utility function for unit testing. Loads observation object directly from JSON.
       Input:
            :filename: (string) observation JSON file.
            :energy_channel: (dict) 
       Output:
            :observation: resultant Observation() object.
    """
    observation_dict = vjson.read_in_json(filename)
    observation = vjson.observation_object_from_json(observation_dict, energy_channel)
    observation.source = filename   
    return observation


def utility_match_sphinx(all_energy_channels, model_names, obs_objs, model_objs):

    utility_setup_logging()

    matched_sphinx, all_observed_thresholds, observed_sep_events =\
        match.match_all_forecasts(all_energy_channels, model_names,
            obs_objs, model_objs)

    return matched_sphinx, all_observed_thresholds, observed_sep_events
    
def utility_get_verbosity():
    """
    Sets verbosity for unit test suite.
    """
    if ('-v' in sys.argv) or ('--verbose' in sys.argv):
        verbosity = 2
    elif ('--quiet' in sys.argv):
        verbosity = 0
    else:
        verbosity = 1
    return verbosity

def utility_delete_output():
    """
    Function to call at the start of each unittest to delete the contents 
    of the output folder 
    """
    shutil.rmtree('./tests/output')

    validate.prepare_outdirs()

def make_docstring_printable(function):
    """
    Decorator method @make_docstring_printable.
    Allows function to call itself to access docstring information.
    Used for verbose unit test outputs.
    """
    def wrapper(*args, **kwargs):
        return function(function, *args, **kwargs)
    return wrapper

def attributes_of_sphinx_obj(keyword, sphinx_obj, energy_channel_key, threshold_key):
    """
    Function that takes the keyword from the SPHINX dataframe and matches the keyword
    to the matched sphinx object to compare/assertEqual to - ensures that the 
    dataframe is being correctly built
    """
    if keyword == "Model": 
        attribute = getattr(sphinx_obj.prediction, 'short_name', None)
    elif keyword == "Energy Channel Key":
        attribute = energy_channel_key
    elif keyword == 'Threshold Key':
        attribute = threshold_key[energy_channel_key][0]
    elif keyword == 'Mismatch Allowed':
        attribute = getattr(sphinx_obj, 'mismatch', None)
    elif keyword == 'Prediction Energy Channel Key':
        if getattr(sphinx_obj, 'mismatch', None) == True:
            attribute = config.mm_pred_ek
        else:
            attribute = energy_channel_key
    elif keyword == 'Prediction Threshold Key':
        if getattr(sphinx_obj, 'mismatch', None) == True:
            attribute = config.mm_pred_tk
        else:
            attribute = threshold_key[energy_channel_key][0]
    elif keyword == 'Forecast Source':
        attribute = getattr(sphinx_obj.prediction, 'source', None)
    elif keyword == 'Forecast Path':
        attribute = getattr(sphinx_obj.prediction, 'path', None)
    elif keyword == 'Forecast Issue Time':
        attribute = getattr(sphinx_obj.prediction, 'issue_time', None)
    elif keyword == 'Prediction Window Start':
        attribute = getattr(sphinx_obj.prediction, 'prediction_window_start', None)        
    elif keyword == 'Prediction Window End':
        attribute = getattr(sphinx_obj.prediction, 'prediction_window_end', None)        
    elif keyword == 'Number of CMEs' or "CME" in keyword:
        num_cmes = len(getattr(sphinx_obj.prediction, 'cmes', None))
        if num_cmes != 0:
            if keyword == 'CME Start Time':
                attribute = getattr(sphinx_obj.prediction.cmes[-1], 'start_time', None)
            elif keyword == 'CME Liftoff Time':
                attribute = getattr(sphinx_obj.prediction.cmes[-1], 'liftoff_time', None)
            elif keyword == 'CME Latitude':
                attribute = getattr(sphinx_obj.prediction.cmes[-1], 'lat', None)
            elif keyword == 'CME Longitude':
                attribute = getattr(sphinx_obj.prediction.cmes[-1], 'lon', None)
            elif keyword == 'CME Speed':
                attribute = getattr(sphinx_obj.prediction.cmes[-1], 'speed', None)
            elif keyword == 'CME Half Width':
                attribute = getattr(sphinx_obj.prediction.cmes[-1], 'half_width', None)
            elif keyword == 'CME PA':
                attribute = getattr(sphinx_obj.prediction.cmes[-1], 'pa', None)
            elif keyword == 'CME Catalog':
                attribute = getattr(sphinx_obj.prediction.cmes[-1], 'catalog', None)
        elif num_cmes == 0 and keyword != 'Number of CMEs':
            attribute = None
        if keyword == "Number of CMEs":
            attribute = num_cmes   
    elif keyword == 'Number of Flares' or "Flare" in keyword:
        num_flares = len(getattr(sphinx_obj.prediction, 'flares', None))
        if num_flares != 0:
            if keyword == 'Flare Start Time':
                attribute = getattr(sphinx_obj.prediction.flares[-1], 'start_time', None)
            elif keyword == 'Flare Peak Time':
                attribute = getattr(sphinx_obj.prediction.flares[-1], 'peak_time', None)
            elif keyword == 'Flare Latitude':
                attribute = getattr(sphinx_obj.prediction.flares[-1], 'lat', None)
            elif keyword == 'Flare Longitude':
                attribute = getattr(sphinx_obj.prediction.flares[-1], 'lon', None)
            elif keyword == 'Flare End Time':
                attribute = getattr(sphinx_obj.prediction.flares[-1], 'end_time', None)
            elif keyword == 'Flare Last Data Time':
                attribute = getattr(sphinx_obj.prediction.flares[-1], 'last_data_time', None)
            elif keyword == 'Flare Intensity':
                attribute = getattr(sphinx_obj.prediction.flares[-1], 'intensity', None)
            elif keyword == 'Flare Integrated Intensity':
                attribute = getattr(sphinx_obj.prediction.flares[-1], 'integrated_intensity', None)
            elif keyword == 'Flare NOAA AR':
                attribute = getattr(sphinx_obj.prediction.flares[-1], 'noaa_region', None)
        elif num_flares == 0 and keyword != 'Number of Flares':
            attribute = None
        if keyword == "Number of Flares":
            attribute = num_flares
    elif keyword == 'Observatory' or keyword == 'Observed Time Profile':
        attribute = []
        for i in range(len(sphinx_obj.prediction_observation_windows_overlap)):
            if i == 0:
                if keyword == 'Observatory':
                    attribute = getattr(sphinx_obj.prediction_observation_windows_overlap[i], "short_name", None)
                else:
                    attribute = getattr(sphinx_obj, "observed_sep_profiles", None)[i]
            else:
                attribute += ','
                if keyword == 'Observatory':
                    attribute += getattr(sphinx_obj.prediction_observation_windows_overlap[i], "short_name", None)
                else:
                    attribute += getattr(sphinx_obj, "observed_sep_profiles", None)[i]
    elif keyword == "Observed SEP All Clear":
        attribute = getattr(sphinx_obj.observed_all_clear, 'all_clear_boolean', None)
    elif keyword == "Observed SEP Probability":
        print('thresh key, energy_key, mm_pred_tk, mm_pred_ek', threshold_key, energy_channel_key, config.mm_pred_tk, config.mm_pred_ek)
        attribute = getattr(sphinx_obj.observed_probability[threshold_key[energy_channel_key][0]], 'probability_value', None)
    elif keyword == 'Observed SEP Threshold Crossing Time':
        attribute = getattr(sphinx_obj.observed_threshold_crossing[threshold_key[energy_channel_key][0]], 'crossing_time', None)        
    elif keyword == 'Observed SEP Start Time':
        attribute = getattr(sphinx_obj, 'observed_start_time', None)[threshold_key[energy_channel_key][0]]
    elif keyword == 'Observed SEP End Time':
        attribute = getattr(sphinx_obj, 'observed_end_time', None)[threshold_key[energy_channel_key][0]]
    elif keyword == 'Observed SEP Duration':
        attribute = getattr(sphinx_obj, 'observed_duration', None)[threshold_key[energy_channel_key][0]]
    elif keyword == 'Observed SEP Fluence':
        attribute = getattr(sphinx_obj.observed_fluence[threshold_key[energy_channel_key][0]], 'fluence', None)
    elif keyword == 'Observed SEP Fluence Units':
        attribute = getattr(sphinx_obj.observed_fluence[threshold_key[energy_channel_key][0]], 'units', None)
    elif keyword == 'Observed SEP Fluence Spectrum':
        attribute = getattr(sphinx_obj.observed_fluence_spectrum[threshold_key[energy_channel_key][0]], 'fluence_spectrum', None)
        # print('OBSERVED FLUENCE HERE', attribute, sphinx_obj.observed_fluence_spectrum[threshold_key[energy_channel_key][0]].fluence_spectrum)
    elif keyword == 'Observed SEP Fluence Spectrum Units':
        attribute = getattr(sphinx_obj.observed_fluence_spectrum[threshold_key[energy_channel_key][0]], 'fluence_units', None)
    elif keyword == "Observed SEP Peak Intensity (Onset Peak)":
        attribute = getattr(sphinx_obj.observed_peak_intensity, 'intensity', None)
    elif keyword == "Observed SEP Peak Intensity (Onset Peak) Units":
        attribute = getattr(sphinx_obj.observed_peak_intensity, 'units', None)
    elif keyword == "Observed SEP Peak Intensity (Onset Peak) Time":
        attribute = getattr(sphinx_obj.observed_peak_intensity, 'time', None)
    elif keyword == "Observed SEP Peak Intensity Max (Max Flux)":
        attribute = getattr(sphinx_obj.observed_peak_intensity_max, 'intensity', None)
    elif keyword == "Observed SEP Peak Intensity Max (Max Flux) Units":
        attribute = getattr(sphinx_obj.observed_peak_intensity_max, 'units', None)
    elif keyword == "Observed SEP Peak Intensity Max (Max Flux) Time":
        attribute = getattr(sphinx_obj.observed_peak_intensity_max, 'time', None)
    elif keyword == "Observed Point Intensity":
        attribute = getattr(sphinx_obj.observed_point_intensity, 'intensity', None)
    elif keyword == "Observed Point Intensity Units":
        attribute = getattr(sphinx_obj.observed_point_intensity, 'units', None)
    elif keyword == "Observed Point Intensity Time":
        attribute = getattr(sphinx_obj.observed_point_intensity, 'time', None)
    elif keyword == "Observed Max Flux in Prediction Window":
        attribute = getattr(sphinx_obj.observed_max_flux_in_prediction_window, 'intensity', None)
    elif keyword == "Observed Max Flux in Prediction Window Units":
        attribute = getattr(sphinx_obj.observed_max_flux_in_prediction_window, 'units', None)
    elif keyword == "Observed Max Flux in Prediction Window Time":
        attribute = getattr(sphinx_obj.observed_max_flux_in_prediction_window, 'time', None)
    elif keyword == "Predicted SEP All Clear":
        attribute = getattr(sphinx_obj, 'return_predicted_all_clear', None)()[0]
    elif keyword == "All Clear Match Status":
        attribute = getattr(sphinx_obj, 'return_predicted_all_clear', None)()[1]
    elif keyword == 'Predicted SEP Probability':
        attribute = getattr(sphinx_obj, 'return_predicted_probability', None)(threshold_key[energy_channel_key][0])[0]
    elif keyword == 'Probability Match Status':
        attribute = getattr(sphinx_obj, 'return_predicted_probability', None)(threshold_key[energy_channel_key][0])[1]
    elif keyword == "Predicted SEP Threshold Crossing Time":
        attribute = getattr(sphinx_obj, 'return_predicted_threshold_crossing_time', None)(threshold_key[energy_channel_key][0])[0]
    elif keyword == "Threshold Crossing Time Match Status":
        attribute = getattr(sphinx_obj, 'return_predicted_threshold_crossing_time', None)(threshold_key[energy_channel_key][0])[1]
    elif keyword == "Predicted SEP Start Time":
        attribute = getattr(sphinx_obj, 'return_predicted_start_time', None)(threshold_key[energy_channel_key][0])[0]
    elif keyword == "Start Time Match Status":
        attribute = getattr(sphinx_obj, 'return_predicted_start_time', None)(threshold_key[energy_channel_key][0])[1]
    elif keyword == "Predicted SEP End Time":
        attribute = getattr(sphinx_obj, 'return_predicted_end_time', None)(threshold_key[energy_channel_key][0])[0]
    elif keyword == "End Time Match Status":
        attribute = getattr(sphinx_obj, 'return_predicted_end_time', None)(threshold_key[energy_channel_key][0])[1]
    elif keyword == 'Predicted SEP Duration':
        attribute = getattr(sphinx_obj, 'return_predicted_duration')(threshold_key[energy_channel_key][0])[0]
    elif keyword == 'Duration Match Status':
        # attribute = getattr(sphinx_obj, 'return_predicted_duration')(threshold_key[energy_channel_key][0])[1]
        attribute = getattr(sphinx_obj, 'return_predicted_end_time', None)(threshold_key[energy_channel_key][0])[1] # doing this for now 
    elif keyword == "Predicted SEP Fluence":
        attribute = getattr(sphinx_obj, 'return_predicted_fluence', None)(threshold_key[energy_channel_key][0])[0]
    elif keyword == "Predicted SEP Fluence Units":
        attribute = getattr(sphinx_obj, 'return_predicted_fluence', None)(threshold_key[energy_channel_key][0])[1]
    elif keyword == "Fluence Match Status":
        attribute = getattr(sphinx_obj, 'return_predicted_fluence', None)(threshold_key[energy_channel_key][0])[2]
    elif keyword == "Predicted SEP Fluence Spectrum":
        attribute = getattr(sphinx_obj, 'return_predicted_fluence_spectrum', None)(threshold_key[energy_channel_key][0])[0]
    elif keyword == "Predicted SEP Fluence Spectrum Units":
        attribute = getattr(sphinx_obj, 'return_predicted_fluence_spectrum', None)(threshold_key[energy_channel_key][0])[1]
    elif keyword == "Fluence Spectrum Match Status":
        attribute = getattr(sphinx_obj, 'return_predicted_fluence_spectrum', None)(threshold_key[energy_channel_key][0])[2]
    elif keyword == "Predicted SEP Peak Intensity (Onset Peak)":
        attribute = getattr(sphinx_obj, 'return_predicted_peak_intensity', None)()[0]
    elif keyword == "Predicted SEP Peak Intensity (Onset Peak) Units":
        attribute = getattr(sphinx_obj, 'return_predicted_peak_intensity', None)()[1]
    elif keyword == "Predicted SEP Peak Intensity (Onset Peak) Time":
        attribute = getattr(sphinx_obj, 'return_predicted_peak_intensity', None)()[2]
    elif keyword == "Peak Intensity Match Status":
        attribute = getattr(sphinx_obj, 'return_predicted_peak_intensity', None)()[3]
    elif keyword == "Predicted SEP Peak Intensity Max (Max Flux)":
        attribute = getattr(sphinx_obj, 'return_predicted_peak_intensity_max', None)()[0]
    elif keyword == "Predicted SEP Peak Intensity Max (Max Flux) Units":
        attribute = getattr(sphinx_obj, 'return_predicted_peak_intensity_max', None)()[1]
    elif keyword == "Predicted SEP Peak Intensity Max (Max Flux) Time":
        attribute = getattr(sphinx_obj, 'return_predicted_peak_intensity_max', None)()[2]
    elif keyword == "Peak Intensity Max Match Status":
        attribute = getattr(sphinx_obj, 'return_predicted_peak_intensity_max', None)()[3]
    elif keyword == "Predicted Point Intensity":
        attribute = getattr(sphinx_obj, 'return_predicted_point_intensity', None)()[0]
    elif keyword == "Predicted Point Intensity Units":
        attribute = getattr(sphinx_obj, 'return_predicted_point_intensity', None)()[1]
    elif keyword == "Predicted Point Intensity Time":
        attribute = getattr(sphinx_obj, 'return_predicted_point_intensity', None)()[2]
    elif keyword == 'Predicted Time Profile':
        try:
            attribute = getattr(sphinx_obj.prediction, 'path', None)
            print(attribute, 'THIS WORKING CORRECTLY?')
            attribute += getattr(sphinx_obj.prediction, 'sep_profile', None)
        except:
            attribute = None
        # print(attribute)
    elif keyword == 'Time Profile Match Status':
        attribute = getattr(sphinx_obj, 'return_predicted_end_time', None)(threshold_key[energy_channel_key][0])[1]
    elif keyword == 'Last Data Time to Issue Time':
        
        attribute = getattr(sphinx_obj.prediction, 'last_data_time_to_issue_time' , None)()
        
        # sys.exit()
    elif keyword == 'Last Eruption Time': 
        attribute = str(getattr(sphinx_obj, 'last_eruption_time' , None))
    elif keyword == 'Last Trigger Time': 
        attribute = str(getattr(sphinx_obj, 'last_trigger_time' , None))
    elif keyword == 'Last Input Time': 
        attribute = str(getattr(sphinx_obj, 'last_input_time' , None))
    elif keyword == 'Threshold Crossed in Prediction Window': 
        attribute = str(getattr(sphinx_obj, 'threshold_crossed_in_pred_win', None)[threshold_key[energy_channel_key][0]])
    elif keyword == 'Eruption before Threshold Crossed':
        attribute = str(getattr(sphinx_obj, 'eruptions_before_threshold_crossing', None)[threshold_key[energy_channel_key][0]])
    elif keyword == 'Time Difference between Eruption and Threshold Crossing':
        attribute = str(getattr(sphinx_obj, 'time_difference_eruptions_threshold_crossing', None)[threshold_key[energy_channel_key][0]])
    elif keyword == 'Triggers before Threshold Crossing':
        attribute = str(getattr(sphinx_obj, 'triggers_before_threshold_crossing', None)[threshold_key[energy_channel_key][0]])
    elif keyword == 'Inputs before Threshold Crossing':
        attribute = str(getattr(sphinx_obj, 'inputs_before_threshold_crossing', None)[threshold_key[energy_channel_key][0]])
    elif keyword == 'Triggers before Peak Intensity':
        attribute = str(getattr(sphinx_obj, 'triggers_before_peak_intensity', None))
    elif keyword == 'Time Difference between Triggers and Peak Intensity':
        attribute = str(getattr(sphinx_obj, 'time_difference_triggers_peak_intensity', None))
    elif keyword == 'Inputs before Peak Intensity':
        attribute = str(getattr(sphinx_obj, 'inputs_before_peak_intensity', None))
    elif keyword == 'Time Difference between Inputs and Peak Intensity':
        attribute = str(getattr(sphinx_obj, 'time_difference_inputs_peak_intensity', None))
    elif keyword == 'Triggers before Peak Intensity Max':
        attribute = str(getattr(sphinx_obj, 'triggers_before_peak_intensity_max', None))
    elif keyword == 'Time Difference between Triggers and Peak Intensity Max':
        attribute = str(getattr(sphinx_obj, 'time_difference_triggers_peak_intensity_max', None))
    elif keyword == 'Inputs before Peak Intensity Max':
        attribute = str(getattr(sphinx_obj, 'inputs_before_peak_intensity_max', None))
    elif keyword == 'Time Difference between Inputs and Peak Intensity Max':
        attribute = str(getattr(sphinx_obj, 'time_difference_inputs_peak_intensity_max', None))
    elif keyword == 'Triggers before SEP End':
        attribute = str(getattr(sphinx_obj, 'triggers_before_sep_end', None)[threshold_key[energy_channel_key][0]])
    elif keyword == 'Inputs before SEP End':
        attribute = str(getattr(sphinx_obj, 'inputs_before_sep_end', None)[threshold_key[energy_channel_key][0]])
    elif keyword == 'Time Difference between Triggers and SEP End':
        attribute = str(getattr(sphinx_obj, 'time_difference_triggers_sep_end', None)[threshold_key[energy_channel_key][0]])
    elif keyword == 'Time Difference between Inputs and SEP End':
        attribute = str(getattr(sphinx_obj, 'time_difference_inputs_sep_end', None)[threshold_key[energy_channel_key][0]])
    elif keyword == 'Prediction Window Overlap with Observed SEP Event':
        attribute = str(getattr(sphinx_obj, 'prediction_window_sep_overlap', None)[threshold_key[energy_channel_key][0]])
    elif keyword == 'Ongoing SEP Event':
        attribute = str(getattr(sphinx_obj, 'observed_ongoing_events', None)[threshold_key[energy_channel_key][0]])







    else:
        attribute = 'Keyword not in sphinx object  ERROR'
   
    return attribute


def fill_all_clear_dict_hit(dict, self):
        """ Fill the all clear metrics dictionary with the 'known' outputs.
        """
        dict["Model"].append('Test_model_0')
        dict["Energy Channel"].append(self.energy_key)
        dict["Threshold"].append(self.obs_thresholds[self.energy_key][0])
        dict["Prediction Energy Channel"].append(self.energy_key)
        dict["Prediction Threshold"].append(self.obs_thresholds[self.energy_key][0])
        dict["All Clear 'True Positives' (Hits)"].append('1') #Hits
        dict["All Clear 'False Positives' (False Alarms)"].append('0') #False Alarms
        dict["All Clear 'True Negatives' (Correct Negatives)"].append('0')  #Correct negatives
        dict["All Clear 'False Negatives' (Misses)"].append('0') #Misses
        dict["N (Total Number of Forecasts)"].append('1')
        dict["Percent Correct"].append('1.0')
        dict["Bias"].append('1.0')
        dict["Hit Rate"].append('1.0')
        dict["False Alarm Rate"].append('')
        dict['False Negative Rate'].append('0.0')
        dict["Frequency of Misses"].append('0.0')
        dict["Frequency of Hits"].append('1.0')
        dict["Probability of Correct Negatives"].append('')
        dict["Frequency of Correct Negatives"].append('')
        dict["False Alarm Ratio"].append('0.0')
        dict["Detection Failure Ratio"].append('')
        dict["Threat Score"].append('1.0') #Critical Success Index
        dict["Odds Ratio"].append('')
        dict["Gilbert Skill Score"].append('') #Equitable Threat Score
        dict["True Skill Statistic"].append('') #Hanssen and Kuipers
                #discriminant (true skill statistic, Peirce's skill score)
        dict["Heidke Skill Score"].append('')
        dict["Odds Ratio Skill Score"].append('')
        dict["Symmetric Extreme Dependency Score"].append('')
        dict["F1 Score"].append('1.0'),
        dict["F2 Score"].append('1.0'),
        dict["Fhalf Score"].append('1.0'),
        dict['Prevalence'].append('1.0'),
        dict['Matthew Correlation Coefficient'].append(''),
        dict['Informedness'].append(''),
        dict['Markedness'].append(''),
        dict['Prevalence Threshold'].append(''),
        dict['Balanced Accuracy'].append(''),
        dict['Fowlkes-Mallows Index'].append('1.0'),
        dict["Number SEP Events Correctly Predicted"].append('1')
        dict["Number SEP Events Missed"].append('0')
        dict["Predicted SEP Events"].append('2000-01-01 01:00:00')
        dict["Missed SEP Events"].append('None')
        return dict


def fill_all_clear_dict_CN(dict, self):
        """ Fill the all clear metrics dictionary with the 'known' outputs.
        """
        dict["Model"].append('Test_model_0')
        dict["Energy Channel"].append(self.energy_key)
        dict["Threshold"].append(self.obs_thresholds[self.energy_key][0])
        dict["Prediction Energy Channel"].append(self.energy_key)
        dict["Prediction Threshold"].append(self.obs_thresholds[self.energy_key][0])
        dict["All Clear 'True Positives' (Hits)"].append('0') #Hits
        dict["All Clear 'False Positives' (False Alarms)"].append('0') #False Alarms
        dict["All Clear 'True Negatives' (Correct Negatives)"].append('1')  #Correct negatives
        dict["All Clear 'False Negatives' (Misses)"].append('0') #Misses
        dict["N (Total Number of Forecasts)"].append('1')
        dict["Percent Correct"].append('1.0')
        dict["Bias"].append('')
        dict["Hit Rate"].append('')
        dict["False Alarm Rate"].append('0.0')
        dict["False Negative Rate"].append('')
        dict["Frequency of Misses"].append('')
        dict["Frequency of Hits"].append('')
        dict["Probability of Correct Negatives"].append('1.0')
        dict["Frequency of Correct Negatives"].append('1.0')
        dict["False Alarm Ratio"].append('')
        dict["Detection Failure Ratio"].append('0.0')
        dict["Threat Score"].append('') #Critical Success Index
        dict["Odds Ratio"].append('')
        dict["Gilbert Skill Score"].append('') #Equitable Threat Score
        dict["True Skill Statistic"].append('') #Hanssen and Kuipers
                #discriminant (true skill statistic, Peirce's skill score)
        dict["Heidke Skill Score"].append('')
        dict["Odds Ratio Skill Score"].append('')
        dict["Symmetric Extreme Dependency Score"].append('')
        dict["F1 Score"].append(''),
        dict["F2 Score"].append(''),
        dict["Fhalf Score"].append(''),
        dict['Prevalence'].append('0.0'),
        dict['Matthew Correlation Coefficient'].append(''),
        dict['Informedness'].append(''),
        dict['Markedness'].append(''),
        dict['Prevalence Threshold'].append(''),
        dict['Balanced Accuracy'].append(''),
        dict['Fowlkes-Mallows Index'].append(''),
        dict["Number SEP Events Correctly Predicted"].append('0')
        dict["Number SEP Events Missed"].append('0')
        dict["Predicted SEP Events"].append('None')
        dict["Missed SEP Events"].append('None')
        return dict


def fill_awt_dict(dict, self):
    """ Metrics for Adanced Warning Time to SEP start, SEP peak, SEP end.
        The "Forecasted Value" field indicates which forecasted quantity
        was used to calculate the AWT.
    """
    dict["Model"].append('Test_model_0')
    dict["Energy Channel"].append(self.energy_key)
    dict["Threshold"].append(self.obs_thresholds[self.energy_key][0])
    dict["Prediction Energy Channel"].append(self.energy_key)
    dict["Prediction Threshold"].append(self.obs_thresholds[self.energy_key][0])
            

    dict["Mean AWT for Predicted SEP All Clear to Observed SEP Threshold Crossing Time"].append('1.0')
    dict["Median AWT for Predicted SEP All Clear to Observed SEP Threshold Crossing Time"].append('1.0')
    dict["Mean AWT for Predicted SEP All Clear to Observed SEP Start Time"].append('1.0')
    dict["Median AWT for Predicted SEP All Clear to Observed SEP Start Time"].append('1.0')

#            #Probability Forecasts - cannot without an explicit threshold
#            "Mean AWT for Probability to Observed Threshold Crossing Time": [],
#            "Median AWT for Probability to Observed Threshold Crossing Time": [],
#            "Mean AWT for Probability to Observed Start Time": [],
#            "Median AWT for Probability to Observed Start Time": [],

            #Threshold Crossing Time Forecasts
    dict["Mean AWT for Predicted SEP Threshold Crossing Time to Observed SEP Threshold Crossing Time"].append('')
    dict["Median AWT for Predicted SEP Threshold Crossing Time to Observed SEP Threshold Crossing Time"].append('')
    dict["Mean AWT for Predicted SEP Threshold Crossing Time to Observed SEP Start Time"].append('')
    dict["Median AWT for Predicted SEP Threshold Crossing Time to Observed SEP Start Time"].append('')

            #Start Time Forecasts
    dict["Mean AWT for Predicted SEP Start Time to Observed SEP Threshold Crossing Time"].append('')
    dict["Median AWT for Predicted SEP Start Time to Observed SEP Threshold Crossing Time"].append('')
    dict["Mean AWT for Predicted SEP Start Time to Observed SEP Start Time"].append('')
    dict["Median AWT for Predicted SEP Start Time to Observed SEP Start Time"].append('')
 
#             #Point Intensity Forecasts
#            "Mean AWT for Predicted Point Intensity to Observed SEP Threshold Crossing Time": [],
#            "Median AWT for Predicted Point Intensity to Observed SEP Threshold Crossing Time": [],
#            "Mean AWT for Predicted Point Intensity to Observed SEP Start Time": [],
#            "Median AWT for Predicted Point Intensity to Observed SEP Start Time": [],
 
 
            #Peak Intensity Forecasts
    dict["Mean AWT for Predicted SEP Peak Intensity (Onset Peak) to Observed SEP Threshold Crossing Time"].append('')
    dict["Median AWT for Predicted SEP Peak Intensity (Onset Peak) to Observed SEP Threshold Crossing Time"].append('')
    dict["Mean AWT for Predicted SEP Peak Intensity (Onset Peak) to Observed SEP Start Time"].append('')
    dict["Median AWT for Predicted SEP Peak Intensity (Onset Peak) to Observed SEP Start Time"].append('')
    dict["Mean AWT for Predicted SEP Peak Intensity (Onset Peak) to Observed SEP Peak Intensity (Onset Peak) Time"].append('')
    dict["Median AWT for Predicted SEP Peak Intensity (Onset Peak) to Observed SEP Peak Intensity (Onset Peak) Time"].append('')

            #Peak Intensity Max Forecasts
    dict["Mean AWT for Predicted SEP Peak Intensity Max (Max Flux) to Observed SEP Threshold Crossing Time"].append('')
    dict["Median AWT for Predicted SEP Peak Intensity Max (Max Flux) to Observed SEP Threshold Crossing Time"].append('')
    dict["Mean AWT for Predicted SEP Peak Intensity Max (Max Flux) to Observed SEP Start Time"].append('')
    dict["Median AWT for Predicted SEP Peak Intensity Max (Max Flux) to Observed SEP Start Time"].append('')
    dict["Mean AWT for Predicted SEP Peak Intensity Max (Max Flux) to Observed SEP Peak Intensity Max (Max Flux) Time"].append('')
    dict["Median AWT for Predicted SEP Peak Intensity Max (Max Flux) to Observed SEP Peak Intensity Max (Max Flux) Time"].append('')

            #End Time Forecasts
    dict["Mean AWT for Predicted SEP End Time to Observed SEP Threshold Crossing Time"].append('')
    dict["Median AWT for Predicted SEP End Time to Observed SEP Threshold Crossing Time"].append('')
    dict["Mean AWT for Predicted SEP End Time to Observed SEP Start Time"].append('')
    dict["Median AWT for Predicted SEP End Time to Observed SEP Start Time"].append('')
    dict["Mean AWT for Predicted SEP End Time to Observed SEP End Time"].append('')
    dict["Median AWT for Predicted SEP End Time to Observed SEP End Time"].append('')
            
    return dict

def fill_probability_dict_highprob(dict, self):
    dict["Model"].append('Test_model_0')
    dict["Energy Channel"].append(self.energy_key)
    dict["Threshold"].append(self.obs_thresholds[self.energy_key][0])
    dict["Prediction Energy Channel"].append(self.energy_key)
    dict["Prediction Threshold"].append(self.obs_thresholds[self.energy_key][0])
    
    dict['ROC Curve Plot'].append("./tests/output/plots/ROC_curve_Test_model_0_min.10.0.max.-1.0.units.MeV_threshold_1.0.pdf")
    dict['Brier Score'].append('0.0')
    dict['Brier Skill Score'].append('1.0')
    dict['Spearman Correlation Coefficient'].append('')
    dict['Area Under ROC Curve'].append('')
    return dict

def fill_probability_dict_lowprob(dict, self):
    dict["Model"].append('Test_model_0')
    dict["Energy Channel"].append(self.energy_key)
    dict["Threshold"].append(self.obs_thresholds[self.energy_key][0])
    dict["Prediction Energy Channel"].append(self.energy_key)
    dict["Prediction Threshold"].append(self.obs_thresholds[self.energy_key][0])
    
    dict['ROC Curve Plot'].append("./tests/output/plots/ROC_curve_Test_model_0_min.10.0.max.-1.0.units.MeV_threshold_1.0.pdf")
    dict['Brier Score'].append('1.0')
    dict['Brier Skill Score'].append('-0.06941692181172066')
    dict['Spearman Correlation Coefficient'].append('')
    dict['Area Under ROC Curve'].append('')
    return dict


def fill_probability_dict_multprob(dict, self):
    dict["Model"].append('Test_model_0')
    dict["Energy Channel"].append(self.energy_key)
    dict["Threshold"].append(self.obs_thresholds[self.energy_key][0])
    dict["Prediction Energy Channel"].append(self.energy_key)
    dict["Prediction Threshold"].append(self.obs_thresholds[self.energy_key][0])
    
    dict['ROC Curve Plot'].append("./tests/output/plots/ROC_curve_Test_model_0_min.10.0.max.-1.0.units.MeV_threshold_1.0.pdf")
    dict['Brier Score'].append('0.5')
    dict['Brier Skill Score'].append('0.4652915390941397')
    dict['Spearman Correlation Coefficient'].append('')
    dict['Area Under ROC Curve'].append('')
    return dict


def fill_peak_intensity_max_dict(dict, self):
    """ Put flux-related metrics into metrics dictionary.
    
    """
    dict["Model"].append('Test_model_0')
    dict["Energy Channel"].append(self.energy_key)
    dict["Threshold"].append(self.obs_thresholds[self.energy_key][0])
    dict["Prediction Energy Channel"].append(self.energy_key)
    dict["Prediction Threshold"].append(self.obs_thresholds[self.energy_key][0])
    dict["Scatter Plot"].append('')
    dict["Linear Regression Slope"].append('')
    dict["Linear Regression y-intercept"].append('')
    dict["Pearson Correlation Coefficient (Linear)"].append('')
    dict["Pearson Correlation Coefficient (Log)"].append('')
    dict["Spearman Correlation Coefficient (Linear)"].append('')
    dict["Mean Error (ME)"].append('0.0')
    dict["Median Error (MedE)"].append('0.0')
    dict["Mean Log Error (MLE)"].append('0.0')
    dict["Median Log Error (MedLE)"].append('0.0')
    dict["Mean Absolute Error (MAE)"].append('0.0')
    dict["Median Absolute Error (MedAE)"].append('0.0')
    dict["Mean Absolute Log Error (MALE)"].append('0.0')
    dict["Median Absolute Log Error (MedALE)"].append('0.0')
    dict["Mean Percent Error (MPE)"].append('0.0')
    dict["Mean Absolute Percent Error (MAPE)"].append('0.0')
    dict["Mean Symmetric Percent Error (MSPE)"].append('0.0')
    dict["Mean Symmetric Absolute Percent Error (SMAPE)"].append('0.0')
    dict["Mean Accuracy Ratio (MAR)"].append('1.0')
    dict["Root Mean Square Error (RMSE)"].append('0.0')
    dict["Root Mean Square Log Error (RMSLE)"].append('0.0')
    dict["Median Symmetric Accuracy (MdSA)"].append('0.0')
    dict.update({"Time Profile Selection Plot": ['']})

    return dict
    # if timeprofplot != None:
    #     if "Time Profile Selection Plot" not in dict.keys():
    #         dict.update({"Time Profile Selection Plot": [timeprofplot]})
    #     else:
    #         dict["Time Profile Selection Plot"].append(timeprofplot)


def fill_peak_intensity_max_mult_dict(dict, self):
    """ Put flux-related metrics into metrics dictionary.
    
    """
    dict["Model"].append('Test_model_0')
    dict["Energy Channel"].append(self.energy_key)
    dict["Threshold"].append(self.obs_thresholds[self.energy_key][0])
    dict["Prediction Energy Channel"].append(self.energy_key)
    dict["Prediction Threshold"].append(self.obs_thresholds[self.energy_key][0])
    dict["Scatter Plot"].append('./tests/output/plots/Correlation_peak_intensity_max_Test_model_0_min.10.0.max.-1.0.units.MeV_threshold_1.0.pdf')
    dict["Linear Regression Slope"].append('-0.25000000000000006')
    dict["Linear Regression y-intercept"].append('-0.25')
    dict["Pearson Correlation Coefficient (Linear)"].append('')
    dict["Pearson Correlation Coefficient (Log)"].append('')
    dict["Spearman Correlation Coefficient (Linear)"].append('')
    dict["Mean Error (ME)"].append('-4.995')
    dict["Median Error (MedE)"].append('-4.995')
    dict["Mean Log Error (MLE)"].append('-1.5')
    dict["Median Log Error (MedLE)"].append('-1.5')
    dict["Mean Absolute Error (MAE)"].append('4.995')
    dict["Median Absolute Error (MedAE)"].append('4.995')
    dict["Mean Absolute Log Error (MALE)"].append('1.5')
    dict["Median Absolute Log Error (MedALE)"].append('1.5')
    dict["Mean Percent Error (MPE)"].append('-0.4995')
    dict["Mean Absolute Percent Error (MAPE)"].append('0.4995')
    dict["Mean Symmetric Percent Error (MSPE)"].append('-0.9980019980019981')
    dict["Mean Symmetric Absolute Percent Error (SMAPE)"].append('0.9980019980019981')
    dict["Mean Accuracy Ratio (MAR)"].append('0.5005')
    dict["Root Mean Square Error (RMSE)"].append('7.06399674405361')
    dict["Root Mean Square Log Error (RMSLE)"].append('2.1213203435596424')
    dict["Median Symmetric Accuracy (MdSA)"].append('30.62277660168379')
    dict.update({"Time Profile Selection Plot": ['']})

    return dict

def fill_peak_intensity_dict(dict, self):
    """ Put flux-related metrics into metrics dictionary.
    
    """
    dict["Model"].append('Test_model_0')
    dict["Energy Channel"].append(self.energy_key)
    dict["Threshold"].append(self.obs_thresholds[self.energy_key][0])
    dict["Prediction Energy Channel"].append(self.energy_key)
    dict["Prediction Threshold"].append(self.obs_thresholds[self.energy_key][0])
    dict["Scatter Plot"].append('')
    dict["Linear Regression Slope"].append('')
    dict["Linear Regression y-intercept"].append('')
    dict["Pearson Correlation Coefficient (Linear)"].append('')
    dict["Pearson Correlation Coefficient (Log)"].append('')
    dict["Spearman Correlation Coefficient (Linear)"].append('')
    dict["Mean Error (ME)"].append('0.0')
    dict["Median Error (MedE)"].append('0.0')
    dict["Mean Log Error (MLE)"].append('0.0')
    dict["Median Log Error (MedLE)"].append('0.0')
    dict["Mean Absolute Error (MAE)"].append('0.0')
    dict["Median Absolute Error (MedAE)"].append('0.0')
    dict["Mean Absolute Log Error (MALE)"].append('0.0')
    dict["Median Absolute Log Error (MedALE)"].append('0.0')
    dict["Mean Percent Error (MPE)"].append('0.0')
    dict["Mean Absolute Percent Error (MAPE)"].append('0.0')
    dict["Mean Symmetric Percent Error (MSPE)"].append('0.0')
    dict["Mean Symmetric Absolute Percent Error (SMAPE)"].append('0.0')
    dict["Mean Accuracy Ratio (MAR)"].append('1.0')
    dict["Root Mean Square Error (RMSE)"].append('0.0')
    dict["Root Mean Square Log Error (RMSLE)"].append('0.0')
    dict["Median Symmetric Accuracy (MdSA)"].append('0.0')
    dict.update({"Time Profile Selection Plot": ['']})

    return dict
    # if timeprofplot != None:
    #     if "Time Profile Selection Plot" not in dict.keys():
    #         dict.update({"Time Profile Selection Plot": [timeprofplot]})
    #     else:
    #         dict["Time Profile Selection Plot"].append(timeprofplot)


def fill_peak_intensity_mult_dict(dict, self):
    """ Put flux-related metrics into metrics dictionary.
    
    """
    dict["Model"].append('Test_model_0')
    dict["Energy Channel"].append(self.energy_key)
    dict["Threshold"].append(self.obs_thresholds[self.energy_key][0])
    dict["Prediction Energy Channel"].append(self.energy_key)
    dict["Prediction Threshold"].append(self.obs_thresholds[self.energy_key][0])
    dict["Scatter Plot"].append('./tests/output/plots/Correlation_peak_intensity_Test_model_0_min.10.0.max.-1.0.units.MeV_threshold_1.0.pdf')
    dict["Linear Regression Slope"].append('-0.25000000000000006')
    dict["Linear Regression y-intercept"].append('-0.25')
    dict["Pearson Correlation Coefficient (Linear)"].append('')
    dict["Pearson Correlation Coefficient (Log)"].append('')
    dict["Spearman Correlation Coefficient (Linear)"].append('')
    dict["Mean Error (ME)"].append('-4.995')
    dict["Median Error (MedE)"].append('-4.995')
    dict["Mean Log Error (MLE)"].append('-1.5')
    dict["Median Log Error (MedLE)"].append('-1.5')
    dict["Mean Absolute Error (MAE)"].append('4.995')
    dict["Median Absolute Error (MedAE)"].append('4.995')
    dict["Mean Absolute Log Error (MALE)"].append('1.5')
    dict["Median Absolute Log Error (MedALE)"].append('1.5')
    dict["Mean Percent Error (MPE)"].append('-0.4995')
    dict["Mean Absolute Percent Error (MAPE)"].append('0.4995')
    dict["Mean Symmetric Percent Error (MSPE)"].append('-0.9980019980019981')
    dict["Mean Symmetric Absolute Percent Error (SMAPE)"].append('0.9980019980019981')
    dict["Mean Accuracy Ratio (MAR)"].append('0.5005')
    dict["Root Mean Square Error (RMSE)"].append('7.06399674405361')
    dict["Root Mean Square Log Error (RMSLE)"].append('2.1213203435596424')
    dict["Median Symmetric Accuracy (MdSA)"].append('30.62277660168379')
    dict.update({"Time Profile Selection Plot": ['']})
    return dict


def fill_peak_intensity_time_dict(dict, self):
    """ Fill in metrics for time
    """
    dict["Model"].append('Test_model_0')
    dict["Energy Channel"].append(self.energy_key)
    dict["Threshold"].append(self.obs_thresholds[self.energy_key][0])
    dict["Prediction Energy Channel"].append(self.energy_key)
    dict["Prediction Threshold"].append(self.obs_thresholds[self.energy_key][0])
    dict["Mean Error (pred - obs)"].append('0.0')
    dict["Median Error (pred - obs)"].append('0.0')
    dict["Mean Absolute Error (|pred - obs|)"].append('0.0')
    dict["Median Absolute Error (|pred - obs|)"].append('0.0')
    return dict

def fill_probability_dict_all(dict, self):
    dict["Model"].append('Test_model_0')
    dict["Energy Channel"].append(self.energy_key)
    dict["Threshold"].append(self.obs_thresholds[self.energy_key][0])
    dict["Prediction Energy Channel"].append(self.energy_key)
    dict["Prediction Threshold"].append(self.obs_thresholds[self.energy_key][0])
    
    dict['ROC Curve Plot'].append("./tests/output/plots/ROC_curve_Test_model_0_min.10.0.max.-1.0.units.MeV_threshold_1.0.pdf")
    dict['Brier Score'].append('0.006799999999999997')
    dict['Brier Skill Score'].append('0.9890982954329874')
    dict['Spearman Correlation Coefficient'].append('1.0')
    dict['Area Under ROC Curve'].append('1.0')
    return dict

def fill_peak_intensity_metrics_dict_all(dict, self):
    dict["Model"].append('Test_model_0')
    dict["Energy Channel"].append(self.energy_key)
    dict["Threshold"].append(self.obs_thresholds[self.energy_key][0])
    dict["Prediction Energy Channel"].append(self.energy_key)
    dict["Prediction Threshold"].append(self.obs_thresholds[self.energy_key][0])
    dict["Scatter Plot"].append('./tests/output/plots/Correlation_peak_intensity_Test_model_0_min.10.0.max.-1.0.units.MeV_threshold_1.0.pdf')
    dict["Linear Regression Slope"].append('0.5103481712895563')
    dict["Linear Regression y-intercept"].append('0.5103481712895562')
    dict["Pearson Correlation Coefficient (Linear)"].append('')
    dict["Pearson Correlation Coefficient (Log)"].append('')
    dict["Spearman Correlation Coefficient (Linear)"].append('')
    dict["Mean Error (ME)"].append('0.5')
    dict["Median Error (MedE)"].append('0.5')
    dict["Mean Log Error (MLE)"].append('0.02069634257911257')
    dict["Median Log Error (MedLE)"].append('0.02069634257911257')
    dict["Mean Absolute Error (MAE)"].append('0.5')
    dict["Median Absolute Error (MedAE)"].append('0.5')
    dict["Mean Absolute Log Error (MALE)"].append('0.02069634257911257')
    dict["Median Absolute Log Error (MedALE)"].append('0.02069634257911257')
    dict["Mean Percent Error (MPE)"].append('0.05')
    dict["Mean Absolute Percent Error (MAPE)"].append('0.05')
    dict["Mean Symmetric Percent Error (MSPE)"].append('0.047619047619047616')
    dict["Mean Symmetric Absolute Percent Error (SMAPE)"].append('0.047619047619047616')
    dict["Mean Accuracy Ratio (MAR)"].append('1.05')
    dict["Root Mean Square Error (RMSE)"].append('0.7071067811865476')
    dict["Root Mean Square Log Error (RMSLE)"].append('0.02926904836690076')
    dict["Median Symmetric Accuracy (MdSA)"].append('0.04880884817015163')
    dict.update({"Time Profile Selection Plot": ['']})

    return dict

def fill_time_profile_dict_all(dict, self):
    """ Put flux-related metrics into metrics dictionary.
    
    """
    dict["Model"].append('Test_model_0')
    dict["Energy Channel"].append(self.energy_key)
    dict["Threshold"].append(self.obs_thresholds[self.energy_key][0])
    dict["Prediction Energy Channel"].append(self.energy_key)
    dict["Prediction Threshold"].append(self.obs_thresholds[self.energy_key][0])
    dict["Scatter Plot"].append('./tests/output/plots/Correlation_time_profile_Test_model_0_min.10.0.max.-1.0.units.MeV_threshold_1.0_20000101T000000.pdf')
    dict["Linear Regression Slope"].append('')
    dict["Linear Regression y-intercept"].append('')
    dict["Pearson Correlation Coefficient (Linear)"].append('-0.16603070802422484')
    dict["Pearson Correlation Coefficient (Log)"].append('-1.1102230246251565e-16')
    dict["Spearman Correlation Coefficient (Linear)"].append('0.0')
    dict["Mean Error (ME)"].append('0.26338502445434836')
    dict["Median Error (MedE)"].append('0.26338502445434836')
    dict["Mean Log Error (MLE)"].append('-0.1884057971014493')
    dict["Median Log Error (MedLE)"].append('-0.1884057971014493')
    dict["Mean Absolute Error (MAE)"].append('4.1868830206983585')
    dict["Median Absolute Error (MedAE)"].append('4.1868830206983585')
    dict["Mean Absolute Log Error (MALE)"].append('0.5072463768115942')
    dict["Median Absolute Log Error (MedALE)"].append('0.5072463768115942')
    dict["Mean Percent Error (MPE)"].append('0.24042966102166793')
    dict["Mean Absolute Percent Error (MAPE)"].append('1.0981586843403988')
    dict["Mean Symmetric Percent Error (MSPE)"].append('-0.3382852235179586')
    dict["Mean Symmetric Absolute Percent Error (SMAPE)"].append('1.0062630593631319')
    dict["Mean Accuracy Ratio (MAR)"].append('1.0981586843403988')
    dict["Root Mean Square Error (RMSE)"].append('4.860100002974809')
    dict["Root Mean Square Log Error (RMSLE)"].append('0.5505339105281907')
    dict["Median Symmetric Accuracy (MdSA)"].append('2.0078825180431')
    dict.update({"Time Profile Selection Plot": ['./tests/output/plots/Time_Profile_Test_model_0_min.10.0.max.-1.0.units.MeV_threshold_1.0_20000101T000000.pdf;./tests/output/plots/Time_Profile_Test_model_0_min.10.0.max.-1.0.units.MeV_threshold_1.0_20000101T000000.pdf']})

    return dict

def fill_all_clear_dict_all(dict, self):
        """ Fill the all clear metrics dictionary with the 'known' outputs.
        """
        dict["Model"].append('Test_model_0')
        dict["Energy Channel"].append(self.energy_key)
        dict["Threshold"].append(self.obs_thresholds[self.energy_key][0])
        dict["Prediction Energy Channel"].append(self.energy_key)
        dict["Prediction Threshold"].append(self.obs_thresholds[self.energy_key][0])
        dict["All Clear 'True Positives' (Hits)"].append('2') #Hits
        dict["All Clear 'False Positives' (False Alarms)"].append('0') #False Alarms
        dict["All Clear 'True Negatives' (Correct Negatives)"].append('1')  #Correct negatives
        dict["All Clear 'False Negatives' (Misses)"].append('0') #Misses
        dict["N (Total Number of Forecasts)"].append('3')
        dict["Percent Correct"].append('1.0')
        dict["Bias"].append('1.0')
        dict["Hit Rate"].append('1.0')
        dict["False Alarm Rate"].append('0.0')
        dict['False Negative Rate'].append('0.0')
        dict["Frequency of Misses"].append('0.0')
        dict["Frequency of Hits"].append('1.0')
        dict["Probability of Correct Negatives"].append('1.0')
        dict["Frequency of Correct Negatives"].append('1.0')
        dict["False Alarm Ratio"].append('0.0')
        dict["Detection Failure Ratio"].append('0.0')
        dict["Threat Score"].append('1.0') #Critical Success Index
        dict["Odds Ratio"].append('inf')
        dict["Gilbert Skill Score"].append('1.0') #Equitable Threat Score
        dict["True Skill Statistic"].append('1.0') #Hanssen and Kuipers
                #discriminant (true skill statistic, Peirce's skill score)
        dict["Heidke Skill Score"].append('1.0')
        dict["Odds Ratio Skill Score"].append('1.0')
        dict["Symmetric Extreme Dependency Score"].append('1.0')
        dict["F1 Score"].append('1.0'),
        dict["F2 Score"].append('1.0'),
        dict["Fhalf Score"].append('1.0'),
        dict['Prevalence'].append('0.6666666666666666'),
        dict['Matthew Correlation Coefficient'].append('1.0'),
        dict['Informedness'].append('1.0'),
        dict['Markedness'].append('1.0'),
        dict['Prevalence Threshold'].append('0.0'),
        dict['Balanced Accuracy'].append('1.0'),
        dict['Fowlkes-Mallows Index'].append('1.0'),
        dict["Number SEP Events Correctly Predicted"].append('1')
        dict["Number SEP Events Missed"].append('0')
        dict["Predicted SEP Events"].append('2000-01-01 01:00:00')
        dict["Missed SEP Events"].append('None')
        return dict

def fill_awt_dict_all(dict, self):
    """ Metrics for Adanced Warning Time to SEP start, SEP peak, SEP end.
        The "Forecasted Value" field indicates which forecasted quantity
        was used to calculate the AWT.
    """
    dict["Model"].append('Test_model_0')
    dict["Energy Channel"].append(self.energy_key)
    dict["Threshold"].append(self.obs_thresholds[self.energy_key][0])
    dict["Prediction Energy Channel"].append(self.energy_key)
    dict["Prediction Threshold"].append(self.obs_thresholds[self.energy_key][0])
            

    dict["Mean AWT for Predicted SEP All Clear to Observed SEP Threshold Crossing Time"].append('1.0')
    dict["Median AWT for Predicted SEP All Clear to Observed SEP Threshold Crossing Time"].append('1.0')
    dict["Mean AWT for Predicted SEP All Clear to Observed SEP Start Time"].append('1.0')
    dict["Median AWT for Predicted SEP All Clear to Observed SEP Start Time"].append('1.0')

#            #Probability Forecasts - cannot without an explicit threshold
#            "Mean AWT for Probability to Observed Threshold Crossing Time": [],
#            "Median AWT for Probability to Observed Threshold Crossing Time": [],
#            "Mean AWT for Probability to Observed Start Time": [],
#            "Median AWT for Probability to Observed Start Time": [],

            #Threshold Crossing Time Forecasts
    dict["Mean AWT for Predicted SEP Threshold Crossing Time to Observed SEP Threshold Crossing Time"].append('1.0')
    dict["Median AWT for Predicted SEP Threshold Crossing Time to Observed SEP Threshold Crossing Time"].append('1.0')
    dict["Mean AWT for Predicted SEP Threshold Crossing Time to Observed SEP Start Time"].append('1.0')
    dict["Median AWT for Predicted SEP Threshold Crossing Time to Observed SEP Start Time"].append('1.0')

            #Start Time Forecasts
    dict["Mean AWT for Predicted SEP Start Time to Observed SEP Threshold Crossing Time"].append('1.0')
    dict["Median AWT for Predicted SEP Start Time to Observed SEP Threshold Crossing Time"].append('1.0')
    dict["Mean AWT for Predicted SEP Start Time to Observed SEP Start Time"].append('1.0')
    dict["Median AWT for Predicted SEP Start Time to Observed SEP Start Time"].append('1.0')
 
#             #Point Intensity Forecasts
#            "Mean AWT for Predicted Point Intensity to Observed SEP Threshold Crossing Time": [],
#            "Median AWT for Predicted Point Intensity to Observed SEP Threshold Crossing Time": [],
#            "Mean AWT for Predicted Point Intensity to Observed SEP Start Time": [],
#            "Median AWT for Predicted Point Intensity to Observed SEP Start Time": [],
 
 
            #Peak Intensity Forecasts
    dict["Mean AWT for Predicted SEP Peak Intensity (Onset Peak) to Observed SEP Threshold Crossing Time"].append('1.0')
    dict["Median AWT for Predicted SEP Peak Intensity (Onset Peak) to Observed SEP Threshold Crossing Time"].append('1.0')
    dict["Mean AWT for Predicted SEP Peak Intensity (Onset Peak) to Observed SEP Start Time"].append('1.0')
    dict["Median AWT for Predicted SEP Peak Intensity (Onset Peak) to Observed SEP Start Time"].append('1.0')
    dict["Mean AWT for Predicted SEP Peak Intensity (Onset Peak) to Observed SEP Peak Intensity (Onset Peak) Time"].append('1.0')
    dict["Median AWT for Predicted SEP Peak Intensity (Onset Peak) to Observed SEP Peak Intensity (Onset Peak) Time"].append('1.0')

            #Peak Intensity Max Forecasts
    dict["Mean AWT for Predicted SEP Peak Intensity Max (Max Flux) to Observed SEP Threshold Crossing Time"].append('1.0')
    dict["Median AWT for Predicted SEP Peak Intensity Max (Max Flux) to Observed SEP Threshold Crossing Time"].append('1.0')
    dict["Mean AWT for Predicted SEP Peak Intensity Max (Max Flux) to Observed SEP Start Time"].append('1.0')
    dict["Median AWT for Predicted SEP Peak Intensity Max (Max Flux) to Observed SEP Start Time"].append('1.0')
    dict["Mean AWT for Predicted SEP Peak Intensity Max (Max Flux) to Observed SEP Peak Intensity Max (Max Flux) Time"].append('1.0')
    dict["Median AWT for Predicted SEP Peak Intensity Max (Max Flux) to Observed SEP Peak Intensity Max (Max Flux) Time"].append('1.0')

            #End Time Forecasts
    dict["Mean AWT for Predicted SEP End Time to Observed SEP Threshold Crossing Time"].append('1.0')
    dict["Median AWT for Predicted SEP End Time to Observed SEP Threshold Crossing Time"].append('1.0')
    dict["Mean AWT for Predicted SEP End Time to Observed SEP Start Time"].append('1.0')
    dict["Median AWT for Predicted SEP End Time to Observed SEP Start Time"].append('1.0')
    dict["Mean AWT for Predicted SEP End Time to Observed SEP End Time"].append('24.0')
    dict["Median AWT for Predicted SEP End Time to Observed SEP End Time"].append('24.0')
            
    return dict


def fill_duration_metrics_dict_all(dict, self):
    dict["Model"].append('Test_model_0')
    dict["Energy Channel"].append(self.energy_key)
    dict["Threshold"].append(self.obs_thresholds[self.energy_key][0])
    dict["Prediction Energy Channel"].append(self.energy_key)
    dict["Prediction Threshold"].append(self.obs_thresholds[self.energy_key][0])
    dict["Mean Error (pred - obs)"].append('-11.0')
    dict["Median Error (pred - obs)"].append('-11.0')
    dict["Mean Absolute Error (|pred - obs|)"].append('11.0')
    dict["Median Absolute Error (|pred - obs|)"].append('11.0')
    return dict

def fill_end_time_metrics_dict_all(dict, self):
    dict["Model"].append('Test_model_0')
    dict["Energy Channel"].append(self.energy_key)
    dict["Threshold"].append(self.obs_thresholds[self.energy_key][0])
    dict["Prediction Energy Channel"].append(self.energy_key)
    dict["Prediction Threshold"].append(self.obs_thresholds[self.energy_key][0])
    dict["Mean Error (pred - obs)"].append('-6.0')
    dict["Median Error (pred - obs)"].append('-6.0')
    dict["Mean Absolute Error (|pred - obs|)"].append('6.0')
    dict["Median Absolute Error (|pred - obs|)"].append('6.0')
    return dict

def fill_last_data_to_issue_time_metrics_dict_all(dict, self):
    dict["Model"].append('Test_model_0')
    dict["Energy Channel"].append(self.energy_key)
    dict["Threshold"].append(self.obs_thresholds[self.energy_key][0])
    dict["Prediction Energy Channel"].append(self.energy_key)
    dict["Prediction Threshold"].append(self.obs_thresholds[self.energy_key][0])
    dict["Mean Error (pred - obs)"].append('30.333333333333332')
    dict["Median Error (pred - obs)"].append('30.0')
    dict["Mean Absolute Error (|pred - obs|)"].append('30.333333333333332')
    dict["Median Absolute Error (|pred - obs|)"].append('30.0')
    return dict

def fill_max_flux_in_pred_win_metrics_dict_all(dict, self):
    dict["Model"].append('Test_model_0')
    dict["Energy Channel"].append(self.energy_key)
    dict["Threshold"].append(self.obs_thresholds[self.energy_key][0])
    dict["Prediction Energy Channel"].append(self.energy_key)
    dict["Prediction Threshold"].append(self.obs_thresholds[self.energy_key][0])
    dict["Scatter Plot"].append('./tests/output/plots/Correlation_max_flux_in_pred_win_Test_model_0_min.10.0.max.-1.0.units.MeV_threshold_1.0.pdf')
    dict["Linear Regression Slope"].append('0.5103481712895563')
    dict["Linear Regression y-intercept"].append('0.5103481712895562')
    dict["Pearson Correlation Coefficient (Linear)"].append('')
    dict["Pearson Correlation Coefficient (Log)"].append('')
    dict["Spearman Correlation Coefficient (Linear)"].append('')
    dict["Mean Error (ME)"].append('0.5')
    dict["Median Error (MedE)"].append('0.5')
    dict["Mean Log Error (MLE)"].append('0.02069634257911257')
    dict["Median Log Error (MedLE)"].append('0.02069634257911257')
    dict["Mean Absolute Error (MAE)"].append('0.5')
    dict["Median Absolute Error (MedAE)"].append('0.5')
    dict["Mean Absolute Log Error (MALE)"].append('0.02069634257911257')
    dict["Median Absolute Log Error (MedALE)"].append('0.02069634257911257')
    dict["Mean Percent Error (MPE)"].append('0.05')
    dict["Mean Absolute Percent Error (MAPE)"].append('0.05')
    dict["Mean Symmetric Percent Error (MSPE)"].append('0.047619047619047616')
    dict["Mean Symmetric Absolute Percent Error (SMAPE)"].append('0.047619047619047616')
    dict["Mean Accuracy Ratio (MAR)"].append('1.05')
    dict["Root Mean Square Error (RMSE)"].append('0.7071067811865476')
    dict["Root Mean Square Log Error (RMSLE)"].append('0.02926904836690076')
    dict["Median Symmetric Accuracy (MdSA)"].append('0.04880884817015163')
    dict.update({"Time Profile Selection Plot": ['./tests/output/plots/Time_Profile_Test_model_0_min.10.0.max.-1.0.units.MeV_threshold_1.0_20000101T000000.pdf']})

    return dict

def fill_peak_intensity_max_time_metrics_dict_all(dict, self):
    dict["Model"].append('Test_model_0')
    dict["Energy Channel"].append(self.energy_key)
    dict["Threshold"].append(self.obs_thresholds[self.energy_key][0])
    dict["Prediction Energy Channel"].append(self.energy_key)
    dict["Prediction Threshold"].append(self.obs_thresholds[self.energy_key][0])
    dict["Mean Error (pred - obs)"].append('11.0')
    dict["Median Error (pred - obs)"].append('11.0')
    dict["Mean Absolute Error (|pred - obs|)"].append('11.0')
    dict["Median Absolute Error (|pred - obs|)"].append('11.0')
    return dict

def fill_peak_intensity_max_metrics_dict_all(dict, self):
    dict["Model"].append('Test_model_0')
    dict["Energy Channel"].append(self.energy_key)
    dict["Threshold"].append(self.obs_thresholds[self.energy_key][0])
    dict["Prediction Energy Channel"].append(self.energy_key)
    dict["Prediction Threshold"].append(self.obs_thresholds[self.energy_key][0])
    dict["Scatter Plot"].append('./tests/output/plots/Correlation_peak_intensity_max_Test_model_0_min.10.0.max.-1.0.units.MeV_threshold_1.0.pdf')
    dict["Linear Regression Slope"].append('0.5103481712895563')
    dict["Linear Regression y-intercept"].append('0.5103481712895562')
    dict["Pearson Correlation Coefficient (Linear)"].append('')
    dict["Pearson Correlation Coefficient (Log)"].append('')
    dict["Spearman Correlation Coefficient (Linear)"].append('')
    dict["Mean Error (ME)"].append('0.5')
    dict["Median Error (MedE)"].append('0.5')
    dict["Mean Log Error (MLE)"].append('0.02069634257911257')
    dict["Median Log Error (MedLE)"].append('0.02069634257911257')
    dict["Mean Absolute Error (MAE)"].append('0.5')
    dict["Median Absolute Error (MedAE)"].append('0.5')
    dict["Mean Absolute Log Error (MALE)"].append('0.02069634257911257')
    dict["Median Absolute Log Error (MedALE)"].append('0.02069634257911257')
    dict["Mean Percent Error (MPE)"].append('0.05')
    dict["Mean Absolute Percent Error (MAPE)"].append('0.05')
    dict["Mean Symmetric Percent Error (MSPE)"].append('0.047619047619047616')
    dict["Mean Symmetric Absolute Percent Error (SMAPE)"].append('0.047619047619047616')
    dict["Mean Accuracy Ratio (MAR)"].append('1.05')
    dict["Root Mean Square Error (RMSE)"].append('0.7071067811865476')
    dict["Root Mean Square Log Error (RMSLE)"].append('0.02926904836690076')
    dict["Median Symmetric Accuracy (MdSA)"].append('0.04880884817015163')
    dict.update({"Time Profile Selection Plot": ['']})

    return dict

def fill_peak_intensity_time_dict_all(dict, self):
    """ Fill in metrics for time
    """
    dict["Model"].append('Test_model_0')
    dict["Energy Channel"].append(self.energy_key)
    dict["Threshold"].append(self.obs_thresholds[self.energy_key][0])
    dict["Prediction Energy Channel"].append(self.energy_key)
    dict["Prediction Threshold"].append(self.obs_thresholds[self.energy_key][0])
    dict["Mean Error (pred - obs)"].append('11.0')
    dict["Median Error (pred - obs)"].append('11.0')
    dict["Mean Absolute Error (|pred - obs|)"].append('11.0')
    dict["Median Absolute Error (|pred - obs|)"].append('11.0')
    return dict

def fill_start_time_dict_all(dict, self):
    """ Fill in metrics for time
    """
    dict["Model"].append('Test_model_0')
    dict["Energy Channel"].append(self.energy_key)
    dict["Threshold"].append(self.obs_thresholds[self.energy_key][0])
    dict["Prediction Energy Channel"].append(self.energy_key)
    dict["Prediction Threshold"].append(self.obs_thresholds[self.energy_key][0])
    dict["Mean Error (pred - obs)"].append('5.0')
    dict["Median Error (pred - obs)"].append('5.0')
    dict["Mean Absolute Error (|pred - obs|)"].append('5.0')
    dict["Median Absolute Error (|pred - obs|)"].append('5.0')
    return dict

def fill_threshold_crossing_time_dict_all(dict, self):
    """ Fill in metrics for time
    """
    dict["Model"].append('Test_model_0')
    dict["Energy Channel"].append(self.energy_key)
    dict["Threshold"].append(self.obs_thresholds[self.energy_key][0])
    dict["Prediction Energy Channel"].append(self.energy_key)
    dict["Prediction Threshold"].append(self.obs_thresholds[self.energy_key][0])
    dict["Mean Error (pred - obs)"].append('5.0')
    dict["Median Error (pred - obs)"].append('5.0')
    dict["Mean Absolute Error (|pred - obs|)"].append('5.0')
    dict["Median Absolute Error (|pred - obs|)"].append('5.0')
    return dict

def fill_fluence_dict_all(dict, self):
    dict["Model"].append('Test_model_0')
    dict["Energy Channel"].append(self.energy_key)
    dict["Threshold"].append(self.obs_thresholds[self.energy_key][0])
    dict["Prediction Energy Channel"].append(self.energy_key)
    dict["Prediction Threshold"].append(self.obs_thresholds[self.energy_key][0])
    dict["Scatter Plot"].append('./tests/output/plots/Correlation_fluence_Test_model_0_min.10.0.max.-1.0.units.MeV_threshold_1.0.pdf')
    dict["Linear Regression Slope"].append('0.42195511876358965')
    dict["Linear Regression y-intercept"].append('3.36735874637887')
    dict["Pearson Correlation Coefficient (Linear)"].append('')
    dict["Pearson Correlation Coefficient (Log)"].append('')
    dict["Spearman Correlation Coefficient (Linear)"].append('')
    dict["Mean Error (ME)"].append('-90152088.20071295')
    dict["Median Error (MedE)"].append('-90152088.20071295')
    dict["Mean Log Error (MLE)"].append('-1.2456543445264794')
    dict["Median Log Error (MedLE)"].append('-1.2456543445264794')
    dict["Mean Absolute Error (MAE)"].append('90152088.20071295')
    dict["Median Absolute Error (MedAE)"].append('90152088.20071295')
    dict["Mean Absolute Log Error (MALE)"].append('1.2456543445264794')
    dict["Median Absolute Log Error (MedALE)"].append('1.2456543445264794')
    dict["Mean Percent Error (MPE)"].append('-0.9432003505386932')
    dict["Mean Absolute Percent Error (MAPE)"].append('0.9432003505386932')
    dict["Mean Symmetric Percent Error (MSPE)"].append('-1.785012610516062')
    dict["Mean Symmetric Absolute Percent Error (SMAPE)"].append('1.785012610516062')
    dict["Mean Accuracy Ratio (MAR)"].append('0.05679964946130683')
    dict["Root Mean Square Error (RMSE)"].append('90152088.20071295')
    dict["Root Mean Square Log Error (RMSLE)"].append('1.2456543445264794')
    dict["Median Symmetric Accuracy (MdSA)"].append('16.605742455879447')
    dict.update({"Time Profile Selection Plot": ['./tests/output/plots/Time_Profile_Test_model_0_min.10.0.max.-1.0.units.MeV_threshold_1.0_20000101T000000.pdf']})

    return dict

class TestAllClear0(unittest.TestCase):
    


    def load_verbosity(self):
        self.verbosity = utility_get_verbosity()
    
    

    def step_0(self): 
        validate.prepare_outdirs()
        


        self.energy_channel = {'min': 10, 'max': -1, 'units': 'MeV'}
        self.energy_key = objh.energy_channel_to_key(self.energy_channel)
        self.all_energy_channels = [self.energy_key] 
        self.model_names = ['Test_model_0']
        observation_json = './tests/files/observations/validation/all_clear/all_clear_false.json'
        observation = utility_load_observation(observation_json, self.energy_channel)
        observation_objects = {self.energy_key: [observation]}
        self.verbosity = utility_get_verbosity()
        forecast_json = './tests/files/forecasts/validation/all_clear/pred_all_clear_false.json'
        forecast = utility_load_forecast(forecast_json, self.energy_channel)
        forecast_objects = {self.energy_key: [forecast]}
        self.sphinx, self.obs_thresholds, self.obs_sep_events = utility_match_sphinx(self.all_energy_channels,\
             self.model_names, observation_objects, forecast_objects)
        self.profname_dict = None
        self.DoResume = False
        
        self.validation_quantity = ['all_clear', 'awt', 'duration', 'end_time', 'fluence', 'max_flux_in_pred_window', 'peak_intensity_max', 'peak_intensity_max_time', 'peak_intensity' \
            'peak_intensity_time', 'probability', 'start_time', 'threshold_crossing', 'time_profile']
        self.quantities_tested = ['all_clear', 'awt']
        self.validation_type = ['All']
        
    def step_1(self):
        """
        Tests that the dataframe is built correctly with the correct fields being filled in/added.
        The observation and forecast have exactly the same observation/prediction windows. 
        Matching requires (at a minimum) that there is a prediction window start/end with an observed
        SEP start time within the prediction window and that the last data time/trigger occur before the
        observed SEP start time.

        Observed all clear is False
        Forecast all clear is False
        """
        self.dataframe = validate.fill_df(self.sphinx, self.model_names, self.all_energy_channels, \
            self.obs_thresholds, self.profname_dict)
        for keywords in self.dataframe:
            # temp = self.sphinx['Test_model_0'][self.energy_key].prediction.short_name\
            
            temp = attributes_of_sphinx_obj(keywords, self.sphinx['Test_model_0'][self.all_energy_channels[0]][0],\
                 self.energy_key, self.obs_thresholds)
            # if keywords == 'Duration Match Status': # Leaving this commented out until Katie determines if the correct logic is being used to determine this match status
            #     print(self.dataframe[keywords][0], temp, 'WHY IS THIS BUGGED')
            self.assertEqual(self.dataframe[keywords][0], temp, 'Error is in keyword ' + keywords)

    def step_2(self):
        print('Test for DoResume feature - this is false right now though so moving on after a quick assertFalse')
        self.assertFalse(self.DoResume)
    
    

    def step_3(self):
        
        with patch('sphinxval.utils.config.outpath', './tests/output'):
            validate.calculate_intuitive_metrics(self.dataframe, self.model_names, self.all_energy_channels, \
                self.obs_thresholds, 'All')
            print('Made it to step 3')
        # for model in self.model_names:
        #     for quantities in self.validation_quantity:
        #         if quantities in self.quantities_tested:
        #             metrics_filename = './output\\csv\\' + quantities + '_metrics' 
        #             self.assertTrue(os.path.isfile(metrics_filename + '.csv'), msg = metrics_filename + '.csv does not exist, check the file is output correctly')
        #             metrics_filename = './output\\pkl\\' + quantities + '_metrics' 
        #             self.assertTrue(os.path.isfile(metrics_filename + '.pkl'), msg = metrics_filename + '.pkl does not exist, check the file is output correctly')
        #         else:
        #             filename =  './output\\csv\\' + quantities + '_metrics.csv'
        #             self.assertFalse(os.path.isfile(metrics_filename), msg = metrics_filename + ' should not exist') 
                
        #         for energy_channels in self.all_energy_channels:
        #             for thresholds in self.obs_thresholds[energy_channels]:
        #                 threshold_shortened = thresholds.rsplit('.')[0]+ '_' + thresholds.rsplit('.')[1] + '.' + thresholds.rsplit('.')[2]

        #                 if quantities == 'awt':
        #                     # pkl_filename = './output\\pkl\\' + quantities + '_selections_' + model + '_' + energy_channels + '_' + threshold_shortened + '.pkl'
        #                     # csv_filename = './output\\csv\\' + quantities + '_selections_' + model + '_' + energy_channels + '_' + threshold_shortened + '.csv'
        #                     pkl_filename = '.\\output\\pkl\\' + quantities + '_selections_' + model + '_' + energy_channels + '_' + threshold_shortened + "_Predicted SEP All Clear.pkl"
        #                     csv_filename = '.\\output\\csv\\' + quantities + '_selections_' + model + '_' + energy_channels + '_' + threshold_shortened + "_Predicted SEP All Clear.csv"
        #                     self.assertTrue(os.path.isfile(pkl_filename) , \
        #                         msg = pkl_filename + ' does not exist, check the file is output correctly')
        #                     self.assertTrue(os.path.isfile(csv_filename), \
        #                         msg = csv_filename + ' does not exist, check the file is output correctly')
                            
        #                 else:
        #                     pkl_filename = './output\\pkl\\' + quantities + '_selections_' + model + '_' + energy_channels + '_' + threshold_shortened + '.pkl'
        #                     csv_filename = './output\\csv\\' + quantities + '_selections_' + model + '_' + energy_channels + '_' + threshold_shortened + '.csv'
        #                     if quantities in self.quantities_tested:
        #                         self.assertTrue(os.path.isfile(pkl_filename), msg = pkl_filename + ' does not exist, check the file is output correctly')
        #                         self.assertTrue(os.path.isfile(csv_filename), msg = csv_filename + ' does not exist, check the file is output correctly')
        #                     else:
        #                         self.assertFalse(os.path.isfile(pkl_filename), msg = pkl_filename + ' should not exist')
        #                         self.assertFalse(os.path.isfile(csv_filename), msg = csv_filename + ' should not exist')


    def step_4(self):
        print('In Step 4')
        test_dict = initialize_all_clear_dict()
        test_dict = fill_all_clear_dict_hit(test_dict, self)
        
        csv_filename = './tests/output\\csv\\all_clear_metrics.csv'
        keyword_all_clear = []
        with open(csv_filename, mode = 'r') as csvfile:
            reading = csv.reader(csvfile, delimiter = ',')
            
            for row in reading:
                if keyword_all_clear == []:
                    keyword_all_clear = row
                else:
                    for j in range(len(row)):
                        if keyword_all_clear[j] == '':
                            pass
                        else:
                            keyword = keyword_all_clear[j]
                            # print(keyword, row[j], test_dict[keyword][0])
                            self.assertEqual(row[j], test_dict[keyword][0], msg = 'This is the keyword ' + keyword)




    def step_5(self):
        print('In Step 5')
        test_dict = initialize_awt_dict()
        test_dict = fill_awt_dict(test_dict, self)

        csv_filename = './tests/output\\csv\\awt_metrics.csv'
        keyword_all_clear = []
        with open(csv_filename, mode = 'r') as csvfile:
            reading = csv.reader(csvfile, delimiter = ',')
            
            for row in reading:
                if keyword_all_clear == []:
                    keyword_all_clear = row
                else:
                    for j in range(len(row)):
                        if keyword_all_clear[j] == '':
                            pass
                        else:
                            keyword = keyword_all_clear[j]
                            # print(keyword, row[j], test_dict[keyword][0])
                            self.assertEqual(row[j], test_dict[keyword][0], msg = 'This is the keyword ' + keyword)

  
    def utility_print_docstring(self, function):
        if self.verbosity == 2:
            print('\n//----------------------------------------------------')
            print(function.__doc__)

    def _steps(self):
        for name in dir(self): # dir() result is implicitly sorted
            if name.startswith("step"):
                yield name, getattr(self, name)
        
    # @make_docstring_printable
    # def test_df_all_clear_1(this, self):
    #     """
    #     Tests that the dataframe is built correctly with the correct fields being filled in/added.
    #     The observation and forecast have exactly the same observation/prediction windows. 
    #     Matching requires (at a minimum) that there is a prediction window start/end with an observed
    #     SEP start time within the prediction window and that the last data time/trigger occur before the
    #     observed SEP start time.
    #     Observed all clear is False
    #     Forecast all clear is False

    #     The tests in this block are:
    #         Observed SEP All Clear is False
    #         Predicted SEP All Clear is False

    #     """
    @patch('sphinxval.utils.config.outpath', './tests/output')

    def test_all_clear_0(self):
        validate.prepare_outdirs()
        print("TestAllClear0")
        for name, step in self._steps():
            try:
                step()
            except Exception as e:
                self.fail("{} failed ({}: {})".format(step, type(e), e))
        utility_delete_output()


class TestAllClear1(unittest.TestCase):

    def load_verbosity(self):
        self.verbosity = utility_get_verbosity()
    
    

    def step_0(self): 
        validate.prepare_outdirs()
        


        self.energy_channel = {'min': 10, 'max': -1, 'units': 'MeV'}
        self.energy_key = objh.energy_channel_to_key(self.energy_channel)
        self.all_energy_channels = [self.energy_key] 
        self.model_names = ['Test_model_0']
        observation_json = './tests/files/observations/validation/all_clear/all_clear_true.json'
        observation = utility_load_observation(observation_json, self.energy_channel)
        observation_objects = {self.energy_key: [observation]}
        self.verbosity = utility_get_verbosity()
        forecast_json = './tests/files/forecasts/validation/all_clear/pred_all_clear_true.json'
        forecast = utility_load_forecast(forecast_json, self.energy_channel)
        forecast_objects = {self.energy_key: [forecast]}
        self.sphinx, self.obs_thresholds, self.obs_sep_events = utility_match_sphinx(self.all_energy_channels,\
             self.model_names, observation_objects, forecast_objects)
        self.profname_dict = None
        self.DoResume = False
        
        self.validation_quantity = ['all_clear', 'awt', 'duration', 'end_time', 'fluence', 'max_flux_in_pred_window', 'peak_intensity_max', 'peak_intensity_max_time', 'peak_intensity' \
            'peak_intensity_time', 'probability', 'start_time', 'threshold_crossing', 'time_profile']
        self.quantities_tested = ['all_clear']
        self.validation_type = ['All']
        
    def step_1(self):
        """
        Tests that the dataframe is built correctly with the correct fields being filled in/added.
        The observation and forecast have exactly the same observation/prediction windows. 
        Matching requires (at a minimum) that there is a prediction window start/end with an observed
        SEP start time within the prediction window and that the last data time/trigger occur before the
        observed SEP start time.
        Observed all clear is False
        Forecast all clear is False
        """
        self.dataframe = validate.fill_df(self.sphinx, self.model_names, self.all_energy_channels, \
            self.obs_thresholds, self.profname_dict)
        for keywords in self.dataframe:
            # temp = self.sphinx['Test_model_0'][self.energy_key].prediction.short_name\
            
            temp = attributes_of_sphinx_obj(keywords, self.sphinx['Test_model_0'][self.all_energy_channels[0]][0],\
                 self.energy_key, self.obs_thresholds)
            # if keywords == 'Duration Match Status': # Leaving this commented out until Katie determines if the correct logic is being used to determine this match status
            #     print(self.dataframe[keywords][0], temp, 'WHY IS THIS BUGGED')
            self.assertEqual(self.dataframe[keywords][0], temp, 'Error is in keyword ' + keywords)

    def step_2(self):
        print('Test for DoResume feature - this is false right now though so moving on after a quick assertFalse')
        self.assertFalse(self.DoResume)
    
    def step_3(self):
        validate.calculate_intuitive_metrics(self.dataframe, self.model_names, self.all_energy_channels, \
            self.obs_thresholds, 'All')
        print('Made it to step 3')
        # for model in self.model_names:
        #     for quantities in self.validation_quantity:
        #         if quantities in self.quantities_tested:
        #             metrics_filename = './output\\csv\\' + quantities + '_metrics' 
        #             self.assertTrue(os.path.isfile(metrics_filename + '.csv'), msg = metrics_filename + '.csv does not exist, check the file is output correctly')
        #             metrics_filename = './output\\pkl\\' + quantities + '_metrics' 
        #             self.assertTrue(os.path.isfile(metrics_filename + '.pkl'), msg = metrics_filename + '.pkl does not exist, check the file is output correctly')
        #         else:
        #             filename =  './output\\csv\\' + quantities + '_metrics.csv'
        #             self.assertFalse(os.path.isfile(metrics_filename), msg = metrics_filename + ' should not exist') 
                
        #         for energy_channels in self.all_energy_channels:
        #             for thresholds in self.obs_thresholds[energy_channels]:
        #                 threshold_shortened = thresholds.rsplit('.')[0]+ '_' + thresholds.rsplit('.')[1] + '.' + thresholds.rsplit('.')[2]

        #                 # if quantities == 'awt':
        #                 #     # pkl_filename = './output\\pkl\\' + quantities + '_selections_' + model + '_' + energy_channels + '_' + threshold_shortened + '.pkl'
        #                 #     # csv_filename = './output\\csv\\' + quantities + '_selections_' + model + '_' + energy_channels + '_' + threshold_shortened + '.csv'
        #                 #     pkl_filename = '.\\output\\pkl\\' + quantities + '_selections_' + model + '_' + energy_channels + '_' + threshold_shortened + "_Predicted SEP All Clear.pkl"
        #                 #     csv_filename = '.\\output\\csv\\' + quantities + '_selections_' + model + '_' + energy_channels + '_' + threshold_shortened + "_Predicted SEP All Clear.csv"
        #                 #     self.assertTrue(os.path.isfile(pkl_filename) , \
        #                 #         msg = pkl_filename + ' does not exist, check the file is output correctly')
        #                 #     self.assertTrue(os.path.isfile(csv_filename), \
        #                 #         msg = csv_filename + ' does not exist, check the file is output correctly')
                            
        #                 # else:
        #                 pkl_filename = './output\\pkl\\' + quantities + '_selections_' + model + '_' + energy_channels + '_' + threshold_shortened + '.pkl'
        #                 csv_filename = './output\\csv\\' + quantities + '_selections_' + model + '_' + energy_channels + '_' + threshold_shortened + '.csv'
        #                 if quantities in self.quantities_tested:
        #                     self.assertTrue(os.path.isfile(pkl_filename), msg = pkl_filename + ' does not exist, check the file is output correctly')
        #                     self.assertTrue(os.path.isfile(csv_filename), msg = csv_filename + ' does not exist, check the file is output correctly')
        #                 else:
        #                     self.assertFalse(os.path.isfile(pkl_filename), msg = pkl_filename + ' should not exist')
        #                     self.assertFalse(os.path.isfile(csv_filename), msg = csv_filename + ' should not exist')


    def step_4(self):
        print('In Step 4')
        test_dict = initialize_all_clear_dict()
        test_dict = fill_all_clear_dict_CN(test_dict, self)

        csv_filename = './tests/output\\csv\\all_clear_metrics.csv'
        keyword_all_clear = []
        with open(csv_filename, mode = 'r') as csvfile:
            reading = csv.reader(csvfile, delimiter = ',')
            
            for row in reading:
                if keyword_all_clear == []:
                    keyword_all_clear = row
                else:
                    for j in range(len(row)):
                        if keyword_all_clear[j] == '':
                            pass
                        else:
                            keyword = keyword_all_clear[j]
                            # print(keyword, row[j], test_dict[keyword][0])
                            self.assertEqual(row[j], test_dict[keyword][0], msg = 'This is the keyword ' + keyword)


    def utility_print_docstring(self, function):
        if self.verbosity == 2:
            print('\n//----------------------------------------------------')
            print(function.__doc__)

    def _steps(self):
        for name in dir(self): # dir() result is implicitly sorted
            if name.startswith("step"):
                yield name, getattr(self, name)
        
    # @make_docstring_printable
    # def test_df_all_clear_1(this, self):
    #     """
    #     Tests that the dataframe is built correctly with the correct fields being filled in/added.
    #     The observation and forecast have exactly the same observation/prediction windows. 
    #     Matching requires (at a minimum) that there is a prediction window start/end with an observed
    #     SEP start time within the prediction window and that the last data time/trigger occur before the
    #     observed SEP start time.
    #     Observed all clear is False
    #     Forecast all clear is False

    #     The tests in this block are:
    #         Observed SEP All Clear is False
    #         Predicted SEP All Clear is False

    #     """
    @patch('sphinxval.utils.config.outpath', './tests/output')

    def test_all_clear_1(self):
        validate.prepare_outdirs()
        print("TestAllClear1")
        for name, step in self._steps():
            try:
                step()
            except Exception as e:
                self.fail("{} failed ({}: {})".format(step, type(e), e))
        utility_delete_output()


class TestAllClearGarbage(unittest.TestCase):

    def load_verbosity(self):
        self.verbosity = utility_get_verbosity()
    
    
    @patch('sphinxval.utils.config.outpath', './tests/output')

    def test_garbage(self): 
        validate.prepare_outdirs()
        print("TestAllClearGarbo")

        with self.assertRaises(NameError, msg = 'Giving purposeful garbage, should exit'):
            self.energy_channel = {'min': 10, 'max': -1, 'units': 'MeV'}
            self.energy_key = objh.energy_channel_to_key(self.energy_channel)
            self.all_energy_channels = [self.energy_key] 
            self.model_names = ['Test_model_0']
            observation_json = './tests/files/observations/validation/all_clear/all_clear_false.json'
            observation = utility_load_observation(observation_json, self.energy_channel)
            observation_objects = {self.energy_key: [observation]}
            self.verbosity = utility_get_verbosity()
            forecast_json = './tests/files/forecasts/validation/all_clear/pred_all_clear_garbage.json'
            forecast = utility_load_forecast(forecast_json, self.energy_channel)
            forecast_objects = {self.energy_key: [forecast]}
            self.sphinx, self.obs_thresholds, self.obs_sep_events = utility_match_sphinx(self.all_energy_channels,\
                self.model_names, observation_objects, forecast_objects)





class TestPeakIntensity0(unittest.TestCase):

    def load_verbosity(self):
        self.verbosity = utility_get_verbosity()
    
    

    def step_0(self): 
        validate.prepare_outdirs()
        self.energy_channel = {'min': 10, 'max': -1, 'units': 'MeV'}
        self.energy_key = objh.energy_channel_to_key(self.energy_channel)
        self.all_energy_channels = [self.energy_key] 
        self.model_names = ['Test_model_0']
        observation_json = './tests/files/observations/validation/all_clear/all_clear_false.json'
        observation = utility_load_observation(observation_json, self.energy_channel)
        observation_objects = {self.energy_key: [observation]}
        self.verbosity = utility_get_verbosity()
        forecast_json = './tests/files/forecasts/validation/onset_peak/pred_all_clear_false.json'
        forecast = utility_load_forecast(forecast_json, self.energy_channel)
        forecast_objects = {self.energy_key: [forecast]}
        self.sphinx, self.obs_thresholds, self.obs_sep_events = utility_match_sphinx(self.all_energy_channels,\
             self.model_names, observation_objects, forecast_objects)
        self.profname_dict = None
        self.DoResume = False
        
        self.validation_quantity = ['all_clear', 'awt', 'duration', 'end_time', 'fluence', 'max_flux_in_pred_window', 'peak_intensity_max', 'peak_intensity_max_time', 'peak_intensity' \
            'peak_intensity_time', 'probability', 'start_time', 'threshold_crossing', 'time_profile']
        self.quantities_tested = ['all_clear', 'peak_intensity']
        self.validation_type = ['All']
        
    def step_1(self):
        """
        Tests that the dataframe is built correctly with the correct fields being filled in/added.
        The observation and forecast have exactly the same observation/prediction windows. 
        Matching requires (at a minimum) that there is a prediction window start/end with an observed
        SEP start time within the prediction window and that the last data time/trigger occur before the
        observed SEP start time.

        Observed all clear is False
        Forecast all clear is False
        """
        self.dataframe = validate.fill_df(self.sphinx, self.model_names, self.all_energy_channels, \
            self.obs_thresholds, self.profname_dict)
        for keywords in self.dataframe:
            # temp = self.sphinx['Test_model_0'][self.energy_key].prediction.short_name\
            
            temp = attributes_of_sphinx_obj(keywords, self.sphinx['Test_model_0'][self.all_energy_channels[0]][0],\
                 self.energy_key, self.obs_thresholds)
            # if keywords == 'Duration Match Status': # Leaving this commented out until Katie determines if the correct logic is being used to determine this match status
            #     print(self.dataframe[keywords][0], temp, 'WHY IS THIS BUGGED')
            self.assertEqual(self.dataframe[keywords][0], temp, 'Error is in keyword ' + keywords)

    def step_2(self):
        print('Test for DoResume feature - this is false right now though so moving on after a quick assertFalse')
        self.assertFalse(self.DoResume)
    
    def step_3(self):
        validate.calculate_intuitive_metrics(self.dataframe, self.model_names, self.all_energy_channels, \
            self.obs_thresholds, 'All')
        print('Made it to step 3')
        # for model in self.model_names:
        #     for quantities in self.validation_quantity:
        #         if quantities in self.quantities_tested:
        #             metrics_filename = './output\\csv\\' + quantities + '_metrics' 
        #             self.assertTrue(os.path.isfile(metrics_filename + '.csv'), msg = metrics_filename + '.csv does not exist, check the file is output correctly')
        #             metrics_filename = './output\\pkl\\' + quantities + '_metrics' 
        #             self.assertTrue(os.path.isfile(metrics_filename + '.pkl'), msg = metrics_filename + '.pkl does not exist, check the file is output correctly')
        #         else:
        #             filename =  './output\\csv\\' + quantities + '_metrics.csv'
        #             self.assertFalse(os.path.isfile(metrics_filename), msg = metrics_filename + ' should not exist') 
                
        #         for energy_channels in self.all_energy_channels:
        #             for thresholds in self.obs_thresholds[energy_channels]:
        #                 threshold_shortened = thresholds.rsplit('.')[0]+ '_' + thresholds.rsplit('.')[1] + '.' + thresholds.rsplit('.')[2]

        #                 if quantities == 'awt':
        #                     # pkl_filename = './output\\pkl\\' + quantities + '_selections_' + model + '_' + energy_channels + '_' + threshold_shortened + '.pkl'
        #                     # csv_filename = './output\\csv\\' + quantities + '_selections_' + model + '_' + energy_channels + '_' + threshold_shortened + '.csv'
        #                     pkl_filename = '.\\output\\pkl\\' + quantities + '_selections_' + model + '_' + energy_channels + '_' + threshold_shortened + "_Predicted SEP All Clear.pkl"
        #                     csv_filename = '.\\output\\csv\\' + quantities + '_selections_' + model + '_' + energy_channels + '_' + threshold_shortened + "_Predicted SEP All Clear.csv"
        #                     self.assertTrue(os.path.isfile(pkl_filename) , \
        #                         msg = pkl_filename + ' does not exist, check the file is output correctly')
        #                     self.assertTrue(os.path.isfile(csv_filename), \
        #                         msg = csv_filename + ' does not exist, check the file is output correctly')
                            
        #                 else:
        #                     pkl_filename = './output\\pkl\\' + quantities + '_selections_' + model + '_' + energy_channels + '_' + threshold_shortened + '.pkl'
        #                     csv_filename = './output\\csv\\' + quantities + '_selections_' + model + '_' + energy_channels + '_' + threshold_shortened + '.csv'
        #                     if quantities in self.quantities_tested:
        #                         self.assertTrue(os.path.isfile(pkl_filename), msg = pkl_filename + ' does not exist, check the file is output correctly')
        #                         self.assertTrue(os.path.isfile(csv_filename), msg = csv_filename + ' does not exist, check the file is output correctly')
        #                     else:
        #                         self.assertFalse(os.path.isfile(pkl_filename), msg = pkl_filename + ' should not exist')
        #                         self.assertFalse(os.path.isfile(csv_filename), msg = csv_filename + ' should not exist')


    def step_4(self):
        print('In Step 4')
        test_dict = initialize_flux_dict()
        test_dict = fill_peak_intensity_dict(test_dict, self)

        csv_filename = './tests/output\\csv\\peak_intensity_metrics.csv'
        keyword_all_clear = []
        with open(csv_filename, mode = 'r') as csvfile:
            reading = csv.reader(csvfile, delimiter = ',')
            
            for row in reading:
                if keyword_all_clear == []:
                    keyword_all_clear = row
                else:
                    for j in range(len(row)):
                        if keyword_all_clear[j] == '':
                            pass
                        else:
                            keyword = keyword_all_clear[j]
                            # print(keyword, row[j], test_dict[keyword][0], "Breaked")
                            self.assertEqual(row[j], test_dict[keyword][0], msg = 'This is the keyword ' + keyword)


    def step_5(self):
        print('In Step 5')
        test_dict = initialize_time_dict()
        test_dict = fill_peak_intensity_time_dict(test_dict, self)

        csv_filename = './tests/output\\csv\\peak_intensity_time_metrics.csv'
        keyword_peak_intensity_time = []
        with open(csv_filename, mode = 'r') as csvfile:
            reading = csv.reader(csvfile, delimiter = ',')
            
            for row in reading:
                if keyword_peak_intensity_time == []:
                    keyword_peak_intensity_time = row
                else:
                    for j in range(len(row)):
                        if keyword_peak_intensity_time[j] == '':
                            pass
                        else:
                            keyword = keyword_peak_intensity_time[j]
                            # print(keyword, row[j], test_dict[keyword][0], "Breaked")
                           
                            self.assertEqual(row[j], test_dict[keyword][0], msg = 'This is the keyword ' + keyword)
  
    def utility_print_docstring(self, function):
        if self.verbosity == 2:
            print('\n//----------------------------------------------------')
            print(function.__doc__)

    def _steps(self):
        for name in dir(self): # dir() result is implicitly sorted
            if name.startswith("step"):
                yield name, getattr(self, name)
        
    @patch('sphinxval.utils.config.outpath', './tests/output')

    def test_peak_intensity_0(self):
        validate.prepare_outdirs()
        for name, step in self._steps():
            try:
                step()
            except Exception as e:
                self.fail("{} failed ({}: {})".format(step, type(e), e))
        utility_delete_output()


class TestPeakIntensityMult(unittest.TestCase):

    def load_verbosity(self):
        self.verbosity = utility_get_verbosity()
    
    

    def step_0(self): 
        validate.prepare_outdirs()
        


        self.energy_channel = {'min': 10, 'max': -1, 'units': 'MeV'}
        self.energy_key = objh.energy_channel_to_key(self.energy_channel)
        self.all_energy_channels = [self.energy_key] 
        self.model_names = ['Test_model_0']
        observation_json = ['./tests/files/observations/validation/all_clear/all_clear_false.json', './tests/files/observations/validation/all_clear/all_clear_true.json']
        observation_objects = {self.energy_key: []}
        for jsons in observation_json:
            observation = utility_load_observation(jsons, self.energy_channel)
            observation_objects[self.energy_key].append(observation)
        
        self.verbosity = utility_get_verbosity()
        forecast_json = ['./tests/files/forecasts/validation/onset_peak/pred_all_clear_false.json', './tests/files/forecasts/validation/onset_peak/pred_all_clear_true.json']
        forecast_objects = {self.energy_key: []}
        for jsons in forecast_json:
            forecast = utility_load_forecast(jsons, self.energy_channel)
            forecast_objects[self.energy_key].append(forecast)
        self.sphinx, self.obs_thresholds, self.obs_sep_events = utility_match_sphinx(self.all_energy_channels, self.model_names, observation_objects, forecast_objects)
        self.profname_dict = None
        self.DoResume = False
        
        self.validation_quantity = ['all_clear', 'awt', 'duration', 'end_time', 'fluence', 'max_flux_in_pred_window', 'peak_intensity_max', 'peak_intensity_max_time', 'peak_intensity' \
            'peak_intensity_time', 'probability', 'start_time', 'threshold_crossing', 'time_profile']
        self.quantities_tested = ['peak_intensity']
        self.validation_type = ['All']
        
    def step_1(self):
        """
        Tests that the dataframe is built correctly with the correct fields being filled in/added.
        The observation and forecast have exactly the same observation/prediction windows. 
        Matching requires (at a minimum) that there is a prediction window start/end with an observed
        SEP start time within the prediction window and that the last data time/trigger occur before the
        observed SEP start time.
        """
        self.dataframe = validate.fill_df(self.sphinx, self.model_names, self.all_energy_channels, \
            self.obs_thresholds, self.profname_dict)
       
        for keywords in self.dataframe:
            # temp = self.sphinx['Test_model_0'][self.energy_key].prediction.short_name\
            
            temp = attributes_of_sphinx_obj(keywords, self.sphinx['Test_model_0'][self.all_energy_channels[0]][0],\
                 self.energy_key, self.obs_thresholds)
            # if keywords == 'Duration Match Status': # Leaving this commented out until Katie determines if the correct logic is being used to determine this match status
            
            self.assertEqual(self.dataframe[keywords][0], temp, 'Error is in keyword ' + keywords)

            temp = attributes_of_sphinx_obj(keywords, self.sphinx['Test_model_0'][self.all_energy_channels[0]][1],\
                 self.energy_key, self.obs_thresholds)
            # if keywords == 'Duration Match Status': # Leaving this commented out until Katie determines if the correct logic is being used to determine this match status
            
            self.assertEqual(self.dataframe[keywords][1], temp, 'Error is in keyword ' + keywords)


    def step_2(self):
        print('Test for DoResume feature - this is false right now though so moving on after a quick assertFalse')
        self.assertFalse(self.DoResume)
    
    def step_3(self):
        validate.calculate_intuitive_metrics(self.dataframe, self.model_names, self.all_energy_channels, \
            self.obs_thresholds, 'All')
        print('Made it to step 3')
        # for model in self.model_names:
        #     for quantities in self.validation_quantity:
        #         if quantities in self.quantities_tested:
        #             metrics_filename = './output\\csv\\' + quantities + '_metrics' 
        #             self.assertTrue(os.path.isfile(metrics_filename + '.csv'), msg = metrics_filename + '.csv does not exist, check the file is output correctly')
        #             metrics_filename = './output\\pkl\\' + quantities + '_metrics' 
        #             self.assertTrue(os.path.isfile(metrics_filename + '.pkl'), msg = metrics_filename + '.pkl does not exist, check the file is output correctly')
        #         else:
        #             filename =  './output\\csv\\' + quantities + '_metrics.csv'
        #             self.assertFalse(os.path.isfile(metrics_filename), msg = metrics_filename + ' should not exist') 
                
        #         for energy_channels in self.all_energy_channels:
        #             for thresholds in self.obs_thresholds[energy_channels]:
        #                 threshold_shortened = thresholds.rsplit('.')[0]+ '_' + thresholds.rsplit('.')[1] + '.' + thresholds.rsplit('.')[2]

        #                 if quantities == 'awt':
        #                     # pkl_filename = './output\\pkl\\' + quantities + '_selections_' + model + '_' + energy_channels + '_' + threshold_shortened + '.pkl'
        #                     # csv_filename = './output\\csv\\' + quantities + '_selections_' + model + '_' + energy_channels + '_' + threshold_shortened + '.csv'
        #                     pkl_filename = '.\\output\\pkl\\' + quantities + '_selections_' + model + '_' + energy_channels + '_' + threshold_shortened + "_Predicted SEP All Clear.pkl"
        #                     csv_filename = '.\\output\\csv\\' + quantities + '_selections_' + model + '_' + energy_channels + '_' + threshold_shortened + "_Predicted SEP All Clear.csv"
        #                     self.assertTrue(os.path.isfile(pkl_filename) , \
        #                         msg = pkl_filename + ' does not exist, check the file is output correctly')
        #                     self.assertTrue(os.path.isfile(csv_filename), \
        #                         msg = csv_filename + ' does not exist, check the file is output correctly')
                            
        #                 else:
        #                     pkl_filename = './output\\pkl\\' + quantities + '_selections_' + model + '_' + energy_channels + '_' + threshold_shortened + '.pkl'
        #                     csv_filename = './output\\csv\\' + quantities + '_selections_' + model + '_' + energy_channels + '_' + threshold_shortened + '.csv'
        #                     if quantities in self.quantities_tested:
        #                         self.assertTrue(os.path.isfile(pkl_filename), msg = pkl_filename + ' does not exist, check the file is output correctly')
        #                         self.assertTrue(os.path.isfile(csv_filename), msg = csv_filename + ' does not exist, check the file is output correctly')
        #                     else:
        #                         self.assertFalse(os.path.isfile(pkl_filename), msg = pkl_filename + ' should not exist')
        #                         self.assertFalse(os.path.isfile(csv_filename), msg = csv_filename + ' should not exist')


    def step_4(self):
        print('In Step 4')
        test_dict = initialize_flux_dict()
        test_dict = fill_peak_intensity_mult_dict(test_dict, self)
        csv_filename = './tests/output\\csv\\peak_intensity_metrics.csv'
        keyword_all_clear = []
        with open(csv_filename, mode = 'r') as csvfile:
            reading = csv.reader(csvfile, delimiter = ',')
            
            for row in reading:
                if keyword_all_clear == []:
                    keyword_all_clear = row
                else:
                    for j in range(len(row)):
                        if keyword_all_clear[j] == '':
                            pass
                        else:
                            keyword = keyword_all_clear[j]
                            # print(keyword, row[j], test_dict[keyword][0])
                            self.assertEqual(row[j], test_dict[keyword][0], msg = 'This is the keyword ' + keyword)




    

    

    def utility_print_docstring(self, function):
        if self.verbosity == 2:
            print('\n//----------------------------------------------------')
            print(function.__doc__)

    def _steps(self):
        for name in dir(self): # dir() result is implicitly sorted
            if name.startswith("step"):
                yield name, getattr(self, name)
        
    # @make_docstring_printable
    # def test_df_all_clear_1(this, self):
    #     """
    #     Tests that the dataframe is built correctly with the correct fields being filled in/added.
    #     The observation and forecast have exactly the same observation/prediction windows. 
    #     Matching requires (at a minimum) that there is a prediction window start/end with an observed
    #     SEP start time within the prediction window and that the last data time/trigger occur before the
    #     observed SEP start time.
    #     Observed all clear is False
    #     Forecast all clear is False

    #     The tests in this block are:
    #         Observed SEP All Clear is False
    #         Predicted SEP All Clear is False

    #     """

    @patch('sphinxval.utils.config.outpath', './tests/output')

    def test_peak_intensity_1(self):
        validate.prepare_outdirs()
        for name, step in self._steps():
            try:
                step()
            except Exception as e:
                self.fail("{} failed ({}: {})".format(step, type(e), e))
        utility_delete_output()





class TestPeakIntensityGarbage(unittest.TestCase):

    def load_verbosity(self):
        self.verbosity = utility_get_verbosity()
    
    
    @patch('sphinxval.utils.config.outpath', './tests/output')

    def test_peak_intensity_garbage(self): 
        validate.prepare_outdirs()
        with self.assertRaises(ValueError, msg = 'Giving purposeful garbage, should exit'):
            self.energy_channel = {'min': 10, 'max': -1, 'units': 'MeV'}
            self.energy_key = objh.energy_channel_to_key(self.energy_channel)
            self.all_energy_channels = [self.energy_key] 
            self.model_names = ['Test_model_0']
            observation_json = './tests/files/observations/validation/all_clear/all_clear_false.json'
            observation = utility_load_observation(observation_json, self.energy_channel)
            observation_objects = {self.energy_key: [observation]}
            self.verbosity = utility_get_verbosity()
            forecast_json = './tests/files/forecasts/validation/onset_peak/pred_garbage.json'
            forecast = utility_load_forecast(forecast_json, self.energy_channel)
            forecast_objects = {self.energy_key: [forecast]}
            self.sphinx, self.obs_thresholds, self.obs_sep_events = utility_match_sphinx(self.all_energy_channels,\
                self.model_names, observation_objects, forecast_objects)
            self.profname_dict = None
            self.DoResume = False
            
            self.validation_quantity = ['all_clear', 'awt', 'duration', 'end_time', 'fluence', 'max_flux_in_pred_window', 'peak_intensity_max', 'peak_intensity_max_time', 'peak_intensity' \
                'peak_intensity_time', 'probability', 'start_time', 'threshold_crossing', 'time_profile']
            self.quantities_tested = ['all_clear', 'peak_intensity']
            self.validation_type = ['All']
        




class TestPeakIntensityMax0(unittest.TestCase):

    def load_verbosity(self):
        self.verbosity = utility_get_verbosity()
    
    

    def step_0(self): 
        validate.prepare_outdirs()
        self.energy_channel = {'min': 10, 'max': -1, 'units': 'MeV'}
        self.energy_key = objh.energy_channel_to_key(self.energy_channel)
        self.all_energy_channels = [self.energy_key] 
        self.model_names = ['Test_model_0']
        observation_json = './tests/files/observations/validation/all_clear/all_clear_false.json'
        observation = utility_load_observation(observation_json, self.energy_channel)
        observation_objects = {self.energy_key: [observation]}
        self.verbosity = utility_get_verbosity()
        forecast_json = './tests/files/forecasts/validation/max_peak/pred_all_clear_false.json'
        forecast = utility_load_forecast(forecast_json, self.energy_channel)
        forecast_objects = {self.energy_key: [forecast]}
        self.sphinx, self.obs_thresholds, self.obs_sep_events = utility_match_sphinx(self.all_energy_channels,\
             self.model_names, observation_objects, forecast_objects)
        self.profname_dict = None
        self.DoResume = False
        
        self.validation_quantity = ['all_clear', 'awt', 'duration', 'end_time', 'fluence', 'max_flux_in_pred_window', 'peak_intensity_max', 'peak_intensity_max_time', 'peak_intensity' \
            'peak_intensity_time', 'probability', 'start_time', 'threshold_crossing', 'time_profile']
        self.quantities_tested = ['all_clear', 'peak_intensity_max']
        self.validation_type = ['All']
        
    def step_1(self):
        """
        Tests that the dataframe is built correctly with the correct fields being filled in/added.
        The observation and forecast have exactly the same observation/prediction windows. 
        Matching requires (at a minimum) that there is a prediction window start/end with an observed
        SEP start time within the prediction window and that the last data time/trigger occur before the
        observed SEP start time.

        Observed all clear is False
        Forecast all clear is False
        """
        self.dataframe = validate.fill_df(self.sphinx, self.model_names, self.all_energy_channels, \
            self.obs_thresholds, self.profname_dict)
        for keywords in self.dataframe:
            # temp = self.sphinx['Test_model_0'][self.energy_key].prediction.short_name\
            
            temp = attributes_of_sphinx_obj(keywords, self.sphinx['Test_model_0'][self.all_energy_channels[0]][0],\
                 self.energy_key, self.obs_thresholds)
            # if keywords == 'Duration Match Status': # Leaving this commented out until Katie determines if the correct logic is being used to determine this match status
            #     print(self.dataframe[keywords][0], temp, 'WHY IS THIS BUGGED')
            self.assertEqual(self.dataframe[keywords][0], temp, 'Error is in keyword ' + keywords)

    def step_2(self):
        print('Test for DoResume feature - this is false right now though so moving on after a quick assertFalse')
        self.assertFalse(self.DoResume)
    
    def step_3(self):
        validate.calculate_intuitive_metrics(self.dataframe, self.model_names, self.all_energy_channels, \
            self.obs_thresholds, 'All')
        print('Made it to step 3')
        # for model in self.model_names:
        #     for quantities in self.validation_quantity:
        #         if quantities in self.quantities_tested:
        #             metrics_filename = './output\\csv\\' + quantities + '_metrics' 
        #             self.assertTrue(os.path.isfile(metrics_filename + '.csv'), msg = metrics_filename + '.csv does not exist, check the file is output correctly')
        #             metrics_filename = './output\\pkl\\' + quantities + '_metrics' 
        #             self.assertTrue(os.path.isfile(metrics_filename + '.pkl'), msg = metrics_filename + '.pkl does not exist, check the file is output correctly')
        #         else:
        #             filename =  './output\\csv\\' + quantities + '_metrics.csv'
        #             self.assertFalse(os.path.isfile(metrics_filename), msg = metrics_filename + ' should not exist') 
                
        #         for energy_channels in self.all_energy_channels:
        #             for thresholds in self.obs_thresholds[energy_channels]:
        #                 threshold_shortened = thresholds.rsplit('.')[0]+ '_' + thresholds.rsplit('.')[1] + '.' + thresholds.rsplit('.')[2]

        #                 if quantities == 'awt':
        #                     # pkl_filename = './output\\pkl\\' + quantities + '_selections_' + model + '_' + energy_channels + '_' + threshold_shortened + '.pkl'
        #                     # csv_filename = './output\\csv\\' + quantities + '_selections_' + model + '_' + energy_channels + '_' + threshold_shortened + '.csv'
        #                     pkl_filename = '.\\output\\pkl\\' + quantities + '_selections_' + model + '_' + energy_channels + '_' + threshold_shortened + "_Predicted SEP All Clear.pkl"
        #                     csv_filename = '.\\output\\csv\\' + quantities + '_selections_' + model + '_' + energy_channels + '_' + threshold_shortened + "_Predicted SEP All Clear.csv"
        #                     self.assertTrue(os.path.isfile(pkl_filename) , \
        #                         msg = pkl_filename + ' does not exist, check the file is output correctly')
        #                     self.assertTrue(os.path.isfile(csv_filename), \
        #                         msg = csv_filename + ' does not exist, check the file is output correctly')
                            
        #                 else:
        #                     pkl_filename = './output\\pkl\\' + quantities + '_selections_' + model + '_' + energy_channels + '_' + threshold_shortened + '.pkl'
        #                     csv_filename = './output\\csv\\' + quantities + '_selections_' + model + '_' + energy_channels + '_' + threshold_shortened + '.csv'
        #                     if quantities in self.quantities_tested:
        #                         self.assertTrue(os.path.isfile(pkl_filename), msg = pkl_filename + ' does not exist, check the file is output correctly')
        #                         self.assertTrue(os.path.isfile(csv_filename), msg = csv_filename + ' does not exist, check the file is output correctly')
        #                     else:
        #                         self.assertFalse(os.path.isfile(pkl_filename), msg = pkl_filename + ' should not exist')
        #                         self.assertFalse(os.path.isfile(csv_filename), msg = csv_filename + ' should not exist')


    def step_4(self):
        print('In Step 4')
        test_dict = initialize_flux_dict()
        test_dict = fill_peak_intensity_max_dict(test_dict, self)

        csv_filename = './tests/output\\csv\\peak_intensity_max_metrics.csv'
        keyword_all_clear = []
        with open(csv_filename, mode = 'r') as csvfile:
            reading = csv.reader(csvfile, delimiter = ',')
            
            for row in reading:
                if keyword_all_clear == []:
                    keyword_all_clear = row
                else:
                    for j in range(len(row)):
                        if keyword_all_clear[j] == '':
                            pass
                        else:
                            keyword = keyword_all_clear[j]
                            # print(keyword, row[j], test_dict[keyword][0], "Breaked")
                            self.assertEqual(row[j], test_dict[keyword][0], msg = 'This is the keyword ' + keyword)

  
    def utility_print_docstring(self, function):
        if self.verbosity == 2:
            print('\n//----------------------------------------------------')
            print(function.__doc__)

    def _steps(self):
        for name in dir(self): # dir() result is implicitly sorted
            if name.startswith("step"):
                yield name, getattr(self, name)
        
    # @make_docstring_printable
    # def test_df_all_clear_1(this, self):
    #     """
    #     Tests that the dataframe is built correctly with the correct fields being filled in/added.
    #     The observation and forecast have exactly the same observation/prediction windows. 
    #     Matching requires (at a minimum) that there is a prediction window start/end with an observed
    #     SEP start time within the prediction window and that the last data time/trigger occur before the
    #     observed SEP start time.
    #     Observed all clear is False
    #     Forecast all clear is False

    #     The tests in this block are:
    #         Observed SEP All Clear is False
    #         Predicted SEP All Clear is False

    #     """

    @patch('sphinxval.utils.config.outpath', './tests/output')

    def test_peak_intensity_max_0(self):
        validate.prepare_outdirs()
        for name, step in self._steps():
            try:
                step()
            except Exception as e:
                self.fail("{} failed ({}: {})".format(step, type(e), e))
        utility_delete_output()


class TestPeakIntensityMaxMult(unittest.TestCase):

    def load_verbosity(self):
        self.verbosity = utility_get_verbosity()
    
    

    def step_0(self): 
        validate.prepare_outdirs()
        


        self.energy_channel = {'min': 10, 'max': -1, 'units': 'MeV'}
        self.energy_key = objh.energy_channel_to_key(self.energy_channel)
        self.all_energy_channels = [self.energy_key] 
        self.model_names = ['Test_model_0']
        observation_json = ['./tests/files/observations/validation/all_clear/all_clear_false.json', './tests/files/observations/validation/all_clear/all_clear_true.json']
        observation_objects = {self.energy_key: []}
        for jsons in observation_json:
            observation = utility_load_observation(jsons, self.energy_channel)
            observation_objects[self.energy_key].append(observation)
        
        self.verbosity = utility_get_verbosity()
        forecast_json = ['./tests/files/forecasts/validation/max_peak/pred_all_clear_false.json', './tests/files/forecasts/validation/max_peak/pred_all_clear_true.json']
        forecast_objects = {self.energy_key: []}
        for jsons in forecast_json:
            forecast = utility_load_forecast(jsons, self.energy_channel)
            forecast_objects[self.energy_key].append(forecast)
        self.sphinx, self.obs_thresholds, self.obs_sep_events = utility_match_sphinx(self.all_energy_channels, self.model_names, observation_objects, forecast_objects)
        self.profname_dict = None
        self.DoResume = False
        
        self.validation_quantity = ['all_clear', 'awt', 'duration', 'end_time', 'fluence', 'max_flux_in_pred_window', 'peak_intensity_max', 'peak_intensity_max_time', 'peak_intensity' \
            'peak_intensity_time', 'probability', 'start_time', 'threshold_crossing', 'time_profile']
        self.quantities_tested = ['peak_intensity_max']
        self.validation_type = ['All']
        
    def step_1(self):
        """
        Tests that the dataframe is built correctly with the correct fields being filled in/added.
        The observation and forecast have exactly the same observation/prediction windows. 
        Matching requires (at a minimum) that there is a prediction window start/end with an observed
        SEP start time within the prediction window and that the last data time/trigger occur before the
        observed SEP start time.
        Observed all clear is False
        Forecast all clear is False
        """
        self.dataframe = validate.fill_df(self.sphinx, self.model_names, self.all_energy_channels, \
            self.obs_thresholds, self.profname_dict)
       
        for keywords in self.dataframe:
            # temp = self.sphinx['Test_model_0'][self.energy_key].prediction.short_name\
            
            temp = attributes_of_sphinx_obj(keywords, self.sphinx['Test_model_0'][self.all_energy_channels[0]][0],\
                 self.energy_key, self.obs_thresholds)
            # if keywords == 'Duration Match Status': # Leaving this commented out until Katie determines if the correct logic is being used to determine this match status
            
            self.assertEqual(self.dataframe[keywords][0], temp, 'Error is in keyword ' + keywords)

            temp = attributes_of_sphinx_obj(keywords, self.sphinx['Test_model_0'][self.all_energy_channels[0]][1],\
                 self.energy_key, self.obs_thresholds)
            # if keywords == 'Duration Match Status': # Leaving this commented out until Katie determines if the correct logic is being used to determine this match status
            
            self.assertEqual(self.dataframe[keywords][1], temp, 'Error is in keyword ' + keywords)


    def step_2(self):
        print('Test for DoResume feature - this is false right now though so moving on after a quick assertFalse')
        self.assertFalse(self.DoResume)
    
    def step_3(self):
        validate.calculate_intuitive_metrics(self.dataframe, self.model_names, self.all_energy_channels, \
            self.obs_thresholds, 'All')
        print('Made it to step 3')
        # for model in self.model_names:
        #     for quantities in self.validation_quantity:
        #         if quantities in self.quantities_tested:
        #             metrics_filename = './output\\csv\\' + quantities + '_metrics' 
        #             self.assertTrue(os.path.isfile(metrics_filename + '.csv'), msg = metrics_filename + '.csv does not exist, check the file is output correctly')
        #             metrics_filename = './output\\pkl\\' + quantities + '_metrics' 
        #             self.assertTrue(os.path.isfile(metrics_filename + '.pkl'), msg = metrics_filename + '.pkl does not exist, check the file is output correctly')
        #         else:
        #             filename =  './output\\csv\\' + quantities + '_metrics.csv'
        #             self.assertFalse(os.path.isfile(metrics_filename), msg = metrics_filename + ' should not exist') 
                
        #         for energy_channels in self.all_energy_channels:
        #             for thresholds in self.obs_thresholds[energy_channels]:
        #                 threshold_shortened = thresholds.rsplit('.')[0]+ '_' + thresholds.rsplit('.')[1] + '.' + thresholds.rsplit('.')[2]

        #                 if quantities == 'awt':
        #                     # pkl_filename = './output\\pkl\\' + quantities + '_selections_' + model + '_' + energy_channels + '_' + threshold_shortened + '.pkl'
        #                     # csv_filename = './output\\csv\\' + quantities + '_selections_' + model + '_' + energy_channels + '_' + threshold_shortened + '.csv'
        #                     pkl_filename = '.\\output\\pkl\\' + quantities + '_selections_' + model + '_' + energy_channels + '_' + threshold_shortened + "_Predicted SEP All Clear.pkl"
        #                     csv_filename = '.\\output\\csv\\' + quantities + '_selections_' + model + '_' + energy_channels + '_' + threshold_shortened + "_Predicted SEP All Clear.csv"
        #                     self.assertTrue(os.path.isfile(pkl_filename) , \
        #                         msg = pkl_filename + ' does not exist, check the file is output correctly')
        #                     self.assertTrue(os.path.isfile(csv_filename), \
        #                         msg = csv_filename + ' does not exist, check the file is output correctly')
                            
        #                 else:
        #                     pkl_filename = './output\\pkl\\' + quantities + '_selections_' + model + '_' + energy_channels + '_' + threshold_shortened + '.pkl'
        #                     csv_filename = './output\\csv\\' + quantities + '_selections_' + model + '_' + energy_channels + '_' + threshold_shortened + '.csv'
        #                     if quantities in self.quantities_tested:
        #                         self.assertTrue(os.path.isfile(pkl_filename), msg = pkl_filename + ' does not exist, check the file is output correctly')
        #                         self.assertTrue(os.path.isfile(csv_filename), msg = csv_filename + ' does not exist, check the file is output correctly')
        #                     else:
        #                         self.assertFalse(os.path.isfile(pkl_filename), msg = pkl_filename + ' should not exist')
        #                         self.assertFalse(os.path.isfile(csv_filename), msg = csv_filename + ' should not exist')


    def step_4(self):
        print('In Step 4')
        test_dict = initialize_flux_dict()
        test_dict = fill_peak_intensity_max_mult_dict(test_dict, self)
        csv_filename = './tests/output\\csv\\peak_intensity_max_metrics.csv'
        keyword_all_clear = []
        with open(csv_filename, mode = 'r') as csvfile:
            reading = csv.reader(csvfile, delimiter = ',')
            
            for row in reading:
                if keyword_all_clear == []:
                    keyword_all_clear = row
                else:
                    for j in range(len(row)):
                        if keyword_all_clear[j] == '':
                            pass
                        else:
                            keyword = keyword_all_clear[j]
                            # print(keyword, row[j], test_dict[keyword][0])
                            self.assertEqual(row[j], test_dict[keyword][0], msg = 'This is the keyword ' + keyword)




    

    

    def utility_print_docstring(self, function):
        if self.verbosity == 2:
            print('\n//----------------------------------------------------')
            print(function.__doc__)

    def _steps(self):
        for name in dir(self): # dir() result is implicitly sorted
            if name.startswith("step"):
                yield name, getattr(self, name)
        
    # @make_docstring_printable
    # def test_df_all_clear_1(this, self):
    #     """
    #     Tests that the dataframe is built correctly with the correct fields being filled in/added.
    #     The observation and forecast have exactly the same observation/prediction windows. 
    #     Matching requires (at a minimum) that there is a prediction window start/end with an observed
    #     SEP start time within the prediction window and that the last data time/trigger occur before the
    #     observed SEP start time.
    #     Observed all clear is False
    #     Forecast all clear is False

    #     The tests in this block are:
    #         Observed SEP All Clear is False
    #         Predicted SEP All Clear is False

    #     """

    @patch('sphinxval.utils.config.outpath', './tests/output')

    def test_peak_intensity_max_1(self):
        validate.prepare_outdirs()
        for name, step in self._steps():
            try:
                step()
            except Exception as e:
                self.fail("{} failed ({}: {})".format(step, type(e), e))
        utility_delete_output()





# class TestPeakIntensityMaxGarbage(unittest.TestCase):

#     def load_verbosity(self):
#         self.verbosity = utility_get_verbosity()
    
    

#     def test_peak_intensity_max_garbage(self): 
#         validate.prepare_outdirs()
#         with self.assertRaises(ValueError, msg = 'Giving purposeful garbage, should exit'):
#             self.energy_channel = {'min': 10, 'max': -1, 'units': 'MeV'}
#             self.energy_key = objh.energy_channel_to_key(self.energy_channel)
#             self.all_energy_channels = [self.energy_key] 
#             self.model_names = ['Test_model_0']
#             observation_json = './tests/files/observations/validation/all_clear/all_clear_false.json'
#             observation = utility_load_observation(observation_json, self.energy_channel)
#             observation_objects = {self.energy_key: [observation]}
#             self.verbosity = utility_get_verbosity()
#             forecast_json = './tests/files/forecasts/validation/max_peak/pred_garbage.json'
#             forecast = utility_load_forecast(forecast_json, self.energy_channel)
#             forecast_objects = {self.energy_key: [forecast]}
#             self.sphinx, self.obs_thresholds, self.obs_sep_events = utility_match_sphinx(self.all_energy_channels,\
#                 self.model_names, observation_objects, forecast_objects)
#             self.profname_dict = None
#             self.DoResume = False
            
#             self.validation_quantity = ['all_clear', 'awt', 'duration', 'end_time', 'fluence', 'max_flux_in_pred_window', 'peak_intensity_max', 'peak_intensity_max_time', 'peak_intensity' \
#                 'peak_intensity_time', 'probability', 'start_time', 'threshold_crossing', 'time_profile']
#             self.quantities_tested = ['all_clear', 'peak_intensity_max']
#             self.validation_type = ['All']
        
   

  
#     def utility_print_docstring(self, function):
#         if self.verbosity == 2:
#             print('\n//----------------------------------------------------')
#             print(function.__doc__)


class TestProbability0(unittest.TestCase):

    def load_verbosity(self):
        self.verbosity = utility_get_verbosity()
    
    

    def step_0(self): 
        validate.prepare_outdirs()
        self.energy_channel = {'min': 10, 'max': -1, 'units': 'MeV'}
        self.energy_key = objh.energy_channel_to_key(self.energy_channel)
        self.all_energy_channels = [self.energy_key] 
        self.model_names = ['Test_model_0']
        observation_json = './tests/files/observations/validation/all_clear/all_clear_false.json'
        observation = utility_load_observation(observation_json, self.energy_channel)
        observation_objects = {self.energy_key: [observation]}
        self.verbosity = utility_get_verbosity()
        forecast_json = './tests/files/forecasts/validation/probability/pred_probability_all_clear_false.json'
        forecast = utility_load_forecast(forecast_json, self.energy_channel)
        forecast_objects = {self.energy_key: [forecast]}
        self.sphinx, self.obs_thresholds, self.obs_sep_events = utility_match_sphinx(self.all_energy_channels,\
             self.model_names, observation_objects, forecast_objects)
        self.profname_dict = None
        self.DoResume = False
        
        self.validation_quantity = ['all_clear', 'awt', 'duration', 'end_time', 'fluence', 'max_flux_in_pred_window', 'peak_intensity_max', 'peak_intensity_max_time', 'peak_intensity' \
            'peak_intensity_time', 'probability', 'start_time', 'threshold_crossing', 'time_profile']
        self.quantities_tested = ['probability']
        self.validation_type = ['All']
        
    def step_1(self):
        """
        Tests that the dataframe is built correctly with the correct fields being filled in/added.
        The observation and forecast have exactly the same observation/prediction windows. 
        Matching requires (at a minimum) that there is a prediction window start/end with an observed
        SEP start time within the prediction window and that the last data time/trigger occur before the
        observed SEP start time.
        Observed all clear is False
        Forecast all clear is False
        """
        self.dataframe = validate.fill_df(self.sphinx, self.model_names, self.all_energy_channels, \
            self.obs_thresholds, self.profname_dict)
        for keywords in self.dataframe:
            # temp = self.sphinx['Test_model_0'][self.energy_key].prediction.short_name\
            
            temp = attributes_of_sphinx_obj(keywords, self.sphinx['Test_model_0'][self.all_energy_channels[0]][0],\
                 self.energy_key, self.obs_thresholds)
            # if keywords == 'Duration Match Status': # Leaving this commented out until Katie determines if the correct logic is being used to determine this match status
            #     print(self.dataframe[keywords][0], temp, 'WHY IS THIS BUGGED')
            self.assertEqual(self.dataframe[keywords][0], temp, 'Error is in keyword ' + keywords)

    def step_2(self):
        print('Test for DoResume feature - this is false right now though so moving on after a quick assertFalse')
        self.assertFalse(self.DoResume)
    
    def step_3(self):
        validate.calculate_intuitive_metrics(self.dataframe, self.model_names, self.all_energy_channels, \
            self.obs_thresholds, 'All')
        print('Made it to step 3')
        # for model in self.model_names:
        #     for quantities in self.validation_quantity:
        #         if quantities in self.quantities_tested:
        #             metrics_filename = './output\\csv\\' + quantities + '_metrics' 
        #             self.assertTrue(os.path.isfile(metrics_filename + '.csv'), msg = metrics_filename + '.csv does not exist, check the file is output correctly')
        #             metrics_filename = './output\\pkl\\' + quantities + '_metrics' 
        #             self.assertTrue(os.path.isfile(metrics_filename + '.pkl'), msg = metrics_filename + '.pkl does not exist, check the file is output correctly')
        #         else:
        #             filename =  './output\\csv\\' + quantities + '_metrics.csv'
        #             self.assertFalse(os.path.isfile(metrics_filename), msg = metrics_filename + ' should not exist') 
                
        #         for energy_channels in self.all_energy_channels:
        #             for thresholds in self.obs_thresholds[energy_channels]:
        #                 threshold_shortened = thresholds.rsplit('.')[0]+ '_' + thresholds.rsplit('.')[1] + '.' + thresholds.rsplit('.')[2]

        #                 if quantities == 'awt':
        #                     # pkl_filename = './output\\pkl\\' + quantities + '_selections_' + model + '_' + energy_channels + '_' + threshold_shortened + '.pkl'
        #                     # csv_filename = './output\\csv\\' + quantities + '_selections_' + model + '_' + energy_channels + '_' + threshold_shortened + '.csv'
        #                     pkl_filename = '.\\output\\pkl\\' + quantities + '_selections_' + model + '_' + energy_channels + '_' + threshold_shortened + "_Predicted SEP All Clear.pkl"
        #                     csv_filename = '.\\output\\csv\\' + quantities + '_selections_' + model + '_' + energy_channels + '_' + threshold_shortened + "_Predicted SEP All Clear.csv"
        #                     self.assertTrue(os.path.isfile(pkl_filename) , \
        #                         msg = pkl_filename + ' does not exist, check the file is output correctly')
        #                     self.assertTrue(os.path.isfile(csv_filename), \
        #                         msg = csv_filename + ' does not exist, check the file is output correctly')
                            
        #                 else:
        #                     pkl_filename = './output\\pkl\\' + quantities + '_selections_' + model + '_' + energy_channels + '_' + threshold_shortened + '.pkl'
        #                     csv_filename = './output\\csv\\' + quantities + '_selections_' + model + '_' + energy_channels + '_' + threshold_shortened + '.csv'
        #                     if quantities in self.quantities_tested:
        #                         self.assertTrue(os.path.isfile(pkl_filename), msg = pkl_filename + ' does not exist, check the file is output correctly')
        #                         self.assertTrue(os.path.isfile(csv_filename), msg = csv_filename + ' does not exist, check the file is output correctly')
        #                     else:
        #                         self.assertFalse(os.path.isfile(pkl_filename), msg = pkl_filename + ' should not exist')
        #                         self.assertFalse(os.path.isfile(csv_filename), msg = csv_filename + ' should not exist')


    def step_4(self):
        print('In Step 4')
        test_dict = initialize_probability_dict()
        test_dict = fill_probability_dict_highprob(test_dict, self)
        csv_filename = './tests/output\\csv\\probability_metrics.csv'
        keyword_all_clear = []
        with open(csv_filename, mode = 'r') as csvfile:
            reading = csv.reader(csvfile, delimiter = ',')
            
            for row in reading:
                if keyword_all_clear == []:
                    keyword_all_clear = row
                else:
                    for j in range(len(row)):
                        if keyword_all_clear[j] == '':
                            pass
                        else:
                            keyword = keyword_all_clear[j]
                            # print(keyword, row[j], test_dict[keyword][0])
                            self.assertEqual(row[j], test_dict[keyword][0], msg = 'This is the keyword ' + keyword)



    

    def utility_print_docstring(self, function):
        if self.verbosity == 2:
            print('\n//----------------------------------------------------')
            print(function.__doc__)

    def _steps(self):
        for name in dir(self): # dir() result is implicitly sorted
            if name.startswith("step"):
                yield name, getattr(self, name)
        
    # @make_docstring_printable
    # def test_df_all_clear_1(this, self):
    #     """
    #     Tests that the dataframe is built correctly with the correct fields being filled in/added.
    #     The observation and forecast have exactly the same observation/prediction windows. 
    #     Matching requires (at a minimum) that there is a prediction window start/end with an observed
    #     SEP start time within the prediction window and that the last data time/trigger occur before the
    #     observed SEP start time.
    #     Observed all clear is False
    #     Forecast all clear is False

    #     The tests in this block are:
    #         Observed SEP All Clear is False
    #         Predicted SEP All Clear is False

    #     """

    @patch('sphinxval.utils.config.outpath', './tests/output')

    def test_prob_0(self):
        validate.prepare_outdirs()
        for name, step in self._steps():
            try:
                step()
            except Exception as e:
                self.fail("{} failed ({}: {})".format(step, type(e), e))
        utility_delete_output()



# class TestProbabilityGarbage(unittest.TestCase):

#     def load_verbosity(self):
#         self.verbosity = utility_get_verbosity()
    
    

#     def test_garbage(self): 
#         validate.prepare_outdirs()
        

#         with self.assertRaises(NameError, msg = 'Giving purposeful garbage, should exit'):
#             self.energy_channel = {'min': 10, 'max': -1, 'units': 'MeV'}
#             self.energy_key = objh.energy_channel_to_key(self.energy_channel)
#             self.all_energy_channels = [self.energy_key] 
#             self.model_names = ['Test_model_0']
#             observation_json = './tests/files/observations/validation/all_clear/all_clear_false.json'
#             observation = utility_load_observation(observation_json, self.energy_channel)
#             observation_objects = {self.energy_key: [observation]}
#             self.verbosity = utility_get_verbosity()
#             forecast_json = './tests/files/forecasts/validation/probability/pred_probability_garbage.json'
#             forecast = utility_load_forecast(forecast_json, self.energy_channel)
#             forecast_objects = {self.energy_key: [forecast]}
#             self.sphinx, self.obs_thresholds, self.obs_sep_events = utility_match_sphinx(self.all_energy_channels,\
#                 self.model_names, observation_objects, forecast_objects)
#             self.profname_dict = None
#             self.DoResume = False
            
#             self.validation_quantity = ['all_clear', 'awt', 'duration', 'end_time', 'fluence', 'max_flux_in_pred_window', 'peak_intensity_max', 'peak_intensity_max_time', 'peak_intensity' \
#                 'peak_intensity_time', 'probability', 'start_time', 'threshold_crossing', 'time_profile']
#             self.quantities_tested = ['probability']
#             self.validation_type = ['All']
        
        


class TestProbabilityMult(unittest.TestCase):

    def load_verbosity(self):
        self.verbosity = utility_get_verbosity()
    
    

    def step_0(self): 
        validate.prepare_outdirs()
        


        self.energy_channel = {'min': 10, 'max': -1, 'units': 'MeV'}
        self.energy_key = objh.energy_channel_to_key(self.energy_channel)
        self.all_energy_channels = [self.energy_key] 
        self.model_names = ['Test_model_0']
        observation_json = ['./tests/files/observations/validation/all_clear/all_clear_false.json', './tests/files/observations/validation/all_clear/all_clear_true.json']
        observation_objects = {self.energy_key: []}
        for jsons in observation_json:
            observation = utility_load_observation(jsons, self.energy_channel)
            observation_objects[self.energy_key].append(observation)
        
        self.verbosity = utility_get_verbosity()
        forecast_json = ['./tests/files/forecasts/validation/probability/pred_probability_all_clear_false.json', './tests/files/forecasts/validation/probability/pred_probability_all_clear_true.json']
        forecast_objects = {self.energy_key: []}
        for jsons in forecast_json:
            forecast = utility_load_forecast(jsons, self.energy_channel)
            forecast_objects[self.energy_key].append(forecast)
        self.sphinx, self.obs_thresholds, self.obs_sep_events = utility_match_sphinx(self.all_energy_channels, self.model_names, observation_objects, forecast_objects)
        self.profname_dict = None
        self.DoResume = False
        
        self.validation_quantity = ['all_clear', 'awt', 'duration', 'end_time', 'fluence', 'max_flux_in_pred_window', 'peak_intensity_max', 'peak_intensity_max_time', 'peak_intensity' \
            'peak_intensity_time', 'probability', 'start_time', 'threshold_crossing', 'time_profile']
        self.quantities_tested = ['probability']
        self.validation_type = ['All']
        
    def step_1(self):
        """
        Tests that the dataframe is built correctly with the correct fields being filled in/added.
        The observation and forecast have exactly the same observation/prediction windows. 
        Matching requires (at a minimum) that there is a prediction window start/end with an observed
        SEP start time within the prediction window and that the last data time/trigger occur before the
        observed SEP start time.
        Observed all clear is False
        Forecast all clear is False
        """
        self.dataframe = validate.fill_df(self.sphinx, self.model_names, self.all_energy_channels, \
            self.obs_thresholds, self.profname_dict)
       
        for keywords in self.dataframe:
            # temp = self.sphinx['Test_model_0'][self.energy_key].prediction.short_name\
            
            temp = attributes_of_sphinx_obj(keywords, self.sphinx['Test_model_0'][self.all_energy_channels[0]][0],\
                 self.energy_key, self.obs_thresholds)
            # if keywords == 'Duration Match Status': # Leaving this commented out until Katie determines if the correct logic is being used to determine this match status
            
            self.assertEqual(self.dataframe[keywords][0], temp, 'Error is in keyword ' + keywords)

            temp = attributes_of_sphinx_obj(keywords, self.sphinx['Test_model_0'][self.all_energy_channels[0]][1],\
                 self.energy_key, self.obs_thresholds)
            # if keywords == 'Duration Match Status': # Leaving this commented out until Katie determines if the correct logic is being used to determine this match status
            
            self.assertEqual(self.dataframe[keywords][1], temp, 'Error is in keyword ' + keywords)


    def step_2(self):
        print('Test for DoResume feature - this is false right now though so moving on after a quick assertFalse')
        self.assertFalse(self.DoResume)
    
    def step_3(self):
        validate.calculate_intuitive_metrics(self.dataframe, self.model_names, self.all_energy_channels, \
            self.obs_thresholds, 'All')
        print('Made it to step 3')
        # for model in self.model_names:
        #     for quantities in self.validation_quantity:
        #         if quantities in self.quantities_tested:
        #             metrics_filename = './output\\csv\\' + quantities + '_metrics' 
        #             self.assertTrue(os.path.isfile(metrics_filename + '.csv'), msg = metrics_filename + '.csv does not exist, check the file is output correctly')
        #             metrics_filename = './output\\pkl\\' + quantities + '_metrics' 
        #             self.assertTrue(os.path.isfile(metrics_filename + '.pkl'), msg = metrics_filename + '.pkl does not exist, check the file is output correctly')
        #         else:
        #             filename =  './output\\csv\\' + quantities + '_metrics.csv'
        #             self.assertFalse(os.path.isfile(metrics_filename), msg = metrics_filename + ' should not exist') 
                
        #         for energy_channels in self.all_energy_channels:
        #             for thresholds in self.obs_thresholds[energy_channels]:
        #                 threshold_shortened = thresholds.rsplit('.')[0]+ '_' + thresholds.rsplit('.')[1] + '.' + thresholds.rsplit('.')[2]

        #                 if quantities == 'awt':
        #                     # pkl_filename = './output\\pkl\\' + quantities + '_selections_' + model + '_' + energy_channels + '_' + threshold_shortened + '.pkl'
        #                     # csv_filename = './output\\csv\\' + quantities + '_selections_' + model + '_' + energy_channels + '_' + threshold_shortened + '.csv'
        #                     pkl_filename = '.\\output\\pkl\\' + quantities + '_selections_' + model + '_' + energy_channels + '_' + threshold_shortened + "_Predicted SEP All Clear.pkl"
        #                     csv_filename = '.\\output\\csv\\' + quantities + '_selections_' + model + '_' + energy_channels + '_' + threshold_shortened + "_Predicted SEP All Clear.csv"
        #                     self.assertTrue(os.path.isfile(pkl_filename) , \
        #                         msg = pkl_filename + ' does not exist, check the file is output correctly')
        #                     self.assertTrue(os.path.isfile(csv_filename), \
        #                         msg = csv_filename + ' does not exist, check the file is output correctly')
                            
        #                 else:
        #                     pkl_filename = './output\\pkl\\' + quantities + '_selections_' + model + '_' + energy_channels + '_' + threshold_shortened + '.pkl'
        #                     csv_filename = './output\\csv\\' + quantities + '_selections_' + model + '_' + energy_channels + '_' + threshold_shortened + '.csv'
        #                     if quantities in self.quantities_tested:
        #                         self.assertTrue(os.path.isfile(pkl_filename), msg = pkl_filename + ' does not exist, check the file is output correctly')
        #                         self.assertTrue(os.path.isfile(csv_filename), msg = csv_filename + ' does not exist, check the file is output correctly')
        #                     else:
        #                         self.assertFalse(os.path.isfile(pkl_filename), msg = pkl_filename + ' should not exist')
        #                         self.assertFalse(os.path.isfile(csv_filename), msg = csv_filename + ' should not exist')


    def step_4(self):
        print('In Step 4')
        test_dict = initialize_probability_dict()
        test_dict = fill_probability_dict_multprob(test_dict, self)
        csv_filename = './tests/output\\csv\\probability_metrics.csv'
        keyword_all_clear = []
        with open(csv_filename, mode = 'r') as csvfile:
            reading = csv.reader(csvfile, delimiter = ',')
            
            for row in reading:
                if keyword_all_clear == []:
                    keyword_all_clear = row
                else:
                    for j in range(len(row)):
                        if keyword_all_clear[j] == '':
                            pass
                        else:
                            keyword = keyword_all_clear[j]
                            # print(keyword, row[j], test_dict[keyword][0])
                            self.assertEqual(row[j], test_dict[keyword][0], msg = 'This is the keyword ' + keyword)




    

    

    def utility_print_docstring(self, function):
        if self.verbosity == 2:
            print('\n//----------------------------------------------------')
            print(function.__doc__)

    def _steps(self):
        for name in dir(self): # dir() result is implicitly sorted
            if name.startswith("step"):
                yield name, getattr(self, name)
        
    # @make_docstring_printable
    # def test_df_all_clear_1(this, self):
    #     """
    #     Tests that the dataframe is built correctly with the correct fields being filled in/added.
    #     The observation and forecast have exactly the same observation/prediction windows. 
    #     Matching requires (at a minimum) that there is a prediction window start/end with an observed
    #     SEP start time within the prediction window and that the last data time/trigger occur before the
    #     observed SEP start time.
    #     Observed all clear is False
    #     Forecast all clear is False

    #     The tests in this block are:
    #         Observed SEP All Clear is False
    #         Predicted SEP All Clear is False

    #     """
    @patch('sphinxval.utils.config.outpath', './tests/output')
    
    def test_prob_1(self):
        validate.prepare_outdirs()
        for name, step in self._steps():
            try:
                step()
            except Exception as e:
                self.fail("{} failed ({}: {})".format(step, type(e), e))
        utility_delete_output()


class Test_AllFields_MultipleForecasts(unittest.TestCase):

    def load_verbosity(self):
        self.verbosity = utility_get_verbosity()
    
    
    def step_1(self):
        """
        Tests that the dataframe is built correctly with the correct fields being filled in/added.
        The observation and forecast have exactly the same observation/prediction windows. 
        Matching requires (at a minimum) that there is a prediction window start/end with an observed
        SEP start time within the prediction window and that the last data time/trigger occur before the
        observed SEP start time.
        Observed all clear is False
        Forecast all clear is False
        """
        validate.prepare_outdirs()
        

        validation_type = ["All", "First", "Last", "Max", "Mean"]
        self.energy_channel = {'min': 10, 'max': -1, 'units': 'MeV'}
        self.energy_key = objh.energy_channel_to_key(self.energy_channel)
        self.all_energy_channels = [self.energy_key] 
        self.model_names = ['Test_model_0']
        observation_json = ['./tests/files/observations/validation/all/all_clear_false.json', './tests/files/observations/validation/all/all_clear_true.json']
        observation_objects = {self.energy_key: []}
        for jsons in observation_json:
            observation = utility_load_observation(jsons, self.energy_channel)
            observation_objects[self.energy_key].append(observation)
        
        self.verbosity = utility_get_verbosity()
        forecast_json = ['./tests/files/forecasts/validation/all/pred_timeprof_all_clear_false.json', './tests/files/forecasts/validation/all/pred_timeprof_all_clear_true.json', './tests/files/forecasts/validation/all/flare_pred_timeprof_all_clear_false.json']
        forecast_objects = {self.energy_key: []}
        for jsons in forecast_json:
            forecast = utility_load_forecast(jsons, self.energy_channel)
            forecast_objects[self.energy_key].append(forecast)
        self.sphinx, self.obs_thresholds, self.obs_sep_events = utility_match_sphinx(self.all_energy_channels, self.model_names, observation_objects, forecast_objects)
        self.profname_dict = None
        self.DoResume = False
        
        self.validation_quantity = ['all_clear', 'awt', 'duration', 'end_time', 'fluence', 'max_flux_in_pred_window', 'peak_intensity_max', 'peak_intensity_max_time', 'peak_intensity' \
            'peak_intensity_time', 'probability', 'start_time', 'threshold_crossing', 'time_profile']
        # self.quantities_tested = ['probability']
        self.validation_type = ["All", "First", "Last", "Max", "Mean"]

        self.dataframe = validate.fill_df(self.sphinx, self.model_names, self.all_energy_channels, \
            self.obs_thresholds, self.profname_dict)
       
        for keywords in self.dataframe:
            # print('Current Keyword',  keywords)
            # temp = self.sphinx['Test_model_0'][self.energy_key].prediction.short_name\
            
            temp = attributes_of_sphinx_obj(keywords, self.sphinx['Test_model_0'][self.all_energy_channels[0]][0],\
                 self.energy_key, self.obs_thresholds)
            print(self.dataframe[keywords][0], temp, keywords)
            if 'SEP Fluence Spectrum' in keywords and "Units" not in keywords:
                try:
                    # print('This is a spectrum - or at least it should be', len(self.dataframe[keywords][0]))
                    for energies in range(len(self.dataframe[keywords][0])):
                        print(self.dataframe[keywords][0][energies]['energy_min'], temp)
                        self.assertEqual(self.dataframe[keywords][0][energies]['energy_min'], temp[energies]['energy_min'], 'Error is in keyword ' + keywords + ' energy_min')
                        self.assertEqual(self.dataframe[keywords][0][energies]['energy_max'], temp[energies]['energy_max'], 'Error is in keyword ' + keywords + ' energy_max')
                        self.assertEqual(self.dataframe[keywords][0][energies]['fluence'], temp[energies]['fluence'], 'Error is in keyword ' + keywords + ' fluence')
                except:
                    print(self.dataframe[keywords][0])
                    self.assertTrue(pd.isna(self.dataframe[keywords][0]))
            elif pd.isna(temp) and pd.isna(self.dataframe[keywords][0]):
                self.assertTrue(pd.isna(self.dataframe[keywords][0]))
            else:
            # if keywords == 'Duration Match Status': # Leaving this commented out until Katie determines if the correct logic is being used to determine this match status
                self.assertEqual(self.dataframe[keywords][0], temp, 'Error is in keyword ' + keywords)

            temp = attributes_of_sphinx_obj(keywords, self.sphinx['Test_model_0'][self.all_energy_channels[0]][1],\
                 self.energy_key, self.obs_thresholds)
            print(self.dataframe[keywords][1], temp, keywords, len(self.sphinx['Test_model_0'][self.all_energy_channels[0]]))
            if 'SEP Fluence Spectrum' in keywords and "Units" not in keywords:
                try:
                    # print('This is a spectrum - or at least it should be', len(self.dataframe[keywords][0]))
                    for energies in range(len(self.dataframe[keywords][1])):
                        self.assertEqual(self.dataframe[keywords][1][energies]['energy_min'], temp[energies]['energy_min'], 'Error is in keyword ' + keywords + ' energy_min')
                        self.assertEqual(self.dataframe[keywords][1][energies]['energy_max'], temp[energies]['energy_max'], 'Error is in keyword ' + keywords + ' energy_max')
                        self.assertEqual(self.dataframe[keywords][1][energies]['fluence'], temp[energies]['fluence'], 'Error is in keyword ' + keywords + ' fluence')
                except:
                    print(self.dataframe[keywords][1])
                    self.assertTrue(pd.isna(self.dataframe[keywords][1]))
            elif pd.isna(temp) and pd.isna(self.dataframe[keywords][1]):
                self.assertTrue(pd.isna(self.dataframe[keywords][1]))
            else:
            # if keywords == 'Duration Match Status': # Leaving this commented out until Katie determines if the correct logic is being used to determine this match status
                
                self.assertEqual(self.dataframe[keywords][1], temp, 'Error is in keyword ' + keywords)
        for type in self.validation_type:
            print('the type is', type)
            validate.calculate_intuitive_metrics(self.dataframe, self.model_names, self.all_energy_channels, \
                self.obs_thresholds, type)

    def step_2(self):
        print('Test for DoResume feature - this is false right now though so moving on after a quick assertFalse')
        self.assertFalse(self.DoResume)
    
    def step_3(self):
        
        print('Made it to step 3')
        # for model in self.model_names:
        #     for quantities in self.validation_quantity:
        #         if quantities in self.quantities_tested:
        #             metrics_filename = './output\\csv\\' + quantities + '_metrics' 
        #             self.assertTrue(os.path.isfile(metrics_filename + '.csv'), msg = metrics_filename + '.csv does not exist, check the file is output correctly')
        #             metrics_filename = './output\\pkl\\' + quantities + '_metrics' 
        #             self.assertTrue(os.path.isfile(metrics_filename + '.pkl'), msg = metrics_filename + '.pkl does not exist, check the file is output correctly')
        #         else:
        #             filename =  './output\\csv\\' + quantities + '_metrics.csv'
        #             self.assertFalse(os.path.isfile(metrics_filename), msg = metrics_filename + ' should not exist') 
                
        #         for energy_channels in self.all_energy_channels:
        #             for thresholds in self.obs_thresholds[energy_channels]:
        #                 threshold_shortened = thresholds.rsplit('.')[0]+ '_' + thresholds.rsplit('.')[1] + '.' + thresholds.rsplit('.')[2]

        #                 if quantities == 'awt':
        #                     # pkl_filename = './output\\pkl\\' + quantities + '_selections_' + model + '_' + energy_channels + '_' + threshold_shortened + '.pkl'
        #                     # csv_filename = './output\\csv\\' + quantities + '_selections_' + model + '_' + energy_channels + '_' + threshold_shortened + '.csv'
        #                     pkl_filename = '.\\output\\pkl\\' + quantities + '_selections_' + model + '_' + energy_channels + '_' + threshold_shortened + "_Predicted SEP All Clear.pkl"
        #                     csv_filename = '.\\output\\csv\\' + quantities + '_selections_' + model + '_' + energy_channels + '_' + threshold_shortened + "_Predicted SEP All Clear.csv"
        #                     self.assertTrue(os.path.isfile(pkl_filename) , \
        #                         msg = pkl_filename + ' does not exist, check the file is output correctly')
        #                     self.assertTrue(os.path.isfile(csv_filename), \
        #                         msg = csv_filename + ' does not exist, check the file is output correctly')
                            
        #                 else:
        #                     pkl_filename = './output\\pkl\\' + quantities + '_selections_' + model + '_' + energy_channels + '_' + threshold_shortened + '.pkl'
        #                     csv_filename = './output\\csv\\' + quantities + '_selections_' + model + '_' + energy_channels + '_' + threshold_shortened + '.csv'
        #                     if quantities in self.quantities_tested:
        #                         self.assertTrue(os.path.isfile(pkl_filename), msg = pkl_filename + ' does not exist, check the file is output correctly')
        #                         self.assertTrue(os.path.isfile(csv_filename), msg = csv_filename + ' does not exist, check the file is output correctly')
        #                     else:
        #                         self.assertFalse(os.path.isfile(pkl_filename), msg = pkl_filename + ' should not exist')
        #                         self.assertFalse(os.path.isfile(csv_filename), msg = csv_filename + ' should not exist')


    def step_4_prob(self):
        print('In Step 4')
        test_dict = initialize_probability_dict()
        test_dict = fill_probability_dict_all(test_dict, self)
        csv_filename = './tests/output\\csv\\probability_metrics.csv'
        keyword_all_clear = []
        with open(csv_filename, mode = 'r') as csvfile:
            reading = csv.reader(csvfile, delimiter = ',')
            
            for row in reading:
                if keyword_all_clear == []:
                    keyword_all_clear = row
                else:
                    for j in range(len(row)):
                        if keyword_all_clear[j] == '':
                            pass
                        else:
                            keyword = keyword_all_clear[j]
                            # print(keyword, row[j], test_dict[keyword][0])
                            self.assertEqual(row[j], test_dict[keyword][0], msg = 'This is the keyword ' + keyword)


    def step_5_peak_int_max(self):
        print('In Step 5')
        test_dict = initialize_flux_dict()
        test_dict = fill_peak_intensity_max_metrics_dict_all(test_dict, self)
        csv_filename = './tests/output\\csv\\peak_intensity_max_metrics.csv'
        keyword_all_clear = []
        with open(csv_filename, mode = 'r') as csvfile:
            reading = csv.reader(csvfile, delimiter = ',')
            
            for row in reading:
                if keyword_all_clear == []:
                    keyword_all_clear = row
                else:
                    for j in range(len(row)):
                        if keyword_all_clear[j] == '':
                            pass
                        else:
                            keyword = keyword_all_clear[j]
                            # print(keyword, row[j], test_dict[keyword][0])
                            self.assertEqual(row[j], test_dict[keyword][0], msg = 'This is the keyword ' + keyword)

    def step_6_time_prof(self):
        print('In Step 6')
        test_dict = initialize_flux_dict()
        test_dict = fill_time_profile_dict_all(test_dict, self)
        csv_filename = './tests/output\\csv\\time_profile_metrics.csv'
        keyword_all_clear = []
        with open(csv_filename, mode = 'r') as csvfile:
            reading = csv.reader(csvfile, delimiter = ',')
            
            for row in reading:
                if keyword_all_clear == []:
                    keyword_all_clear = row
                else:
                    for j in range(len(row)):
                        if keyword_all_clear[j] == '':
                            pass
                        else:
                            keyword = keyword_all_clear[j]
                            # print(keyword, row[j], test_dict[keyword][0])
                            self.assertEqual(row[j], test_dict[keyword][0], msg = 'This is the keyword ' + keyword)

    def step_7_all_clear(self):
        print('In Step 7')
        test_dict = initialize_all_clear_dict()
        test_dict = fill_all_clear_dict_all(test_dict, self)
        csv_filename = './tests/output\\csv\\all_clear_metrics.csv'
        keyword_all_clear = []
        with open(csv_filename, mode = 'r') as csvfile:
            reading = csv.reader(csvfile, delimiter = ',')
            
            for row in reading:
                if keyword_all_clear == []:
                    keyword_all_clear = row
                else:
                    for j in range(len(row)):
                        if keyword_all_clear[j] == '':
                            pass
                        else:
                            keyword = keyword_all_clear[j]
                            # print(keyword, row[j], test_dict[keyword][0])
                            self.assertEqual(row[j], test_dict[keyword][0], msg = 'This is the keyword ' + keyword)

    
    def step_8_awt(self):
        print('In Step 8')
        test_dict = initialize_awt_dict()
        test_dict = fill_awt_dict_all(test_dict, self)
        csv_filename = './tests/output\\csv\\awt_metrics.csv'
        keyword_all_clear = []
        with open(csv_filename, mode = 'r') as csvfile:
            reading = csv.reader(csvfile, delimiter = ',')
            
            for row in reading:
                if keyword_all_clear == []:
                    keyword_all_clear = row
                else:
                    for j in range(len(row)):
                        if keyword_all_clear[j] == '':
                            pass
                        else:
                            keyword = keyword_all_clear[j]
                            # print(keyword, row[j], test_dict[keyword][0])
                            self.assertEqual(row[j], test_dict[keyword][0], msg = 'This is the keyword ' + keyword)

    def step_9_duration(self):
        # print('In Step 8')
        test_dict = initialize_time_dict()
        test_dict = fill_duration_metrics_dict_all(test_dict, self)
        csv_filename = './tests/output\\csv\\duration_metrics.csv'
        keyword_all_clear = []
        with open(csv_filename, mode = 'r') as csvfile:
            reading = csv.reader(csvfile, delimiter = ',')
            
            for row in reading:
                if keyword_all_clear == []:
                    keyword_all_clear = row
                else:
                    for j in range(len(row)):
                        if keyword_all_clear[j] == '':
                            pass
                        else:
                            keyword = keyword_all_clear[j]
                            # print(keyword, row[j], test_dict[keyword][0])
                            self.assertEqual(row[j], test_dict[keyword][0], msg = 'This is the keyword ' + keyword)
    
    def step_10_end_time(self):
        print('In Step 10')
        test_dict = initialize_time_dict()
        test_dict = fill_end_time_metrics_dict_all(test_dict, self)
        csv_filename = './tests/output\\csv\\end_time_metrics.csv'
        keyword_all_clear = []
        with open(csv_filename, mode = 'r') as csvfile:
            reading = csv.reader(csvfile, delimiter = ',')
            
            for row in reading:
                if keyword_all_clear == []:
                    keyword_all_clear = row
                else:
                    for j in range(len(row)):
                        if keyword_all_clear[j] == '':
                            pass
                        else:
                            keyword = keyword_all_clear[j]
                            # print(keyword, row[j], test_dict[keyword][0])
                            self.assertEqual(row[j], test_dict[keyword][0], msg = 'This is the keyword ' + keyword)

    def step_11_last_data_to_issue_time(self):
        print('In Step 11')
        test_dict = initialize_time_dict()
        test_dict = fill_last_data_to_issue_time_metrics_dict_all(test_dict, self)
        csv_filename = './tests/output\\csv\\last_data_to_issue_time_metrics.csv'
        keyword_all_clear = []
        with open(csv_filename, mode = 'r') as csvfile:
            reading = csv.reader(csvfile, delimiter = ',')
            
            for row in reading:
                if keyword_all_clear == []:
                    keyword_all_clear = row
                else:
                    for j in range(len(row)):
                        if keyword_all_clear[j] == '':
                            pass
                        else:
                            keyword = keyword_all_clear[j]
                            # print(keyword, row[j], test_dict[keyword][0])
                            self.assertEqual(row[j], test_dict[keyword][0], msg = 'This is the keyword ' + keyword)

    def step_12_max_flux_pred_win(self):
        print('In Step 12')
        test_dict = initialize_flux_dict()
        test_dict = fill_max_flux_in_pred_win_metrics_dict_all(test_dict, self)
        csv_filename = './tests/output\\csv\\max_flux_in_pred_win_metrics.csv'
        keyword_all_clear = []
        with open(csv_filename, mode = 'r') as csvfile:
            reading = csv.reader(csvfile, delimiter = ',')
            
            for row in reading:
                if keyword_all_clear == []:
                    keyword_all_clear = row
                else:
                    for j in range(len(row)):
                        if keyword_all_clear[j] == '':
                            pass
                        else:
                            keyword = keyword_all_clear[j]
                            # print(keyword, row[j], test_dict[keyword][0])
                            self.assertEqual(row[j], test_dict[keyword][0], msg = 'This is the keyword ' + keyword)    

    def step_13_peak_int_max_time(self):
        print('In Step 13')
        test_dict = initialize_time_dict()
        test_dict = fill_peak_intensity_max_time_metrics_dict_all(test_dict, self)
        csv_filename = './tests/output\\csv\\peak_intensity_max_time_metrics.csv'
        keyword_all_clear = []
        with open(csv_filename, mode = 'r') as csvfile:
            reading = csv.reader(csvfile, delimiter = ',')
            
            for row in reading:
                if keyword_all_clear == []:
                    keyword_all_clear = row
                else:
                    for j in range(len(row)):
                        if keyword_all_clear[j] == '':
                            pass
                        else:
                            keyword = keyword_all_clear[j]
                            # print(keyword, row[j], test_dict[keyword][0])
                            self.assertEqual(row[j], test_dict[keyword][0], msg = 'This is the keyword ' + keyword)    
    
    def step_14_peak_int(self):
        print('In Step 14')
        test_dict = initialize_flux_dict()
        test_dict = fill_peak_intensity_metrics_dict_all(test_dict, self)
        csv_filename = './tests/output\\csv\\peak_intensity_metrics.csv'
        keyword_all_clear = []
        with open(csv_filename, mode = 'r') as csvfile:
            reading = csv.reader(csvfile, delimiter = ',')
            
            for row in reading:
                if keyword_all_clear == []:
                    keyword_all_clear = row
                else:
                    for j in range(len(row)):
                        if keyword_all_clear[j] == '':
                            pass
                        else:
                            keyword = keyword_all_clear[j]
                            # print(keyword, row[j], test_dict[keyword][0])
                            self.assertEqual(row[j], test_dict[keyword][0], msg = 'This is the keyword ' + keyword)    

    def step_15_peak_int_time(self):
        print('In Step 15')
        test_dict = initialize_time_dict()
        test_dict = fill_peak_intensity_time_dict_all(test_dict, self)
        csv_filename = './tests/output\\csv\\peak_intensity_time_metrics.csv'
        keyword_all_clear = []
        with open(csv_filename, mode = 'r') as csvfile:
            reading = csv.reader(csvfile, delimiter = ',')
            
            for row in reading:
                if keyword_all_clear == []:
                    keyword_all_clear = row
                else:
                    for j in range(len(row)):
                        if keyword_all_clear[j] == '':
                            pass
                        else:
                            keyword = keyword_all_clear[j]
                            # print(keyword, row[j], test_dict[keyword][0])
                            self.assertEqual(row[j], test_dict[keyword][0], msg = 'This is the keyword ' + keyword)    

    def step_16_start_time(self):
        print('In Step 16')
        test_dict = initialize_time_dict()
        test_dict = fill_start_time_dict_all(test_dict, self)
        csv_filename = './tests/output\\csv\\start_time_metrics.csv'
        keyword_all_clear = []
        with open(csv_filename, mode = 'r') as csvfile:
            reading = csv.reader(csvfile, delimiter = ',')
            
            for row in reading:
                if keyword_all_clear == []:
                    keyword_all_clear = row
                else:
                    for j in range(len(row)):
                        if keyword_all_clear[j] == '':
                            pass
                        else:
                            keyword = keyword_all_clear[j]
                            # print(keyword, row[j], test_dict[keyword][0])
                            self.assertEqual(row[j], test_dict[keyword][0], msg = 'This is the keyword ' + keyword)    
    
    def step_17_thresh_crossing_time(self):
        print('In Step 17')
        test_dict = initialize_time_dict()
        test_dict = fill_threshold_crossing_time_dict_all(test_dict, self)
        csv_filename = './tests/output\\csv\\threshold_crossing_metrics.csv'
        keyword_all_clear = []
        with open(csv_filename, mode = 'r') as csvfile:
            reading = csv.reader(csvfile, delimiter = ',')
            
            for row in reading:
                if keyword_all_clear == []:
                    keyword_all_clear = row
                else:
                    for j in range(len(row)):
                        if keyword_all_clear[j] == '':
                            pass
                        else:
                            keyword = keyword_all_clear[j]
                            # print(keyword, row[j], test_dict[keyword][0])
                            self.assertEqual(row[j], test_dict[keyword][0], msg = 'This is the keyword ' + keyword)    
    
    def step_18_fluence(self):
        print('In Step 18')
        test_dict = initialize_flux_dict()
        test_dict = fill_fluence_dict_all(test_dict, self)
        csv_filename = './tests/output\\csv\\fluence_metrics.csv'
        keyword_all_clear = []
        with open(csv_filename, mode = 'r') as csvfile:
            reading = csv.reader(csvfile, delimiter = ',')
            
            for row in reading:
                if keyword_all_clear == []:
                    keyword_all_clear = row
                else:
                    for j in range(len(row)):
                        if keyword_all_clear[j] == '':
                            pass
                        else:
                            keyword = keyword_all_clear[j]
                            # print(keyword, row[j], test_dict[keyword][0])
                            self.assertEqual(row[j], test_dict[keyword][0], msg = 'This is the keyword ' + keyword)    

    def utility_print_docstring(self, function):
        if self.verbosity == 2:
            print('\n//----------------------------------------------------')
            print(function.__doc__)

    def _steps(self):
        for name in dir(self): # dir() result is implicitly sorted
            if name.startswith("step"):
                yield name, getattr(self, name)
        
    # @make_docstring_printable
    # def test_df_all_clear_1(this, self):
    #     """
    #     Tests that the dataframe is built correctly with the correct fields being filled in/added.
    #     The observation and forecast have exactly the same observation/prediction windows. 
    #     Matching requires (at a minimum) that there is a prediction window start/end with an observed
    #     SEP start time within the prediction window and that the last data time/trigger occur before the
    #     observed SEP start time.
    #     Observed all clear is False
    #     Forecast all clear is False

    #     The tests in this block are:
    #         Observed SEP All Clear is False
    #         Predicted SEP All Clear is False

    #     """
    @patch('sphinxval.utils.config.outpath', './tests/output')
    
    def test_all(self):
        validate.prepare_outdirs()
        for name, step in self._steps():
            try:
                step()
            except Exception as e:
                self.fail("{} failed ({}: {})".format(step, type(e), e))
        # utility_delete_output()


