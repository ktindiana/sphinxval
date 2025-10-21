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

from . import config_tests




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
from datetime import datetime
import numpy as np

from unittest.mock import patch
import shutil # using this to delete the contents of the output folder each run - since the unittest is based on the existence/creation of certain files each loop

logger = logging.getLogger(__name__)

def utility_setup_logging():
    # Create the tests/logs/ directory if it does not yet exist
    with patch('sphinxval.utils.config.logpath', './tests/logs'):
        if not os.path.exists(config.logpath):
            os.mkdir(config.logpath)

        config_file = pathlib.Path('./tests/log_config_tests.json')
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

    evaluated_sphinx, not_evaluated_sphinx, all_observed_thresholds, observed_sep_events =\
        match.match_all_forecasts(all_energy_channels, model_names,
            obs_objs, model_objs)
            

    return evaluated_sphinx, not_evaluated_sphinx, all_observed_thresholds, observed_sep_events
    
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


def utility_load_objects(obs_filenames, forecast_filenames):
    # This is used in the mismatch unit test since it needs
    # the sphinx objects in a slightly different configuration
    # due to how we structure the prediction energy and
    # prediction threshold keys to accomodate for the mismatch
    # energy channel. Need to mock the config to config_tests
    # since this function needs some of the variables from
    # config_tests
    import types
    from mock import Mock
    import sys
    from . import config_tests
    from sphinxval.utils import config

    module_name = 'config'
    mocked_config = types.ModuleType(module_name)
    sys.modules[module_name] = mocked_config


    mocked_config = Mock(name='sphinxval.utils.' + module_name)
    obs_jsons = vjson.read_json_list(obs_filenames)
    model_jsons = vjson.read_json_list(forecast_filenames)
    
    logger.info("STATS: Observation json files read in: " + str(len(obs_jsons)))
    logger.info("STATS: Forecast json files read in: " + str(len(model_jsons)))

    
    #Find energy channels available in the OBSERVATIONS
    #Only channels with observed values will be validated
    all_energy_channels = vjson.identify_all_energy_channels(obs_jsons, 'observation')
    
    obs_objs = {}
    model_objs = {}
    for channel in all_energy_channels:
        key = objh.energy_channel_to_key(channel)
        obs_objs.update({key: []})
        model_objs.update({key:[]})

    #If mismatch allowed in config.py, save observation and model
    #objects under separate keys specifically for mismatched energy
    #channels and thresholds
    mm_key = ""
    logger.debug('This should be true - load objects from json')
    logger.debug(config_tests.do_mismatch)
    if config_tests.do_mismatch:
        obs_objs.update({config_tests.mm_energy_key: []})
        model_objs.update({config_tests.mm_energy_key: []})


    #Load observation objects
    for json in obs_jsons:
        for channel in all_energy_channels:
            key = objh.energy_channel_to_key(channel)
            obj = vjson.observation_object_from_json(json, channel)
            
            logger.debug("Created OBSERVATION object from json " + obj.source   + ", " + str(channel))
            logger.debug("Observation window start: " + str(obj.observation_window_start))
            #skip if energy block wasn't present in json
            if not pd.isnull(obj.observation_window_start):
                obs_objs[key].append(obj)
                logger.debug("Adding " + obj.source + " to dictionary under "
                    "key " + key)
        
            if config_tests.do_mismatch:
                if key == config_tests.mm_obs_ek:
                    obj = vjson.observation_object_from_json(json, channel)
                    if not pd.isnull(obj.observation_window_start):
                        obs_objs[config_tests.mm_energy_key].append(obj)
                        logger.debug("Adding " + obj.source + " to dictionary under key " + config_tests.mm_energy_key)
            

    #Load json objects
    for json in model_jsons:
        short_name = json["sep_forecast_submission"]["model"]["short_name"]
        for channel in all_energy_channels:
            key = objh.energy_channel_to_key(channel)
            obj, is_good = vjson.forecast_object_from_json(json, channel)
            #At this point, may not be a good object if the forecast needed to use
            #a mismatched energy channel. Check that first before determine
            #outcome of object.
            #If the object is good, include here
            if not pd.isnull(obj.prediction_window_start):
                model_objs[key].append(obj)
                logger.debug("Created FORECAST object from json " + str(obj.source)  + ", " + key)
                logger.debug("Prediction window start: " + str(obj.prediction_window_start))
 
            if not is_good:
                logger.warning("Note issue with creating FORECAST object from json " + str(obj.source)  + ", " + key)

            #If mismatched observation and prediction energy channels
            #enabled, then find the correct prediction energy channel
            #to load.
            if config_tests.do_mismatch:
                if config_tests.mm_model in short_name:
                    if channel == config_tests.mm_obs_energy_channel:
                        pred_channel = config_tests.mm_pred_energy_channel
                        obj, is_good = vjson.forecast_object_from_json(json, pred_channel)
 
                        if not is_good:
                            logger.warning("Note issue with creating FORECAST object from json " + str(obj.source)  + ", mismatch channel" + str(pred_channel))

                        #skip if energy block wasn't present in json
                        if not pd.isnull(obj.prediction_window_start):
                            model_objs[config_tests.mm_energy_key].append(obj)
                            logger.debug("Adding " + obj.source + " to dictionary under key " + key)




    #Convert all_energy_channels to an array of string keys
    for i in range(len(all_energy_channels)):
        all_energy_channels[i] = objh.energy_channel_to_key(all_energy_channels[i])
        
    if config_tests.do_mismatch:
        all_energy_channels.append(config_tests.mm_energy_key)
    logger.debug('IN VJSON CHECKING MISMATCH')
    logger.debug(all_energy_channels)
    
    del obs_jsons
    del model_jsons

    for channel in all_energy_channels:
        logger.info("STATS: Observation objects created for : " + channel + ", " + str(len(obs_objs[channel])))
        logger.info("STATS: Forecast objects created for : " + channel + ", " + str(len(model_objs[channel])))

    return all_energy_channels, obs_objs, model_objs



def attributes_of_sphinx_obj(keyword, sphinx_obj, energy_channel_key, threshold_key):
    """
    Function that takes the keyword from the SPHINX dataframe and matches the keyword
    to the matched sphinx object to compare/assertEqual to - ensures that the 
    dataframe is being correctly built

    Some of these had similar structure/location within the matched SPHINX object so they
    are collapsed into dictionaries of Keyword to name of the object in the matched SPHINX
    object. These dictionaries are titled based on the depth into the matched sphinx object
    (i.e. depth_prediction means that those keywords are in the sphinx_obj.prediction 
    object). Additional information about how to extract the proper data from the sphinx
    object is also in the dictionary title, like if the data is a string, or if the
    threshold is needed to extract the data. 
    Anything that wasn't easy to extract into the dictionaries are at the end in 
    the form of an if/elif/else structure.
    """

    logger.debug(str(keyword))
    depth_prediction = {
        'Model': 'short_name',
        'Forecast Source' : 'source',
        'Forecast Path': 'path',
        'Forecast Issue Time': 'issue_time',
        'Prediction Window Start': 'prediction_window_start',
        'Prediction Window End': 'prediction_window_end',
        'Original Model Short Name': 'original_short_name'
    }
    if keyword in depth_prediction:
        return getattr(sphinx_obj.prediction, depth_prediction[keyword], None)

    depth_prediction_string = {
        'All Thresholds in Prediction': 'all_thresholds'
    }
    if keyword in depth_prediction_string:
        return str(getattr(sphinx_obj.prediction, depth_prediction_string[keyword], None))

    depth_top = {
        'Mismatch Allowed': 'mismatch',
        'Evaluation Status': 'not_evaluated'
    }
    if keyword in depth_top:
        return getattr(sphinx_obj, depth_top[keyword], None)

    depth_cmes = {
        'Number of CMEs': 'cmes',
        'CME Start Time': 'start_time',
        'CME Liftoff Time': 'liftoff_time',
        'CME Latitude' : 'lat',
        'CME Longitude': 'lon',
        'CME Speed': 'speed',
        'CME Half Width': 'half_width',
        'CME PA': 'pa',
        'CME Catalog': 'catalog'
    }
    if keyword in depth_cmes:
        if keyword == 'Number of CMEs':
            return len(getattr(sphinx_obj.prediction, depth_cmes[keyword], None))
        else:
            return getattr(sphinx_obj.prediction.cmes[-1], depth_cmes[keyword], None)

    depth_flare = {
        'Number of Flares': 'flares',
        'Flare Start Time': 'start_time',
        'Flare Peak Time': 'peak_time',
        'Flare Latitude' : "lat",
        'Flare Longitude': 'lon',
        'Flare End Time': 'end_time',
        'Flare Last Data Time': 'last_data_time',
        'Flare Intensity': 'intensity',
        'Flare Integrated Intensity': 'integrated_intensity',
        'Flare NOAA AR': 'noaa_region'
    }
    if keyword in depth_flare:
        if keyword == 'Number of Flares':
            return len(getattr(sphinx_obj.prediction, depth_flare[keyword], None))
        else:
            logger.debug(depth_flare[keyword])
            return getattr(sphinx_obj.prediction.flares, depth_flare[keyword], None)

    depth_top_string = {
        'Last Eruption Time': 'last_eruption_time',
        'Last Trigger Time': 'last_trigger_time',
        'Last Input Time': 'last_input_time',
        'Triggers before Peak Intensity': 'triggers_before_peak_intensity',
        'Inputs before Peak Intensity': 'inputs_before_peak_intensity',
        'Time Difference between Triggers and Peak Intensity': 'time_difference_triggers_peak_intensity',
        'Time Difference between Inputs and Peak Intensity': 'time_difference_inputs_peak_intensity',
        'Time Difference between Triggers and Peak Intensity Max': 'time_difference_triggers_peak_intensity_max',
        'Triggers before Peak Intensity Max': 'triggers_before_peak_intensity_max', 
        'Inputs before Peak Intensity Max': 'inputs_before_peak_intensity_max', 
        'Time Difference between Inputs and Peak Intensity Max': 'time_difference_inputs_peak_intensity_max'
    }
    if keyword in depth_top_string:
        return str(getattr(sphinx_obj, depth_top_string[keyword], None))

    depth_top_threshold = {
        'Observed SEP Start Time': 'observed_start_time',
        'Observed SEP End Time': 'observed_end_time'
    }
    if keyword in depth_top_threshold:
        return getattr(sphinx_obj, depth_top_threshold[keyword], None)[threshold_key[energy_channel_key][0]]

    depth_top_prediction_threshold = {
        'Predicted SEP Probability': 'return_predicted_probability',
        'Probability Match Status': 'return_predicted_probability',
        'Predicted SEP Threshold Crossing Time' : 'return_predicted_threshold_crossing_time',
        'Threshold Crossing Time Match Status': 'return_predicted_threshold_crossing_time',
        'Predicted SEP Start Time': 'return_predicted_start_time',
        'Start Time Match Status': 'return_predicted_start_time',
        'Predicted SEP End Time': 'return_predicted_end_time',
        'End Time Match Status': 'return_predicted_end_time',
        'Predicted SEP Duration': 'return_predicted_duration',
        'Duration Match Status': 'return_predicted_end_time',
        'Time Profile Match Status': 'return_predicted_end_time'
    }
    if keyword in depth_top_prediction_threshold:
        if 'Match Status' in keyword:
            return getattr(sphinx_obj, depth_top_prediction_threshold[keyword], None)(threshold_key[energy_channel_key][0])[1]
        else:
            return getattr(sphinx_obj, depth_top_prediction_threshold[keyword], None)(threshold_key[energy_channel_key][0])[0]

    depth_top_threshold_strings ={
        'Threshold Crossed in Prediction Window': 'threshold_crossed_in_pred_win',
        'Eruption before Threshold Crossed': 'eruptions_before_threshold_crossing',
        'Time Difference between Eruption and Threshold Crossing': 'time_difference_eruptions_threshold_crossing',
        'Triggers before Threshold Crossing': 'triggers_before_threshold_crossing',
        'Inputs before Threshold Crossing': 'inputs_before_threshold_crossing',
        'Triggers before SEP End': 'triggers_before_sep_end',
        'Inputs before SEP End': 'inputs_before_sep_end',
        'Time Difference between Triggers and SEP End': 'time_difference_triggers_sep_end',
        'Time Difference between Inputs and SEP End': 'time_difference_inputs_sep_end',
        'Prediction Window Overlap with Observed SEP Event': 'prediction_window_sep_overlap',
        'Ongoing SEP Event': 'observed_ongoing_events',
        'Eruption in Range': 'is_eruption_in_range'
    }
    if keyword in depth_top_threshold_strings:
        return str(getattr(sphinx_obj, depth_top_threshold_strings[keyword], None)[threshold_key[energy_channel_key][0]])
    

        
    
    if keyword == "Energy Channel Key":
       return energy_channel_key
    elif keyword == 'Threshold Key':
        return threshold_key[energy_channel_key][0]
    elif keyword == 'Prediction Energy Channel Key':
        if getattr(sphinx_obj, 'mismatch', None) == True:
            return config_tests.mm_pred_ek
        else:
            return energy_channel_key
    elif keyword == 'Prediction Threshold Key':
        if getattr(sphinx_obj, 'mismatch', None) == True:
            return config_tests.mm_pred_tk
        else:
            return threshold_key[energy_channel_key][0]
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
        return attribute    
    elif keyword == "Observed SEP All Clear":
        return getattr(sphinx_obj.observed_all_clear, 'all_clear_boolean', None)
    elif keyword == "Observed SEP Probability":
        return getattr(sphinx_obj.observed_probability[threshold_key[energy_channel_key][0]], 'probability_value', None)
    elif keyword == 'Observed SEP Threshold Crossing Time':
        return getattr(sphinx_obj.observed_threshold_crossing[threshold_key[energy_channel_key][0]], 'crossing_time', None)        
    elif keyword == 'Observed SEP Duration':
        return getattr(sphinx_obj, 'observed_duration', None)[threshold_key[energy_channel_key][0]]
    elif keyword == 'Observed SEP Fluence':
        return getattr(sphinx_obj.observed_fluence[threshold_key[energy_channel_key][0]], 'fluence', None)
    elif keyword == 'Observed SEP Fluence Units':
        return getattr(sphinx_obj.observed_fluence[threshold_key[energy_channel_key][0]], 'units', None)
    elif keyword == 'Observed SEP Fluence Spectrum':
        return getattr(sphinx_obj.observed_fluence_spectrum[threshold_key[energy_channel_key][0]], 'fluence_spectrum', None)
    elif keyword == 'Observed SEP Fluence Spectrum Units':
        return getattr(sphinx_obj.observed_fluence_spectrum[threshold_key[energy_channel_key][0]], 'fluence_units', None)
    elif keyword == "Observed SEP Peak Intensity (Onset Peak)":
        return getattr(sphinx_obj.observed_peak_intensity, 'intensity', None)
    elif keyword == "Observed SEP Peak Intensity (Onset Peak) Units":
        return getattr(sphinx_obj.observed_peak_intensity, 'units', None)
    elif keyword == "Observed SEP Peak Intensity (Onset Peak) Time":
        return getattr(sphinx_obj.observed_peak_intensity, 'time', None)
    elif keyword == "Observed SEP Peak Intensity Max (Max Flux)":
        return getattr(sphinx_obj.observed_peak_intensity_max, 'intensity', None)
    elif keyword == "Observed SEP Peak Intensity Max (Max Flux) Units":
        return getattr(sphinx_obj.observed_peak_intensity_max, 'units', None)
    elif keyword == "Observed SEP Peak Intensity Max (Max Flux) Time":
        return getattr(sphinx_obj.observed_peak_intensity_max, 'time', None)
    elif keyword == "Observed Point Intensity":
        return getattr(sphinx_obj.observed_point_intensity, 'intensity', None)
    elif keyword == "Observed Point Intensity Units":
        return getattr(sphinx_obj.observed_point_intensity, 'units', None)
    elif keyword == "Observed Point Intensity Time":
        return getattr(sphinx_obj.observed_point_intensity, 'time', None)
    elif keyword == "Observed Max Flux in Prediction Window":
        return getattr(sphinx_obj.observed_max_flux_in_prediction_window, 'intensity', None)
    elif keyword == "Observed Max Flux in Prediction Window Units":
        return getattr(sphinx_obj.observed_max_flux_in_prediction_window, 'units', None)
    elif keyword == "Observed Max Flux in Prediction Window Time":
        return getattr(sphinx_obj.observed_max_flux_in_prediction_window, 'time', None)
    elif keyword == "Predicted SEP All Clear":
        return getattr(sphinx_obj, 'return_predicted_all_clear', None)()[0]
    elif keyword == "All Clear Match Status":
        return getattr(sphinx_obj, 'return_predicted_all_clear', None)()[1]
    elif keyword == 'Predicted SEP All Clear Probability Threshold':
        return getattr(sphinx_obj.prediction.all_clear, 'probability_threshold', None)
    elif 'Fluence' in keyword:
        if 'Spectrum' in keyword:
            temp = 'return_predicted_fluence_spectrum'
        else:
            temp = 'return_predicted_fluence'
        if 'Units' in keyword:
            return getattr(sphinx_obj, temp, None)(threshold_key[energy_channel_key][0])[1]
        elif 'Match Status' in keyword:
            return getattr(sphinx_obj, temp, None)(threshold_key[energy_channel_key][0])[2]
        else:
            return getattr(sphinx_obj, temp, None)(threshold_key[energy_channel_key][0])[0]
    elif 'Peak Intensity' in keyword:
        if 'Onset Peak' in keyword:
            temp = 'return_predicted_peak_intensity'
        else:
            temp = 'return_predicted_peak_intensity_max'
        if 'Units' in keyword:
            return getattr(sphinx_obj, temp, None)()[1]
        elif 'Time' in keyword:
            return getattr(sphinx_obj, temp, None)()[2]
        elif 'Match Status' in keyword:
            return getattr(sphinx_obj, temp, None)()[3]    
        else:
            return getattr(sphinx_obj, temp, None)()[0]
    elif keyword == "Predicted Point Intensity":
        return getattr(sphinx_obj, 'return_predicted_point_intensity', None)()[0]
    elif keyword == "Predicted Point Intensity Units":
        return getattr(sphinx_obj, 'return_predicted_point_intensity', None)()[1]
    elif keyword == "Predicted Point Intensity Time":
        return getattr(sphinx_obj, 'return_predicted_point_intensity', None)()[2]
    elif keyword == 'Predicted Time Profile':
        try:
            attribute = getattr(sphinx_obj.prediction, 'path', None)
            attribute += getattr(sphinx_obj.prediction, 'sep_profile', None)
            return attribute
        except:
            return None
    elif keyword == 'Last Data Time to Issue Time':
        return getattr(sphinx_obj.prediction, 'last_data_time_to_issue_time' , None)()
    
    elif keyword == 'All Threshold Crossing Times':
        attribute = getattr(sphinx_obj, 'all_threshold_crossing_times', None)[threshold_key[energy_channel_key][0]]
        if len(attribute) == 1:
            if pd.isnull(attribute[0]): return str(attribute[0])
            else:
                attribute = str([str(datetime.strptime(str(attribute[0]), '%Y-%m-%d %H:%M:%S') )])
    
    
        return attribute

    elif keyword == 'Trigger Advance Time':
        return 'NaT'


def assert_equal_table(self, filename, test_dict):
    keyword_all_clear = []
    print(filename)
    with open(filename, mode = 'r') as csvfile:
        reading = pd.read_csv(csvfile, delimiter = ',')
    
        for rows in reading.index:
            # print('this is rows ', rows)
            # print(reading.iloc[rows])
            for keywords in reading.columns:
                # print('this is keywords ', keywords)
                # print()
                if 'Unnamed' in keywords:
                    pass
                else:
                    # keyword = keyword_all_clear
                    # print(reading.iloc[rows][keywords], type(reading.iloc[rows][keywords]))#, column)#, test_dict[keyword][0])
                    if pd.isna(reading.iloc[rows][keywords]):# and pd.isna(test_dict[keywords][0]):
                        self.assertTrue(pd.isna(reading.iloc[rows][keywords]))
                        self.assertTrue(test_dict[keywords][0], msg = 'This test_dict element should be a nan for keyword ' + keywords) 
                    elif isinstance(reading.iloc[rows][keywords], str):
                        self.assertEqual(reading.iloc[rows][keywords], test_dict[keywords][0], msg = 'This is the keyword ' + keywords) 
                    else:
                        print(keywords)
                        self.assertAlmostEqual(reading.iloc[rows][keywords], float(test_dict[keywords][0]), msg = 'This is the keyword ' + keywords)
                        
                               
