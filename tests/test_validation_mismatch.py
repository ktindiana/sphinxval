# SUBROUTINES RELATED TO WORKFLOW UNIT TESTING
# from tests import config_tests
import types
from mock import Mock
import sys
from . import config_tests
from sphinxval.utils import config

module_name = 'config'
mocked_config = types.ModuleType(module_name)
sys.modules[module_name] = mocked_config
mocked_config = Mock(name='sphinxval.utils.' + module_name)

from sphinxval.utils import units_handler as vunits
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



logger = logging.getLogger(__name__)



"""
Mismatch unit test requires a slightly different 
set up from the regular validation unit test
since the dataframe takes different values
from the config file (which I mock so that I can
more easily control the values of). Also since
this is mismatch the output file names may contain
_mm at the end 
"""


def utility_setup_logging():
    # Create the tests/logs/ directory if it does not yet exist
    if not os.path.exists(config_tests.logpath):
        os.mkdir(config_tests.logpath)

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

def attributes_of_mismatch_sphinx_obj(keyword, sphinx_obj, energy_channel_key, threshold_key, mm_pred_energy_channel_key, mm_pred_thresh_key):
    """
    Function that takes the keyword from the SPHINX dataframe and matches the keyword
    to the matched sphinx object to compare/assertEqual to - ensures that the 
    dataframe is being correctly built
    MISMATCH TEST
    """
    # if getattr(sphinx_obj, 'mismatch', None) == True and keyword != 'Threshold Key':
    #     mismatch_tk = config.mm_pred_tk
    #     mismatch_ek = 
    # energy_channel_key = obs_energy_channel_key
    # threshold_key = obs_threshold_key

    if keyword == "Model": 
        attribute = getattr(sphinx_obj.prediction, 'short_name', None)
    elif keyword == "Energy Channel Key":
        attribute = energy_channel_key
    elif keyword == 'Threshold Key':
        # if getattr(sphinx_obj, 'mismatch', None) == True
        #     attribute = 
        attribute = threshold_key[energy_channel_key][0]
    elif keyword == 'Mismatch Allowed':
        attribute = True
        # attribute = getattr(sphinx_obj, 'mismatch', None)
    elif keyword == 'Prediction Energy Channel Key':
        if getattr(sphinx_obj, 'mismatch', None) == True:
            
            # attribute = mm_pred_energy_channel
            attribute = mm_pred_energy_channel_key
        else:
            attribute = energy_channel_key
    elif keyword == 'Prediction Threshold Key':
        if getattr(sphinx_obj, 'mismatch', None) == True:
            attribute = config_tests.mm_pred_tk
    
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
        attribute = getattr(sphinx_obj.observed_probability, 'probability_value', None)
    elif keyword == 'Observed SEP Threshold Crossing Time':
        attribute = getattr(sphinx_obj.observed_threshold_crossing, 'crossing_time', None)        
    elif keyword == 'Observed SEP Start Time':
        attribute = getattr(sphinx_obj, 'observed_start_time', '')
    elif keyword == 'Observed SEP End Time':
        attribute = getattr(sphinx_obj, 'observed_end_time', None)
    elif keyword == 'Observed SEP Duration':
        attribute = getattr(sphinx_obj, 'observed_duration', None)
    elif keyword == 'Observed SEP Fluence':
        attribute = getattr(sphinx_obj.observed_fluence, 'fluence', None)
    elif keyword == 'Observed SEP Fluence Units':
        attribute = getattr(sphinx_obj.observed_fluence, 'units', None)
    elif keyword == 'Observed SEP Fluence Spectrum':
        attribute = getattr(sphinx_obj.observed_fluence_spectrum, 'fluence_spectrum', None)
    elif keyword == 'Observed SEP Fluence Spectrum Units':
        attribute = getattr(sphinx_obj.observed_fluence_spectrum, 'fluence_units', None)
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
            attribute += getattr(sphinx_obj.prediction, 'sep_profile', None)
        except:
            attribute = None
    elif keyword == 'Time Profile Match Status':
        attribute = getattr(sphinx_obj, 'return_predicted_end_time', None)(threshold_key[energy_channel_key][0])[1]
    elif keyword == 'Last Data Time to Issue Time':
        attribute = getattr(sphinx_obj.prediction, 'last_data_time_to_issue_time' , None)()
    elif keyword == 'Last Eruption Time':
        attribute = str(getattr(sphinx_obj, 'last_eruption_time' , None))
    elif keyword == 'Last Trigger Time':
        attribute = str(getattr(sphinx_obj, 'last_trigger_time' , None))
    elif keyword == 'Last Input Time': 
        attribute = str(getattr(sphinx_obj, 'last_input_time' , None))
    elif keyword == 'Threshold Crossed in Prediction Window': 
        attribute = getattr(sphinx_obj, 'threshold_crossed_in_pred_win', None)
    elif keyword == 'Eruption before Threshold Crossed':
        attribute = getattr(sphinx_obj, 'eruptions_before_threshold_crossing', None)
    elif keyword == 'Time Difference between Eruption and Threshold Crossing':
        attribute = getattr(sphinx_obj, 'time_difference_eruptions_threshold_crossing', None)
    elif keyword == 'Triggers before Threshold Crossing':
        attribute = getattr(sphinx_obj, 'triggers_before_threshold_crossing', None)
    elif keyword == 'Inputs before Threshold Crossing':
        attribute = getattr(sphinx_obj, 'inputs_before_threshold_crossing', None)
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
        attribute = getattr(sphinx_obj, 'triggers_before_sep_end', None)
    elif keyword == 'Inputs before SEP End':
        attribute = getattr(sphinx_obj, 'inputs_before_sep_end', None)
    elif keyword == 'Time Difference between Triggers and SEP End':
        attribute = getattr(sphinx_obj, 'time_difference_triggers_sep_end', None)
    elif keyword == 'Time Difference between Inputs and SEP End':
        attribute = getattr(sphinx_obj, 'time_difference_inputs_sep_end', None)
    elif keyword == 'Prediction Window Overlap with Observed SEP Event':
        attribute = getattr(sphinx_obj, 'prediction_window_sep_overlap', None)
    elif keyword == 'Ongoing SEP Event':
        attribute = getattr(sphinx_obj, 'observed_ongoing_events', None)
    else:
        attribute = 'Keyword not in sphinx object  ERROR'
    if attribute == {}:
        attribute = None 
    return attribute




class Test_AllFields_Mismatch(unittest.TestCase):

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
        self.obs_energy_channel = {'min': 10, 'max': -1, 'units': 'MeV'}
        self.obs_energy_key = "min." +str(float(self.obs_energy_channel['min'])) + ".max." \
                + str(float(self.obs_energy_channel['max'])) + ".units." \
                + vunits.convert_units_to_string(self.obs_energy_channel['units'])
        self.all_energy_channels = [self.obs_energy_key] 
        self.model_names = ['Test_model_0']
        observation_json = ['./tests/files/observations/validation/mismatch/all_clear_false.json', './tests/files/observations/validation/mismatch/all_clear_false_02.json']
        observation_objects = {self.obs_energy_key: []}
        for jsons in observation_json:
            observation = utility_load_observation(jsons, self.obs_energy_channel)
            observation_objects[self.obs_energy_key].append(observation)
        
        self.verbosity = utility_get_verbosity()
        forecast_json = ['./tests/files/forecasts/validation/mismatch/pred_all_clear_false_mismatch.json','./tests/files/forecasts/validation/mismatch/pred_all_clear_false_mismatch_02.json']
        self.pred_energy_channel = config_tests.mm_pred_energy_channel
        self.pred_energy_key = "min." +str(float(self.pred_energy_channel['min'])) + ".max." \
                + str(float(self.pred_energy_channel['max'])) + ".units." \
                + vunits.convert_units_to_string(self.pred_energy_channel['units'])
        forecast_objects = {config_tests.mm_pred_ek: []}
        for jsons in forecast_json:
            forecast = utility_load_forecast(jsons, config_tests.mm_pred_energy_channel)
            forecast_objects[self.pred_energy_key].append(forecast)
        self.all_energy_channels = [self.obs_energy_key] 

        self.pred_thresh_key = {self.pred_energy_key: [config_tests.mm_pred_tk]}
        

        self.sphinx, self.obs_thresholds, self.obs_sep_events = utility_match_sphinx(self.all_energy_channels, self.model_names, observation_objects, forecast_objects)
        
        
        logger.debug('Testing mismatched tk and ek')
        logger.debug(config_tests.mm_obs_energy_channel)
        logger.debug(config_tests.mm_obs_threshold)
        logger.debug(config_tests.mm_pred_tk)
        logger.debug(config.do_mismatch)
        
        
        
        
        
        self.profname_dict = None
        self.DoResume = False
        print('mm pred thresh key',self.pred_thresh_key)
        print('obs thresh key', self.obs_thresholds)
        self.validation_quantity = ['all_clear', 'awt', 'duration', 'end_time', 'fluence', 'max_flux_in_pred_window', 'peak_intensity_max', 'peak_intensity_max_time', 'peak_intensity' \
            'peak_intensity_time', 'probability', 'start_time', 'threshold_crossing', 'time_profile']
        # self.quantities_tested = ['probability']
        self.validation_type = ["All", "First", "Last", "Max", "Mean"]

        self.dataframe = validate.fill_df(self.sphinx, self.model_names, self.all_energy_channels, \
            self.obs_thresholds, self.profname_dict)
       
        for keywords in self.dataframe:
           
            
            temp = attributes_of_mismatch_sphinx_obj(keywords, self.sphinx['Test_model_0'][self.all_energy_channels[0]][0],\
                 self.obs_energy_key, self.obs_thresholds, self.pred_energy_key, self.pred_thresh_key)
            print('this is what should be equal - dataframe, temp, keywords', self.dataframe[keywords][0], temp, keywords)
            if 'SEP Fluence Spectrum' in keywords and "Units" not in keywords:
                try:
                    # print('This is a spectrum - or at least it should be', len(self.dataframe[keywords][0]))
                    for energies in range(len(self.dataframe[keywords][0])):
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

           
        
        

    def step_2(self):
        """
        step 2 writes the dataframe to a file and then checks that those files exist
        """
        validate.write_df(self.dataframe, "SPHINX_dataframe")
        
        self.assertTrue(os.path.isfile('.\\tests\\output\\csv\\SPHINX_dataframe.csv'), msg = 'SPHINX_dataframe.csv does not exist, check the file is output correctly')
        self.assertTrue(os.path.isfile('.\\tests\\output\\pkl\\SPHINX_dataframe.pkl'), msg = 'SPHINX_dataframe.pkl does not exist, check the file is output correctly')
    
    def step_3(self):
        """
        step 3 does the actual validation workflow on the mismatch dataframe, and then tests its output files
        exist
        """
        
        validate.calculate_intuitive_metrics(self.dataframe, self.model_names, self.all_energy_channels, \
                self.obs_thresholds, 'All')

        self.validation_quantity = ['awt', 'duration', 'end_time', 'fluence', 'max_flux_in_pred_window', 'peak_intensity_max', 'peak_intensity_max_time', 'peak_intensity' \
            'peak_intensity_time', 'probability', 'start_time', 'threshold_crossing', 'time_profile']
        for model in self.model_names:
            for quantities in self.validation_quantity:
               
                metrics_filename = '.\\tests\\output\\csv\\' + quantities + '_metrics' 
                self.assertTrue(os.path.isfile(metrics_filename + '.csv'), msg = metrics_filename + '.csv does not exist, check the file is output correctly')
                metrics_filename = '.\\tests\\output\\pkl\\' + quantities + '_metrics' 
                self.assertTrue(os.path.isfile(metrics_filename + '.pkl'), msg = metrics_filename + '.pkl does not exist, check the file is output correctly')
                
                
                for energy_channels in self.all_energy_channels:
                    for thresholds in self.obs_thresholds[energy_channels]:
                        threshold_shortened = thresholds.rsplit('.')[0]+ '_' + thresholds.rsplit('.')[1] + '.' + thresholds.rsplit('.')[2]

                        if quantities == 'awt':
                            # pkl_filename = './output\\pkl\\' + quantities + '_selections_' + model + '_' + energy_channels + '_' + threshold_shortened + '.pkl'
                            # csv_filename = './output\\csv\\' + quantities + '_selections_' + model + '_' + energy_channels + '_' + threshold_shortened + '.csv'
                            pkl_filename = '.\\tests\\output\\pkl\\' + quantities + '_selections_' + model + '_' + energy_channels + '_' + threshold_shortened + "_Predicted SEP All Clear.pkl"
                            csv_filename = '.\\tests\\output\\csv\\' + quantities + '_selections_' + model + '_' + energy_channels + '_' + threshold_shortened + "_Predicted SEP All Clear.csv"
                            self.assertTrue(os.path.isfile(pkl_filename) , \
                                msg = pkl_filename + ' does not exist, check the file is output correctly')
                            self.assertTrue(os.path.isfile(csv_filename), \
                                msg = csv_filename + ' does not exist, check the file is output correctly')
                            
                        # else:
                        pkl_filename = '.\\tests\\output\\pkl\\' + quantities + '_selections_' + model + '_' + energy_channels + '_' + threshold_shortened + '.pkl'
                        csv_filename = '.\\tests\\output\\csv\\' + quantities + '_selections_' + model + '_' + energy_channels + '_' + threshold_shortened + '.csv'
                    
                        self.assertTrue(os.path.isfile(pkl_filename), msg = pkl_filename + ' does not exist, check the file is output correctly')
                        self.assertTrue(os.path.isfile(csv_filename), msg = csv_filename + ' does not exist, check the file is output correctly')
        
    
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
    # @patch('sphinxval.utils.config', 'config_tests')
    @patch('sphinxval.utils.config.outpath', './tests/output')
    @patch('sphinxval.utils.config.logpath', './tests/logs')
    @patch('sphinxval.utils.config.do_mismatch', True)
    @patch('sphinxval.utils.config.mm_model', 'Test_model_0')
    @patch('sphinxval.utils.config.mm_pred_energy_channel', config_tests.mm_pred_energy_channel)
    @patch('sphinxval.utils.config.mm_pred_threshold', config_tests.mm_pred_threshold)
    @patch('sphinxval.utils.config.mm_obs_energy_channel', config_tests.mm_obs_energy_channel)
    @patch('sphinxval.utils.config.mm_obs_threshold', config_tests.mm_obs_threshold)
    @patch('sphinxval.utils.config.mm_obs_ek', config_tests.mm_obs_ek)
    @patch('sphinxval.utils.config.mm_obs_tk', config_tests.mm_obs_tk)
    
    def test_all(self):
        validate.prepare_outdirs()
        utility_delete_output()
        
        

        
        for name, step in self._steps():
            try:
                step()
            except Exception as e:
                self.fail("{} failed ({}: {})".format(step, type(e), e))
        # utility_delete_output()
