# SUBROUTINES RELATED TO WORKFLOW UNIT TESTING
from sphinxval.utils import config
from sphinxval.utils import object_handler as objh
from sphinxval.utils import validation
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
import json

""" 
utils/test_match.py contains subroutines to run unit tests on match.py functions
"""

# HELPER FUNCTIONS
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

def utility_get_unique_dicts(dict_list):
    """
    Extracts the unique dictionaries from a list of dictionaries.
    """
    unique_set = set()
    unique_list = []
    for i in range(0, len(dict_list)):
        frozen_set = frozenset(dict_list[i].items())
        if frozen_set not in unique_set:
            unique_set.add(frozen_set)
            unique_list.append(dict_list[i])
    return unique_list
    
def utility_convert_dict_to_key_energy(d):
    """
    Converts dict to key (energy channel).
    """
    string = 'min.' + '{:.1f}'.format(d['min']) + '.max.' + '{:.1f}'.format(d['max']) + '.units.' + str(d['units'])
    return string

def utility_convert_dict_to_key_flux(d):
    """
    Converts dict to key (flux threshold).
    """
    string = 'threshold.' + '{:.1f}'.format(d['threshold']) + '.units.' + str(d['threshold_units'])
    return string

def make_docstring_printable(function):
    """
    Decorator method @make_docstring_printable.
    Allows function to call itself to access docstring information.
    Used for verbose unit test outputs.
    """
    def wrapper(*args, **kwargs):
        return function(function, *args, **kwargs)
    return wrapper

def tag(*tags):
    """
    Decorator to add tags to a test class or method.
    """
    def decorator(obj):
        setattr(obj, 'tags', set(tags))
        return obj
    return decorator

class LoadMatch(unittest.TestCase):
    """
    Class for loading main observation, SPHINX objects
    """
    def load_verbosity(self):
        self.verbosity = utility_get_verbosity()
    
    def load_energy(self, energy_channel):
        self.energy_channel = energy_channel
        self.energy_key = objh.energy_channel_to_key(self.energy_channel)
        self.all_energy_channels = [self.energy_key] 
        
    def load_observation(self, observation_json):
        self.observation_json = observation_json
        self.observation = utility_load_observation(self.observation_json, self.energy_channel)
        self.observation_objects = {self.energy_key : [self.observation]}
        self.observation_values = match.compile_all_obs(self.all_energy_channels, self.observation_objects)
        self.threshold = self.observation_values[self.energy_key]['thresholds'][0]
        self.threshold_key = objh.threshold_to_key(self.threshold)

    def load_matched_sphinx_and_inputs(self, forecasts, all_forecast_thresholds):
        """
        Loads inputs and SPHINX object to test match.py functions.
        """
        forecast_objects = {}
        forecast_objects[self.energy_key] = []
        for i in range(0, len(forecasts)):
            forecast_objects[self.energy_key].append(forecasts[i])
        self.model_names = ['unit_test']
        matched_sphinx = {}
        matched_sphinx, observed_sep_events = match.setup_match_all_forecasts(self.all_energy_channels, 
                                                            self.observation_objects,
                                                            self.observation_values,
                                                            forecast_objects, 
                                                            self.model_names)
        return matched_sphinx, observed_sep_events

    def load_sphinx_and_inputs(self, forecast, all_forecast_thresholds):
        """
        Loads inputs and SPHINX object to test match.py functions.
        """
        sphinx = objh.initialize_sphinx(forecast)
        forecast_objects = {self.energy_key : [forecast]}
        self.model_names = ['unit_test']
        matched_sphinx = {}
        matched_sphinx, observed_sep_events = match.setup_match_all_forecasts(self.all_energy_channels, 
                                                            self.observation_objects,
                                                            self.observation_values,
                                                            forecast_objects, 
                                                            self.model_names)
        sphinx = matched_sphinx['unit_test'][self.energy_key][0]
        return sphinx, observed_sep_events
    
    def utility_print_docstring(self, function):
        if self.verbosity == 2:
            print('\n//----------------------------------------------------')
            print(function.__doc__)




############################### TESTS ##################################
# sphinx.py --> match.match_all_forecasts --> match.does_win_overlap --> match.pred_and_obs_overlap
class TestPredAndObsOverlap(unittest.TestCase):
    """
    Unit test class for pred_and_obs_overlap function in match.py
    """
    
    def setUp(self):        
        self.energy_channel = {'min': 10, 'max': -1, 'units': 'MeV'}
        energy_key = objh.energy_channel_to_key(self.energy_channel)
        all_energy_channels = [energy_key] 
        observation_json = './tests/files/observations/match/pred_and_obs_overlap/pred_and_obs_overlap.json'
        observation = utility_load_observation(observation_json, self.energy_channel)
        observation_objects = {energy_key: [observation]}
        observation_values = match.compile_all_obs(all_energy_channels, observation_objects)
        self.obs = observation_values[energy_key]['dataframes'][0]
        self.verbosity = utility_get_verbosity()
    
    def utility_print_docstring(self, function):
        if self.verbosity == 2:
            print('\n//----------------------------------------------------')
            print(function.__doc__)
    
    @make_docstring_printable
    def test_pred_and_obs_overlap_1(this, self):
        """
        The observation and forecast have exactly the same observation/prediction windows.
        The function should evaluate to [True].
        """
        self.utility_print_docstring(this)
        forecast_json = './tests/files/forecasts/match/pred_and_obs_overlap/pred_and_obs_overlap_1.json'
        forecast = utility_load_forecast(forecast_json, self.energy_channel)
        self.assertEqual(match.pred_and_obs_overlap(forecast, self.obs), [True], 
                         'Forecast/Observation objects with identical prediction/observation windows marked as non-overlapping.') 
    
    @make_docstring_printable
    def test_pred_and_obs_overlap_2(this, self):
        """
        The observation and forecast have observation and prediction windows that are different, but overlap.
        The function should evaluate to [True].
        """
        self.utility_print_docstring(this)
        forecast_json = './tests/files/forecasts/match/pred_and_obs_overlap/pred_and_obs_overlap_2.json'
        forecast = utility_load_forecast(forecast_json, self.energy_channel)
        self.assertEqual(match.pred_and_obs_overlap(forecast, self.obs), [True], 
                         'Forecast/Observation objects with overlapping prediction/observation windows marked as non-overlapping.') 
    
    @make_docstring_printable
    def test_pred_and_obs_overlap_3(this, self):
        """
        The observation and forecast have observation and prediction windows that do not overlap.
        The function should evaluate to [False].
        """
        self.utility_print_docstring(this)
        forecast_json = './tests/files/forecasts/match/pred_and_obs_overlap/pred_and_obs_overlap_3.json'
        forecast = utility_load_forecast(forecast_json, self.energy_channel)
        self.assertEqual(match.pred_and_obs_overlap(forecast, self.obs), [False], 
                         'Forecast/Observation objects with non-overlapping prediction/observation windows marked as overlapping.') 

# sphinx.py --> match.match_all_forecasts --> match.match_observed_onset_peak
class TestMatchObservedOnsetPeak(LoadMatch):
    """
    Unit test class for match_observed_onset_peak function in match.py
    """
    def setUp(self):
        energy_channel = {'min': 10, 'max': -1, 'units': 'MeV'}
        observation_json = './tests/files/observations/match/match_observed_onset_peak/match_observed_onset_peak.json'
        self.load_verbosity()
        self.load_energy(energy_channel)
        self.load_observation(observation_json)
    
    def utility_print_inputs(self, sphinx, forecast_threshold_key, i):
        if self.verbosity == 2:
            print('')
            print('===== PRINT INPUTS =====')
            print('sphinx.peak_intensity_match_status =', sphinx.peak_intensity_match_status)
            print('sphinx.is_eruption_in_range =', sphinx.is_eruption_in_range[forecast_threshold_key][i])
            print('sphinx.triggers_before_peak_intensity =', sphinx.triggers_before_peak_intensity[i])
            print('sphinx.inputs_before_peak_intensity =', sphinx.inputs_before_peak_intensity[i]) 
            print('sphinx.is_win_overlap =', sphinx.is_win_overlap[i], '(always True)')
            print('sphinx.prediction_window_sep_overlap =', sphinx.prediction_window_sep_overlap[forecast_threshold_key][i])
            print('==========')
            print('')
            
    def utility_print_outputs(self, sphinx, function_evaluations):
        if self.verbosity == 2:
            print('')
            print('===== PRINT OUTPUTS =====')
            print('sphinx.observed_match_peak_intensity_source =', sphinx.observed_match_peak_intensity_source)
            print('sphinx.observed_peak_intensity =', sphinx.observed_peak_intensity)
            print('sphinx.peak_intensity_match_status =', sphinx.peak_intensity_match_status)
            print('function_evaluations =', function_evaluations)
            print('==========')
            print('')
            print('----------------------------------------------------//\n\n\n\n\n\n')
    
    def utility_test_match_observed_onset_peak(self, function, forecast_json):
        """
        Obtains SPHINX object and function evaluations given the forecast JSON.
        """
        self.utility_print_docstring(function)
        forecast = utility_load_forecast(forecast_json, self.energy_channel)
        # BUILD UP SPHINX OBJECT USING FORECAST AND OBSERVATION JSONS
        all_forecast_thresholds = forecast.identify_all_thresholds()
        sphinx, _ = self.load_sphinx_and_inputs(forecast, all_forecast_thresholds)
        function_evaluations = []
        for forecast_threshold_index in range(len(all_forecast_thresholds)):
            forecast_threshold = all_forecast_thresholds[forecast_threshold_index]
            forecast_threshold_key = objh.threshold_to_key(forecast_threshold)
            for i in sphinx.overlapping_indices:
                self.utility_print_inputs(sphinx, forecast_threshold_key, i)
                function_evaluation = match.match_observed_onset_peak(sphinx,
                                                                    self.observation_objects[self.energy_key][i],
                                                                    sphinx.is_win_overlap[i],
                                                                    sphinx.is_eruption_in_range[forecast_threshold_key][i],
                                                                    sphinx.triggers_before_peak_intensity[i],
                                                                    sphinx.inputs_before_peak_intensity[i],
                                                                    sphinx.prediction_window_sep_overlap[forecast_threshold_key][i])
                function_evaluations.append(function_evaluation)
        self.utility_print_outputs(sphinx, function_evaluations)
        return sphinx, function_evaluations 
    
    @make_docstring_printable
    def test_match_observed_onset_peak_1(this, self):
        """
        The forecast/observation pair has the following attributes:
           -- Prediction window overlaps with observation window
                prediction window start:  2000-01-01T00:00:00Z
                prediction window end:    2000-01-01T01:00:00Z
                observation window start: 2000-01-01T00:00:00Z
                observation window end:   2000-01-01T01:00:00Z
           -- The last eruption occurred between 24 hours and 8 minutes prior to threshold crossing
                CME start:                2000-01-01T00:00:00Z
                threshold crossing:       2000-01-01T00:16:00Z
           -- The prediction window overlaps with an SEP event
                SEP start:                2000-01-01T00:16:00Z
                SEP end:                  2000-01-01T00:35:00Z
        The function should evaluate to [True]
        sphinx.peak_intensity_match_status should be 'SEP Event'
        """
        forecast_json = './tests/files/forecasts/match/match_observed_onset_peak/match_observed_onset_peak_1.json'
        sphinx, function_evaluations = self.utility_test_match_observed_onset_peak(this, forecast_json)
        self.assertEqual(sphinx.observed_peak_intensity, self.observation.peak_intensity)
        self.assertEqual(sphinx.peak_intensity_match_status, 'SEP Event')
        self.assertEqual(function_evaluations, [True])
        
    @make_docstring_printable
    def test_match_observed_onset_peak_2(this, self):
        """
        The forecast/observation pair has the following attributes:
           -- Prediction window overlaps with observation window
                prediction window start:  2000-01-01T00:36:00Z
                prediction window end:    2000-01-01T01:30:00Z
                observation window start: 2000-01-01T00:00:00Z
                observation window end:   2000-01-01T01:00:00Z
           -- The last eruption occurred between 24 hours and 8 minutes prior to threshold crossing
                CME start:                2000-01-01T00:00:00Z
                threshold crossing:       2000-01-01T00:16:00Z
           -- The prediction window does not overlap with an SEP event
                SEP start:                2000-01-01T00:16:00Z
                SEP end:                  2000-01-01T00:35:00Z
        The function should evaluate to [False]
        sphinx.peak_intensity_match_status should be 'No SEP Event'
        """
        forecast_json = './tests/files/forecasts/match/match_observed_onset_peak/match_observed_onset_peak_2.json'
        sphinx, function_evaluations = self.utility_test_match_observed_onset_peak(this, forecast_json)
        self.assertEqual(function_evaluations, [False])
        self.assertEqual(sphinx.observed_match_peak_intensity_source, None)
        self.assertEqual(sphinx.peak_intensity_match_status, 'No SEP Event')
    
    @make_docstring_printable
    def test_match_observed_onset_peak_3(this, self):
        """
        The forecast/observation pair has the following attributes:
           -- Prediction window overlaps with observation window
                prediction window start:  2000-01-01T00:00:00Z
                prediction window end:    2000-01-01T01:00:00Z
                observation window start: 2000-01-01T00:00:00Z
                observation window end:   2000-01-01T01:00:00Z
           -- The last eruption was out of range, > 24 hours prior to threshold crossing
                CME start:                1999-12-29T00:00:00Z
                threshold crossing:       2000-01-01T00:16:00Z
           -- The prediction window overlaps with an SEP event
                SEP start:                2000-01-01T00:16:00Z
                SEP end:                  2000-01-01T00:35:00Z
        The function should evaluate to [False]
        sphinx.peak_intensity_match_status should be 'Eruption Out of Range'
        """
        forecast_json = './tests/files/forecasts/match/match_observed_onset_peak/match_observed_onset_peak_3.json'
        sphinx, function_evaluations = self.utility_test_match_observed_onset_peak(this, forecast_json)
        self.assertEqual(function_evaluations, [False])
        self.assertEqual(sphinx.observed_match_peak_intensity_source, None)
        self.assertEqual(sphinx.peak_intensity_match_status, 'Eruption Out of Range')
    
    @make_docstring_printable
    def test_match_observed_onset_peak_4(this, self):
        """
        The forecast/observation pair has the following attributes:
           -- Prediction window overlaps with observation window
                prediction window start:  2000-01-01T00:36:00Z
                prediction window end:    2000-01-01T01:30:00Z
                observation window start: 2000-01-01T00:00:00Z
                observation window end:   2000-01-01T01:00:00Z
           -- The last eruption occurred after the onset peak time
                CME start:                2000-01-01T00:31:00Z
                peak time:                2000-01-01T00:30:00Z
           -- The prediction window overlaps with an SEP event
                SEP start:                2000-01-01T00:16:00Z
                SEP end:                  2000-01-01T00:35:00Z
        The function should evaluate to [False]
        sphinx.peak_intensity_match_status should be 'Trigger/Input after Observed Phenomenon'
        """
        forecast_json = './tests/files/forecasts/match/match_observed_onset_peak/match_observed_onset_peak_4.json'
        sphinx, function_evaluations = self.utility_test_match_observed_onset_peak(this, forecast_json)
        self.assertEqual(function_evaluations, [False])
        self.assertEqual(sphinx.observed_match_peak_intensity_source, None)
        self.assertEqual(sphinx.peak_intensity_match_status, 'Trigger/Input after Observed Phenomenon')
    
    @make_docstring_printable
    def test_match_observed_onset_peak_5(this, self):
        """
        The forecast/observation pair has the following attributes:
           -- Prediction window overlaps with observation window
                prediction window start:  2000-01-01T00:36:00Z
                prediction window end:    2000-01-01T01:30:00Z
                observation window start: 2000-01-01T00:00:00Z
                observation window end:   2000-01-01T01:00:00Z
           -- The input occurred after the onset peak time
                magnetogram last data:    2000-01-01T00:31:00Z
                peak time:                2000-01-01T00:30:00Z
           -- The prediction window overlaps with an SEP event
                SEP start:                2000-01-01T00:16:00Z
                SEP end:                  2000-01-01T00:35:00Z
        The function should evaluate to [False]
        sphinx.peak_intensity_match_status should be 'Trigger/Input after Observed Phenomenon'
        """
        forecast_json = './tests/files/forecasts/match/match_observed_onset_peak/match_observed_onset_peak_5.json'
        sphinx, function_evaluations = self.utility_test_match_observed_onset_peak(this, forecast_json)
        self.assertEqual(function_evaluations, [False])
        self.assertEqual(sphinx.observed_match_peak_intensity_source, None)
        self.assertEqual(sphinx.peak_intensity_match_status, 'Trigger/Input after Observed Phenomenon')
    
    @make_docstring_printable
    def test_match_observed_onset_peak_6(this, self):
        """
        The forecast/observation pair has the following attributes:
           -- Prediction window overlaps with both observation windows
                prediction window start:    2000-01-01T00:00:00Z
                prediction window end:      2000-01-01T01:00:00Z
                observation windows start:  2000-01-01T00:00:00Z
                observation windows end:    2000-01-01T01:00:00Z
           -- The last eruption occurred between 24 hours and 8 minutes prior to threshold crossing
                CME start:                  2000-01-01T00:00:00Z
                peak time 1:                2000-01-01T00:30:00Z
                peak time 2:                2000-01-01T00:31:00Z
           -- The prediction window overlaps with an SEP event
                SEP start:                2000-01-01T00:16:00Z
                SEP end:                  2000-01-01T00:35:00Z
        The function should evaluate to [True, None] (forecast matches to first )
        sphinx.peak_intensity_match_status should be 'SEP Event'
        """
        observation_json = './tests/files/observations/match/match_observed_onset_peak/match_observed_onset_peak_6.json'
        observation = utility_load_observation(observation_json, self.energy_channel) # SAME ENERGY CHANNEL
        self.observation_objects[self.energy_key].append(observation)
        self.observation_values = match.compile_all_obs(self.all_energy_channels, self.observation_objects)
        forecast_json = './tests/files/forecasts/match/match_observed_onset_peak/match_observed_onset_peak_6.json'
        sphinx, function_evaluations = self.utility_test_match_observed_onset_peak(this, forecast_json)
        self.assertEqual(function_evaluations, [True, None])
        self.assertEqual(sphinx.peak_intensity_match_status, 'SEP Event')

    @make_docstring_printable
    def test_match_observed_onset_peak_7(this, self):
        """
        The forecast/observation pair has the following attributes:
           -- Prediction window overlaps with observation window
                prediction window start:  2000-01-01T00:00:00Z
                prediction window end:    2000-01-01T00:32:00Z
                observation window start: 2000-01-01T00:00:00Z
                observation window end:   2000-01-01T01:00:00Z
           -- The trigger occurred before the onset peak time
                magnetogram last data:    2000-01-01T00:00:00Z
                peak time:                2000-01-01T00:30:00Z
           -- The prediction window overlaps with an SEP event
                SEP start:                2000-01-01T00:16:00Z
                SEP end:                  2000-01-01T00:35:00Z
        The function should evaluate to [True]
        sphinx.peak_intensity_match_status should be 'SEP Event'
        """
        forecast_json = './tests/files/forecasts/match/match_observed_onset_peak/match_observed_onset_peak_7.json'
        sphinx, function_evaluations = self.utility_test_match_observed_onset_peak(this, forecast_json)
        self.assertEqual(function_evaluations, [True])
        self.assertEqual(sphinx.peak_intensity_match_status, 'SEP Event')
   
    @make_docstring_printable
    def test_match_observed_onset_peak_8(this, self):
        """
        The forecast/observation pair has the following attributes:
           -- Prediction window overlaps with observation window
                prediction window start:  2000-01-01T00:00:00Z
                prediction window end:    2000-01-01T00:18:00Z
                observation window start: 2000-01-01T00:00:00Z
                observation window end:   2000-01-01T01:00:00Z
           -- The trigger occurred before the onset peak time
                magnetogram last data:    2000-01-01T00:00:00Z
                peak time:                2000-01-01T00:30:00Z
           -- The prediction window overlaps with an SEP event (but does not overlap with peak)
                SEP start:                2000-01-01T00:16:00Z
                SEP end:                  2000-01-01T00:35:00Z
        The function should evaluate to [True]
        sphinx.peak_intensity_match_status should be 'SEP Event'
        """
        forecast_json = './tests/files/forecasts/match/match_observed_onset_peak/match_observed_onset_peak_8.json'
        sphinx, function_evaluations = self.utility_test_match_observed_onset_peak(this, forecast_json)
        self.assertEqual(function_evaluations, [True])
        self.assertEqual(sphinx.peak_intensity_match_status, 'SEP Event')
 



# sphinx.py --> match.match_all_forecasts --> match.match_observed_max_flux
class TestMatchObservedMaxFlux(LoadMatch):
    """
    Unit test class for match_observed_max_flux function in match.py
    """
    def setUp(self):
        energy_channel = {'min': 10, 'max': -1, 'units': 'MeV'}
        observation_json = './tests/files/observations/match/match_observed_max_flux/match_observed_max_flux.json'
        self.load_verbosity()
        self.load_energy(energy_channel)
        self.load_observation(observation_json)
    
    def utility_print_inputs(self, sphinx, forecast_threshold_key, i):
        if self.verbosity == 2:
            print('')
            print('===== PRINT INPUTS =====')
            print(sphinx.triggers_before_peak_intensity_max)
            print('sphinx.peak_intensity_max_match_status =', sphinx.peak_intensity_max_match_status)
            print('is_eruption_in_range =', sphinx.is_eruption_in_range[forecast_threshold_key][i])
            print('triggers_before_peak_intensity_max =', sphinx.triggers_before_peak_intensity_max[i])
            print('triggers_before_peak_intensity_max =', sphinx.triggers_before_peak_intensity_max[i]) 
            print('is_win_overlap =', sphinx.is_win_overlap[i], '(always True)')
            print('prediction_window_sep_overlap =', sphinx.prediction_window_sep_overlap[forecast_threshold_key][i])
            print('==========')
            print('')
            
    def utility_print_outputs(self, sphinx, function_evaluations):
        if self.verbosity == 2:
            print('')
            print('===== PRINT OUTPUTS =====')
            print('sphinx.observed_match_peak_intensity_max_source =', sphinx.observed_match_peak_intensity_max_source)
            print('sphinx.observed_peak_intensity_max =', sphinx.observed_peak_intensity_max)
            print('sphinx.peak_intensity_match_status =', sphinx.peak_intensity_max_match_status)
            print('function_evaluations =', function_evaluations)
            print('==========')
            print('')
            print('----------------------------------------------------//\n\n\n\n\n\n')
        
    def utility_test_match_observed_max_flux(self, function, forecast_json):
        """
        Obtains SPHINX object and function evaluations given the forecast JSON.
        """
        self.utility_print_docstring(function)
        forecast = utility_load_forecast(forecast_json, self.energy_channel)
        # BUILD UP SPHINX OBJECT USING FORECAST AND OBSERVATION JSONS
        all_forecast_thresholds = forecast.identify_all_thresholds()
        sphinx, _ = self.load_sphinx_and_inputs(forecast, all_forecast_thresholds)
        function_evaluations = []
        for forecast_threshold_index in range(len(all_forecast_thresholds)):
            forecast_threshold = all_forecast_thresholds[forecast_threshold_index]
            forecast_threshold_key = objh.threshold_to_key(forecast_threshold)
            for i in sphinx.overlapping_indices:
                self.utility_print_inputs(sphinx, forecast_threshold_key, i)
                function_evaluation = match.match_observed_max_flux(sphinx,
                                                                    self.observation_objects[self.energy_key][i],
                                                                    sphinx.is_win_overlap[i],
                                                                    sphinx.is_eruption_in_range[forecast_threshold_key][i],
                                                                    sphinx.triggers_before_peak_intensity_max[i],
                                                                    sphinx.inputs_before_peak_intensity_max[i],
                                                                    sphinx.prediction_window_sep_overlap[forecast_threshold_key][i])
                function_evaluations.append(function_evaluation)
        self.utility_print_outputs(sphinx, function_evaluations)
        return sphinx, function_evaluations 
        
    @make_docstring_printable
    def test_match_observed_max_flux_1(this, self):
        """
        The forecast/observation pair has the following attributes:
           -- Prediction window overlaps with observation window
                prediction window start:  2000-01-01T00:00:00Z
                prediction window end:    2000-01-01T01:00:00Z
                observation window start: 2000-01-01T00:00:00Z
                observation window end:   2000-01-01T01:00:00Z
           -- The last eruption occurred between 24 hours and 8 minutes prior to threshold crossing
                CME start:                2000-01-01T00:00:00Z
                threshold crossing:       2000-01-01T00:16:00Z
           -- The prediction window overlaps with an SEP event
                SEP start:                2000-01-01T00:16:00Z
                SEP end:                  2000-01-01T00:35:00Z
        The function should evaluate to [True]
        sphinx.peak_intensity_max_match_status should be 'SEP Event'
        """
        forecast_json = './tests/files/forecasts/match/match_observed_max_flux/match_observed_max_flux_1.json'
        sphinx, function_evaluations = self.utility_test_match_observed_max_flux(this, forecast_json)
        self.assertEqual(sphinx.observed_peak_intensity_max, self.observation.peak_intensity_max)
        self.assertEqual(sphinx.peak_intensity_max_match_status, 'SEP Event')
        self.assertEqual(function_evaluations, [True])
        
    @make_docstring_printable
    def test_match_observed_max_flux_2(this, self):
        """
        The forecast/observation pair has the following attributes:
           -- Prediction window overlaps with observation window
                prediction window start:  2000-01-01T00:36:00Z
                prediction window end:    2000-01-01T01:30:00Z
                observation window start: 2000-01-01T00:00:00Z
                observation window end:   2000-01-01T01:00:00Z
           -- The last eruption occurred between 24 hours and 8 minutes prior to threshold crossing
                CME start:                2000-01-01T00:00:00Z
                threshold crossing:       2000-01-01T00:16:00Z
           -- The prediction window does not overlap with an SEP event
                SEP start:                2000-01-01T00:16:00Z
                SEP end:                  2000-01-01T00:35:00Z
        The function should evaluate to [False]
        sphinx.peak_intensity_max_match_status should be 'No SEP Event'
        """
        forecast_json = './tests/files/forecasts/match/match_observed_max_flux/match_observed_max_flux_2.json'
        sphinx, function_evaluations = self.utility_test_match_observed_max_flux(this, forecast_json)
        self.assertEqual(function_evaluations, [False])
        self.assertEqual(sphinx.observed_match_peak_intensity_max_source, None)
        self.assertEqual(sphinx.peak_intensity_max_match_status, 'No SEP Event')
    
    @make_docstring_printable
    def test_match_observed_max_flux_3(this, self):
        """
        The forecast/observation pair has the following attributes:
           -- Prediction window overlaps with observation window
                prediction window start:  2000-01-01T00:00:00Z
                prediction window end:    2000-01-01T01:00:00Z
                observation window start: 2000-01-01T00:00:00Z
                observation window end:   2000-01-01T01:00:00Z
           -- The last eruption was out of range, > 24 hours prior to threshold crossing
                CME start:                1999-12-29T00:00:00Z
                threshold crossing:       2000-01-01T00:16:00Z
           -- The prediction window overlaps with an SEP event
                SEP start:                2000-01-01T00:16:00Z
                SEP end:                  2000-01-01T00:35:00Z
        The function should evaluate to [False]
        sphinx.peak_intensity_max_match_status should be 'Eruption Out of Range'
        """
        forecast_json = './tests/files/forecasts/match/match_observed_max_flux/match_observed_max_flux_3.json'
        sphinx, function_evaluations = self.utility_test_match_observed_max_flux(this, forecast_json)
        self.assertEqual(function_evaluations, [False])
        self.assertEqual(sphinx.observed_match_peak_intensity_max_source, None)
        self.assertEqual(sphinx.peak_intensity_max_match_status, 'Eruption Out of Range')
    
    @make_docstring_printable
    def test_match_observed_max_flux_4(this, self):
        """
        The forecast/observation pair has the following attributes:
           -- Prediction window overlaps with observation window
                prediction window start:  2000-01-01T00:36:00Z
                prediction window end:    2000-01-01T01:30:00Z
                observation window start: 2000-01-01T00:00:00Z
                observation window end:   2000-01-01T01:00:00Z
           -- The last eruption occurred after the onset peak time
                CME start:                2000-01-01T00:31:00Z
                peak time:                2000-01-01T00:30:00Z
           -- The prediction window overlaps with an SEP event
                SEP start:                2000-01-01T00:16:00Z
                SEP end:                  2000-01-01T00:35:00Z
        The function should evaluate to [False]
        sphinx.peak_intensity_max_match_status should be 'Trigger/Input after Observed Phenomenon'
        """
        forecast_json = './tests/files/forecasts/match/match_observed_max_flux/match_observed_max_flux_4.json'
        sphinx, function_evaluations = self.utility_test_match_observed_max_flux(this, forecast_json)
        self.assertEqual(function_evaluations, [False])
        self.assertEqual(sphinx.observed_match_peak_intensity_max_source, None)
        self.assertEqual(sphinx.peak_intensity_max_match_status, 'Trigger/Input after Observed Phenomenon')
    
    @make_docstring_printable
    def test_match_observed_max_flux_5(this, self):
        """
        The forecast/observation pair has the following attributes:
           -- Prediction window overlaps with observation window
                prediction window start:  2000-01-01T00:36:00Z
                prediction window end:    2000-01-01T01:30:00Z
                observation window start: 2000-01-01T00:00:00Z
                observation window end:   2000-01-01T01:00:00Z
           -- The trigger occurred after the onset peak time
                magnetogram last data:    2000-01-01T00:31:00Z
                peak time:                2000-01-01T00:30:00Z
           -- The prediction window overlaps with an SEP event
                SEP start:                2000-01-01T00:16:00Z
                SEP end:                  2000-01-01T00:35:00Z
        The function should evaluate to [False]
        sphinx.peak_intensity_max_match_status should be 'Trigger/Input after Observed Phenomenon'
        """
        forecast_json = './tests/files/forecasts/match/match_observed_max_flux/match_observed_max_flux_5.json'
        sphinx, function_evaluations = self.utility_test_match_observed_max_flux(this, forecast_json)
        self.assertEqual(function_evaluations, [False])
        self.assertEqual(sphinx.observed_match_peak_intensity_max_source, None)
        self.assertEqual(sphinx.peak_intensity_max_match_status, 'Trigger/Input after Observed Phenomenon')
    
    @make_docstring_printable
    def test_match_observed_max_flux_6(this, self):
        """
        The forecast/observation pair has the following attributes:
           -- Prediction window overlaps with both observation windows
                prediction window start:    2000-01-01T00:00:00Z
                prediction window end:      2000-01-01T01:00:00Z
                observation windows start:  2000-01-01T00:00:00Z
                observation windows end:    2000-01-01T01:00:00Z
           -- The last eruption occurred between 24 hours and 8 minutes prior to threshold crossing
                CME start:                  2000-01-01T00:00:00Z
                peak time 1:                2000-01-01T00:30:00Z
                peak time 2:                2000-01-01T00:31:00Z
           -- The prediction window overlaps with an SEP event
                SEP start:                2000-01-01T00:16:00Z
                SEP end:                  2000-01-01T00:35:00Z
        The function should evaluate to [True, None] (forecast matches to first )
        sphinx.peak_intensity_max_match_status should be 'SEP Event'
        """
        observation_json = './tests/files/observations/match/match_observed_max_flux/match_observed_max_flux_6.json'
        observation = utility_load_observation(observation_json, self.energy_channel) # SAME ENERGY CHANNEL
        self.observation_objects[self.energy_key].append(observation)
        self.observation_values = match.compile_all_obs(self.all_energy_channels, self.observation_objects)
        forecast_json = './tests/files/forecasts/match/match_observed_max_flux/match_observed_max_flux_6.json'
        sphinx, function_evaluations = self.utility_test_match_observed_max_flux(this, forecast_json)
        self.assertEqual(function_evaluations, [True, None])
        self.assertEqual(sphinx.peak_intensity_max_match_status, 'SEP Event')


class TestForecastValidity(LoadMatch):
    """
    Unit test classes for checking forecast JSON validity.
    """

    def setUp(self):
        energy_channel = {'min': 10, 'max': -1, 'units': 'MeV'}
        self.load_energy(energy_channel)
        self.load_verbosity()

    @make_docstring_printable
    def test_forecast_validity_1(this, self):
        """
        The forecast should be invalid because the trigger comes after the issue time. 
        No matchable forecast/observation pair exists.
        """
        forecast_json = './tests/files/forecasts/forecast_validity/forecast_validity_1.json'
        self.utility_print_docstring(this)
        forecast = utility_load_forecast(forecast_json, self.energy_channel)
        forecast.valid_forecast()
        self.assertEqual(forecast.valid, False)
 
    @make_docstring_printable
    def test_forecast_validity_2(this, self):
        """
        The forecast should be invalid because the trigger/input block does not exist. 
        No matchable forecast/observation pair exists.
        """
        forecast_json = './tests/files/forecasts/forecast_validity/forecast_validity_2.json'
        self.utility_print_docstring(this)
        forecast = utility_load_forecast(forecast_json, self.energy_channel)
        forecast.valid_forecast()
        self.assertEqual(forecast.valid, False)

    @make_docstring_printable
    def test_forecast_validity_3(this, self):
        """
        The forecast should be invalid because the trigger/input block is empty. 
        No matchable forecast/observation pair exists.
        """
        forecast_json = './tests/files/forecasts/forecast_validity/forecast_validity_3.json'
        self.utility_print_docstring(this)
        forecast = utility_load_forecast(forecast_json, self.energy_channel)
        forecast.valid_forecast()
        self.assertEqual(forecast.valid, False)
  
    @make_docstring_printable
    def test_forecast_validity_4(this, self):
        """
        The forecast should be invalid because the issue time does not exist. 
        No matchable forecast/observation pair exists.
        """
        forecast_json = './tests/files/forecasts/forecast_validity/forecast_validity_4.json'
        self.utility_print_docstring(this)
        self.assertRaises(KeyError, utility_load_forecast, forecast_json, self.energy_channel)
        '''
        forecast = utility_load_forecast(forecast_json, self.energy_channel)
        forecast.valid_forecast()
        self.assertEqual(forecast.valid, False)
        '''
 
    @make_docstring_printable
    def test_forecast_validity_5(this, self):
        """
        The forecast should be invalid because the issue time is empty. 
        No matchable forecast/observation pair exists.
        """
        forecast_json = './tests/files/forecasts/forecast_validity/forecast_validity_5.json'
        self.utility_print_docstring(this)
        forecast = utility_load_forecast(forecast_json, self.energy_channel)
        self.assertRaises(TypeError, forecast.valid_forecast)
        #self.assertEqual(forecast.valid, False)


# sphinx.py --> match.match_all_forecasts --> match.match_all_clear
class TestMatchAllClear(LoadMatch):
    """
    Unit test class for match_all_clear function in match.py
    """        
    def setUp(self):
        energy_channel = {'min': 10, 'max': -1, 'units': 'MeV'}
        observation_json = './tests/files/observations/match/match_all_clear/match_all_clear.json'
        self.load_verbosity()
        self.load_energy(energy_channel)
        self.load_observation(observation_json)
    
    def utility_print_inputs(self, sphinx, forecast_threshold_key, i):
        if self.verbosity == 2:
            print('')
            print('===== PRINT INPUTS =====')
            print('sphinx.observed_all_clear.all_clear_boolean =', sphinx.observed_all_clear.all_clear_boolean)
            print('sphinx.all_clear_match_status =', sphinx.all_clear_match_status)
            print('is_eruption_in_range =', sphinx.is_eruption_in_range[forecast_threshold_key][i])
            print('is_win_overlap =', sphinx.is_win_overlap[i], '(always True)')
            print('observed_ongoing_events =', sphinx.observed_ongoing_events[forecast_threshold_key][i])
            print('trigger_input_start =', sphinx.trigger_input_start[forecast_threshold_key][i])
            print('threshold_crossed_in_pred_win =', sphinx.threshold_crossed_in_pred_win[forecast_threshold_key][i])
            print('==========')
            print('')
            
    def utility_print_outputs(self, sphinx, function_evaluations):
        if self.verbosity == 2:
            print('')
            print('===== PRINT OUTPUTS =====')
            print('sphinx.all_clear_match_status =', sphinx.all_clear_match_status)
            print('sphinx.observed_all_clear.all_clear_boolean =', sphinx.observed_all_clear.all_clear_boolean)
            print('function_evaluations =', function_evaluations)
            print('==========')
            print('')
            print('----------------------------------------------------//\n\n\n\n\n\n')
    
    def utility_test_match_all_clear(self, function, forecast_json):
        """
        Obtains SPHINX object and function evaluations given the forecast JSON.
        """
        self.utility_print_docstring(function)
        forecast = utility_load_forecast(forecast_json, self.energy_channel)
        # BUILD UP SPHINX OBJECT USING FORECAST AND OBSERVATION JSONS
        all_forecast_thresholds = forecast.identify_all_thresholds()
        sphinx, _ = self.load_sphinx_and_inputs(forecast, all_forecast_thresholds)
        function_evaluations = []
        for forecast_threshold_index in range(len(all_forecast_thresholds)):
            forecast_threshold = all_forecast_thresholds[forecast_threshold_index]
            forecast_threshold_key = objh.threshold_to_key(forecast_threshold)
            for i in sphinx.overlapping_indices:
                self.utility_print_inputs(sphinx, forecast_threshold_key, i)
                function_evaluation = match.match_all_clear(sphinx,
                                                            self.observation_objects[self.energy_key][i],
                                                            sphinx.is_win_overlap[i],
                                                            sphinx.is_eruption_in_range[forecast_threshold_key][i],
                                                            sphinx.trigger_input_start[forecast_threshold_key][i],
                                                            sphinx.threshold_crossed_in_pred_win[forecast_threshold_key][i],
                                                            sphinx.observed_ongoing_events[forecast_threshold_key][i])
                function_evaluations.append(function_evaluation)
        self.utility_print_outputs(sphinx, function_evaluations)
        return sphinx, function_evaluations
    
    @make_docstring_printable
    def test_match_all_clear_1(this, self):
        """
        The forecast/observation pair has the following attributes:
           -- Prediction window overlaps with observation window
                prediction window start:  2000-01-01T00:00:00Z
                prediction window end:    2000-01-01T01:00:00Z
                observation window start: 2000-01-01T00:00:00Z
                observation window end:   2000-01-01T01:00:00Z
           -- No ongoing SEP event at start of prediction window
                event start:              2000-01-01T00:15:00Z
                event end:                2000-01-01T00:20:00Z
           -- The trigger (CME observation) for the forecast occurred before the threshold crossing 
                CME start:                2000-01-01T00:00:00Z
           -- A threshold crossing occurred within the prediction window
                threshold crossing:       2000-01-01T00:15:00Z
        The function should evaluate to [False] (All Clear = False is observed within the prediction window)
        sphinx.all_clear_match_status should be 'SEP Event'
        """
        forecast_json = './tests/files/forecasts/match/match_all_clear/match_all_clear_1.json'
        sphinx, function_evaluations = self.utility_test_match_all_clear(this, forecast_json)
        self.assertEqual(sphinx.all_clear_match_status, 'SEP Event')
        self.assertEqual(function_evaluations, [False])
        
    @make_docstring_printable
    def test_match_all_clear_2(this, self):
        """
        The forecast/observation pair has the following attributes:
           -- Prediction window overlaps with observation window
                prediction window start:  2000-01-01T00:30:00Z
                prediction window end:    2000-01-01T01:30:00Z
                observation window start: 2000-01-01T00:00:00Z
                observation window end:   2000-01-01T01:00:00Z
           -- No ongoing SEP event at start of prediction window
                event start:              2000-01-01T00:15:00Z
                event end:                2000-01-01T00:20:00Z
           -- The trigger (CME observation) for the forecast occurred before the threshold crossing 
                start:                    2000-01-01T00:00:00Z
           -- A threshold crossing occurred prior to the start of the prediction window
                threshold crossing:       2000-01-01T00:15:00Z
        The function should evaluate to [True] (All Clear = True is observed within the prediction window)
        sphinx.all_clear_match_status should be 'No SEP Event'
        """
        forecast_json = './tests/files/forecasts/match/match_all_clear/match_all_clear_2.json'
        sphinx, function_evaluations = self.utility_test_match_all_clear(this, forecast_json)
        self.assertEqual(sphinx.all_clear_match_status, 'No SEP Event')
        self.assertEqual(function_evaluations, [True])
    
    @make_docstring_printable
    def test_match_all_clear_3(this, self):
        """
        REVISION NEEDED? -- Should we allow forecasts whose prediction window extends to times prior to their issue time? Their last data time?
        The forecast/observation pair has the following attributes:
           -- Prediction window overlaps with observation window
                prediction window start:  2000-01-01T00:00:00Z # Is this even possible? Do any forecasts declare prediction windows that happened prior to the issue time?
                prediction window end:    2000-01-01T01:00:00Z # 
                observation window start: 2000-01-01T00:00:00Z
                observation window end:   2000-01-01T01:00:00Z
           -- No ongoing SEP event at start of prediction window
                event start:              2000-01-01T00:15:00Z
                event end:                2000-01-01T00:20:00Z
           -- The trigger (CME observation) for the forecast occurred after the threshold crossing 
                CME start:                2000-01-01T00:30:00Z
           -- The threshold crossing was observed within the prediction window
                threshold crossing:       2000-01-01T00:15:00Z
        The function should evaluate to [None]
        sphinx.all_clear_match_status should be 'Trigger/Input observed after Phenomenon'
        """
        forecast_json = './tests/files/forecasts/match/match_all_clear/match_all_clear_3.json'
        sphinx, function_evaluations = self.utility_test_match_all_clear(this, forecast_json)
        self.assertEqual(sphinx.all_clear_match_status, 'Trigger/Input after Observed Phenomenon')
        self.assertEqual(function_evaluations, [None])

    @make_docstring_printable
    def test_match_all_clear_4(this, self):
        """
        The forecast/observation pair has the following attributes:
           -- Prediction window overlaps with observation window
                prediction window start:  2000-01-01T00:00:00Z
                prediction window end:    2000-01-01T01:00:00Z
                observation window start: 2000-01-01T00:00:00Z
                observation window end:   2000-01-01T01:00:00Z
           -- No ongoing SEP event at start of prediction window
                event start:              2000-01-01T00:15:00Z
                event end:                2000-01-01T00:20:00Z
           -- The trigger (CME observation) for the forecast occurred long before (20 years) the threshold crossing
                CME start:                1980-01-01T00:30:00Z
           -- The threshold crossing was observed within the prediction window
                threshold crossing:       2000-01-01T00:15:00Z
        The function should evaluate to [True]
        sphinx.all_clear_match_status should be 'Eruption Out of Range' -- trigger occurs 20 years prior to threshold crossing
        """
        forecast_json = './tests/files/forecasts/match/match_all_clear/match_all_clear_4.json'
        sphinx, function_evaluations = self.utility_test_match_all_clear(this, forecast_json)
        self.assertEqual(sphinx.all_clear_match_status, 'Eruption Out of Range')
        self.assertEqual(function_evaluations, [True])

    @make_docstring_printable
    def test_match_all_clear_5(this, self):
        """
        The forecast/observation pair has the following attributes:
           -- Prediction window overlaps with observation window
                prediction window start:  2000-01-01T00:00:00Z
                prediction window end:    2000-01-01T01:00:00Z
                observation window start: 2000-01-01T00:00:00Z
                observation window end:   2000-01-01T01:00:00Z
           -- No ongoing SEP event at start of prediction window
                event start:              2000-01-01T00:15:00Z
                event end:                2000-01-01T00:20:00Z
           -- The trigger (CME observation) for the forecast occurred just before (1 minute) the threshold crossing
                CME start:                2000-01-01T00:14:00Z
           -- The threshold crossing was observed within the prediction window
                threshold crossing:       2000-01-01T00:15:00Z
        The function should evaluate to [True]
        sphinx.all_clear_match_status should be 'Eruption Out of Range' -- trigger occurs < 8 minutes prior to threshold crossing (violates speed limit of the universe)
        """
        forecast_json = './tests/files/forecasts/match/match_all_clear/match_all_clear_5.json'
        sphinx, function_evaluations = self.utility_test_match_all_clear(this, forecast_json)
        self.assertEqual(sphinx.all_clear_match_status, 'Eruption Out of Range')
        self.assertEqual(function_evaluations, [True])

    @make_docstring_printable
    def test_match_all_clear_6(this, self):
        """
        The forecast/observation pair has the following attributes:
           -- Prediction window overlaps with observation window
                prediction window start:  2000-01-01T00:17:00Z
                prediction window end:    2000-01-01T01:00:00Z
                observation window start: 2000-01-01T00:00:00Z
                observation window end:   2000-01-01T01:00:00Z
           -- There is an ongoing SEP event at start of prediction window
                event start:              2000-01-01T00:15:00Z
                event end:                2000-01-01T00:20:00Z
           -- The trigger (CME observation) and input (magnetogram image) for the forecast occurred before the threshold crossing
                CME start:                1999-12-31T23:00:00Z
                Magnetogram 1:            2000-01-01T00:00:00Z
                Magnetogram 2:            2000-01-01T00:00:01Z
           -- The threshold crossing was observed within the prediction window
                threshold crossing:       2000-01-01T00:15:00Z
        The function should evaluate to [None] ()
        sphinx.all_clear_match_status should be 'Ongoing SEP Event' 
        """
        forecast_json = './tests/files/forecasts/match/match_all_clear/match_all_clear_6.json'
        sphinx, function_evaluations = self.utility_test_match_all_clear(this, forecast_json)
        self.assertEqual(sphinx.all_clear_match_status, 'Ongoing SEP Event')
        self.assertEqual(function_evaluations, [None])

    @make_docstring_printable
    def test_match_all_clear_7(this, self):
        """
        The forecast/observation pair has the following attributes:
           -- Prediction window overlaps with observation window
                prediction window start:  2000-01-01T00:00:00Z
                prediction window end:    2000-01-01T01:00:00Z
                observation window start: 2000-01-01T00:00:00Z
                observation window end:   2000-01-01T01:00:00Z
           -- The trigger (CME observation) for the forecast occurred before the both threshold crossings
                CME start:                2000-01-01T00:00:00Z
           -- There are two SEP events in the prediction window
                event 1 start:            2000-01-01T00:15:00Z
                event 1 end:              2000-01-01T00:20:00Z
                event 2 start:            2000-01-01T00:25:00Z
                event 2 end:              2000-01-01T00:28:00Z
           -- The threshold crossing was observed within the prediction window
                threshold crossing:       2000-01-01T00:15:00Z
           -- In another observation JSON, a different threshold crossing was observed within the same prediction window
                threshold crossing:       2000-01-01T00:25:00Z
        The function should evaluate to [False, None] (the forecast correctly identifies All Clear = False, evidenced by first observation, second observation is ignored)
        sphinx.all_clear_match_status should be unchanged from 'SEP Event'
        sphinx.observed_all_clear.all_clear_boolean should be False from first forecast-observation match
        """
        observation_json = './tests/files/observations/match/match_all_clear/match_all_clear_7.json'
        observation = utility_load_observation(observation_json, self.energy_channel) # SAME ENERGY CHANNEL
        self.observation_objects[self.energy_key].append(observation)
        self.observation_values = match.compile_all_obs(self.all_energy_channels, self.observation_objects)        
        forecast_json = './tests/files/forecasts/match/match_all_clear/match_all_clear_7.json'
        sphinx, function_evaluations = self.utility_test_match_all_clear(this, forecast_json)
        self.assertEqual(sphinx.all_clear_match_status, 'SEP Event')
        self.assertEqual(sphinx.observed_all_clear.all_clear_boolean, False)
        self.assertEqual(function_evaluations, [False, None])
        
    @make_docstring_printable
    def test_match_all_clear_8(this, self):
        """
        The forecast/observation pair has the following attributes:
            -- Prediction window overlaps with observation window
                prediction window start: 2000-01-01T00:18:00Z
                prediction window end:   2000-01-01T01:00:00Z
                observation window start: 2000-01-01T00:00:00Z
                observation window end:   2000-01-01T01:00:00Z
           -- Ongoing SEP event at the start of the prediction window
                event start:            2000-01-01T00:15:00Z
                event end:              2000-01-01T00:20:00Z
           -- The trigger (CME observation) for the forecast occurred after the threshold crossing.
                CME start: 2000-01-01T00:17:00Z
           -- The threshold crossing was not observed within the prediction window
                threshold crossing:       2000-01-01T00:15:00Z
        The function should evaluate to [None]
        """
        forecast_json = './tests/files/forecasts/match/match_all_clear/match_all_clear_8.json'
        sphinx, function_evaluations = self.utility_test_match_all_clear(this, forecast_json)
        self.assertEqual(sphinx.all_clear_match_status, 'Ongoing SEP Event')
        self.assertEqual(function_evaluations, [None])

    @make_docstring_printable
    def test_match_all_clear_9(this, self):
        """
        The forecast/observation pair has the following attributes:
           -- Prediction window overlaps with observation window
                prediction window start:  2000-01-01T00:18:00Z
                prediction window end:    2000-01-01T01:00:00Z
                observation window start: 2000-01-01T00:00:00Z
                observation window end:   2000-01-01T01:00:00Z
                observation window start: 2000-01-01T00:25:00Z
                observation window end:   2000-01-01T01:00:00Z
           -- The trigger (CME observation) for the forecast occurred after the first, and before the second threshold crossing
                CME start:                2000-01-01T00:17:00Z
           -- There are two SEP events in the prediction window. One of the events is ongoing at the start of the prediction window.
                event 1 start:            2000-01-01T00:15:00Z
                event 1 end:              2000-01-01T00:20:00Z
                event 2 start:            2000-01-01T00:40:00Z
                event 2 end:              2000-01-01T00:59:00Z
           -- The second threshold crossing was observed within the prediction window
                threshold crossing:       2000-01-01T00:40:00Z
           The function should evaluate to [None, False] (the prediction window starts after the first event's threshold crossing time; the second event is correctly identified as All Clear = False)
        sphinx.all_clear_match_status should end up as 'SEP Event'
        sphinx.observed_all_clear.all_clear_boolean should be False from second forecast-observation match
        """
        observation_json = './tests/files/observations/match/match_all_clear/match_all_clear_9.json'
        observation = utility_load_observation(observation_json, self.energy_channel) # SAME ENERGY CHANNEL
        self.observation_objects[self.energy_key].append(observation)
        self.observation_values = match.compile_all_obs(self.all_energy_channels, self.observation_objects)        
        forecast_json = './tests/files/forecasts/match/match_all_clear/match_all_clear_9.json'
        sphinx, function_evaluations = self.utility_test_match_all_clear(this, forecast_json)
        self.assertEqual(sphinx.all_clear_match_status, 'SEP Event')
        self.assertEqual(sphinx.observed_all_clear.all_clear_boolean, False)
        self.assertEqual(function_evaluations, [None, False])
 
    @make_docstring_printable
    def test_match_all_clear_10(this, self):
        """
        The forecast/observation pair has the following attributes:
           -- Prediction window overlaps with observation window
                prediction window start:  2000-01-01T00:00:00Z
                prediction window end:    2000-01-01T00:45:00Z
                observation window start: 2000-01-01T00:00:00Z
                observation window end:   2000-01-01T01:00:00Z
                observation window start: 2000-01-01T00:25:00Z
                observation window end:   2000-01-01T01:00:00Z
           -- The trigger (CME observation) for the forecast occurred after the first, and before the second threshold crossing
                CME start:                2000-01-01T00:00:00Z
           -- There are two SEP events in the prediction window. One of the events is ongoing at the start of the prediction window.
                event 1 start:            2000-01-01T00:15:00Z
                event 1 end:              2000-01-01T00:20:00Z
                event 2 start:            2000-01-01T00:40:00Z
                event 2 end:              2000-01-01T00:59:00Z
           -- The second threshold crossing was observed within the prediction window
                threshold crossing:       2000-01-01T00:40:00Z
           The function should evaluate to [False, None] (the prediction window starts before the first event's threshold crossing time and stops in the middle of the second event; the second event is ignored)
        sphinx.all_clear_match_status should end up as 'SEP Event'
        sphinx.observed_all_clear.all_clear_boolean should be False from second forecast-observation match
        """
        observation_json = './tests/files/observations/match/match_all_clear/match_all_clear_10.json'
        observation = utility_load_observation(observation_json, self.energy_channel) # SAME ENERGY CHANNEL
        self.observation_objects[self.energy_key].append(observation)
        self.observation_values = match.compile_all_obs(self.all_energy_channels, self.observation_objects)        
        forecast_json = './tests/files/forecasts/match/match_all_clear/match_all_clear_10.json'
        sphinx, function_evaluations = self.utility_test_match_all_clear(this, forecast_json)
        self.assertEqual(sphinx.all_clear_match_status, 'SEP Event')
        self.assertEqual(sphinx.observed_all_clear.all_clear_boolean, False)
        self.assertEqual(function_evaluations, [False, None])

class TestNoMatchingThreshold(LoadMatch):
    """
    Unit test class for match_all_clear function in match.py
    """        
    def setUp(self):
        energy_channel = {'min': 10, 'max': -1, 'units': 'MeV'}
        observation_json = './tests/files/observations/match/match_all_clear/match_all_clear.json'
        self.load_verbosity()
        self.load_energy(energy_channel)
        self.load_observation(observation_json)
    
    def utility_print_inputs(self, sphinx, forecast_threshold_key, i):
        if self.verbosity == 2:
            print('')
            print('===== PRINT INPUTS =====')
            print('sphinx.observed_all_clear.all_clear_boolean =', sphinx.observed_all_clear.all_clear_boolean)
            print('sphinx.all_clear_match_status =', sphinx.all_clear_match_status)
            print('is_win_overlap =', sphinx.is_win_overlap[i], '(always True)')
            print('observed_ongoing_events =', sphinx.observed_ongoing_events[forecast_threshold_key][i])
            print('trigger_input_start =', sphinx.trigger_input_start[forecast_threshold_key][i])
            print('threshold_crossed_in_pred_win =', sphinx.threshold_crossed_in_pred_win[forecast_threshold_key][i])
            print('==========')
            print('')
            
    def utility_print_outputs(self, sphinx, function_evaluations):
        if self.verbosity == 2:
            print('')
            print('===== PRINT OUTPUTS =====')
            print('sphinx.all_clear_match_status =', sphinx.all_clear_match_status)
            print('sphinx.observed_all_clear.all_clear_boolean =', sphinx.observed_all_clear.all_clear_boolean)
            print('function_evaluations =', function_evaluations)
            print('==========')
            print('')
            print('----------------------------------------------------//\n\n\n\n\n\n')
    
    def utility_test_no_matching_threshold(self, function, forecast_json):
        """
        Obtains SPHINX object and function evaluations given the forecast JSON.
        """
        self.utility_print_docstring(function)
        forecast = utility_load_forecast(forecast_json, self.energy_channel)
        # BUILD UP SPHINX OBJECT USING FORECAST AND OBSERVATION JSONS
        all_forecast_thresholds = forecast.identify_all_thresholds()
        sphinx, _ = self.load_sphinx_and_inputs(forecast, all_forecast_thresholds)
        return sphinx






class TestMatchSEPQuantities(LoadMatch):
    """
    Unit test class for match_sep_quantities function in match.py
    """   
    def setUp(self):
        energy_channel = {'min': 10, 'max': -1, 'units': 'MeV'}
        observation_json = './tests/files/observations/match/match_sep_quantities/match_sep_quantities.json'
        self.load_verbosity()
        self.load_energy(energy_channel)
        self.load_observation(observation_json)
    
    # sphinx.py --> match.match_all_forecasts --> match.match_sep_quantities
    def utility_print_inputs(self, sphinx, forecast_threshold_key, i):
        if self.verbosity == 2:
            print('')
            print('===== PRINT INPUTS =====')
            print('sphinx.sep_match_status[thresh_key] =', sphinx.sep_match_status[forecast_threshold_key])
            print('is_eruption_in_range =', sphinx.is_eruption_in_range[forecast_threshold_key][i])
            print('is_win_overlap =', sphinx.is_win_overlap[i], '(always True)')
            print('observed_ongoing_events =', sphinx.observed_ongoing_events[forecast_threshold_key][i])
            print('trigger_input_start =', sphinx.trigger_input_start[forecast_threshold_key][i])
            print('threshold_crossed_in_pred_win =', sphinx.threshold_crossed_in_pred_win[forecast_threshold_key][i])
            print('==========')
            print('')
            
    def utility_print_outputs(self, sphinx, function_evaluations):
        if self.verbosity == 2:
            print('')
            print('===== PRINT OUTPUTS =====')
            print('sphinx.sep_match_status[thresh_key] =', sphinx.sep_match_status[self.threshold_key])
            print('function_evaluations =', function_evaluations)
            print('==========')
            print('')
            print('----------------------------------------------------//\n\n\n\n\n\n')
        
    def utility_test_match_sep_quantities(self, function, forecast_json):
        """
        Obtains SPHINX object and function evaluations given the forecast JSON.
        """
        self.utility_print_docstring(function)
        forecast = utility_load_forecast(forecast_json, self.energy_channel)
        # BUILD UP SPHINX OBJECT USING FORECAST AND OBSERVATION JSONS
        all_forecast_thresholds = forecast.identify_all_thresholds()
        sphinx, _ = self.load_sphinx_and_inputs(forecast, all_forecast_thresholds)
        function_evaluations = []
        for forecast_threshold_index in range(len(all_forecast_thresholds)):
            forecast_threshold = all_forecast_thresholds[forecast_threshold_index]
            forecast_threshold_key = objh.threshold_to_key(forecast_threshold)
            for i in sphinx.overlapping_indices:
                self.utility_print_inputs(sphinx, forecast_threshold_key, i)
                function_evaluation = match.match_sep_quantities(sphinx,
                                                                 self.observation_objects[self.energy_key][i],
                                                                 forecast_threshold,
                                                                 sphinx.is_win_overlap[i],
                                                                 sphinx.is_eruption_in_range[forecast_threshold_key][i],
                                                                 sphinx.trigger_input_start[forecast_threshold_key][i],
                                                                 sphinx.threshold_crossed_in_pred_win[forecast_threshold_key][i],
                                                                 sphinx.observed_ongoing_events[forecast_threshold_key][i])
                function_evaluations.append(function_evaluation)
        self.utility_print_outputs(sphinx, function_evaluations)
        return sphinx, function_evaluations

    @make_docstring_printable
    def test_match_sep_quantities_1(this, self):
        """
        The forecast/observation pair has the following attributes:
           -- Prediction window overlaps with observation window
                prediction window start:  2000-01-01T00:00:00Z
                prediction window end:    2000-01-01T01:00:00Z
                observation window start: 2000-01-01T00:00:00Z
                observation window end:   2000-01-01T01:00:00Z
           -- An eruption occurred less than 24 hours and more than/equal to 8 minutes prior to the threshold crossing
                CME start:                2000-01-01T00:00:00Z
           -- A SEP event occurs in the prediction window
                SEP start:                2000-01-01T00:15:00Z
                SEP end:                  2000-01-01T00:20:00Z
           -- The threshold crossing was observed within the prediction window
                threshold crossing:       2000-01-01T00:15:00Z
        The function should evaluate to [True]
        sphinx.sep_match_status[self.threshold_key] should be 'SEP Event'
        """
        forecast_json = './tests/files/forecasts/match/match_sep_quantities/match_sep_quantities_1.json'
        sphinx, function_evaluations = self.utility_test_match_sep_quantities(this, forecast_json)
        self.assertEqual(sphinx.sep_match_status[self.threshold_key], 'SEP Event')
        self.assertEqual(function_evaluations, [True])
        
    @make_docstring_printable
    def test_match_sep_quantities_2(this, self):
        """
        The forecast/observation pair has the following attributes:
           -- Prediction window overlaps with observation window
                prediction window start:  2000-01-01T00:00:00Z
                prediction window end:    2000-01-01T01:00:00Z
                observation window start: 2000-01-01T00:00:00Z
                observation window end:   2000-01-01T01:00:00Z
           -- An eruption more than 24 hours prior to the threshold crossing
                CME start:                1999-12-29T00:00:00Z
           -- A SEP event occurs in the prediction window
                SEP start:                2000-01-01T00:15:00Z
                SEP end:                  2000-01-01T00:20:00Z
           -- The threshold crossing was observed within the prediction window
                threshold crossing:       2000-01-01T00:15:00Z
        The function should evaluate to [None]
        sphinx.sep_match_status[self.threshold_key] should be 'Eruption Out of Range'
        """
        forecast_json = './tests/files/forecasts/match/match_sep_quantities/match_sep_quantities_2.json'
        sphinx, function_evaluations = self.utility_test_match_sep_quantities(this, forecast_json)
        self.assertEqual(sphinx.sep_match_status[self.threshold_key], 'Eruption Out of Range')
        self.assertEqual(function_evaluations, [None])

    @make_docstring_printable
    def test_match_sep_quantities_3(this, self):
        """
        The forecast/observation pair has the following attributes:
           -- Prediction window overlaps with observation window
                prediction window start:  2000-01-01T00:00:00Z
                prediction window end:    2000-01-01T01:00:00Z
                observation window start: 2000-01-01T00:00:00Z
                observation window end:   2000-01-01T01:00:00Z
           -- CME starts after threshold crossing
                CME start:                2000-01-01T00:16:00Z
                threshold crossing:       2000-01-01T00:15:00Z
           -- A SEP event occurs in the prediction window
                SEP start:                2000-01-01T00:15:00Z
                SEP end:                  2000-01-01T00:20:00Z
           -- The threshold crossing was observed within the prediction window
                threshold crossing:       2000-01-01T00:15:00Z
        The function should evaluate to [None]
        sphinx.sep_match_status[self.threshold_key] should be 'Eruption Out of Range'
        """
        forecast_json = './tests/files/forecasts/match/match_sep_quantities/match_sep_quantities_3.json'
        sphinx, function_evaluations = self.utility_test_match_sep_quantities(this, forecast_json)
        self.assertEqual(sphinx.sep_match_status[self.threshold_key], 'Trigger/Input after Observed Phenomenon')
        self.assertEqual(function_evaluations, [None])
        
    @make_docstring_printable
    def test_match_sep_quantities_4(this, self):
        """
        The forecast/observation pair has the following attributes:
           -- Prediction window overlaps with observation window
                prediction window start:  2000-01-01T00:21:00Z
                prediction window end:    2000-01-01T01:00:00Z
                observation window start: 2000-01-01T00:00:00Z
                observation window end:   2000-01-01T01:00:00Z
           -- CME starts after threshold crossing
                CME start:                2000-01-01T00:00:00Z
           -- A SEP event occurs in the prediction window
                SEP start:                2000-01-01T00:15:00Z
                SEP end:                  2000-01-01T00:20:00Z
           -- The threshold crossing was observed within the prediction window
                threshold crossing:       2000-01-01T00:15:00Z
        The function should evaluate to [False]
        sphinx.sep_match_status[self.threshold_key] should be 'No SEP Event'
        """
        forecast_json = './tests/files/forecasts/match/match_sep_quantities/match_sep_quantities_4.json'
        sphinx, function_evaluations = self.utility_test_match_sep_quantities(this, forecast_json)
        self.assertEqual(sphinx.sep_match_status[self.threshold_key], 'No SEP Event')
        self.assertEqual(function_evaluations, [False])

    @make_docstring_printable
    def test_match_sep_quantities_5(this, self):
        """
        The forecast/observation pair has the following attributes:
           -- Prediction window overlaps with observation window
                prediction window start:  2000-01-01T00:16:00Z
                prediction window end:    2000-01-01T01:00:00Z
                observation window start: 2000-01-01T00:00:00Z
                observation window end:   2000-01-01T01:00:00Z
           -- CME starts after threshold crossing
                CME start:                2000-01-01T00:00:00Z
           -- A SEP event occurs in the prediction window
                SEP start:                2000-01-01T00:15:00Z
                SEP end:                  2000-01-01T00:20:00Z
           -- The threshold crossing was observed within the prediction window
                threshold crossing:       2000-01-01T00:15:00Z
        The function should evaluate to [False]
        sphinx.sep_match_status[self.threshold_key] should be 'Ongoing SEP Event'
        """
        forecast_json = './tests/files/forecasts/match/match_sep_quantities/match_sep_quantities_5.json'
        sphinx, function_evaluations = self.utility_test_match_sep_quantities(this, forecast_json)
        self.assertEqual(sphinx.sep_match_status[self.threshold_key], 'Ongoing SEP Event')
        self.assertEqual(function_evaluations, [False])

    @make_docstring_printable
    def test_match_sep_quantities_6(this, self):
        """
        The forecast/observation pair has the following attributes:
           -- Prediction window overlaps with observation window
                prediction window start:  2000-01-01T00:00:00Z
                prediction window end:    2000-01-01T01:00:00Z
                observation window start: 2000-01-01T00:00:00Z
                observation window end:   2000-01-01T01:00:00Z
           -- There are two SEP events in the prediction window
                event 1 start:            2000-01-01T00:15:00Z
                event 1 end:              2000-01-01T00:20:00Z
                event 2 start:            2000-01-01T00:25:00Z
                event 2 end:              2000-01-01T00:28:00Z
           -- The threshold crossing was observed within the prediction window
                threshold crossing:       2000-01-01T00:15:00Z
           -- In another observation JSON, a different threshold crossing was observed within the same prediction window
                threshold crossing:       2000-01-01T00:25:00Z
        The function should evaluate to [False, None] (the forecast correctly identifies All Clear = False, evidenced by first observation, second observation is ignored)
        sphinx.all_clear_match_status should be unchanged from 'SEP Event'
        sphinx.observed_all_clear.all_clear_boolean should be False from first forecast-observation match
        """
        observation_json = './tests/files/observations/match/match_sep_quantities/match_sep_quantities_6.json'
        observation = utility_load_observation(observation_json, self.energy_channel) # SAME ENERGY CHANNEL
        self.observation_objects[self.energy_key].append(observation)
        self.observation_values = match.compile_all_obs(self.all_energy_channels, self.observation_objects)        
        forecast_json = './tests/files/forecasts/match/match_sep_quantities/match_sep_quantities_6.json'
        sphinx, function_evaluations = self.utility_test_match_sep_quantities(this, forecast_json)
        self.assertEqual(sphinx.sep_match_status[self.threshold_key], 'SEP Event')
        self.assertEqual(function_evaluations, [True, None])

class TestMatchSEPEndTime(LoadMatch):
    """
    Unit test class for match_sep_end_time function in match.py
    """   
    def setUp(self):
        energy_channel = {'min': 10, 'max': -1, 'units': 'MeV'}
        observation_json = './tests/files/observations/match/match_sep_end_time/match_sep_end_time.json'
        self.load_verbosity()
        self.load_energy(energy_channel)
        self.load_observation(observation_json)
    
    # sphinx.py --> match.match_all_forecasts --> match.match_sep_end_time
    def utility_print_inputs(self, sphinx, forecast_threshold_key, i):
        if self.verbosity == 2:
            print('')
            print('===== PRINT INPUTS =====')
            print('sphinx.end_time_match_status[thresh_key] =', sphinx.end_time_match_status[forecast_threshold_key])
            print('is_win_overlap =', sphinx.is_win_overlap[i], '(always True)')
            print('is_eruption_in_range =', sphinx.is_eruption_in_range[forecast_threshold_key][i])
            print('trigger_input_end =', sphinx.trigger_input_end[forecast_threshold_key][i])
            print('prediction_window_sep_overlap =', sphinx.prediction_window_sep_overlap[forecast_threshold_key][i])
            print('==========')
            print('')
            
    def utility_print_outputs(self, sphinx, function_evaluations):
        if self.verbosity == 2:
            print('')
            print('===== PRINT OUTPUTS =====')
            print('sphinx.sep_match_status[thresh_key] =', sphinx.sep_match_status[self.threshold_key])
            print('function_evaluations =', function_evaluations)
            print('==========')
            print('')
            print('----------------------------------------------------//\n\n\n\n\n\n')
        
    def utility_test_match_sep_end_time(self, function, forecast_json):
        """
        Obtains SPHINX object and function evaluations given the forecast JSON.
        """
        self.utility_print_docstring(function)
        forecast = utility_load_forecast(forecast_json, self.energy_channel)
        # BUILD UP SPHINX OBJECT USING FORECAST AND OBSERVATION JSONS
        all_forecast_thresholds = forecast.identify_all_thresholds()
        sphinx, _ = self.load_sphinx_and_inputs(forecast, all_forecast_thresholds)
        function_evaluations = []
        for forecast_threshold_index in range(len(all_forecast_thresholds)):
            forecast_threshold = all_forecast_thresholds[forecast_threshold_index]
            forecast_threshold_key = objh.threshold_to_key(forecast_threshold)
            for i in sphinx.overlapping_indices:
                self.utility_print_inputs(sphinx, forecast_threshold_key, i)
                function_evaluation = match.match_sep_end_time(sphinx,
                                                               self.observation_objects[self.energy_key][i],
                                                               forecast_threshold,
                                                               sphinx.is_win_overlap[i],
                                                               sphinx.is_eruption_in_range[forecast_threshold_key][i],
                                                               sphinx.trigger_input_end[forecast_threshold_key][i],
                                                               sphinx.prediction_window_sep_overlap[forecast_threshold_key][i])
                function_evaluations.append(function_evaluation)
        self.utility_print_outputs(sphinx, function_evaluations)
        return sphinx, function_evaluations

    @make_docstring_printable
    def test_match_sep_end_time_1(this, self):
        """
        The forecast/observation pair has the following attributes:
           -- Prediction window overlaps with observation window
                prediction window start:  2000-01-01T00:00:00Z
                prediction window end:    2000-01-01T01:00:00Z
                observation window start: 2000-01-01T00:00:00Z
                observation window end:   2000-01-01T01:00:00Z
           -- An eruption occurred less than 24 hours and more than/equal to 8 minutes prior to the threshold crossing
                CME start:                2000-01-01T00:00:00Z
           -- A SEP event occurs in the prediction window
                SEP start:                2000-01-01T00:15:00Z
                SEP end:                  2000-01-01T00:20:00Z
           -- The threshold crossing was observed within the prediction window
                threshold crossing:       2000-01-01T00:15:00Z
        The function should evaluate to [True]
        sphinx.sep_match_status[self.threshold_key] should be 'SEP Event'
        """
        forecast_json = './tests/files/forecasts/match/match_sep_end_time/match_sep_end_time_1.json'
        sphinx, function_evaluations = self.utility_test_match_sep_end_time(this, forecast_json)
        self.assertEqual(sphinx.end_time_match_status[self.threshold_key], 'SEP Event')
        self.assertEqual(function_evaluations, [True])

    @make_docstring_printable
    def test_match_sep_end_time_2(this, self):
        """
        The forecast/observation pair has the following attributes:
           -- Prediction window overlaps with observation window
                prediction window start:  2000-01-01T00:00:00Z
                prediction window end:    2000-01-01T01:00:00Z
                observation window start: 2000-01-01T00:00:00Z
                observation window end:   2000-01-01T01:00:00Z
           -- An eruption more than 24 hours prior to the threshold crossing
                CME start:                1999-12-29T00:00:00Z
           -- A SEP event occurs in the prediction window
                SEP start:                2000-01-01T00:15:00Z
                SEP end:                  2000-01-01T00:20:00Z
           -- The threshold crossing was observed within the prediction window
                threshold crossing:       2000-01-01T00:15:00Z
        The function should evaluate to [None]
        sphinx.sep_match_status[self.threshold_key] should be 'Eruption Out of Range'
        """
        forecast_json = './tests/files/forecasts/match/match_sep_end_time/match_sep_end_time_2.json'
        sphinx, function_evaluations = self.utility_test_match_sep_end_time(this, forecast_json)
        self.assertEqual(sphinx.end_time_match_status[self.threshold_key], 'Eruption Out of Range')
        self.assertEqual(function_evaluations, [None])

    @make_docstring_printable
    def test_match_sep_end_time_3(this, self):
        """
        The forecast/observation pair has the following attributes:
           -- Prediction window overlaps with observation window
                prediction window start:  2000-01-01T00:00:00Z
                prediction window end:    2000-01-01T01:00:00Z
                observation window start: 2000-01-01T00:00:00Z
                observation window end:   2000-01-01T01:00:00Z
           -- CME starts after threshold crossing, after SEP event ends
                CME start:                2000-01-01T00:21:00Z
                threshold crossing:       2000-01-01T00:15:00Z
           -- A SEP event occurs in the prediction window
                SEP start:                2000-01-01T00:15:00Z
                SEP end:                  2000-01-01T00:20:00Z
           -- The threshold crossing was observed within the prediction window
                threshold crossing:       2000-01-01T00:15:00Z
        The function should evaluate to [None]
        sphinx.sep_match_status[self.threshold_key] should be 'Trigger/Input after Observed Phenomenon'
        """
        forecast_json = './tests/files/forecasts/match/match_sep_end_time/match_sep_end_time_3.json'
        sphinx, function_evaluations = self.utility_test_match_sep_end_time(this, forecast_json)
        self.assertEqual(sphinx.end_time_match_status[self.threshold_key], 'Trigger/Input after Observed Phenomenon')
        self.assertEqual(function_evaluations, [None])
        
    @make_docstring_printable
    def test_match_sep_end_time_4(this, self):
        """
        The forecast/observation pair has the following attributes:
           -- Prediction window overlaps with observation window
                prediction window start:  2000-01-01T00:21:00Z
                prediction window end:    2000-01-01T01:00:00Z
                observation window start: 2000-01-01T00:00:00Z
                observation window end:   2000-01-01T01:00:00Z
           -- CME starts after threshold crossing
                CME start:                2000-01-01T00:00:00Z
           -- A SEP event occurs in the prediction window
                SEP start:                2000-01-01T00:15:00Z
                SEP end:                  2000-01-01T00:20:00Z
           -- The threshold crossing was observed within the prediction window
                threshold crossing:       2000-01-01T00:15:00Z
        The function should evaluate to [False]
        sphinx.sep_match_status[self.threshold_key] should be 'No SEP Event'
        """
        forecast_json = './tests/files/forecasts/match/match_sep_end_time/match_sep_end_time_4.json'
        sphinx, function_evaluations = self.utility_test_match_sep_end_time(this, forecast_json)
        self.assertEqual(sphinx.end_time_match_status[self.threshold_key], 'No SEP Event')
        self.assertEqual(function_evaluations, [False])
        
    @make_docstring_printable
    def test_match_sep_end_time_5(this, self):
        """
        The forecast/observation pair has the following attributes:
           -- Prediction window overlaps with observation window
                prediction window start:  2000-01-01T00:00:00Z
                prediction window end:    2000-01-01T01:00:00Z
                observation window start: 2000-01-01T00:00:00Z
                observation window end:   2000-01-01T01:00:00Z
           -- An eruption occurred less than 24 hours and more than/equal to 8 minutes prior to the threshold crossing
                CME start:                2000-01-01T00:00:00Z
           -- A SEP event occurs in the prediction window
                SEP start:                2000-01-01T00:15:00Z
                SEP end:                  2000-01-01T00:20:00Z
           -- The threshold crossing was observed within the prediction window
                threshold crossing:       2000-01-01T00:15:00Z
        The function should evaluate to [True]
        sphinx.sep_match_status[self.threshold_key] should be 'SEP Event'
        The observed fluence should be 1 cm^-2*sr^-1.
        #The observed fluence spectrum should be ...
        """
        forecast_json = './tests/files/forecasts/match/match_sep_end_time/match_sep_end_time_5.json'
        sphinx, function_evaluations = self.utility_test_match_sep_end_time(this, forecast_json)
        self.assertEqual(sphinx.end_time_match_status[self.threshold_key], 'SEP Event')
        self.assertEqual(function_evaluations, [True])
        self.assertEqual(sphinx.observed_fluence[self.threshold_key].fluence, 1)
        self.assertEqual(sphinx.observed_fluence[self.threshold_key].units, 'cm^-2*sr^-1')
        #self.assertEqual(sphinx.observed_fluence_spectrum[self.threshold_key].fluence_spectrum)

    @make_docstring_printable
    def test_match_sep_end_time_6(this, self):
        """
        The forecast/observation pair has the following attributes:
           -- Prediction window overlaps with observation window
                prediction window start:  2000-01-01T00:00:00Z
                prediction window end:    2000-01-01T01:00:00Z
                observation window start: 2000-01-01T00:00:00Z
                observation window end:   2000-01-01T01:00:00Z
           -- There are two SEP events in the prediction window
                event 1 start:            2000-01-01T00:15:00Z
                event 1 end:              2000-01-01T00:20:00Z
                event 2 start:            2000-01-01T00:25:00Z
                event 2 end:              2000-01-01T00:28:00Z
           -- The threshold crossing was observed within the prediction window
                threshold crossing:       2000-01-01T00:15:00Z
           -- In another observation JSON, a different threshold crossing was observed within the same prediction window
                threshold crossing:       2000-01-01T00:25:00Z
        The function should evaluate to [True, None]
        """
        observation_json = './tests/files/observations/match/match_sep_end_time/match_sep_end_time_6.json'
        observation = utility_load_observation(observation_json, self.energy_channel) # SAME ENERGY CHANNEL
        self.observation_objects[self.energy_key].append(observation)
        self.observation_values = match.compile_all_obs(self.all_energy_channels, self.observation_objects)        
        forecast_json = './tests/files/forecasts/match/match_sep_end_time/match_sep_end_time_6.json'
        sphinx, function_evaluations = self.utility_test_match_sep_end_time(this, forecast_json)
        self.assertEqual(sphinx.end_time_match_status[self.threshold_key], 'SEP Event')
        self.assertEqual(function_evaluations, [True, None])
    
    @make_docstring_printable
    def test_match_sep_end_time_7(this, self):
        """
        Tests the function's ability to recognize 'fluence_spectrum' entries in observation JSON.
        """
        forecast_json = './tests/files/forecasts/match/match_sep_end_time/match_sep_end_time_7.json'
        sphinx, _ = self.utility_test_match_sep_end_time(this, forecast_json)
        self.assertEqual(sphinx.observed_fluence_spectrum[self.threshold_key].label, 'fluence_spectrum')        
        
class TestCalculateDerivedQuantities(LoadMatch):
    """
    Unit test class for calculate_derived_quantities function in match.py
    """   
    def setUp(self):
        energy_channel = {'min': 10, 'max': -1, 'units': 'MeV'}
        observation_json = './tests/files/observations/match/calculate_derived_quantities/calculate_derived_quantities.json'
        self.load_verbosity()
        self.load_energy(energy_channel)
        self.load_observation(observation_json)
    
    # sphinx.py --> match.match_all_forecasts --> match.calculate_derived_quantities
    def utility_print_inputs(self, sphinx, forecast_threshold_key, i):
        if self.verbosity == 2:
            print('')
            print('===== PRINT INPUTS =====')
            print('sphinx.prediction.point_intensity.intensity =', sphinx.prediction.point_intensity.intensity)
            print('sphinx.prediction.peak_intensity.intensity =', sphinx.prediction.peak_intensity.intensity)
            print('sphinx.prediction.peak_intensity_max.intensity =', sphinx.prediction.peak_intensity_max.intensity)
            print('sphinx.prediction.prediction_window_start =', sphinx.prediction.prediction_window_start)
            print('sphinx.prediction.prediction_window_end =', sphinx.prediction.prediction_window_end)
            print('sphinx.end_time_match_status =', sphinx.end_time_match_status)
            print('sphinx.prediction.peak_intensity.units =', sphinx.prediction.peak_intensity.units)
            print('sphinx.prediction.peak_intensity_max.units =', sphinx.prediction.peak_intensity_max.units)
            print('sphinx.prediction.point_intensity.time =', sphinx.prediction.point_intensity.time)
            print('sphinx.prediction.point_intensity.units =', sphinx.prediction.point_intensity.units)
            print('==========')
            print('')
            
    def utility_print_outputs(self, sphinx, function_evaluations):
        if self.verbosity == 2:
            print('')
            print('===== PRINT OUTPUTS =====')
            print('function_evaluations =', function_evaluations)
            print('==========')
            print('')
            print('----------------------------------------------------//\n\n\n\n\n\n')
    
    def utility_test_calculate_derived_quantities(self, function, forecast_jsons):
        """
        Obtains matched_sphinx object and function evaluations given the forecast JSON.
        """
        self.utility_print_docstring(function)
        function_evaluations = []
        forecast_threshold_list = []
        forecasts = []
        for i in range(0, len(forecast_jsons)):
            forecast = utility_load_forecast(forecast_jsons[i], self.energy_channel)
            forecasts.append(forecast)
            this_forecast_thresholds = forecast.identify_all_thresholds()
            for j in range(0, len(this_forecast_thresholds)):
                forecast_threshold_list.append(this_forecast_thresholds[j])
        all_forecast_thresholds = utility_get_unique_dicts(forecast_threshold_list)
        matched_sphinx, observed_sep_events = self.load_matched_sphinx_and_inputs(forecasts, all_forecast_thresholds)
        initial_matched_sphinx = matched_sphinx.copy()
        function_evaluations = []
        last_fcast_shortname = ''
        energy_key = self.energy_key
        observation_objs = self.observation_objects[self.energy_key]
        for fcast in forecasts:
            if fcast.short_name != last_fcast_shortname or energy_key != last_energy_key:
                forecast_index = 0
            fcast.valid_forecast(verbose=True)
            if fcast.valid == False:
                continue
            sphinx = matched_sphinx[fcast.short_name][energy_key][forecast_index]
            all_fcast_thresholds = fcast.identify_all_thresholds()
            for f_thresh in all_fcast_thresholds:
                fcast_thresh = f_thresh
                if sphinx.mismatch:
                    if f_thresh == cfg.mm_pred_threshold:
                        fcast_thresh = cfg.mm_obs_threshold
                    else:
                        continue
                if fcast_thresh not in self.observation_values[energy_key]['thresholds']:
                    continue        
                thresh_key = objh.threshold_to_key(fcast_thresh)
                for i in sphinx.overlapping_indices:
                    peak_criteria = match.match_observed_onset_peak(sphinx,
                        observation_objs[i], sphinx.is_win_overlap[i],
                        sphinx.is_eruption_in_range[thresh_key][i],
                        sphinx.triggers_before_peak_intensity[i],
                        sphinx.inputs_before_peak_intensity[i], 
                        sphinx.prediction_window_sep_overlap[thresh_key][i])
                    max_criteria = match.match_observed_max_flux(sphinx,
                        observation_objs[i], sphinx.is_win_overlap[i],
                        sphinx.is_eruption_in_range[thresh_key][i], 
                        sphinx.triggers_before_peak_intensity_max[i],
                        sphinx.triggers_before_peak_intensity_max[i], 
                        sphinx.prediction_window_sep_overlap[thresh_key][i])
                    all_clear_status = match.match_all_clear(sphinx,
                        observation_objs[i], sphinx.is_win_overlap[i],
                        sphinx.is_eruption_in_range[thresh_key][i], 
                        sphinx.trigger_input_start[thresh_key][i],
                        sphinx.threshold_crossed_in_pred_win[thresh_key][i], 
                        sphinx.observed_ongoing_events[thresh_key][i])
                    sep_status = match.match_sep_quantities(sphinx, observation_objs[i], fcast_thresh, sphinx.is_win_overlap[i],
                        sphinx.is_eruption_in_range[thresh_key][i], 
                        sphinx.trigger_input_start[thresh_key][i],
                        sphinx.threshold_crossed_in_pred_win[thresh_key][i], 
                        sphinx.observed_ongoing_events[thresh_key][i])
                    if sep_status == True:
                        if sphinx.observed_threshold_crossing[thresh_key].crossing_time\
                        not in observed_sep_events[fcast.short_name][energy_key][thresh_key]:
                            observed_sep_events[fcast.short_name][energy_key][thresh_key].append(sphinx.observed_threshold_crossing[thresh_key].crossing_time)
                    end_status = match.match_sep_end_time(sphinx, observation_objs[i], 
                        fcast_thresh, 
                        sphinx.is_win_overlap[i],
                        sphinx.is_eruption_in_range[thresh_key][i], 
                        sphinx.trigger_input_end[thresh_key][i],
                        sphinx.prediction_window_sep_overlap[thresh_key][i])
                    function_evaluation = match.calculate_derived_quantities(sphinx)
                    function_evaluations.append(function_evaluation)
            matched_sphinx[fcast.short_name][energy_key][forecast_index] = sphinx
            last_fcast_shortname = fcast.short_name + ''
            last_energy_key = energy_key + ''
            forecast_index += 1
        self.utility_print_outputs(matched_sphinx, function_evaluations)
        return matched_sphinx, function_evaluations
    
    @make_docstring_printable
    def test_calculate_derived_quantities_1(this, self):
        """
        The forecast/observation pair has the following attributes:
            There are no "intensity" metrics in the forecast JSON. The function should evaluate to [None].
        """
        forecast_json = ['./tests/files/forecasts/match/calculate_derived_quantities/calculate_derived_quantities_1.json']
        matched_sphinx, function_evaluations = self.utility_test_calculate_derived_quantities(this, forecast_json)
        self.assertEqual(function_evaluations, [None])
    
    @make_docstring_printable
    def test_calculate_derived_quantities_2(this, self):
        """
        The forecast/observation pair has the following attributes:
            There is now a "point_intensity" dictionary in the forecast JSON.
            The members of the dictionary are 'intensity' and 'time'. 
            The function should evaluate to [True].
        """
        forecast_jsons = ['./tests/files/forecasts/match/calculate_derived_quantities/calculate_derived_quantities_2.json']
        matched_sphinx, function_evaluations = self.utility_test_calculate_derived_quantities(this, forecast_jsons)
        self.assertEqual(function_evaluations, [True]) # WE REACHED status = True
        self.assertEqual(matched_sphinx['unit_test'][self.energy_key][0].observed_max_flux_in_prediction_window.intensity, 10.0)
        self.assertEqual(matched_sphinx['unit_test'][self.energy_key][0].observed_max_flux_in_prediction_window.time, vjson.zulu_to_time('2000-01-01T00:15:00Z'))
        self.assertEqual(matched_sphinx['unit_test'][self.energy_key][0].max_flux_in_prediction_window_match_status, matched_sphinx['unit_test'][self.energy_key][0].end_time_match_status)
        self.assertEqual(matched_sphinx['unit_test'][self.energy_key][0].observed_point_intensity.intensity, 10.0)
        self.assertEqual(matched_sphinx['unit_test'][self.energy_key][0].observed_point_intensity.time, vjson.zulu_to_time('2000-01-01T00:15:00Z'))
        self.assertEqual(matched_sphinx['unit_test'][self.energy_key][0].observed_point_intensity.units, matched_sphinx['unit_test'][self.energy_key][0].prediction.point_intensity.units)
        
    @make_docstring_printable
    def test_calculate_derived_quantities_3(this, self):
        """
        The forecast/observation pair has the following attributes:
            No prediction window specified.
            The function should evaluate to [] (forecast skipped because it is invalid).
        """
        forecast_jsons = ['./tests/files/forecasts/match/calculate_derived_quantities/calculate_derived_quantities_3.json']
        matched_sphinx, function_evaluations = self.utility_test_calculate_derived_quantities(this, forecast_jsons)
        self.assertEqual(function_evaluations, [])
        
    @make_docstring_printable
    def test_calculate_derived_quantities_4(this, self):
        """
        The forecast/observation pair has the following attributes:
            There is a "peak_intensity" dictionary in the forecast JSON.
            The only member of the dictionary is 'intensity'.
            The function should evaluate to [True].
        """
        forecast_jsons = ['./tests/files/forecasts/match/calculate_derived_quantities/calculate_derived_quantities_4.json']
        matched_sphinx, function_evaluations = self.utility_test_calculate_derived_quantities(this, forecast_jsons)
        self.assertEqual(function_evaluations, [True])
        
    @make_docstring_printable
    def test_calculate_derived_quantities_5(this, self):
        """
        The forecast/observation pair has the following attributes:
            There is a "peak_intensity" dictionary in the forecast JSON.
            There is also a "peak_intensity_max" dictionary in the forecast JSON.
            The only member of the dictionary is 'intensity'.
            The function should evaluate to [True].
        """
        forecast_jsons = ['./tests/files/forecasts/match/calculate_derived_quantities/calculate_derived_quantities_5.json']
        matched_sphinx, function_evaluations = self.utility_test_calculate_derived_quantities(this, forecast_jsons)
        self.assertEqual(function_evaluations, [True])

 
class TestReviseEruptionMatches(LoadMatch):
    """
    Unit test class for calculate_derived_quantities function in match.py
    """   
    def setUp(self):
        energy_channel = {'min': 10, 'max': -1, 'units': 'MeV'}
        observation_json = './tests/files/observations/match/revise_eruption_matches/revise_eruption_matches.json'
        self.load_verbosity()
        self.load_energy(energy_channel)
        self.load_observation(observation_json)
        
    def utility_test_revise_eruption_matches(self, function, forecast_jsons):
        """
        Obtains SPHINX object and function evaluations given the forecast JSON.
        """
        self.utility_print_docstring(function)
        forecast_threshold_list = []
        forecasts = []
        for i in range(0, len(forecast_jsons)):
            forecast = utility_load_forecast(forecast_jsons[i], self.energy_channel)
            forecasts.append(forecast)
            this_forecast_thresholds = forecast.identify_all_thresholds()
            for j in range(0, len(this_forecast_thresholds)):
                forecast_threshold_list.append(this_forecast_thresholds[j])
        all_forecast_thresholds = utility_get_unique_dicts(forecast_threshold_list)
        matched_sphinx, observed_sep_events = self.load_matched_sphinx_and_inputs(forecasts, all_forecast_thresholds)
        initial_matched_sphinx = matched_sphinx.copy()
        function_evaluations = []
        last_fcast_shortname = ''
        observation_objs = self.observation_objects[self.energy_key]
        for fcast in forecasts:
            if fcast.short_name != last_fcast_shortname or self.energy_key != last_energy_key:
                forecast_index = 0
            fcast.valid_forecast(verbose=True)
            if fcast.valid == False:
                continue
            sphinx = matched_sphinx[fcast.short_name][self.energy_key][forecast_index]
            all_fcast_thresholds = fcast.identify_all_thresholds()
            for f_thresh in all_fcast_thresholds:
                fcast_thresh = f_thresh
                if sphinx.mismatch:
                    if f_thresh == cfg.mm_pred_threshold:
                        fcast_thresh = cfg.mm_obs_threshold
                    else:
                        continue
                if fcast_thresh not in self.observation_values[self.energy_key]['thresholds']:
                    continue        
                thresh_key = objh.threshold_to_key(fcast_thresh)
                for i in sphinx.overlapping_indices:
                    peak_criteria = match.match_observed_onset_peak(sphinx,
                        observation_objs[i], sphinx.is_win_overlap[i],
                        sphinx.is_eruption_in_range[thresh_key][i],
                        sphinx.triggers_before_peak_intensity[i],
                        sphinx.inputs_before_peak_intensity[i], 
                        sphinx.prediction_window_sep_overlap[thresh_key][i])
                    max_criteria = match.match_observed_max_flux(sphinx,
                        observation_objs[i], sphinx.is_win_overlap[i],
                        sphinx.is_eruption_in_range[thresh_key][i], 
                        sphinx.triggers_before_peak_intensity_max[i],
                        sphinx.triggers_before_peak_intensity_max[i], 
                        sphinx.prediction_window_sep_overlap[thresh_key][i])
                    all_clear_status = match.match_all_clear(sphinx,
                        observation_objs[i], sphinx.is_win_overlap[i],
                        sphinx.is_eruption_in_range[thresh_key][i], 
                        sphinx.trigger_input_start[thresh_key][i],
                        sphinx.threshold_crossed_in_pred_win[thresh_key][i], 
                        sphinx.observed_ongoing_events[thresh_key][i])
                    sep_status = match.match_sep_quantities(sphinx, observation_objs[i], fcast_thresh, sphinx.is_win_overlap[i],
                        sphinx.is_eruption_in_range[thresh_key][i], 
                        sphinx.trigger_input_start[thresh_key][i],
                        sphinx.threshold_crossed_in_pred_win[thresh_key][i], 
                        sphinx.observed_ongoing_events[thresh_key][i])
                    if sep_status == True:
                        if sphinx.observed_threshold_crossing[thresh_key].crossing_time\
                        not in observed_sep_events[fcast.short_name][self.energy_key][thresh_key]:
                            observed_sep_events[fcast.short_name][self.energy_key][thresh_key].append(sphinx.observed_threshold_crossing[thresh_key].crossing_time)
                    end_status = match.match_sep_end_time(sphinx, observation_objs[i], 
                        fcast_thresh, 
                        sphinx.is_win_overlap[i],
                        sphinx.is_eruption_in_range[thresh_key][i], 
                        sphinx.trigger_input_end[thresh_key][i],
                        sphinx.prediction_window_sep_overlap[thresh_key][i])
                    derived_status = match.calculate_derived_quantities(sphinx)
            matched_sphinx[fcast.short_name][self.energy_key][forecast_index] = sphinx
            last_fcast_shortname = fcast.short_name + ''
            last_energy_key = self.energy_key + ''
            forecast_index += 1
        match.revise_eruption_matches(matched_sphinx, self.all_energy_channels, self.observation_values, self.model_names, observed_sep_events)        
        return matched_sphinx, initial_matched_sphinx
        
    @make_docstring_printable
    def test_revise_eruption_matches_1(this, self):
        """
        The forecast does not use eruptions as triggers.
        The matched_sphinx object should be completely unchanged.
        """
        forecast_jsons = ['./tests/files/forecasts/match/revise_eruption_matches/revise_eruption_matches_1.json']
        matched_sphinx, initial_matched_sphinx = self.utility_test_revise_eruption_matches(this, forecast_jsons)
        self.assertEqual(matched_sphinx, initial_matched_sphinx)
        
    @make_docstring_printable
    def test_revise_eruption_matches_2(this, self):
        """
        The forecast uses CME start time as a trigger.
        The matched_sphinx object should be completely unchanged, as there is only one forecast.
        """
        forecast_jsons = ['./tests/files/forecasts/match/revise_eruption_matches/revise_eruption_matches_2.json']
        matched_sphinx, initial_matched_sphinx = self.utility_test_revise_eruption_matches(this, forecast_jsons)
        self.assertEqual(matched_sphinx, initial_matched_sphinx)
    
    @make_docstring_printable
    def test_revise_eruption_matches_3(this, self):
        """
        The forecast uses CME start time as a trigger.
        The matched_sphinx object should be completely unchanged.
        There is one forecast who's CME occurs prior to the observed 
            threshold crossing in two forecast files under consideration,
            but the...[] 
        """
        forecast_jsons = ['./tests/files/forecasts/match/revise_eruption_matches/revise_eruption_matches_3_1.json',
                          './tests/files/forecasts/match/revise_eruption_matches/revise_eruption_matches_3_2.json']
        matched_sphinx, initial_matched_sphinx = self.utility_test_revise_eruption_matches(this, forecast_jsons)
        self.assertEqual(matched_sphinx, initial_matched_sphinx)

class TestMatchAllForecasts(LoadMatch):
    """
    Unit test class for match_all_forecasts function in match.py
    """   
    def setUp(self):
        energy_channel = {'min': 10, 'max': -1, 'units': 'MeV'}
        observation_json = './tests/files/observations/match/match_all_forecasts/match_all_forecasts.json'
        self.load_verbosity()
        self.load_energy(energy_channel)
        self.load_observation(observation_json)
        
    def utility_test_match_all_forecasts(self, function, forecast_jsons, model_names=['unit_test']):
        forecast_objects = {}
        for energy_key in self.all_energy_channels:
            forecast_objects[energy_key] = []
            for fcast_json in forecast_jsons:
                forecast_object = utility_load_forecast(fcast_json, self.energy_channel)
                forecast_objects[energy_key].append(forecast_object)
        matched_sphinx, all_obs_thresholds, observed_sep_events = match.match_all_forecasts(self.all_energy_channels, model_names, self.observation_objects, forecast_objects)
        return matched_sphinx, all_obs_thresholds, observed_sep_events
    
    @make_docstring_printable
    def test_match_all_forecasts_1(this, self):
        """
        The forecast/observation pair has the following attributes:
           -- Prediction window overlaps with observation window
                prediction window start:  2000-01-01T00:00:00Z
                prediction window end:    2000-01-01T01:00:00Z
                observation window start: 2000-01-01T00:00:00Z
                observation window end:   2000-01-01T01:00:00Z
           -- No ongoing SEP event at start of prediction window
                event start:              2000-01-01T00:15:00Z
                event end:                2000-01-01T00:20:00Z
           -- The trigger (CME observation) for the forecast occurred before the threshold crossing 
                CME start:                2000-01-01T00:00:00Z
           -- A threshold crossing occurred within the prediction window
                threshold crossing:       2000-01-01T00:15:00Z
        sphinx.peak_intensity_match_status
        sphinx.all_clear_match_status should be 'SEP Event'
        
        """
        forecast_jsons = ['./tests/files/forecasts/match/match_all_forecasts/match_all_forecasts_1.json']
        matched_sphinx, all_obs_thresholds, observed_sep_events = self.utility_test_match_all_forecasts(this, forecast_jsons)
        
        sphinx = matched_sphinx['unit_test'][self.energy_key][0]
        self.assertEqual(sphinx.peak_intensity_match_status, 'SEP Event')
        self.assertEqual(sphinx.peak_intensity_max_match_status, 'SEP Event')
        self.assertEqual(sphinx.all_clear_match_status, 'SEP Event')
        self.assertEqual(sphinx.sep_match_status[self.threshold_key], 'SEP Event')
        self.assertEqual(sphinx.end_time_match_status[self.threshold_key], 'SEP Event')

    @make_docstring_printable
    def test_match_all_forecasts_2(this, self):
        """
        The forecast/observation pair has the following attributes:
           -- Prediction flux threshold is > 3 pfu
           -- Prediction energy threshold is > 10 MeV
           -- Observation flux threshold is > 10 pfu
           -- Observation energy threshold is > 10 MeV
           -- Prediction window overlaps with observation window
                prediction window start:  2000-01-01T00:00:00Z
                prediction window end:    2000-01-01T01:00:00Z
                observation window start: 2000-01-01T00:00:00Z
                observation window end:   2000-01-01T01:00:00Z
           -- No ongoing SEP event at start of prediction window
                event start:              2000-01-01T00:15:00Z
                event end:                2000-01-01T00:20:00Z
           -- The trigger (CME observation) for the forecast occurred before the threshold crossing 
                CME start:                2000-01-01T00:00:00Z
           -- A threshold crossing occurred within the prediction window
                threshold crossing:       2000-01-01T00:15:00Z
        sphinx.all_clear_match_status should be 'No Matching Threshold'
        """
        forecast_jsons = ['./tests/files/forecasts/match/match_all_forecasts/match_all_forecasts_2.json']
        matched_sphinx, all_obs_thresholds, observed_sep_events = self.utility_test_match_all_forecasts(this, forecast_jsons)
        sphinx = matched_sphinx['unit_test'][self.energy_key][0]
        self.assertEqual(sphinx.all_clear_match_status, 'No Matching Threshold')

    '''
    @tag('skip_setup')
    @make_docstring_printable
    def test_match_all_forecasts_3(this, self):
        """
        """
        energy_channel = {'min': 10, 'max': -1, 'units': 'MeV'}
        observation_json = './tests/files/observations/match/match_all_forecasts/match_all_forecasts_3.json'
        self.load_verbosity()
        self.load_energy(energy_channel)
        self.load_observation(observation_json)
        

        forecast_jsons = ['./tests/files/forecasts/match/match_all_forecasts/match_all_forecasts_3.json']
        matched_sphinx, all_obs_thresholds, observed_sep_events = self.utility_test_match_all_forecasts(this, forecast_jsons, model_names=['SAWS-ASPECS flare']) 
        sphinx = matched_sphinx['SAWS-ASPECS flare'][self.energy_key][0]
        self.assertEqual(sphinx.all_clear_match_status, 'Eruption Out of Range')

        self.setUp()
    '''

    @tag('skip_setup')
    @make_docstring_printable    
    def test_match_all_forecasts_4_1(this, self):
        """
        The purpose of this test is to determine the behavior of a forecast/observation pair where the forecast contains a field that is supposed to be associated with a flux threshold, but does not list a flux threshold.
        In this situation, the flux threshold should be defined using a flux threshold extracted from the observation, at the appropriate energy channel.
        This test only handles the case where there is a single flux threshold associated with the energy channel in question.
        """
        # LOAD THE OBSERVATION JSON
        energy_channel = {'min': 100, 'max': -1, 'units': 'MeV'}
        observation_json = './tests/files/observations/match/match_all_forecasts/match_all_forecasts_4.json'
        self.load_verbosity()
        self.load_energy(energy_channel)
        self.load_observation(observation_json)
        
        forecast_jsons = ['./tests/files/forecasts/match/match_all_forecasts/match_all_forecasts_4_fake.json']
        matched_sphinx, all_obs_thresholds, observed_sep_events = self.utility_test_match_all_forecasts(this, forecast_jsons, model_names=['SEPSTER (Parker Spiral)'])
        sphinx = matched_sphinx['SEPSTER (Parker Spiral)']
        energy_channel_key = utility_convert_dict_to_key_energy(energy_channel)
        self.assertEqual(utility_convert_dict_to_key_flux(sphinx[energy_channel_key][0].thresholds[0]), all_obs_thresholds[energy_channel_key][0])

    @tag('skip_setup')
    @make_docstring_printable    
    def test_match_all_forecasts_4_2(this, self):
        """
        The purpose of this test is to determine the behavior of a forecast/observation pair where the forecast does not contain a field with a flux threshold.
        In this situation, the flux threshold should be defined using a flux threshold extracted from the observation, at the appropriate energy channel.
        This test only handles the case where there is a single flux threshold associated with the energy channel in question.
        """
        # LOAD THE OBSERVATION JSON
        energy_channel = {'min': 100, 'max': -1, 'units': 'MeV'}
        observation_json = './tests/files/observations/match/match_all_forecasts/match_all_forecasts_4.json'
        self.load_verbosity()
        self.load_energy(energy_channel)
        self.load_observation(observation_json) 
        forecast_jsons = ['./tests/files/forecasts/match/match_all_forecasts/match_all_forecasts_4_cut.json']
        matched_sphinx, all_obs_thresholds, observed_sep_events = self.utility_test_match_all_forecasts(this, forecast_jsons, model_names=['SEPSTER (Parker Spiral)'])
        sphinx = matched_sphinx['SEPSTER (Parker Spiral)']
        energy_channel_key = utility_convert_dict_to_key_energy(energy_channel)
        self.assertEqual(utility_convert_dict_to_key_flux(sphinx[energy_channel_key][0].thresholds[0]), all_obs_thresholds[energy_channel_key][0])





    @tag('skip_setup')
    @make_docstring_printable
    def test_match_all_forecasts_5(this, self):
        """
        The purpose of this test is to determine the behavior of a forecast/observation pair where the forecast does not contain a field with a flux threshold.
        In this situation, the flux threshold should be defined using a flux threshold extracted from the observation, at the appropriate energy channel.
        This test handles the case where there are multiple flux thresholds associated with the energy channel in question.
        """
        




