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
    forecast = vjson.forecast_object_from_json(forecast_dict, energy_channel)    
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

def make_docstring_printable(function):
    """
    Decorator method @make_docstring_printable.
    Allows function to call itself to access docstring information.
    Used for verbose unit test outputs.
    """
    def wrapper(*args, **kwargs):
        return function(function, *args, **kwargs)
    return wrapper

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

    def load_sphinx_and_inputs(self, forecast, all_forecast_thresholds):
        """
        Loads inputs and SPHINX object to test match.match_all_clear() function.
        """
        sphinx = objh.initialize_sphinx(forecast)
        forecast_objects = {self.energy_key : [forecast]}
        model_names = ['unit_test']
        matched_sphinx = {}
        matched_sphinx, _ = match.setup_match_all_forecasts(self.all_energy_channels, 
                                                            self.observation_objects,
                                                            self.observation_values,
                                                            forecast_objects, 
                                                            model_names)
        sphinx = matched_sphinx['unit_test'][self.energy_key][0]
        return sphinx
    
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
            print('sphinx.is_trigger_before_onset_peak =', sphinx.is_trigger_before_onset_peak[i])
            print('sphinx.is_input_before_onset_peak =', sphinx.is_input_before_onset_peak[i]) 
            print('sphinx.is_win_overlap =', sphinx.is_win_overlap[i], '(always True)')
            print('sphinx.is_pred_sep_overlap =', sphinx.is_pred_sep_overlap[forecast_threshold_key][i])
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
        sphinx = self.load_sphinx_and_inputs(forecast, all_forecast_thresholds)
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
                                                                    sphinx.is_trigger_before_onset_peak[i],
                                                                    sphinx.is_input_before_onset_peak[i],
                                                                    sphinx.is_pred_sep_overlap[forecast_threshold_key][i])
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
           -- The last eruption occurred between 48 hours and 15 minutes prior to threshold crossing
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
        self.assertEqual(sphinx.observed_match_peak_intensity_source, self.observation.source, '')
        self.assertEqual(sphinx.observed_peak_intensity, self.observation.peak_intensity, '')
        self.assertEqual(sphinx.peak_intensity_match_status, 'SEP Event', '')
        self.assertEqual(function_evaluations, [True], '')
        
    @make_docstring_printable
    def test_match_observed_onset_peak_2(this, self):
        """
        The forecast/observation pair has the following attributes:
           -- Prediction window overlaps with observation window
                prediction window start:  2000-01-01T00:36:00Z
                prediction window end:    2000-01-01T01:30:00Z
                observation window start: 2000-01-01T00:00:00Z
                observation window end:   2000-01-01T01:00:00Z
           -- The last eruption occurred between 48 hours and 15 minutes prior to threshold crossing
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
        self.assertEqual(function_evaluations, [False], '')
        self.assertEqual(sphinx.observed_match_peak_intensity_source, None, '')
        self.assertEqual(sphinx.peak_intensity_match_status, 'No SEP Event', '')
    
    @make_docstring_printable
    def test_match_observed_onset_peak_3(this, self):
        """
        The forecast/observation pair has the following attributes:
           -- Prediction window overlaps with observation window
                prediction window start:  2000-01-01T00:00:00Z
                prediction window end:    2000-01-01T01:00:00Z
                observation window start: 2000-01-01T00:00:00Z
                observation window end:   2000-01-01T01:00:00Z
           -- The last eruption was out of range, > 48 hours prior to threshold crossing
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
        self.assertEqual(function_evaluations, [False], '')
        self.assertEqual(sphinx.observed_match_peak_intensity_source, None, '')
        self.assertEqual(sphinx.peak_intensity_match_status, 'Eruption Out of Range', '')
    
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
        self.assertEqual(function_evaluations, [False], '')
        self.assertEqual(sphinx.observed_match_peak_intensity_source, None, '')
        self.assertEqual(sphinx.peak_intensity_match_status, 'Trigger/Input after Observed Phenomenon', '')
    
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
        self.assertEqual(function_evaluations, [False], '')
        self.assertEqual(sphinx.observed_match_peak_intensity_source, None, '')
        self.assertEqual(sphinx.peak_intensity_match_status, 'Trigger/Input after Observed Phenomenon', '')
    
    @make_docstring_printable
    def test_match_observed_onset_peak_6(this, self):
        """
        The forecast/observation pair has the following attributes:
           -- Prediction window overlaps with both observation windows
                prediction window start:    2000-01-01T00:00:00Z
                prediction window end:      2000-01-01T01:00:00Z
                observation windows start:  2000-01-01T00:00:00Z
                observation windows end:    2000-01-01T01:00:00Z
           -- The last eruption occurred between 48 hours and 15 minutes prior to threshold crossing
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
        self.assertEqual(function_evaluations, [True, None], '')
        self.assertEqual(sphinx.peak_intensity_match_status, 'SEP Event', '')

# sphinx.py --> match.match_all_forecasts --> match.match_observed_onset_peak
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
            print(sphinx.is_trigger_before_max_time)
            print('sphinx.peak_intensity_max_match_status =', sphinx.peak_intensity_max_match_status)
            print('is_eruption_in_range =', sphinx.is_eruption_in_range[forecast_threshold_key][i])
            print('is_trigger_before_max_time =', sphinx.is_trigger_before_max_time[i])
            print('is_input_before_max_time =', sphinx.is_input_before_max_time[i]) 
            print('is_win_overlap =', sphinx.is_win_overlap[i], '(always True)')
            print('is_pred_sep_overlap =', sphinx.is_pred_sep_overlap[forecast_threshold_key][i])
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
        sphinx = self.load_sphinx_and_inputs(forecast, all_forecast_thresholds)
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
                                                                    sphinx.is_trigger_before_max_time[i],
                                                                    sphinx.is_input_before_max_time[i],
                                                                    sphinx.is_pred_sep_overlap[forecast_threshold_key][i])
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
           -- The last eruption occurred between 48 hours and 15 minutes prior to threshold crossing
                CME start:                2000-01-01T00:00:00Z
                threshold crossing:       2000-01-01T00:16:00Z
           -- The prediction window overlaps with an SEP event
                SEP start:                2000-01-01T00:16:00Z
                SEP end:                  2000-01-01T00:35:00Z
        The function should evaluate to [True]
        sphinx.peak_intensity_match_status should be 'SEP Event'
        """
        forecast_json = './tests/files/forecasts/match/match_observed_max_flux/match_observed_max_flux_1.json'
        sphinx, function_evaluations = self.utility_test_match_observed_max_flux(this, forecast_json)
        self.assertEqual(sphinx.observed_match_peak_intensity_max_source, self.observation.source, '')
        self.assertEqual(sphinx.observed_peak_intensity_max, self.observation.peak_intensity_max, '')
        self.assertEqual(sphinx.peak_intensity_max_match_status, 'SEP Event', '')
        self.assertEqual(function_evaluations, [True], '')
        
    @make_docstring_printable
    def test_match_observed_max_flux_2(this, self):
        """
        The forecast/observation pair has the following attributes:
           -- Prediction window overlaps with observation window
                prediction window start:  2000-01-01T00:36:00Z
                prediction window end:    2000-01-01T01:30:00Z
                observation window start: 2000-01-01T00:00:00Z
                observation window end:   2000-01-01T01:00:00Z
           -- The last eruption occurred between 48 hours and 15 minutes prior to threshold crossing
                CME start:                2000-01-01T00:00:00Z
                threshold crossing:       2000-01-01T00:16:00Z
           -- The prediction window does not overlap with an SEP event
                SEP start:                2000-01-01T00:16:00Z
                SEP end:                  2000-01-01T00:35:00Z
        The function should evaluate to [False]
        sphinx.peak_intensity_match_status should be 'No SEP Event'
        """
        forecast_json = './tests/files/forecasts/match/match_observed_max_flux/match_observed_max_flux_2.json'
        sphinx, function_evaluations = self.utility_test_match_observed_max_flux(this, forecast_json)
        self.assertEqual(function_evaluations, [False], '')
        self.assertEqual(sphinx.observed_match_peak_intensity_max_source, None, '')
        self.assertEqual(sphinx.peak_intensity_max_match_status, 'No SEP Event', '')
    
    @make_docstring_printable
    def test_match_observed_max_flux_3(this, self):
        """
        The forecast/observation pair has the following attributes:
           -- Prediction window overlaps with observation window
                prediction window start:  2000-01-01T00:00:00Z
                prediction window end:    2000-01-01T01:00:00Z
                observation window start: 2000-01-01T00:00:00Z
                observation window end:   2000-01-01T01:00:00Z
           -- The last eruption was out of range, > 48 hours prior to threshold crossing
                CME start:                1999-12-29T00:00:00Z
                threshold crossing:       2000-01-01T00:16:00Z
           -- The prediction window overlaps with an SEP event
                SEP start:                2000-01-01T00:16:00Z
                SEP end:                  2000-01-01T00:35:00Z
        The function should evaluate to [False]
        sphinx.peak_intensity_match_status should be 'Eruption Out of Range'
        """
        forecast_json = './tests/files/forecasts/match/match_observed_max_flux/match_observed_max_flux_3.json'
        sphinx, function_evaluations = self.utility_test_match_observed_max_flux(this, forecast_json)
        self.assertEqual(function_evaluations, [False], '')
        self.assertEqual(sphinx.observed_match_peak_intensity_max_source, None, '')
        self.assertEqual(sphinx.peak_intensity_max_match_status, 'Eruption Out of Range', '')
    
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
        sphinx.peak_intensity_match_status should be 'Trigger/Input after Observed Phenomenon'
        """
        forecast_json = './tests/files/forecasts/match/match_observed_max_flux/match_observed_max_flux_4.json'
        sphinx, function_evaluations = self.utility_test_match_observed_max_flux(this, forecast_json)
        self.assertEqual(function_evaluations, [False], '')
        self.assertEqual(sphinx.observed_match_peak_intensity_max_source, None, '')
        self.assertEqual(sphinx.peak_intensity_max_match_status, 'Trigger/Input after Observed Phenomenon', '')
    
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
        sphinx.peak_intensity_match_status should be 'Trigger/Input after Observed Phenomenon'
        """
        forecast_json = './tests/files/forecasts/match/match_observed_max_flux/match_observed_max_flux_5.json'
        sphinx, function_evaluations = self.utility_test_match_observed_max_flux(this, forecast_json)
        self.assertEqual(function_evaluations, [False], '')
        self.assertEqual(sphinx.observed_match_peak_intensity_max_source, None, '')
        self.assertEqual(sphinx.peak_intensity_max_match_status, 'Trigger/Input after Observed Phenomenon', '')
    
    @make_docstring_printable
    def test_match_observed_max_flux_6(this, self):
        """
        The forecast/observation pair has the following attributes:
           -- Prediction window overlaps with both observation windows
                prediction window start:    2000-01-01T00:00:00Z
                prediction window end:      2000-01-01T01:00:00Z
                observation windows start:  2000-01-01T00:00:00Z
                observation windows end:    2000-01-01T01:00:00Z
           -- The last eruption occurred between 48 hours and 15 minutes prior to threshold crossing
                CME start:                  2000-01-01T00:00:00Z
                peak time 1:                2000-01-01T00:30:00Z
                peak time 2:                2000-01-01T00:31:00Z
           -- The prediction window overlaps with an SEP event
                SEP start:                2000-01-01T00:16:00Z
                SEP end:                  2000-01-01T00:35:00Z
        The function should evaluate to [True, None] (forecast matches to first )
        sphinx.peak_intensity_match_status should be 'SEP Event'
        """
        observation_json = './tests/files/observations/match/match_observed_max_flux/match_observed_max_flux_6.json'
        observation = utility_load_observation(observation_json, self.energy_channel) # SAME ENERGY CHANNEL
        self.observation_objects[self.energy_key].append(observation)
        self.observation_values = match.compile_all_obs(self.all_energy_channels, self.observation_objects)
        forecast_json = './tests/files/forecasts/match/match_observed_max_flux/match_observed_max_flux_6.json'
        sphinx, function_evaluations = self.utility_test_match_observed_max_flux(this, forecast_json)
        self.assertEqual(function_evaluations, [True, None], '')
        self.assertEqual(sphinx.peak_intensity_max_match_status, 'SEP Event', '')

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
            print('is_sep_ongoing =', sphinx.is_sep_ongoing[forecast_threshold_key][i])
            print('trigger_input_start =', sphinx.trigger_input_start[forecast_threshold_key][i])
            print('contains_thresh_cross =', sphinx.contains_thresh_cross[forecast_threshold_key][i])
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
        sphinx = self.load_sphinx_and_inputs(forecast, all_forecast_thresholds)
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
                                                            sphinx.contains_thresh_cross[forecast_threshold_key][i],
                                                            sphinx.is_sep_ongoing[forecast_threshold_key][i])
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
                CME start:                    2000-01-01T00:00:00Z
           -- A threshold crossing occurred within the prediction window
                threshold crossing:       2000-01-01T00:15:00Z
        The function should evaluate to [False] (All Clear = False is observed within the prediction window)
        sphinx.all_clear_match_status should be 'SEP Event'
        """
        forecast_json = './tests/files/forecasts/match/match_all_clear/match_all_clear_1.json'
        sphinx, function_evaluations = self.utility_test_match_all_clear(this, forecast_json)
        self.assertEqual(sphinx.all_clear_match_status, 'SEP Event', '')
        self.assertEqual(function_evaluations, [False], '')
        
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
        self.assertEqual(sphinx.all_clear_match_status, 'No SEP Event', '')
        self.assertEqual(function_evaluations, [True], '')
    
    @make_docstring_printable
    def test_match_all_clear_3(this, self):
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
                start:                    2000-01-01T00:00:00Z
                peak:                     2000-01-01T00:10:00Z
           -- The threshold crossing was observed within the prediction window
                threshold crossing:       2000-01-01T00:15:00Z
        The function should evaluate to [False] (All Clear = False is observed within the prediction window)
        sphinx.all_clear_match_status should be 'SEP Event'
        """
        forecast_json = './tests/files/forecasts/match/match_all_clear/match_all_clear_3.json'
        sphinx, function_evaluations = self.utility_test_match_all_clear(this, forecast_json)
        self.assertEqual(sphinx.all_clear_match_status, 'SEP Event', '')
        self.assertEqual(function_evaluations, [False], '')

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
           -- The trigger (CME observation) for the forecast occurred after the threshold crossing
                start:                    2000-01-01T00:30:00Z
                peak:                     2000-01-01T01:00:00Z
           -- The threshold crossing was observed within the prediction window
                threshold crossing:       2000-01-01T00:15:00Z
        The forecast itself should be invalid because the trigger comes after the threshold crossing. 
        No matchable forecast/observation pair exists.
        """
        forecast_json = './tests/files/forecasts/match/match_all_clear/match_all_clear_4.json'
        self.utility_print_docstring(this)
        forecast = utility_load_forecast(forecast_json, self.energy_channel)
        forecast.valid_forecast()
        self.assertEqual(forecast.valid, False, '')
        
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
           -- The trigger (CME observation) for the forecast occurred long before (20 years) the threshold crossing
                start:                    1980-01-01T00:30:00Z
                peak:                     1980-01-01T01:00:00Z
           -- The threshold crossing was observed within the prediction window
                threshold crossing:       2000-01-01T00:15:00Z
        The function should evaluate to [True] (forecast is invalid due to triggering after threshold crossing)
        sphinx.all_clear_match_status should be 'Eruption Out of Range' -- trigger occurs 20 years prior to threshold crossing
        """
        forecast_json = './tests/files/forecasts/match/match_all_clear/match_all_clear_5.json'
        sphinx, function_evaluations = self.utility_test_match_all_clear(this, forecast_json)
        self.assertEqual(sphinx.all_clear_match_status, 'Eruption Out of Range', '')
        self.assertEqual(function_evaluations, [True], '')

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
                CME peak:                 1999-12-31T23:10:00Z
                Magnetogram 1:            2000-01-01T00:00:00Z
                Magnetogram 2:            2000-01-01T00:00:01Z
           -- The threshold crossing was observed within the prediction window
                threshold crossing:       2000-01-01T00:15:00Z
        The function should evaluate to [None] ()
        sphinx.all_clear_match_status should be 'Ongoing SEP Event' 
        """
        forecast_json = './tests/files/forecasts/match/match_all_clear/match_all_clear_6.json'
        sphinx, function_evaluations = self.utility_test_match_all_clear(this, forecast_json)
        self.assertEqual(sphinx.all_clear_match_status, 'Ongoing SEP Event', '')
        self.assertEqual(function_evaluations, [None], '')

    @make_docstring_printable
    def test_match_all_clear_7(this, self):
        """
        The forecast/observation pair has the following attributes:
           -- Prediction window overlaps with observation window
                prediction window start:  2000-01-01T00:17:00Z
                prediction window end:    2000-01-01T01:00:00Z
                observation window start: 2000-01-01T00:00:00Z
                observation window end:   2000-01-01T01:00:00Z
           -- Ongoing SEP event at the start of the prediction window
                event start:            2000-01-01T00:15:00Z
                event end:              2000-01-01T00:20:00Z
           -- No trigger/input information
           -- The threshold crossing was not observed within the prediction window
                threshold crossing:       2000-01-01T00:15:00Z
        The function should evaluate to [None] (forecast is invalid due to no trigger)
        sphinx.all_clear_match_status should be '' -- no trigger, rejection of JSON
        """
        forecast_json = './tests/files/forecasts/match/match_all_clear/match_all_clear_7.json'
        sphinx, function_evaluations = self.utility_test_match_all_clear(this, forecast_json)
        self.assertEqual(sphinx.all_clear_match_status, '', '')
        self.assertEqual(function_evaluations, [None], '')
    
    @make_docstring_printable
    def test_match_all_clear_8(this, self):
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
        observation_json = './tests/files/observations/match/match_all_clear/match_all_clear_8.json'
        observation = utility_load_observation(observation_json, self.energy_channel) # SAME ENERGY CHANNEL
        self.observation_objects[self.energy_key].append(observation)
        self.observation_values = match.compile_all_obs(self.all_energy_channels, self.observation_objects)        
        forecast_json = './tests/files/forecasts/match/match_all_clear/match_all_clear_8.json'
        sphinx, function_evaluations = self.utility_test_match_all_clear(this, forecast_json)
        self.assertEqual(sphinx.all_clear_match_status, 'SEP Event', '')
        self.assertEqual(sphinx.observed_all_clear.all_clear_boolean, False, '')
        self.assertEqual(function_evaluations, [False, None], '')

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
            print('is_sep_ongoing =', sphinx.is_sep_ongoing[forecast_threshold_key][i])
            print('trigger_input_start =', sphinx.trigger_input_start[forecast_threshold_key][i])
            print('contains_thresh_cross =', sphinx.contains_thresh_cross[forecast_threshold_key][i])
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
        sphinx = self.load_sphinx_and_inputs(forecast, all_forecast_thresholds)
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
                                                                 sphinx.contains_thresh_cross[forecast_threshold_key][i],
                                                                 sphinx.is_sep_ongoing[forecast_threshold_key][i])
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
           -- An eruption occurred less than 48 hours and more than/equal to 15 minutes prior to the threshold crossing
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
        self.assertEqual(sphinx.sep_match_status[self.threshold_key], 'SEP Event', '')
        self.assertEqual(function_evaluations, [True], '')
        
    @make_docstring_printable
    def test_match_sep_quantities_2(this, self):
        """
        The forecast/observation pair has the following attributes:
           -- Prediction window overlaps with observation window
                prediction window start:  2000-01-01T00:00:00Z
                prediction window end:    2000-01-01T01:00:00Z
                observation window start: 2000-01-01T00:00:00Z
                observation window end:   2000-01-01T01:00:00Z
           -- An eruption more than 48 hours prior to the threshold crossing
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
        self.assertEqual(sphinx.sep_match_status[self.threshold_key], 'Eruption Out of Range', '')
        self.assertEqual(function_evaluations, [None], '')

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
        self.assertEqual(sphinx.sep_match_status[self.threshold_key], 'Trigger/Input after Observed Phenomenon', '')
        self.assertEqual(function_evaluations, [None], '')
        
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
        self.assertEqual(sphinx.sep_match_status[self.threshold_key], 'No SEP Event', '')
        self.assertEqual(function_evaluations, [False], '')

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
        self.assertEqual(sphinx.sep_match_status[self.threshold_key], 'Ongoing SEP Event', '')
        self.assertEqual(function_evaluations, [False], '')

