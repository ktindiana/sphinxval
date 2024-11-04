# SUBROUTINES RELATED TO WORKFLOW UNIT TESTING
# Need to keep the mock module call here since it fixes
# the circular import issue I was having 
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
from sphinxval.utils import object_handler as objh

from . import utils_test


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
class Test_AllFields_Mismatch(unittest.TestCase):

    def load_verbosity(self):
        self.verbosity = utils_test.utility_get_verbosity()
    

    
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
        # self.all_energy_channels = [self.obs_energy_key] 
        self.model_names = ['Test_model_0']
        observation_json = ['./tests/files/observations/validation/mismatch/all_clear_false.json', './tests/files/observations/validation/mismatch/all_clear_false_02.json']

        self.verbosity = utils_test.utility_get_verbosity()
        forecast_json = ['./tests/files/forecasts/validation/mismatch/pred_all_clear_false_mismatch.json','./tests/files/forecasts/validation/mismatch/pred_all_clear_false_mismatch_02.json']
        self.pred_energy_channel = config_tests.mm_pred_energy_channel
        self.pred_energy_key = "min." +str(float(self.pred_energy_channel['min'])) + ".max." \
                + str(float(self.pred_energy_channel['max'])) + ".units." \
                + vunits.convert_units_to_string(self.pred_energy_channel['units'])
        forecast_objects = {config_tests.mm_pred_ek: [], config_tests.mm_energy_key: []}
        
        self.all_energy_channels, observation_objects, forecast_objects = utils_test.utility_load_objects(observation_json, forecast_json)
        self.sphinx, self.obs_thresholds, self.obs_sep_events = utils_test.utility_match_sphinx(self.all_energy_channels, self.model_names, observation_objects, forecast_objects)
        

        # Keeping this block but commenting out since it's useful for bugfixing
        # logger.debug('Looking at the set up after the patches')
        # logger.debug(config.mm_pred_threshold)
        # logger.debug(config_tests.mm_pred_threshold)
        # logger.debug('Testing mismatched tk and ek')
        # logger.debug(config_tests.mm_obs_energy_channel)
        # logger.debug(config_tests.mm_obs_threshold)
        # logger.debug(config_tests.mm_pred_tk)
        # logger.debug(config.do_mismatch)
        
        self.profname_dict = None
        
        self.validation_quantity = ['all_clear', 'awt', 'duration', 'end_time', 'fluence', 'max_flux_in_pred_window', 'peak_intensity_max', 'peak_intensity_max_time', 'peak_intensity' \
            'peak_intensity_time', 'probability', 'start_time', 'threshold_crossing', 'time_profile']
        
        self.validation_type = ["All", "First", "Last", "Max", "Mean"]

        self.dataframe = validate.fill_sphinx_df(self.sphinx, self.model_names, self.all_energy_channels, \
            self.obs_thresholds, self.profname_dict)
       
        for keywords in self.dataframe:
           
            logger.debug(len(self.sphinx['Test_model_0'][self.all_energy_channels[1]]))
            temp = utils_test.attributes_of_sphinx_obj(keywords, self.sphinx['Test_model_0'][self.all_energy_channels[1]][0],\
                 config_tests.mm_energy_key, self.obs_thresholds)#, self.pred_energy_key)
            logger.debug('this is what should be equal - dataframe, temp, keywords')
            logger.debug(self.dataframe[keywords][2])
            logger.debug(temp)
            logger.debug(keywords)
            if 'SEP Fluence Spectrum' in keywords and "Units" not in keywords:
                try:
                    
                    for energies in range(len(self.dataframe[keywords][2])):
                        self.assertEqual(self.dataframe[keywords][2][energies]['energy_min'], temp[energies]['energy_min'], 'Error is in keyword ' + keywords + ' energy_min')
                        self.assertEqual(self.dataframe[keywords][2][energies]['energy_max'], temp[energies]['energy_max'], 'Error is in keyword ' + keywords + ' energy_max')
                        self.assertEqual(self.dataframe[keywords][2][energies]['fluence'], temp[energies]['fluence'], 'Error is in keyword ' + keywords + ' fluence')
                except:
                    
                    self.assertTrue(pd.isna(self.dataframe[keywords][2]))
            elif keywords == 'All Threshold Crossing Times': 
                pass
            elif pd.isna(temp) and pd.isna(self.dataframe[keywords][2]):
                self.assertTrue(pd.isna(self.dataframe[keywords][2]))
            else:
                self.assertEqual(self.dataframe[keywords][2], temp, 'Error is in keyword ' + keywords)


    def step_2(self):
        """
        step 2 writes the dataframe to a file and then checks that those files exist
        """
        validate.write_df(self.dataframe, "SPHINX_dataframe")
        
        self.assertTrue(os.path.isfile('./tests/output/csv/SPHINX_dataframe.csv'), msg = 'SPHINX_dataframe.csv does not exist, check the file is output correctly')
        self.assertTrue(os.path.isfile('./tests/output/pkl/SPHINX_dataframe.pkl'), msg = 'SPHINX_dataframe.pkl does not exist, check the file is output correctly')
    
    def step_3(self):
        """
        step 3 does the actual validation workflow on the mismatch dataframe, and then tests its output files
        exist
        """
        
        validate.calculate_intuitive_metrics(self.dataframe, self.model_names, self.all_energy_channels, \
                self.obs_thresholds, 'All')

        self.validation_quantity = ['awt', 'duration', 'end_time', 'fluence', 'max_flux_in_pred_win', 'peak_intensity_max', 'peak_intensity_max_time', 'peak_intensity', \
            'peak_intensity_time', 'probability', 'start_time', 'threshold_crossing', 'time_profile']
        for model in self.model_names:
            for quantities in self.validation_quantity:
               
                metrics_filename = './tests/output/csv/' + quantities + '_metrics' 
                self.assertTrue(os.path.isfile(metrics_filename + '.csv'), msg = metrics_filename + '.csv does not exist, check the file is output correctly')
                metrics_filename = './tests/output/pkl/' + quantities + '_metrics' 
                self.assertTrue(os.path.isfile(metrics_filename + '.pkl'), msg = metrics_filename + '.pkl does not exist, check the file is output correctly')
                
                
                energy_channels = self.all_energy_channels[1]
                for thresholds in self.obs_thresholds[energy_channels]:
            
                    threshold_shortened = thresholds.rsplit('.')[0]+ '_' + thresholds.rsplit('.')[1] + '.' + thresholds.rsplit('.')[2]
                    logger.debug(quantities)
                    if quantities == 'awt':
                        pkl_filename = './tests/output/pkl/' + quantities + '_selections_' + model + '_' + energy_channels + '_' + threshold_shortened + "_Predicted SEP All Clear_mm.pkl"
                        csv_filename = './tests/output/csv/' + quantities + '_selections_' + model + '_' + energy_channels + '_' + threshold_shortened + "_Predicted SEP All Clear_mm.csv"
                        self.assertTrue(os.path.isfile(pkl_filename) , \
                            msg = pkl_filename + ' does not exist, check the file is output correctly')
                        self.assertTrue(os.path.isfile(csv_filename), \
                            msg = csv_filename + ' does not exist, check the file is output correctly')
                    elif quantities == 'threshold_crossing':
                        pkl_filename = './tests/output/pkl/' + quantities + '_time_selections_' + model + '_' + energy_channels + '_' + threshold_shortened + "_mm.pkl"
                        csv_filename = './tests/output/csv/' + quantities + '_time_selections_' + model + '_' + energy_channels + '_' + threshold_shortened + "_mm.csv"
                        self.assertTrue(os.path.isfile(pkl_filename) , \
                            msg = pkl_filename + ' does not exist, check the file is output correctly')
                        self.assertTrue(os.path.isfile(csv_filename), \
                            msg = csv_filename + ' does not exist, check the file is output correctly')
                    
                    else:
                        pkl_filename = './tests/output/pkl/' + quantities + '_selections_' + model + '_' + energy_channels + '_' + threshold_shortened + '_mm.pkl'
                        csv_filename = './tests/output/csv/' + quantities + '_selections_' + model + '_' + energy_channels + '_' + threshold_shortened + '_mm.csv'
                    
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
        
    @patch('sphinxval.utils.config.outpath', './tests/output')
    @patch('sphinxval.utils.config.do_mismatch', True)
    @patch('sphinxval.utils.config.mm_model', 'Test_model_0')
    @patch('sphinxval.utils.config.mm_pred_energy_channel', config_tests.mm_pred_energy_channel)
    @patch('sphinxval.utils.config.mm_pred_threshold', config_tests.mm_pred_threshold)
    @patch('sphinxval.utils.config.mm_obs_energy_channel', config_tests.mm_obs_energy_channel)
    @patch('sphinxval.utils.config.mm_obs_threshold', config_tests.mm_obs_threshold)
    @patch('sphinxval.utils.config.mm_obs_ek', config_tests.mm_obs_ek)
    @patch('sphinxval.utils.config.mm_obs_tk', config_tests.mm_obs_tk)
    @patch('sphinxval.utils.config.mm_pred_ek', config_tests.mm_pred_ek)
    @patch('sphinxval.utils.config.mm_pred_tk', config_tests.mm_pred_tk)
    @patch('sphinxval.utils.config.mm_energy_key', config_tests.mm_obs_ek + "_" + config_tests.mm_pred_ek)
    
    def test_all(self):
        validate.prepare_outdirs()
        utils_test.utility_delete_output()
        utils_test.utility_setup_logging()
        validate.prepare_outdirs()

        for name, step in self._steps():
            try:
                step()
            except Exception as e:
                self.fail("{} failed ({}: {})".format(step, type(e), e))
        utils_test.utility_delete_output()  # Comment out when you want to bug fix since you may want to have the sphinx dataframe to look at
