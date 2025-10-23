# SUBROUTINES RELATED TO WORKFLOW UNIT TESTING
# from tests import config_tests
import types
from mock import Mock
import sys
from . import config_tests
from sphinxval.utils import config

from sphinxval.utils import units_handler as vunits
from sphinxval.utils import validation as validate
from sphinxval.utils import metrics
from sphinxval.utils import time_profile as profile
from sphinxval.utils import resume
from sphinxval.utils import plotting_tools as plt_tools

from sphinxval.utils import match
from sphinxval.utils import validation_json_handler as vjson
from sphinxval.utils import classes as cl

from . import utils


from astropy import units
import unittest
import pprint
import pandas as pd
import numpy as np
import os
import csv
import logging
import logging.config
import pathlib
import json
import pickle
from datetime import datetime

from unittest.mock import patch
from mock import mock_open
import shutil # using this to delete the contents of the output folder each run - since the unittest is based on the existence/creation of certain files each loop

logger = logging.getLogger(__name__)

"""
Resume Validation Workflow Unit Test
"""


def mock_df_populate(dict):
    """ Set up a pandas df to hold each possible quantity,
        each observed energy channel, and predicted and
        observed values.
        
    """
    
    dict["Model"].append("Test_model_resume_old")
    dict["Energy Channel Key"].append("min.10.0.max.-1.0.units.MeV")
    dict["Threshold Key"].append("threshold.1.0.units.1 / (cm2 s sr)")
    dict["Mismatch Allowed"].append(False)
    dict['Evaluation Status'].append('Forecast is evaluated')
    dict["Prediction Energy Channel Key"].append("min.10.0.max.-1.0.units.MeV")
    dict["Prediction Threshold Key"].append("threshold.1.0.units.1 / (cm2 s sr)")
    dict["Forecast Source"].append("/home/m_sphinx/data/forecasts/enlil2.9e/2022/SEPMOD.20220402_000000.20220402_174512.20220402_172002/json/SEPMOD.2022-04-02T000000Z.2022-04-02T174804Z.json")
    dict["Forecast Path"].append("/home/m_sphinx/data/forecasts/enlil2.9e/2022/SEPMOD.20220402_000000.20220402_174512.20220402_172002/json/")
    dict["Forecast Issue Time"].append(datetime(year = 2022, month = 4, day = 2, hour =  17, minute = 48, second =4))
    dict["Prediction Window Start"].append(datetime(year = 2022, month = 4, day = 2, hour= 0, minute = 0))
    dict["Prediction Window End"].append(datetime(year =2022, month=4, day=9, hour= 0, minute=1))
            
            #OBSERVATIONS
    dict["Number of CMEs"].append("1")
    dict["CME Start Time"].append(datetime(year = 2022, month = 4, day = 2, hour = 13, minute =38)) #Timestamp of 1st coronagraph image CME is visible in
    dict["CME Liftoff Time"].append(""), #Timestamp of coronagraph image with 1st indication of CME liftoff (used by CACTUS)
    dict["CME Latitude"].append(-15.0)
    dict["CME Longitude"].append(54.0)
    dict["CME Speed"].append(1370.0)
    dict["CME Half Width"].append(45.0)
    dict["CME PA"].append("")
    dict["CME Catalog"].append("2022-04-02T13:38:00-CME-001")
    dict["Number of Flares"].append("0")
    dict["Flare Latitude"].append("")
    dict["Flare Longitude"].append("")
    dict["Flare Start Time"].append("")
    dict["Flare Peak Time"].append("")
    dict["Flare End Time"].append("")
    dict["Flare Last Data Time"].append("")
    dict["Flare Intensity"].append("")
    dict["Flare Integrated Intensity"].append("")
    dict["Flare NOAA AR"].append("")
    dict["Observatory"].append("GOES-16,GOES-16")
    dict["Observed Time Profile"].append("./tests/files/observations/validation/resume/GOES-16_integral.2022-04-02T000000Z.10MeV.txt,./tests/files/observations/validation/resume/GOES-16_integral.2022-04-05T034500Z.10MeV.txt") #string of comma
    #separated filenames ".\tests\files\observations\validation\resume\GOES-16_integral.2022-04-05T034500Z.10MeV.txt"
    dict["Observed SEP All Clear"].append("False")
    dict["Observed SEP Probability"].append(1.0)
    dict["Observed SEP Threshold Crossing Time"].append(datetime(year = 2022, month = 4, day = 2,  hour = 14, minute = 40))
    dict["Observed SEP Start Time"].append(datetime(year = 2022, month = 4, day = 2,  hour = 14, minute = 40))
    dict["Observed SEP End Time"].append(datetime(year = 2022, month = 4, day = 3,  hour = 0, minute = 10))
    dict["Observed SEP Duration"].append(9.5)
    dict["Observed SEP Fluence"].append(7608871.653746902)
    dict["Observed SEP Fluence Units"].append("1 / cm2")
    dict["Observed SEP Fluence Spectrum"].append("[{'energy_min': 1, 'energy_max': -1, 'fluence': 82103217.4768976}, {'energy_min': 5, 'energy_max': -1, 'fluence': 17401577.903146088}, {'energy_min': 10, 'energy_max': -1, 'fluence': 7608871.653746902}, {'energy_min': 30, 'energy_max': -1, 'fluence': 1637019.4533599885}, {'energy_min': 50, 'energy_max': -1, 'fluence': 644477.238211527}, {'energy_min': 100, 'energy_max': -1, 'fluence': 216056.99133072246}]")
    dict["Observed SEP Fluence Spectrum Units"].append("1 / cm2")
    dict["Observed SEP Peak Intensity (Onset Peak)"].append(32.18270492553711)
    dict["Observed SEP Peak Intensity (Onset Peak) Units"].append("1 / cm2")
    dict["Observed SEP Peak Intensity (Onset Peak) Time"].append(datetime(year = 2022, month = 4, day = 2, hour = 16, minute = 0))
    dict["Observed SEP Peak Intensity Max (Max Flux)"].append(32.18270492553711)
    dict["Observed SEP Peak Intensity Max (Max Flux) Units"].append("1 / cm2")
    dict["Observed SEP Peak Intensity Max (Max Flux) Time"].append(datetime(year = 2022, month = 4, day = 2, hour = 16, minute = 0))

    dict["Observed Point Intensity"].append(None)
    dict["Observed Point Intensity Units"].append(None)
    dict["Observed Point Intensity Time"].append(None)
    dict["Observed Max Flux in Prediction Window"].append(32.18270492553711)
    dict["Observed Max Flux in Prediction Window Units"].append("1 / cm2")
    dict["Observed Max Flux in Prediction Window Time"].append(datetime(year = 2022, month = 4, day = 2, hour = 16, minute = 0))
            
            #PREDICTIONS
    dict["Predicted SEP All Clear"].append("False")
    dict["All Clear Match Status"].append("SEP Event")
    dict["Predicted SEP Probability"].append(None)
    dict["Probability Match Status"].append("SEP Event")
    dict["Predicted SEP All Clear Probability Threshold"].append(None)
    dict["Predicted SEP Threshold Crossing Time"].append(np.datetime64('NaT'))
    dict["Threshold Crossing Time Match Status"].append("SEP Event")
    dict["Predicted SEP Start Time"].append(np.datetime64("NaT") )
    dict["Start Time Match Status"].append("SEP Event")
    dict["Predicted SEP End Time"].append(np.datetime64('NaT'))
    dict["End Time Match Status"].append("SEP Event")
    dict["Predicted SEP Duration"].append(48)
    dict["Duration Match Status"].append("SEP Event")
    dict["Predicted SEP Fluence"].append(123)
    dict["Predicted SEP Fluence Units"].append("1 / cm2")
    dict["Fluence Match Status"].append("SEP Event")
    dict["Predicted SEP Fluence Spectrum"].append(None)
    dict["Predicted SEP Fluence Spectrum Units"].append("")
    dict["Fluence Spectrum Match Status"].append("SEP Event")
    dict["Predicted SEP Peak Intensity (Onset Peak)"].append(58.82)
    dict["Predicted SEP Peak Intensity (Onset Peak) Units"].append("1 / cm2")
    dict["Predicted SEP Peak Intensity (Onset Peak) Time"].append(datetime(year = 2022, month = 4, day = 2, hour = 20, minute =30))
    dict["Peak Intensity Match Status"].append("SEP Event")
    dict["Predicted SEP Peak Intensity Max (Max Flux)"].append(58.82)
    dict["Predicted SEP Peak Intensity Max (Max Flux) Units"].append("1 / cm2")
    dict["Predicted SEP Peak Intensity Max (Max Flux) Time"].append(datetime(year = 2022, month = 4, day = 2, hour = 20, minute =30))
    dict["Peak Intensity Max Match Status"].append("SEP Event")
            
    dict["Predicted Point Intensity"].append(None)
    dict["Predicted Point Intensity Units"].append(None)
    dict["Predicted Point Intensity Time"].append("SEP Event")

    dict["Predicted Time Profile"].append("./tests/files/forecasts/validation/resume/SEPMOD.2022-04-02T000000Z.2022-04-02T174804Z.10mev.txt")
    dict["Time Profile Match Status"].append("SEP Event")
            
    dict["Last Data Time to Issue Time"].append(250.06666666666666)
          
            #MATCHING INFORMATION
    dict["Last Eruption Time"].append("2022-04-02 13:38:00") #Last time for flare/CME
    dict["Last Trigger Time"].append("2022-04-02 13:38:00")
    dict["Last Input Time"].append("None")
    dict["Threshold Crossed in Prediction Window"].append([True, False])
    dict["Eruption before Threshold Crossed"].append([True, None])
    dict["Time Difference between Eruption and Threshold Crossing"].append([-1.0333333333333334, np.nan])
    dict["Triggers before Threshold Crossing"].append([True, None])
    dict["Inputs before Threshold Crossing"].append([None, None])
    dict["Triggers before Peak Intensity"].append([True, None])
    dict["Time Difference between Triggers and Peak Intensity"].append([-2.3666666666666667, np.nan])
    dict["Inputs before Peak Intensity"].append([None, None])
    dict["Time Difference between Inputs and Peak Intensity"].append([None, None])
    dict["Triggers before Peak Intensity Max"].append([True, True])
    dict["Time Difference between Triggers and Peak Intensity Max"].append([-2.3666666666666667, -73.2])
    dict["Inputs before Peak Intensity Max"].append([None, None])
    dict["Time Difference between Inputs and Peak Intensity Max"].append([None, None])
    dict["Triggers before SEP End"].append([True, None])
    dict["Time Difference between Triggers and SEP End"].append([-10.533333333333333, np.nan])
    dict["Inputs before SEP End"].append([None, None])
    dict["Time Difference between Inputs and SEP End"].append([None, None])
    dict["Prediction Window Overlap with Observed SEP Event"].append([True, False])
    dict["Ongoing SEP Event"].append([False, None])
    dict["All Thresholds in Prediction"].append([config_tests.mm_pred_threshold])
    dict["All Threshold Crossing Times"].append([np.datetime64('NaT')])
    dict["Eruption in Range"].append([True])
    dict["Original Model Short Name"].append("Test_model_resume_old")
    dict['Trigger Advance Time'].append(['NaT'])

    return dict

def mock_profile_output():
    """
    Creating a pair of dictionaries that are fed into the Resume structure
    and added to the observed_profiles.json and model_profiles.json
    outputs. 
    """
    observed_profile_dict = {
        "./tests/files/observations/validation/resume/GOES-16_integral.2022-04-02T000000Z.10MeV.txt": {
            "dates": [
                    "I'm",
                    "Cheating",
                    "Out",
                    "of",
                    "Laziness",
                    "No",
                    "One",
                    "Looks",
                    "Here",
                    "Anyway"
                ],
            "fluxes":
                [
                    0.0,
                    0.1,
                    0.2,
                    8,
                    6,
                    7,
                    5,
                    3,
                    0,
                    9
                ]

        },
        "./tests/files/observations/validation/resume/GOES-16_integral.2022-04-05T034500Z.10MeV.txt": {
            "dates": [
                "This is a date",
                "This is a different date",
                "Oh look another date"
            ],
            "fluxes":[
                0,
                1,
                2,
            ]
        }
    }


    model_profile_dict = {
        "./tests/files/forecasts/validation/resume/SEPMOD.2022-04-02T000000Z.2022-04-02T174804Z.10mev.txt":{
            "dates": [
                'the',
                'actual',
                'content',
                'here',
                'doesnt',
                'matter',
                'since',
                'im',
                'just',
                'checking',
                'existence',
                'and',
                'that',
                'it',
                'works',
                'not',
                'actually',
                'running',
                'on',
                'this'
            ],
            'fluxes': [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10
            ]
        }
    }

    return observed_profile_dict, model_profile_dict


class Test_Resume(unittest.TestCase):

    def load_verbosity(self):
        self.verbosity = utils.utility_get_verbosity()
    

    
    def step_1(self):
        """
        Step 1 prepares the matched_sphinx object from the input jsons as we normally would
        do in the sphinx workflow
        """
        validate.prepare_outdirs()
        

        validation_type = ["All", "First", "Last", "Max", "Mean"]
        self.obs_energy_channel = {'min': 10, 'max': -1, 'units': 'MeV'}
        self.obs_energy_key = "min." +str(float(self.obs_energy_channel['min'])) + ".max." \
                + str(float(self.obs_energy_channel['max'])) + ".units." \
                + vunits.convert_units_to_string(self.obs_energy_channel['units'])
        self.all_energy_channels = [self.obs_energy_key] 
        self.model_names = ['Test_model_0']
        observation_json = ['./tests/files/observations/validation/all/all_clear_false.json', './tests/files/observations/validation/all/all_clear_true.json']
        observation_objects = {self.obs_energy_key: []}
        for jsons in observation_json:
            observation = utils.utility_load_observation(jsons, self.obs_energy_channel)
            observation_objects[self.obs_energy_key].append(observation)
        
        self.verbosity = utils.utility_get_verbosity()
        forecast_json = ['./tests/files/forecasts/validation/all/pred_timeprof_all_clear_false.json', './tests/files/forecasts/validation/all/pred_timeprof_all_clear_true.json']
        self.pred_energy_channel = config_tests.mm_pred_energy_channel
        self.pred_energy_key = "min." +str(float(self.pred_energy_channel['min'])) + ".max." \
                + str(float(self.pred_energy_channel['max'])) + ".units." \
                + vunits.convert_units_to_string(self.pred_energy_channel['units'])
        forecast_objects = {config_tests.mm_pred_ek: []}
        for jsons in forecast_json:
            forecast = utils.utility_load_forecast(jsons, config_tests.mm_pred_energy_channel)
            forecast_objects[self.pred_energy_key].append(forecast)
        self.all_energy_channels = [self.obs_energy_key] 

        self.pred_thresh_key = {self.pred_energy_key: [config_tests.mm_pred_tk]}
        
        self.sphinx, self.not_eval_sphinx, self.obs_thresholds, self.obs_sep_events = utils.utility_match_sphinx(self.all_energy_channels, self.model_names, observation_objects, forecast_objects)
        self.profname_dict = None

        
        
        self.validation_type = ["All", "First", "Last", "Max", "Mean"]

        self.dataframe = validate.fill_sphinx_df(self.sphinx,  \
            self.obs_thresholds, self.profname_dict)
       
        for keywords in self.dataframe:
           
            logger.debug(str(self.dataframe[keywords][0]))
            temp = utils.attributes_of_sphinx_obj(keywords, self.sphinx['Test_model_0'][self.all_energy_channels[0]][0],\
                 self.obs_energy_key, self.obs_thresholds)
            logger.debug(str(self.dataframe[keywords][0]) + ' ' + str(temp) + ' ' + str(keywords))
            if 'SEP Fluence Spectrum' in keywords and "Units" not in keywords:
                try:
                    for energies in range(len(self.dataframe[keywords][0])):
                        self.assertEqual(self.dataframe[keywords][0][energies]['energy_min'], temp[energies]['energy_min'], 'Error is in keyword ' + keywords + ' energy_min')
                        self.assertEqual(self.dataframe[keywords][0][energies]['energy_max'], temp[energies]['energy_max'], 'Error is in keyword ' + keywords + ' energy_max')
                        self.assertEqual(self.dataframe[keywords][0][energies]['fluence'], temp[energies]['fluence'], 'Error is in keyword ' + keywords + ' fluence')
                except:
                    self.assertTrue(pd.isna(self.dataframe[keywords][0]))
            elif pd.isna(temp) and pd.isna(self.dataframe[keywords][0]):
                self.assertTrue(pd.isna(self.dataframe[keywords][0]))
            else:    
                self.assertEqual(self.dataframe[keywords][0], temp, 'Error is in keyword ' + keywords)
   

    def step_2(self):
        """ 
        step 2 creates a fake pickle file via mock and then the mocked file is 'read in' where
        we tell it what data is seen - via the mock_df_populate. Once this happens
        we go back to the normal resume schema which concatenates the mock data
        with the dataframe from step 1. Test is to show the length of this new data frame
        is equal to the length of the fake df and the step 1 df
        """
        self.Resume = './tests/files/fake_resume_dataframe.pkl'
        mock_dict = validate.initialize_sphinx_dict()
        data = mock_df_populate(mock_dict)
        
        df = pd.DataFrame(data = data)
            # if 'ValueError' occurs here, make sure the dictionary that is initialized in validation.py has the same
            # length/information as what get put in there form mock_df_populate. New columns to the dataframe will
            # cause the ValueError
        logger.debug('mock df data')
        logger.debug(df)
        read_data = pickle.dumps(df)
        mockOpen = mock_open(read_data=read_data)
        with patch('builtins.open', mockOpen):
            r_df = resume.read_in_df(self.Resume)
        
        logger.debug('r_df')
        logger.debug(r_df)
        logger.debug('self.dataframe')
        logger.debug(self.dataframe)
        
        self.df = pd.concat([r_df, self.dataframe], ignore_index=True)
        logger.debug('df (concat the dfs)')
        logger.debug(self.df)

        logger.debug("RESUME: Completed concatenation and removed any duplicates. Writing SPHINX_dataframe to file.")
        logger.debug(self.model_names)
        self.model_names = resume.identify_unique(self.df, 'Model')
        logger.debug(self.model_names)
        self.all_energy_channels = resume.identify_unique(self.df, 'Energy Channel Key')
        self.all_observed_thresholds = resume.identify_thresholds_per_energy_channel(self.df)
        validate.write_df(self.df, "SPHINX_dataframe")
        logger.debug("Completed writing SPHINX_dataframe to file.")
        self.assertAlmostEqual(len(self.df), len(r_df)+len(self.dataframe), msg = 'The dataframe from the resume feature is not equal to the "old" dataframe and the new dataframe')
    
    def step_3(self):
        """
        step 3 uses the step 2 dataframe to follow the rest of the normal validation workflow.
        The tests here look to the output files created as a check to ensure the workflow
        is followed as expected - since we have 2 models here you expect metric outputs
        for the various quantities as well as the selections for those quantities for each model

        """
        validate.calculate_intuitive_metrics(self.df, self.model_names, self.all_energy_channels, \
                self.all_observed_thresholds, 'All')
        # taking the 'easy way' of doing this test- just going to check that there are outputs for each
        # of the models that are created - should be files for 'Test_model_0' and "Test_model_resume_old"
        # the validation_quantity array lists the expected quantities that should have been calculated based
        # on the input json and the resume stuff, so there may be files created it doesn't test for, like end
        # time since both models don't output for that
        self.validation_quantity = ['all_clear','duration', 'fluence', 'max_flux_in_pred_win', 'peak_intensity_max', 'peak_intensity_max_time', 'peak_intensity', \
            'peak_intensity_time', 'time_profile']
        for model in self.model_names:
            for quantities in self.validation_quantity:
               
                metrics_filename = './tests/output/csv/' + quantities + '_metrics' 
                self.assertTrue(os.path.isfile(metrics_filename + '.csv'), msg = metrics_filename + '.csv does not exist, check the file is output correctly')
                metrics_filename = './tests/output/pkl/' + quantities + '_metrics' 
                self.assertTrue(os.path.isfile(metrics_filename + '.pkl'), msg = metrics_filename + '.pkl does not exist, check the file is output correctly')
                
                
                for energy_channels in self.all_energy_channels:
                    for thresholds in self.obs_thresholds[energy_channels]:
                        threshold_shortened = thresholds.rsplit('.')[0]+ '_' + thresholds.rsplit('.')[1] + '.' + thresholds.rsplit('.')[2]

                        pkl_filename = './tests/output/pkl/' + quantities + '_selections_' + model + '_' + energy_channels + '_' + threshold_shortened + '.pkl'
                        csv_filename = './tests/output/csv/' + quantities + '_selections_' + model + '_' + energy_channels + '_' + threshold_shortened + '.csv'
                    
                        self.assertTrue(os.path.isfile(pkl_filename), msg = pkl_filename + ' does not exist, check the file is output correctly')
                        self.assertTrue(os.path.isfile(csv_filename), msg = csv_filename + ' does not exist, check the file is output correctly')
                            
    def step_4(self):
        self.r_obs_prof, self.r_model_prof = mock_profile_output()


        # self.r_obs
        # self.r_mod
        validate.profile_output(self.df, self.r_obs_prof, self.r_model_prof)
        obs_prof_filename = './tests/output/json/observed_profiles.json'
        model_prof_filename = './tests/output/json/model_profiles.json'
        self.assertTrue(os.path.isfile(obs_prof_filename), msg = obs_prof_filename + ' does not exist, check the file is output correctly')
        self.assertTrue(os.path.isfile(model_prof_filename), msg = model_prof_filename + ' does not exist, check the file is output correctly')
        
        
    
    def utility_print_docstring(self, function):
        if self.verbosity == 2:
            print('\n//----------------------------------------------------')
            print(function.__doc__)

    def _steps(self):
        for name in dir(self): # dir() result is implicitly sorted
            if name.startswith("step"):
                yield name, getattr(self, name)
        

    # @patch('sphinxval.utils.config', 'config_tests')
    @patch('sphinxval.utils.config.outpath', './tests/output')
    @patch('sphinxval.utils.config.logpath', './tests/logs')
    @patch('sphinxval.utils.config.do_mismatch', False)
    @patch('sphinxval.utils.config.mm_model', 'Test_model_0')
    @patch('sphinxval.utils.config.mm_pred_energy_channel', config_tests.mm_pred_energy_channel)
    @patch('sphinxval.utils.config.mm_pred_threshold', config_tests.mm_pred_threshold)
    @patch('sphinxval.utils.config.mm_obs_energy_channel', config_tests.mm_obs_energy_channel)
    @patch('sphinxval.utils.config.mm_obs_threshold', config_tests.mm_obs_threshold)
    @patch('sphinxval.utils.config.mm_obs_ek', config_tests.mm_obs_ek)
    @patch('sphinxval.utils.config.mm_obs_tk', config_tests.mm_obs_tk)
    
    def test_all(self):
        utils.utility_delete_output()
        validate.prepare_outdirs()
        for name, step in self._steps():
            try:
                step()
            except Exception as e:
                self.fail("{} failed ({}: {})".format(step, type(e), e))
        
