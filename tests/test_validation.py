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
from . import utils


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
            fill_sphinx_df(matched_sphinx, model_names, all_energy_channels,
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
    testing the various output of fill_sphinx_df and the dictionary for each of the
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
            'Mean Ratio': [],
            'Median Ratio': [],
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
            "Percentage within an Order of Magnitude (%)":[],
            "Percentage within a factor of 2 (%)":[]
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
            "Mean AWT Efficiency for Predicted SEP All Clear to Observed SEP Threshold Crossing Time": [],

            #Threshold Crossing Time Forecasts
            "Mean AWT for Predicted SEP Threshold Crossing Time to Observed SEP Threshold Crossing Time": [],
            "Median AWT for Predicted SEP Threshold Crossing Time to Observed SEP Threshold Crossing Time": [],
            "Mean AWT for Predicted SEP Threshold Crossing Time to Observed SEP Start Time": [],
            "Median AWT for Predicted SEP Threshold Crossing Time to Observed SEP Start Time": [],
            "Mean AWT Efficiency for Predicted SEP Threshold Crossing Time to Observed SEP Threshold Crossing Time": [],

            #Start Time Forecasts
            "Mean AWT for Predicted SEP Start Time to Observed SEP Threshold Crossing Time": [],
            "Median AWT for Predicted SEP Start Time to Observed SEP Threshold Crossing Time": [],
            "Mean AWT for Predicted SEP Start Time to Observed SEP Start Time": [],
            "Median AWT for Predicted SEP Start Time to Observed SEP Start Time": [],
            "Mean AWT Efficiency for Predicted SEP Start Time to Observed SEP Threshold Crossing Time": [],
 
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
        dict["False Alarm Rate"].append(np.nan)
        dict['False Negative Rate'].append('0.0')
        dict["Frequency of Misses"].append('0.0')
        dict["Frequency of Hits"].append('1.0')
        dict["Probability of Correct Negatives"].append(np.nan)
        dict["Frequency of Correct Negatives"].append(np.nan)
        dict["False Alarm Ratio"].append('0.0')
        dict["Detection Failure Ratio"].append(np.nan)
        dict["Threat Score"].append('1.0') #Critical Success Index
        dict["Odds Ratio"].append(np.nan)
        dict["Gilbert Skill Score"].append(np.nan) #Equitable Threat Score
        dict["True Skill Statistic"].append(np.nan) #Hanssen and Kuipers
                #discriminant (true skill statistic, Peirce's skill score)
        dict["Heidke Skill Score"].append(np.nan)
        dict["Odds Ratio Skill Score"].append(np.nan)
        dict["Symmetric Extreme Dependency Score"].append(np.nan)
        dict["F1 Score"].append('1.0'),
        dict["F2 Score"].append('1.0'),
        dict["Fhalf Score"].append('1.0'),
        dict['Prevalence'].append('1.0'),
        dict['Matthew Correlation Coefficient'].append(np.nan),
        dict['Informedness'].append(np.nan),
        dict['Markedness'].append(np.nan),
        dict['Prevalence Threshold'].append(np.nan),
        dict['Balanced Accuracy'].append(np.nan),
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
        dict["Bias"].append(np.nan)
        dict["Hit Rate"].append(np.nan)
        dict["False Alarm Rate"].append('0.0')
        dict["False Negative Rate"].append(np.nan)
        dict["Frequency of Misses"].append(np.nan)
        dict["Frequency of Hits"].append(np.nan)
        dict["Probability of Correct Negatives"].append('1.0')
        dict["Frequency of Correct Negatives"].append('1.0')
        dict["False Alarm Ratio"].append(np.nan)
        dict["Detection Failure Ratio"].append('0.0')
        dict["Threat Score"].append(np.nan) #Critical Success Index
        dict["Odds Ratio"].append(np.nan)
        dict["Gilbert Skill Score"].append(np.nan) #Equitable Threat Score
        dict["True Skill Statistic"].append(np.nan) #Hanssen and Kuipers
                #discriminant (true skill statistic, Peirce's skill score)
        dict["Heidke Skill Score"].append(np.nan)
        dict["Odds Ratio Skill Score"].append(np.nan)
        dict["Symmetric Extreme Dependency Score"].append(np.nan)
        dict["F1 Score"].append(np.nan),
        dict["F2 Score"].append(np.nan),
        dict["Fhalf Score"].append(np.nan),
        dict['Prevalence'].append('0.0'),
        dict['Matthew Correlation Coefficient'].append(np.nan),
        dict['Informedness'].append(np.nan),
        dict['Markedness'].append(np.nan),
        dict['Prevalence Threshold'].append(np.nan),
        dict['Balanced Accuracy'].append(np.nan),
        dict['Fowlkes-Mallows Index'].append(np.nan),
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
    dict["Mean AWT Efficiency for Predicted SEP All Clear to Observed SEP Threshold Crossing Time"].append('0.0')

            #Threshold Crossing Time Forecasts
    dict["Mean AWT for Predicted SEP Threshold Crossing Time to Observed SEP Threshold Crossing Time"].append(np.nan)
    dict["Median AWT for Predicted SEP Threshold Crossing Time to Observed SEP Threshold Crossing Time"].append(np.nan)
    dict["Mean AWT for Predicted SEP Threshold Crossing Time to Observed SEP Start Time"].append(np.nan)
    dict["Median AWT for Predicted SEP Threshold Crossing Time to Observed SEP Start Time"].append(np.nan)
    dict['Mean AWT Efficiency for Predicted SEP Threshold Crossing Time to Observed SEP Threshold Crossing Time'].append(np.nan)

            #Start Time Forecasts
    dict["Mean AWT for Predicted SEP Start Time to Observed SEP Threshold Crossing Time"].append(np.nan)
    dict["Median AWT for Predicted SEP Start Time to Observed SEP Threshold Crossing Time"].append(np.nan)
    dict["Mean AWT for Predicted SEP Start Time to Observed SEP Start Time"].append(np.nan)
    dict["Median AWT for Predicted SEP Start Time to Observed SEP Start Time"].append(np.nan)
    dict["Mean AWT Efficiency for Predicted SEP Start Time to Observed SEP Threshold Crossing Time"].append(np.nan)
 
            #Peak Intensity Forecasts
    dict["Mean AWT for Predicted SEP Peak Intensity (Onset Peak) to Observed SEP Threshold Crossing Time"].append(np.nan)
    dict["Median AWT for Predicted SEP Peak Intensity (Onset Peak) to Observed SEP Threshold Crossing Time"].append(np.nan)
    dict["Mean AWT for Predicted SEP Peak Intensity (Onset Peak) to Observed SEP Start Time"].append(np.nan)
    dict["Median AWT for Predicted SEP Peak Intensity (Onset Peak) to Observed SEP Start Time"].append(np.nan)
    dict["Mean AWT for Predicted SEP Peak Intensity (Onset Peak) to Observed SEP Peak Intensity (Onset Peak) Time"].append(np.nan)
    dict["Median AWT for Predicted SEP Peak Intensity (Onset Peak) to Observed SEP Peak Intensity (Onset Peak) Time"].append(np.nan)

            #Peak Intensity Max Forecasts
    dict["Mean AWT for Predicted SEP Peak Intensity Max (Max Flux) to Observed SEP Threshold Crossing Time"].append(np.nan)
    dict["Median AWT for Predicted SEP Peak Intensity Max (Max Flux) to Observed SEP Threshold Crossing Time"].append(np.nan)
    dict["Mean AWT for Predicted SEP Peak Intensity Max (Max Flux) to Observed SEP Start Time"].append(np.nan)
    dict["Median AWT for Predicted SEP Peak Intensity Max (Max Flux) to Observed SEP Start Time"].append(np.nan)
    dict["Mean AWT for Predicted SEP Peak Intensity Max (Max Flux) to Observed SEP Peak Intensity Max (Max Flux) Time"].append(np.nan)
    dict["Median AWT for Predicted SEP Peak Intensity Max (Max Flux) to Observed SEP Peak Intensity Max (Max Flux) Time"].append(np.nan)

            #End Time Forecasts
    dict["Mean AWT for Predicted SEP End Time to Observed SEP Threshold Crossing Time"].append(np.nan)
    dict["Median AWT for Predicted SEP End Time to Observed SEP Threshold Crossing Time"].append(np.nan)
    dict["Mean AWT for Predicted SEP End Time to Observed SEP Start Time"].append(np.nan)
    dict["Median AWT for Predicted SEP End Time to Observed SEP Start Time"].append(np.nan)
    dict["Mean AWT for Predicted SEP End Time to Observed SEP End Time"].append(np.nan)
    dict["Median AWT for Predicted SEP End Time to Observed SEP End Time"].append(np.nan)
            
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
    dict['Spearman Correlation Coefficient'].append(np.nan)
    dict['Area Under ROC Curve'].append(np.nan)
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
    dict['Spearman Correlation Coefficient'].append(np.nan)
    dict['Area Under ROC Curve'].append(np.nan)
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
    dict['Spearman Correlation Coefficient'].append(np.nan)
    dict['Area Under ROC Curve'].append(np.nan)
    return dict


def fill_peak_intensity_max_dict(dict, self):
    """ Put flux-related metrics into metrics dictionary.
    
    """
    dict["Model"].append('Test_model_0')
    dict["Energy Channel"].append(self.energy_key)
    dict["Threshold"].append(self.obs_thresholds[self.energy_key][0])
    dict["Prediction Energy Channel"].append(self.energy_key)
    dict["Prediction Threshold"].append(self.obs_thresholds[self.energy_key][0])
    dict["Scatter Plot"].append(np.nan)
    dict["Linear Regression Slope"].append(np.nan)
    dict["Linear Regression y-intercept"].append(np.nan)
    dict["Pearson Correlation Coefficient (Linear)"].append(np.nan)
    dict["Pearson Correlation Coefficient (Log)"].append(np.nan)
    dict["Spearman Correlation Coefficient (Linear)"].append(np.nan)
    dict['Mean Ratio'].append('1.0')
    dict['Median Ratio'].append('1.0')
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
    dict["Percentage within an Order of Magnitude (%)"].append('1.0')
    dict["Percentage within a factor of 2 (%)"].append('1.0')
    dict.update({"Time Profile Selection Plot": [np.nan]})

    return dict


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
    dict["Pearson Correlation Coefficient (Linear)"].append(np.nan)
    dict["Pearson Correlation Coefficient (Log)"].append(np.nan)
    dict["Spearman Correlation Coefficient (Linear)"].append(np.nan)
    dict['Mean Ratio'].append('0.5005')
    dict['Median Ratio'].append('0.5005')
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
    dict["Percentage within an Order of Magnitude (%)"].append('0.5')
    dict["Percentage within a factor of 2 (%)"].append('0.5')
    dict.update({"Time Profile Selection Plot": [np.nan]})

    return dict

def fill_peak_intensity_dict(dict, self):
    """ Put flux-related metrics into metrics dictionary.
    
    """
    dict["Model"].append('Test_model_0')
    dict["Energy Channel"].append(self.energy_key)
    dict["Threshold"].append(self.obs_thresholds[self.energy_key][0])
    dict["Prediction Energy Channel"].append(self.energy_key)
    dict["Prediction Threshold"].append(self.obs_thresholds[self.energy_key][0])
    dict["Scatter Plot"].append(np.nan)
    dict["Linear Regression Slope"].append(np.nan)
    dict["Linear Regression y-intercept"].append(np.nan)
    dict["Pearson Correlation Coefficient (Linear)"].append(np.nan)
    dict["Pearson Correlation Coefficient (Log)"].append(np.nan)
    dict["Spearman Correlation Coefficient (Linear)"].append(np.nan)
    dict['Mean Ratio'].append('1.0')
    dict['Median Ratio'].append('1.0')
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
    dict["Percentage within an Order of Magnitude (%)"].append('1.0')
    dict["Percentage within a factor of 2 (%)"].append('1.0')
    dict.update({"Time Profile Selection Plot": [np.nan]})

    return dict


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
    dict["Pearson Correlation Coefficient (Linear)"].append(np.nan)
    dict["Pearson Correlation Coefficient (Log)"].append(np.nan)
    dict["Spearman Correlation Coefficient (Linear)"].append(np.nan)
    dict['Mean Ratio'].append('0.5005')
    dict['Median Ratio'].append('0.5005')
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
    dict["Percentage within an Order of Magnitude (%)"].append('0.5')
    dict["Percentage within a factor of 2 (%)"].append('0.5')
    dict.update({"Time Profile Selection Plot": [np.nan]})
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
    dict["Pearson Correlation Coefficient (Linear)"].append(np.nan)
    dict["Pearson Correlation Coefficient (Log)"].append(np.nan)
    dict["Spearman Correlation Coefficient (Linear)"].append(np.nan)
    dict['Mean Ratio'].append('1.05')
    dict['Median Ratio'].append('1.05')
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
    dict["Percentage within an Order of Magnitude (%)"].append('1.0')
    dict["Percentage within a factor of 2 (%)"].append('1.0')
    dict.update({"Time Profile Selection Plot": [np.nan]})

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
    dict["Linear Regression Slope"].append(np.nan)
    dict["Linear Regression y-intercept"].append(np.nan)
    dict["Pearson Correlation Coefficient (Linear)"].append('-0.16603070802422484')
    dict["Pearson Correlation Coefficient (Log)"].append('-5.551115123125783e-17')
    dict["Spearman Correlation Coefficient (Linear)"].append('0.0')
    dict['Mean Ratio'].append('1.240429661021668')
    dict['Median Ratio'].append('1.240429661021668')
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
    dict["Percentage within an Order of Magnitude (%)"].append('1.0')
    dict["Percentage within a factor of 2 (%)"].append('0.3333333333333333')
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
    dict["Mean AWT Efficiency for Predicted SEP All Clear to Observed SEP Threshold Crossing Time"].append('0.0')

            #Threshold Crossing Time Forecasts
    dict["Mean AWT for Predicted SEP Threshold Crossing Time to Observed SEP Threshold Crossing Time"].append('1.0')
    dict["Median AWT for Predicted SEP Threshold Crossing Time to Observed SEP Threshold Crossing Time"].append('1.0')
    dict["Mean AWT for Predicted SEP Threshold Crossing Time to Observed SEP Start Time"].append('1.0')
    dict["Median AWT for Predicted SEP Threshold Crossing Time to Observed SEP Start Time"].append('1.0')
    dict['Mean AWT Efficiency for Predicted SEP Threshold Crossing Time to Observed SEP Threshold Crossing Time'].append('0.0')

            #Start Time Forecasts
    dict["Mean AWT for Predicted SEP Start Time to Observed SEP Threshold Crossing Time"].append('1.0')
    dict["Median AWT for Predicted SEP Start Time to Observed SEP Threshold Crossing Time"].append('1.0')
    dict["Mean AWT for Predicted SEP Start Time to Observed SEP Start Time"].append('1.0')
    dict["Median AWT for Predicted SEP Start Time to Observed SEP Start Time"].append('1.0')
    dict["Mean AWT Efficiency for Predicted SEP Start Time to Observed SEP Threshold Crossing Time"].append('0.0')
  
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
    dict["Pearson Correlation Coefficient (Linear)"].append(np.nan)
    dict["Pearson Correlation Coefficient (Log)"].append(np.nan)
    dict["Spearman Correlation Coefficient (Linear)"].append(np.nan)
    dict['Mean Ratio'].append('1.05')
    dict['Median Ratio'].append('1.05')
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
    dict["Percentage within an Order of Magnitude (%)"].append('1.0')
    dict["Percentage within a factor of 2 (%)"].append('1.0')
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
    dict["Pearson Correlation Coefficient (Linear)"].append(np.nan)
    dict["Pearson Correlation Coefficient (Log)"].append(np.nan)
    dict["Spearman Correlation Coefficient (Linear)"].append(np.nan)
    dict['Mean Ratio'].append('1.05')
    dict['Median Ratio'].append('1.05')
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
    dict["Percentage within an Order of Magnitude (%)"].append('1.0')
    dict["Percentage within a factor of 2 (%)"].append('1.0')
    dict.update({"Time Profile Selection Plot": [np.nan]})

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
    dict["Pearson Correlation Coefficient (Linear)"].append(np.nan)
    dict["Pearson Correlation Coefficient (Log)"].append(np.nan)
    dict["Spearman Correlation Coefficient (Linear)"].append(np.nan)
    dict['Mean Ratio'].append('0.0567996494613068')
    dict['Median Ratio'].append('0.0567996494613068')
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
    dict["Percentage within an Order of Magnitude (%)"].append('0.0')
    dict["Percentage within a factor of 2 (%)"].append('0.0')
    dict.update({"Time Profile Selection Plot": ['./tests/output/plots/Time_Profile_Test_model_0_min.10.0.max.-1.0.units.MeV_threshold_1.0_20000101T000000.pdf']})

    return dict

class TestAllClear0(unittest.TestCase):
    

    def load_verbosity(self):
        self.verbosity = utils.utility_get_verbosity()
    
    
    def step_0(self): 
        validate.prepare_outdirs()
        self.energy_channel = {'min': 10, 'max': -1, 'units': 'MeV'}
        self.energy_key = objh.energy_channel_to_key(self.energy_channel)
        self.all_energy_channels = [self.energy_key] 
        self.model_names = ['Test_model_0']
        observation_json = './tests/files/observations/validation/all_clear/all_clear_false.json'
        observation = utils.utility_load_observation(observation_json, self.energy_channel)
        observation_objects = {self.energy_key: [observation]}
        self.verbosity = utils.utility_get_verbosity()
        forecast_json = './tests/files/forecasts/validation/all_clear/pred_all_clear_false.json'
        forecast = utils.utility_load_forecast(forecast_json, self.energy_channel)
        forecast_objects = {self.energy_key: [forecast]}
        self.sphinx, self.not_eval_sphinx, self.obs_thresholds, self.obs_sep_events = utils.utility_match_sphinx(self.all_energy_channels,\
             self.model_names, observation_objects, forecast_objects)
        self.profname_dict = None
        self.DoResume = False
        
        self.validation_quantity = ['all_clear', 'awt', 'duration', 'end_time', 'fluence', 'max_flux_in_pred_window', 'peak_intensity_max', 'peak_intensity_max_time', 'peak_intensity' \
            'peak_intensity_time', 'probability', 'start_time', 'threshold_crossing', 'time_profile']
        self.quantities_tested = ['all_clear', 'awt']
        self.validation_type = ['All']
        
    def step_1(self):
       
        self.dataframe = validate.fill_sphinx_df(self.sphinx,  \
            self.obs_thresholds, self.profname_dict)
        for keywords in self.dataframe:            
            temp = utils.attributes_of_sphinx_obj(keywords, self.sphinx['Test_model_0'][self.all_energy_channels[0]][0],\
                 self.energy_key, self.obs_thresholds)
            if pd.isnull(temp) and pd.isnull(self.dataframe[keywords][0]):
                self.assertTrue(pd.isnull(self.dataframe[keywords][0]))
            else:
                self.assertEqual(self.dataframe[keywords][0], temp, 'Error is in keyword ' + keywords)

    def step_2(self):
        validate.calculate_intuitive_metrics(self.dataframe, self.model_names, self.all_energy_channels, \
                self.obs_thresholds, 'All')
    

    def step_3(self):
        test_dict = initialize_all_clear_dict()
        test_dict = fill_all_clear_dict_hit(test_dict, self)
        csv_filename = os.path.join(config.outpath, 'csv', 'all_clear_metrics.csv')
        utils.assert_equal_table(self, csv_filename, test_dict)




    def step_5(self):
        
        test_dict = initialize_awt_dict()
        test_dict = fill_awt_dict(test_dict, self)
        csv_filename = os.path.join(config.outpath, 'csv', 'awt_metrics.csv')
        
        utils.assert_equal_table(self, csv_filename, test_dict)

  
    def utility_print_docstring(self, function):
        if self.verbosity == 2:
            print('\n//----------------------------------------------------')
            print(function.__doc__)

    def _steps(self):
        for name in dir(self): # dir() result is implicitly sorted
            if name.startswith("step"):
                yield name, getattr(self, name)
        
    @patch('sphinxval.utils.config.outpath', './tests/output')

    def test_all_clear_0(self):
        validate.prepare_outdirs()
        for name, step in self._steps():
            try:
                step()
            except Exception as e:
                self.fail("{} failed ({}: {})".format(step, type(e), e))
        utils.utility_delete_output()


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
        observation = utils.utility_load_observation(observation_json, self.energy_channel)
        observation_objects = {self.energy_key: [observation]}
        self.verbosity = utils.utility_get_verbosity()
        forecast_json = './tests/files/forecasts/validation/all_clear/pred_all_clear_true.json'
        forecast = utils.utility_load_forecast(forecast_json, self.energy_channel)
        forecast_objects = {self.energy_key: [forecast]}
        self.sphinx, self.not_eval_sphinx, self.obs_thresholds, self.obs_sep_events = utils.utility_match_sphinx(self.all_energy_channels,\
             self.model_names, observation_objects, forecast_objects)
        self.profname_dict = None
        self.DoResume = False
        
        self.validation_quantity = ['all_clear', 'awt', 'duration', 'end_time', 'fluence', 'max_flux_in_pred_window', 'peak_intensity_max', 'peak_intensity_max_time', 'peak_intensity' \
            'peak_intensity_time', 'probability', 'start_time', 'threshold_crossing', 'time_profile']
        self.quantities_tested = ['all_clear']
        self.validation_type = ['All']
        
    def step_1(self):
        
        self.dataframe = validate.fill_sphinx_df(self.sphinx,  \
            self.obs_thresholds, self.profname_dict)
        for keywords in self.dataframe:
            # temp = self.sphinx['Test_model_0'][self.energy_key].prediction.short_name\
            
            temp = utils.attributes_of_sphinx_obj(keywords, self.sphinx['Test_model_0'][self.all_energy_channels[0]][0],\
                 self.energy_key, self.obs_thresholds)
            if pd.isnull(temp) and pd.isnull(self.dataframe[keywords][0]):
                self.assertTrue(pd.isnull(self.dataframe[keywords][0]))
            elif keywords == 'All Threshold Crossing Times':
                self.assertEqual(self.dataframe[keywords][0], str([temp]))
            else:
                self.assertEqual(self.dataframe[keywords][0], temp, 'Error is in keyword ' + keywords)

    def step_2(self):
        validate.calculate_intuitive_metrics(self.dataframe, self.model_names, self.all_energy_channels, \
            self.obs_thresholds, 'All')
    
    def step_3(self):
        test_dict = initialize_all_clear_dict()
        test_dict = fill_all_clear_dict_CN(test_dict, self)

        csv_filename = os.path.join(config.outpath, 'csv', 'all_clear_metrics.csv')
        utils.assert_equal_table(self, csv_filename, test_dict)


    def utility_print_docstring(self, function):
        if self.verbosity == 2:
            print('\n//----------------------------------------------------')
            print(function.__doc__)

    def _steps(self):
        for name in dir(self): # dir() result is implicitly sorted
            if name.startswith("step"):
                yield name, getattr(self, name)

    @patch('sphinxval.utils.config.outpath', './tests/output')

    def test_all_clear_1(self):
        validate.prepare_outdirs()
        for name, step in self._steps():
            try:
                step()
            except Exception as e:
                self.fail("{} failed ({}: {})".format(step, type(e), e))
        utils.utility_delete_output()


class TestAllClearGarbage(unittest.TestCase):

    def load_verbosity(self):
        self.verbosity = utils.utility_get_verbosity()
    
    
    @patch('sphinxval.utils.config.outpath', './tests/output')

    def test_garbage(self): 
        validate.prepare_outdirs()

        with self.assertRaises(NameError, msg = 'Giving purposeful garbage, should exit'):
            self.energy_channel = {'min': 10, 'max': -1, 'units': 'MeV'}
            self.energy_key = objh.energy_channel_to_key(self.energy_channel)
            self.all_energy_channels = [self.energy_key] 
            self.model_names = ['Test_model_0']
            observation_json = './tests/files/observations/validation/all_clear/all_clear_false.json'
            observation = utils.utility_load_observation(observation_json, self.energy_channel)
            observation_objects = {self.energy_key: [observation]}
            self.verbosity = utils.utility_get_verbosity()
            forecast_json = './tests/files/forecasts/validation/all_clear/pred_all_clear_garbage.json'
            forecast = utils.utility_load_forecast(forecast_json, self.energy_channel)
            forecast_objects = {self.energy_key: [forecast]}
            self.sphinx, self.not_eval_sphinx, self.obs_thresholds, self.obs_sep_events = utils.utility_match_sphinx(self.all_energy_channels,\
                self.model_names, observation_objects, forecast_objects)





class TestPeakIntensity0(unittest.TestCase):

    def load_verbosity(self):
        self.verbosity = utils.utility_get_verbosity()
    
    

    def step_0(self): 
        validate.prepare_outdirs()
        self.energy_channel = {'min': 10, 'max': -1, 'units': 'MeV'}
        self.energy_key = objh.energy_channel_to_key(self.energy_channel)
        self.all_energy_channels = [self.energy_key] 
        self.model_names = ['Test_model_0']
        observation_json = './tests/files/observations/validation/all_clear/all_clear_false.json'
        observation = utils.utility_load_observation(observation_json, self.energy_channel)
        observation_objects = {self.energy_key: [observation]}
        self.verbosity = utils.utility_get_verbosity()
        forecast_json = './tests/files/forecasts/validation/onset_peak/pred_all_clear_false.json'
        forecast = utils.utility_load_forecast(forecast_json, self.energy_channel)
        forecast_objects = {self.energy_key: [forecast]}
        self.sphinx, self.not_eval_sphinx, self.obs_thresholds, self.obs_sep_events = utils.utility_match_sphinx(self.all_energy_channels,\
             self.model_names, observation_objects, forecast_objects)
        self.profname_dict = None
        self.DoResume = False
        
        self.validation_quantity = ['all_clear', 'awt', 'duration', 'end_time', 'fluence', 'max_flux_in_pred_window', 'peak_intensity_max', 'peak_intensity_max_time', 'peak_intensity' \
            'peak_intensity_time', 'probability', 'start_time', 'threshold_crossing', 'time_profile']
        self.quantities_tested = ['all_clear', 'peak_intensity']
        self.validation_type = ['All']
        
    def step_1(self):
        
        self.dataframe = validate.fill_sphinx_df(self.sphinx,  \
            self.obs_thresholds, self.profname_dict)
        for keywords in self.dataframe:
            # temp = self.sphinx['Test_model_0'][self.energy_key].prediction.short_name\
            
            temp = utils.attributes_of_sphinx_obj(keywords, self.sphinx['Test_model_0'][self.all_energy_channels[0]][0],\
                 self.energy_key, self.obs_thresholds)
            if pd.isnull(temp) and pd.isnull(self.dataframe[keywords][0]):
                self.assertTrue(pd.isnull(self.dataframe[keywords][0]))
            
            else:
                self.assertEqual(self.dataframe[keywords][0], temp, 'Error is in keyword ' + keywords)

    def step_2(self):
        validate.calculate_intuitive_metrics(self.dataframe, self.model_names, self.all_energy_channels, \
            self.obs_thresholds, 'All')
    
    def step_3(self):
        test_dict = initialize_flux_dict()
        test_dict = fill_peak_intensity_dict(test_dict, self)

        csv_filename = os.path.join(config.outpath, 'csv', 'peak_intensity_metrics.csv')
        utils.assert_equal_table(self, csv_filename, test_dict)


    def step_4(self):
        test_dict = initialize_time_dict()
        test_dict = fill_peak_intensity_time_dict(test_dict, self)
        csv_filename = os.path.join(config.outpath, 'csv', 'peak_intensity_time_metrics.csv')
        utils.assert_equal_table(self, csv_filename, test_dict)
  
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
        utils.utility_delete_output()


class TestPeakIntensityMult(unittest.TestCase):

    def load_verbosity(self):
        self.verbosity = utils.utility_get_verbosity()
    
    

    def step_0(self): 
        validate.prepare_outdirs()
        


        self.energy_channel = {'min': 10, 'max': -1, 'units': 'MeV'}
        self.energy_key = objh.energy_channel_to_key(self.energy_channel)
        self.all_energy_channels = [self.energy_key] 
        self.model_names = ['Test_model_0']
        observation_json = ['./tests/files/observations/validation/all_clear/all_clear_false.json', './tests/files/observations/validation/all_clear/all_clear_true.json']
        observation_objects = {self.energy_key: []}
        for jsons in observation_json:
            observation = utils.utility_load_observation(jsons, self.energy_channel)
            observation_objects[self.energy_key].append(observation)
        
        self.verbosity = utils.utility_get_verbosity()
        forecast_json = ['./tests/files/forecasts/validation/onset_peak/pred_all_clear_false.json', './tests/files/forecasts/validation/onset_peak/pred_all_clear_true.json']
        forecast_objects = {self.energy_key: []}
        for jsons in forecast_json:
            forecast = utils.utility_load_forecast(jsons, self.energy_channel)
            forecast_objects[self.energy_key].append(forecast)
        self.sphinx, self.not_eval_sphinx, self.obs_thresholds, self.obs_sep_events = utils.utility_match_sphinx(self.all_energy_channels, self.model_names, observation_objects, forecast_objects)
        self.profname_dict = None
        self.DoResume = False
        
        self.validation_quantity = ['all_clear', 'awt', 'duration', 'end_time', 'fluence', 'max_flux_in_pred_window', 'peak_intensity_max', 'peak_intensity_max_time', 'peak_intensity' \
            'peak_intensity_time', 'probability', 'start_time', 'threshold_crossing', 'time_profile']
        self.quantities_tested = ['peak_intensity']
        self.validation_type = ['All']
        
    def step_1(self):
        self.dataframe = validate.fill_sphinx_df(self.sphinx,  \
            self.obs_thresholds, self.profname_dict)
       
        for keywords in self.dataframe:
            
            temp = utils.attributes_of_sphinx_obj(keywords, self.sphinx['Test_model_0'][self.all_energy_channels[0]][0],\
                 self.energy_key, self.obs_thresholds)
            
            if keywords == 'All Threshold Crossing Times': 
                
                temp[0] = str(datetime.strptime(str(temp[0]), '%Y-%m-%d %H:%M:%S') )
                temp[1] = 'NaT'
                self.assertEqual(self.dataframe[keywords][0], str(temp))
            elif pd.isnull(temp) and pd.isnull(self.dataframe[keywords][0]):
                self.assertTrue(pd.isnull(self.dataframe[keywords][0]))
            else:
                self.assertEqual(self.dataframe[keywords][0], temp, 'Error is in keyword ' + keywords)

            temp = utils.attributes_of_sphinx_obj(keywords, self.sphinx['Test_model_0'][self.all_energy_channels[0]][1],\
                 self.energy_key, self.obs_thresholds)
            
            
            # specific fix for this keyword since this can have the value of "['Timestamp(YYYY-MM-DD)', 'NaT']" which it incredibly
            # hard to deal with since its a string of a an array of strings... 
            if keywords == 'All Threshold Crossing Times': 
                temp[0] = str(datetime.strptime(str(temp[0]), '%Y-%m-%d %H:%M:%S') )
                temp[1] = 'NaT'
                self.assertEqual(self.dataframe[keywords][1], str(temp))
            elif pd.isnull(temp) and pd.isnull(self.dataframe[keywords][1]):
                self.assertTrue(pd.isnull(self.dataframe[keywords][1]))
            else:
                self.assertEqual(self.dataframe[keywords][1], temp, 'Error is in keyword ' + keywords)



    def step_2(self):
        validate.calculate_intuitive_metrics(self.dataframe, self.model_names, self.all_energy_channels, \
            self.obs_thresholds, 'All')
    
    def step_3(self):    
        test_dict = initialize_flux_dict()
        test_dict = fill_peak_intensity_mult_dict(test_dict, self)
        csv_filename = os.path.join(config.outpath, 'csv', 'peak_intensity_metrics.csv')
        utils.assert_equal_table(self, csv_filename, test_dict)


    def utility_print_docstring(self, function):
        if self.verbosity == 2:
            print('\n//----------------------------------------------------')
            print(function.__doc__)

    def _steps(self):
        for name in dir(self): # dir() result is implicitly sorted
            if name.startswith("step"):
                yield name, getattr(self, name)
        

    @patch('sphinxval.utils.config.outpath', './tests/output')

    def test_peak_intensity_1(self):
        validate.prepare_outdirs()
        for name, step in self._steps():
            try:
                step()
            except Exception as e:
                self.fail("{} failed ({}: {})".format(step, type(e), e))
        utils.utility_delete_output()

        




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
        observation = utils.utility_load_observation(observation_json, self.energy_channel)
        observation_objects = {self.energy_key: [observation]}
        self.verbosity = utils.utility_get_verbosity()
        forecast_json = './tests/files/forecasts/validation/max_peak/pred_all_clear_false.json'
        forecast = utils.utility_load_forecast(forecast_json, self.energy_channel)
        forecast_objects = {self.energy_key: [forecast]}
        self.sphinx, self.not_eval_sphinx, self.obs_thresholds, self.obs_sep_events = utils.utility_match_sphinx(self.all_energy_channels,\
             self.model_names, observation_objects, forecast_objects)
        self.profname_dict = None
        self.DoResume = False
        
        self.validation_quantity = ['all_clear', 'awt', 'duration', 'end_time', 'fluence', 'max_flux_in_pred_window', 'peak_intensity_max', 'peak_intensity_max_time', 'peak_intensity' \
            'peak_intensity_time', 'probability', 'start_time', 'threshold_crossing', 'time_profile']
        self.quantities_tested = ['all_clear', 'peak_intensity_max']
        self.validation_type = ['All']
        
    def step_1(self):
        self.dataframe = validate.fill_sphinx_df(self.sphinx,  \
            self.obs_thresholds, self.profname_dict)
        for keywords in self.dataframe:
        
            
            temp = utils.attributes_of_sphinx_obj(keywords, self.sphinx['Test_model_0'][self.all_energy_channels[0]][0],\
                 self.energy_key, self.obs_thresholds)
            if pd.isnull(temp) and pd.isnull(self.dataframe[keywords][0]):
                self.assertTrue(pd.isnull(self.dataframe[keywords][0]))
            else:
                self.assertEqual(self.dataframe[keywords][0], temp, 'Error is in keyword ' + keywords)
            

    def step_2(self):
        validate.calculate_intuitive_metrics(self.dataframe, self.model_names, self.all_energy_channels, \
            self.obs_thresholds, 'All')
    
    def step_3(self):
        test_dict = initialize_flux_dict()
        test_dict = fill_peak_intensity_max_dict(test_dict, self)
        csv_filename = os.path.join(config.outpath, 'csv', 'peak_intensity_max_metrics.csv')
        utils.assert_equal_table(self, csv_filename, test_dict)

  
    def utility_print_docstring(self, function):
        if self.verbosity == 2:
            print('\n//----------------------------------------------------')
            print(function.__doc__)

    def _steps(self):
        for name in dir(self): # dir() result is implicitly sorted
            if name.startswith("step"):
                yield name, getattr(self, name)
        
   
    @patch('sphinxval.utils.config.outpath', './tests/output')

    def test_peak_intensity_max_0(self):
        validate.prepare_outdirs()
        for name, step in self._steps():
            try:
                step()
            except Exception as e:
                self.fail("{} failed ({}: {})".format(step, type(e), e))
        utils.utility_delete_output()


class TestPeakIntensityMaxMult(unittest.TestCase):

    def load_verbosity(self):
        self.verbosity = utils.utility_get_verbosity()
    
    

    def step_0(self): 
        validate.prepare_outdirs()
        


        self.energy_channel = {'min': 10, 'max': -1, 'units': 'MeV'}
        self.energy_key = objh.energy_channel_to_key(self.energy_channel)
        self.all_energy_channels = [self.energy_key] 
        self.model_names = ['Test_model_0']
        observation_json = ['./tests/files/observations/validation/all_clear/all_clear_false.json', './tests/files/observations/validation/all_clear/all_clear_true.json']
        observation_objects = {self.energy_key: []}
        for jsons in observation_json:
            observation = utils.utility_load_observation(jsons, self.energy_channel)
            observation_objects[self.energy_key].append(observation)
        
        self.verbosity = utils.utility_get_verbosity()
        forecast_json = ['./tests/files/forecasts/validation/max_peak/pred_all_clear_false.json', './tests/files/forecasts/validation/max_peak/pred_all_clear_true.json']
        forecast_objects = {self.energy_key: []}
        for jsons in forecast_json:
            forecast = utils.utility_load_forecast(jsons, self.energy_channel)
            forecast_objects[self.energy_key].append(forecast)
        self.sphinx, self.not_eval_sphinx, self.obs_thresholds, self.obs_sep_events = utils.utility_match_sphinx(self.all_energy_channels, self.model_names, observation_objects, forecast_objects)
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
        self.dataframe = validate.fill_sphinx_df(self.sphinx,  \
            self.obs_thresholds, self.profname_dict)
       
        for keywords in self.dataframe:
            temp = utils.attributes_of_sphinx_obj(keywords, self.sphinx['Test_model_0'][self.all_energy_channels[0]][0],\
                 self.energy_key, self.obs_thresholds)
            if keywords == 'All Threshold Crossing Times': 
                
                temp[0] = str(datetime.strptime(str(temp[0]), '%Y-%m-%d %H:%M:%S') )
                temp[1] = 'NaT'

                self.assertEqual(self.dataframe[keywords][0], str(temp))
            elif pd.isnull(temp) and pd.isnull(self.dataframe[keywords][0]):
                self.assertTrue(pd.isnull(self.dataframe[keywords][0]))
            else:
                self.assertEqual(self.dataframe[keywords][0], temp, 'Error is in keyword ' + keywords)

            temp = utils.attributes_of_sphinx_obj(keywords, self.sphinx['Test_model_0'][self.all_energy_channels[0]][1],\
                 self.energy_key, self.obs_thresholds)
            
            
            # specific fix for this keyword since this can have the value of "['Timestamp(YYYY-MM-DD)', 'NaT']" which it incredibly
            # hard to deal with since its a string of a an array of strings... 
            if keywords == 'All Threshold Crossing Times': 
                temp[0] = str(datetime.strptime(str(temp[0]), '%Y-%m-%d %H:%M:%S') )
                temp[1] = 'NaT'
                self.assertEqual(self.dataframe[keywords][1], str(temp))
            elif pd.isnull(temp) and pd.isnull(self.dataframe[keywords][1]):
                self.assertTrue(pd.isnull(self.dataframe[keywords][1]))
            else:
                self.assertEqual(self.dataframe[keywords][1], temp, 'Error is in keyword ' + keywords)

    def step_2(self):
        validate.calculate_intuitive_metrics(self.dataframe, self.model_names, self.all_energy_channels, \
            self.obs_thresholds, 'All')
        validate.write_df(self.dataframe, "SPHINX_dataframe")
        
    
    def step_3(self):
        test_dict = initialize_flux_dict()
        test_dict = fill_peak_intensity_max_mult_dict(test_dict, self)
        csv_filename = os.path.join(config.outpath, 'csv', 'peak_intensity_max_metrics.csv')
        
        utils.assert_equal_table(self, csv_filename, test_dict)

        
    def utility_print_docstring(self, function):
        if self.verbosity == 2:
            print('\n//----------------------------------------------------')
            print(function.__doc__)

    def _steps(self):
        for name in dir(self): # dir() result is implicitly sorted
            if name.startswith("step"):
                yield name, getattr(self, name)


    @patch('sphinxval.utils.config.outpath', './tests/output')

    def test_peak_intensity_max_1(self):
        validate.prepare_outdirs()
        for name, step in self._steps():
            try:
                step()
            except Exception as e:
                self.fail("{} failed ({}: {})".format(step, type(e), e))
        utils.utility_delete_output()



class TestProbability0(unittest.TestCase):

    def load_verbosity(self):
        self.verbosity = utils.utility_get_verbosity()
    
    

    def step_0(self): 
        validate.prepare_outdirs()
        self.energy_channel = {'min': 10, 'max': -1, 'units': 'MeV'}
        self.energy_key = objh.energy_channel_to_key(self.energy_channel)
        self.all_energy_channels = [self.energy_key] 
        self.model_names = ['Test_model_0']
        observation_json = './tests/files/observations/validation/all_clear/all_clear_false.json'
        observation = utils.utility_load_observation(observation_json, self.energy_channel)
        observation_objects = {self.energy_key: [observation]}
        self.verbosity = utils.utility_get_verbosity()
        forecast_json = './tests/files/forecasts/validation/probability/pred_probability_all_clear_false.json'
        forecast = utils.utility_load_forecast(forecast_json, self.energy_channel)
        forecast_objects = {self.energy_key: [forecast]}
        self.sphinx, self.not_eval_sphinx, self.obs_thresholds, self.obs_sep_events = utils.utility_match_sphinx(self.all_energy_channels,\
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
        self.dataframe = validate.fill_sphinx_df(self.sphinx,  \
            self.obs_thresholds, self.profname_dict)
        for keywords in self.dataframe:
            # temp = self.sphinx['Test_model_0'][self.energy_key].prediction.short_name\
            temp = utils.attributes_of_sphinx_obj(keywords, self.sphinx['Test_model_0'][self.all_energy_channels[0]][0],\
                 self.energy_key, self.obs_thresholds)
            if pd.isnull(temp) and pd.isnull(self.dataframe[keywords][0]):
                self.assertTrue(pd.isnull(self.dataframe[keywords][0]))
            else:
                self.assertEqual(self.dataframe[keywords][0], temp, 'Error is in keyword ' + keywords)


    def step_2(self):
        self.assertFalse(self.DoResume)
    
    def step_3(self):
        validate.calculate_intuitive_metrics(self.dataframe, self.model_names, self.all_energy_channels, \
            self.obs_thresholds, 'All')
        
       

    def step_4(self):
        test_dict = initialize_probability_dict()
        test_dict = fill_probability_dict_highprob(test_dict, self)
        csv_filename = os.path.join(config.outpath, 'csv', 'probability_metrics.csv')
        utils.assert_equal_table(self, csv_filename, test_dict)



    

    def utility_print_docstring(self, function):
        if self.verbosity == 2:
            print('\n//----------------------------------------------------')
            print(function.__doc__)

    def _steps(self):
        for name in dir(self): # dir() result is implicitly sorted
            if name.startswith("step"):
                yield name, getattr(self, name)
        


    @patch('sphinxval.utils.config.outpath', './tests/output')

    def test_prob_0(self):
        validate.prepare_outdirs()
        for name, step in self._steps():
            try:
                step()
            except Exception as e:
                self.fail("{} failed ({}: {})".format(step, type(e), e))
        utils.utility_delete_output()




class TestProbabilityMult(unittest.TestCase):

    def load_verbosity(self):
        self.verbosity = utils.utility_get_verbosity()
    
    

    def step_0(self): 
        validate.prepare_outdirs()
        


        self.energy_channel = {'min': 10, 'max': -1, 'units': 'MeV'}
        self.energy_key = objh.energy_channel_to_key(self.energy_channel)
        self.all_energy_channels = [self.energy_key] 
        self.model_names = ['Test_model_0']
        observation_json = ['./tests/files/observations/validation/all_clear/all_clear_false.json', './tests/files/observations/validation/all_clear/all_clear_true.json']
        observation_objects = {self.energy_key: []}
        for jsons in observation_json:
            observation = utils.utility_load_observation(jsons, self.energy_channel)
            observation_objects[self.energy_key].append(observation)
        
        self.verbosity = utils.utility_get_verbosity()
        forecast_json = ['./tests/files/forecasts/validation/probability/pred_probability_all_clear_false.json', './tests/files/forecasts/validation/probability/pred_probability_all_clear_true.json']
        forecast_objects = {self.energy_key: []}
        for jsons in forecast_json:
            forecast = utils.utility_load_forecast(jsons, self.energy_channel)
            forecast_objects[self.energy_key].append(forecast)
        self.sphinx, self.not_eval_sphinx, self.obs_thresholds, self.obs_sep_events = utils.utility_match_sphinx(self.all_energy_channels, self.model_names, observation_objects, forecast_objects)
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
        self.dataframe = validate.fill_sphinx_df(self.sphinx,  \
            self.obs_thresholds, self.profname_dict)
       
        for keywords in self.dataframe:
            # temp = self.sphinx['Test_model_0'][self.energy_key].prediction.short_name\
            
            temp = utils.attributes_of_sphinx_obj(keywords, self.sphinx['Test_model_0'][self.all_energy_channels[0]][0],\
                 self.energy_key, self.obs_thresholds)
            
            # specific fix for this keyword since this can have the value of "['NaT']" which is not pd.NaT which means 
            # if does not get caught by the isnull(). Probably a temp fix on my end
            if keywords == 'All Threshold Crossing Times': 
                
                temp[0] = str(datetime.strptime(str(temp[0]), '%Y-%m-%d %H:%M:%S') )
                temp[1] = 'NaT'
                self.assertEqual(self.dataframe[keywords][0], str(temp))
            elif pd.isnull(temp) and pd.isnull(self.dataframe[keywords][0]):
                self.assertTrue(pd.isnull(self.dataframe[keywords][0]))
            else:
                self.assertEqual(self.dataframe[keywords][0], temp, 'Error is in keyword ' + keywords)

            temp = utils.attributes_of_sphinx_obj(keywords, self.sphinx['Test_model_0'][self.all_energy_channels[0]][1],\
                 self.energy_key, self.obs_thresholds)
            
            
            # specific fix for this keyword since this can have the value of "['Timestamp(YYYY-MM-DD)', 'NaT']" which it incredibly
            # hard to deal with since its a string of a an array of strings... 
            if keywords == 'All Threshold Crossing Times': 
                temp[0] = str(datetime.strptime(str(temp[0]), '%Y-%m-%d %H:%M:%S') )
                temp[1] = 'NaT'
                self.assertEqual(self.dataframe[keywords][1], str(temp))
            elif pd.isnull(temp) and pd.isnull(self.dataframe[keywords][1]):
                self.assertTrue(pd.isnull(self.dataframe[keywords][1]))
            else:
                self.assertEqual(self.dataframe[keywords][1], temp, 'Error is in keyword ' + keywords)



    def step_2(self):
        validate.calculate_intuitive_metrics(self.dataframe, self.model_names, self.all_energy_channels, \
            self.obs_thresholds, 'All')
    
    def step_3(self):
        test_dict = initialize_probability_dict()
        test_dict = fill_probability_dict_multprob(test_dict, self)
        csv_filename = os.path.join(config.outpath, 'csv', 'probability_metrics.csv')
        utils.assert_equal_table(self, csv_filename, test_dict)
 

    def utility_print_docstring(self, function):
        if self.verbosity == 2:
            print('\n//----------------------------------------------------')
            print(function.__doc__)

    def _steps(self):
        for name in dir(self): # dir() result is implicitly sorted
            if name.startswith("step"):
                yield name, getattr(self, name)
        
   
    @patch('sphinxval.utils.config.outpath', './tests/output')
    
    def test_prob_1(self):
        validate.prepare_outdirs()
        for name, step in self._steps():
            try:
                step()
            except Exception as e:
                self.fail("{} failed ({}: {})".format(step, type(e), e))
        utils.utility_delete_output()


class TestShortNameChanger(unittest.TestCase):

    def load_verbosity(self):
        self.verbosity = utility_get_verbosity()
    
    

    def step_0(self): 
        validate.prepare_outdirs()
        # print(sphinxval.utils.config.shortname_grouping)
        self.energy_channel = {'min': 10, 'max': -1, 'units': 'MeV'}
        self.energy_key = objh.energy_channel_to_key(self.energy_channel)
        self.all_energy_channels = [self.energy_key] 
        self.model_names = ['new_shortname_for_testing']
        observation_json = './tests/files/observations/validation/all_clear/all_clear_false.json'
        observation = utils.utility_load_observation(observation_json, self.energy_channel)
        observation_objects = {self.energy_key: [observation]}
        self.verbosity = utils.utility_get_verbosity()
        forecast_json = './tests/files/forecasts/validation/max_peak/pred_all_clear_false.json'
        forecast = utils.utility_load_forecast(forecast_json, self.energy_channel)
        forecast_objects = {self.energy_key: [forecast]}
        self.sphinx, self.not_eval_sphinx, self.obs_thresholds, self.obs_sep_events = utils.utility_match_sphinx(self.all_energy_channels,\
             self.model_names, observation_objects, forecast_objects)
        self.profname_dict = None
        self.DoResume = False
        
        self.validation_quantity = ['all_clear', 'awt', 'duration', 'end_time', 'fluence', 'max_flux_in_pred_window', 'peak_intensity_max', 'peak_intensity_max_time', 'peak_intensity' \
            'peak_intensity_time', 'probability', 'start_time', 'threshold_crossing', 'time_profile']
        self.quantities_tested = ['all_clear', 'peak_intensity_max']
        self.validation_type = ['All']
        
    def step_1(self):
        self.dataframe = validate.fill_sphinx_df(self.sphinx,  \
            self.obs_thresholds, self.profname_dict)
        for keywords in self.dataframe:
            temp = utils.attributes_of_sphinx_obj(keywords, self.sphinx['new_shortname_for_testing'][self.all_energy_channels[0]][0],\
                 self.energy_key, self.obs_thresholds)
            if pd.isnull(temp) and pd.isnull(self.dataframe[keywords][0]):
                self.assertTrue(pd.isnull(self.dataframe[keywords][0]))
            else:
                self.assertEqual(self.dataframe[keywords][0], temp, 'Error is in keyword ' + keywords)
            
  
    def utility_print_docstring(self, function):
        if self.verbosity == 2:
            print('\n//----------------------------------------------------')
            print(function.__doc__)

    def _steps(self):
        for name in dir(self): # dir() result is implicitly sorted
            if name.startswith("step"):
                yield name, getattr(self, name)
        
   
    @patch('sphinxval.utils.config.outpath', './tests/output')
    @patch('sphinxval.utils.config.shortname_grouping', [('Test_model_0.*', 'new_shortname_for_testing')])

    def test_shortname_change(self):
        validate.prepare_outdirs()
        for name, step in self._steps():
            try:
                step()
            except Exception as e:
                self.fail("{} failed ({}: {})".format(step, type(e), e))
        utils.utility_delete_output()




class Test_AllFields_MultipleForecasts(unittest.TestCase):

    def load_verbosity(self):
        self.verbosity = utils.utility_get_verbosity()
    
    
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
            observation = utils.utility_load_observation(jsons, self.energy_channel)
            observation_objects[self.energy_key].append(observation)
        
        self.verbosity = utils.utility_get_verbosity()
        forecast_json = ['./tests/files/forecasts/validation/all/pred_timeprof_all_clear_false.json', './tests/files/forecasts/validation/all/pred_timeprof_all_clear_true.json', './tests/files/forecasts/validation/all/flare_pred_timeprof_all_clear_false.json']
        forecast_objects = {self.energy_key: []}
        for jsons in forecast_json:
            forecast = utils.utility_load_forecast(jsons, self.energy_channel)
            forecast_objects[self.energy_key].append(forecast)
        self.sphinx, self.not_eval_sphinx, self.obs_thresholds, self.obs_sep_events = utils.utility_match_sphinx(self.all_energy_channels, self.model_names, observation_objects, forecast_objects)
        self.profname_dict = None
        self.DoResume = False
        
        self.validation_quantity = ['all_clear', 'awt', 'duration', 'end_time', 'fluence', 'max_flux_in_pred_window', 'peak_intensity_max', 'peak_intensity_max_time', 'peak_intensity' \
            'peak_intensity_time', 'probability', 'start_time', 'threshold_crossing', 'time_profile']
        self.validation_type = ["All", "First", "Last", "Max", "Mean"]

        self.dataframe = validate.fill_sphinx_df(self.sphinx,  \
            self.obs_thresholds, self.profname_dict)
       
        for keywords in self.dataframe:
            
            temp = utils.attributes_of_sphinx_obj(keywords, self.sphinx['Test_model_0'][self.all_energy_channels[0]][0],\
                 self.energy_key, self.obs_thresholds)
            
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

            temp = utils.attributes_of_sphinx_obj(keywords, self.sphinx['Test_model_0'][self.all_energy_channels[0]][1],\
                 self.energy_key, self.obs_thresholds)
            
            if 'SEP Fluence Spectrum' in keywords and "Units" not in keywords:
                try:
                    for energies in range(len(self.dataframe[keywords][1])):
                        self.assertEqual(self.dataframe[keywords][1][energies]['energy_min'], temp[energies]['energy_min'], 'Error is in keyword ' + keywords + ' energy_min')
                        self.assertEqual(self.dataframe[keywords][1][energies]['energy_max'], temp[energies]['energy_max'], 'Error is in keyword ' + keywords + ' energy_max')
                        self.assertEqual(self.dataframe[keywords][1][energies]['fluence'], temp[energies]['fluence'], 'Error is in keyword ' + keywords + ' fluence')
                except:
                    
                    self.assertTrue(pd.isna(self.dataframe[keywords][1]))
            elif pd.isna(temp) and pd.isna(self.dataframe[keywords][1]):
                self.assertTrue(pd.isna(self.dataframe[keywords][1]))
            elif keywords == 'All Threshold Crossing Times':
                self.assertEqual(self.dataframe[keywords][1], str(['NaT']))
            else:        
                self.assertEqual(self.dataframe[keywords][1], temp, 'Error is in keyword ' + keywords)
        for type in self.validation_type:
            
            validate.calculate_intuitive_metrics(self.dataframe, self.model_names, self.all_energy_channels, \
                self.obs_thresholds, type)

    def step_2(self):
        self.assertFalse(self.DoResume)
    
    def step_3(self):
        """
        step 3 writes the dataframe to a file and then checks that those files exist
        """
        validate.write_df(self.dataframe, "SPHINX_dataframe")
        
        self.assertTrue(os.path.isfile('./tests/output/csv/SPHINX_dataframe.csv'), msg = 'SPHINX_dataframe.csv does not exist, check the file is output correctly')
        self.assertTrue(os.path.isfile('./tests/output/pkl/SPHINX_dataframe.pkl'), msg = 'SPHINX_dataframe.pkl does not exist, check the file is output correctly')

        

    def step_4_prob(self):
        
        test_dict = initialize_probability_dict()
        test_dict = fill_probability_dict_all(test_dict, self)
        csv_filename = os.path.join(config.outpath, 'csv', 'probability_metrics.csv')

        utils.assert_equal_table(self, csv_filename, test_dict)
     


    def step_5_peak_int_max(self):
        
        test_dict = initialize_flux_dict()
        test_dict = fill_peak_intensity_max_metrics_dict_all(test_dict, self)
        csv_filename = os.path.join(config.outpath, 'csv', 'peak_intensity_max_metrics.csv')
   
        utils.assert_equal_table(self, csv_filename, test_dict)
        

    def step_6_time_prof(self):
       
        test_dict = initialize_flux_dict()
        test_dict = fill_time_profile_dict_all(test_dict, self)
        csv_filename = os.path.join(config.outpath, 'csv', 'time_profile_metrics.csv')
  
        utils.assert_equal_table(self, csv_filename, test_dict)
       

    def step_7_all_clear(self):
       
        test_dict = initialize_all_clear_dict()
        test_dict = fill_all_clear_dict_all(test_dict, self)
        csv_filename = os.path.join(config.outpath, 'csv', 'all_clear_metrics.csv')
        
        utils.assert_equal_table(self, csv_filename, test_dict)


    
    def step_8_awt(self):
        
        test_dict = initialize_awt_dict()
        test_dict = fill_awt_dict_all(test_dict, self)
        csv_filename = os.path.join(config.outpath, 'csv', 'awt_metrics.csv')
       
        utils.assert_equal_table(self, csv_filename, test_dict)
    

    def step_9_duration(self):
        
        test_dict = initialize_time_dict()
        test_dict = fill_duration_metrics_dict_all(test_dict, self)
        csv_filename = os.path.join(config.outpath, 'csv', 'duration_metrics.csv')
   
        utils.assert_equal_table(self, csv_filename, test_dict)
     
    
    def step_10_end_time(self):
        
        test_dict = initialize_time_dict()
        test_dict = fill_end_time_metrics_dict_all(test_dict, self)
        csv_filename = os.path.join(config.outpath, 'csv', 'end_time_metrics.csv')
      
        utils.assert_equal_table(self, csv_filename, test_dict)
    

    def step_11_last_data_to_issue_time(self):
        
        test_dict = initialize_time_dict()
        test_dict = fill_last_data_to_issue_time_metrics_dict_all(test_dict, self)
        csv_filename = os.path.join(config.outpath, 'csv', 'last_data_to_issue_time_metrics.csv')
    
        utils.assert_equal_table(self, csv_filename, test_dict)
        

    def step_12_max_flux_pred_win(self):
        
        test_dict = initialize_flux_dict()
        test_dict = fill_max_flux_in_pred_win_metrics_dict_all(test_dict, self)
        csv_filename = os.path.join(config.outpath, 'csv', 'max_flux_in_pred_win_metrics.csv')
        
        utils.assert_equal_table(self, csv_filename, test_dict)
        

    def step_13_peak_int_max_time(self):
        
        test_dict = initialize_time_dict()
        test_dict = fill_peak_intensity_max_time_metrics_dict_all(test_dict, self)
        csv_filename = os.path.join(config.outpath, 'csv', 'peak_intensity_max_time_metrics.csv')
        
        utils.assert_equal_table(self, csv_filename, test_dict)
       
    
    def step_14_peak_int(self):
        
        test_dict = initialize_flux_dict()
        test_dict = fill_peak_intensity_metrics_dict_all(test_dict, self)
        csv_filename = os.path.join(config.outpath, 'csv', 'peak_intensity_metrics.csv')
        
        utils.assert_equal_table(self, csv_filename, test_dict)
        
    def step_15_peak_int_time(self):
        
        test_dict = initialize_time_dict()
        test_dict = fill_peak_intensity_time_dict_all(test_dict, self)
        csv_filename = os.path.join(config.outpath, 'csv', 'peak_intensity_time_metrics.csv')
        utils.assert_equal_table(self, csv_filename, test_dict)

    def step_16_start_time(self):
        
        test_dict = initialize_time_dict()
        test_dict = fill_start_time_dict_all(test_dict, self)
        csv_filename = os.path.join(config.outpath, 'csv', 'start_time_metrics.csv')
        utils.assert_equal_table(self, csv_filename, test_dict)
       
    
    def step_17_thresh_crossing_time(self):
        
        test_dict = initialize_time_dict()
        test_dict = fill_threshold_crossing_time_dict_all(test_dict, self)
        csv_filename = os.path.join(config.outpath, 'csv', 'threshold_crossing_metrics.csv')
       
        utils.assert_equal_table(self, csv_filename, test_dict)
        
    
    def step_18_fluence(self):
        
        test_dict = initialize_flux_dict()
        test_dict = fill_fluence_dict_all(test_dict, self)
        csv_filename = os.path.join(config.outpath, 'csv', 'fluence_metrics.csv')
        utils.assert_equal_table(self, csv_filename, test_dict)


    def step_19_profiledicts(self):

        validate.profile_output(self.dataframe, None, None)
        self.assertTrue(os.path.isfile('./tests/output/json/model_profiles.json'), msg = 'model_profiles.json does not exist, check the file is output correctly')
        self.assertTrue(os.path.isfile('./tests/output/json/observed_profiles.json'), msg = 'observed_profiles.json does not exist, check the file is output correctly')


    def utility_print_docstring(self, function):
        if self.verbosity == 2:
            print('\n//----------------------------------------------------')
            print(function.__doc__)

    def _steps(self):
        for name in dir(self): # dir() result is implicitly sorted
            if name.startswith("step"):
                yield name, getattr(self, name)
        

    @patch('sphinxval.utils.config.outpath', './tests/output')
    
    def test_all(self):
        validate.prepare_outdirs()
        for name, step in self._steps():
            try:
                step()
            except Exception as e:
                self.fail("{} failed ({}: {})".format(step, type(e), e))
        utils.utility_delete_output()


