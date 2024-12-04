from . import classes as cl
from . import units_handler as vunits
import sys
import datetime
from astropy import units as u
import logging
import pandas as pd
import re

__author__ = "Katie Whitman"
__maintainer__ = "Katie Whitman"
__email__ = "kathryn.whitman@nasa.gov"

""" utils/object_handler.py contains subroutines to interact
    with class objects defined in utils/classes.py.
    
"""

#Create logger
logger = logging.getLogger(__name__)

def build_model_list(all_model, shortname_grouping):
    """ Identify all of the models represented in the list from the entry in
        ['sep_forecast_submission']['model']['short_name']
        
        INPUTS:
        
        :all_model: dict of array of Forecast objects order by
            energy channel
            
        OUTPUTS:
        
        :model_names: (string 1xn array) list of SEP model names
            
    """
    model_names = []
    keys = all_model.keys() #energy channels
    for key in keys:
        models = all_model[key] #array
        for fcast in models:
            name = fcast.short_name
            if name == "" or name == None: continue
            if name not in model_names:
                model_names.append(name)

    logger.info("All models identified are: ")
    logger.info(model_names)
    
    return model_names


def energy_channel_to_key(energy_channel):
    """ Want to organize observations and forecasts according
        to energy channel to reduce uneccesary elements in loops.
        
        Turn the energy channel into a string key that can
        be used to organize a dictionary.
        
    Inputs:
    
        :energy_channel: (dict)
            {'min': 10, 'max': -1, 'units': Unit("MeV")}
    
    Output:
    
        :key: (string)
    
    """

    units = energy_channel['units']
    if isinstance(units,str):
        units = vunits.convert_string_to_units(units)
        
    str_units = vunits.convert_units_to_string(units)

    key = "min." +str(float(energy_channel['min'])) + ".max." \
        + str(float(energy_channel['max'])) + ".units." \
        + str_units
    
    return key


def threshold_to_key(threshold):
    """ Want to organize observations and forecasts according
        to energy channel and thresholds.
        
        Turn the threshold into a string key that can
        be used to organize a dictionary.
        
    Inputs:
    
        :threshold: (dict)
            {'threshold': 10, 'threshold_units': Unit("1 / (cm2 s sr)")}
    
    Output:
    
        :key: (string) e.g. "threshold.10.0.units.1 / (cm2 s sr)"
    
    """

    units = threshold['threshold_units']
    if isinstance(units,str):
        units = vunits.convert_string_to_units(units)
        
    str_units = vunits.convert_units_to_string(units)

    key = "threshold." +str(float(threshold['threshold'])) \
        + ".units." + str_units
    
    return key


def key_to_threshold(key):
    """ Want to organize observations and forecasts according
        to energy channel and thresholds.
        
        Turn a string key into a threshold that can
        be used to compare against data.
        
    Inputs:
    
        :key: (string) e.g. "threshold.10.0.units.1 / (cm2 s sr)"
    
    Output:
    
        :threshold: (dict)
            {'threshold': 10, 'threshold_units': Unit("1 / (cm2 s sr)")}

    """

    key2 = key.replace("threshold.",'')
    key2 = key2.replace(".units.",',')
    key2 = key2.split(',')
    
    unts = u.Unit(key2[1])
    threshold = {'threshold':float(key2[0]), 'threshold_units': unts}
    
    return threshold



def compare_energy_channels(channel1, channel2):
    """ Returns true if energy channels are the same,
        False if they are not.
        
    """
    ek1 = energy_channel_to_key(channel1)
    ek2 = energy_channel_to_key(channel2)
    
    if ek1 == ek2:
        return True
    else:
        return False


def compare_thresholds(thresh1, thresh2):
    """ Returns true if thresholds are the same,
        False if they are not.
        
    """
    tk1 = threshold_to_key(thresh1)
    tk2 = threshold_to_key(thresh2)
    
    if tk1 == tk2:
        return True
    else:
        return False


def identify_all_thresholds(all_obj):
    """ Find all the thresholds applied to a given energy channel.
        Thresholds are applied in:
        All clear
        Event lengths
        Fluence spectra
        Threshold crossings
        Probabilities
    
    Inputs:
    
        :all_obj: (array of Forecast or Observation objects)
        
    Outputs:
    
        :all_thresholds: (array of dict)
            [{'threshold': 10, 'threshold_units': Unit('pfu')}]
    
    """
    all_thresholds = []
    
    for obj in all_obj:
        thresh = obj.all_clear.threshold
        units = obj.all_clear.threshold_units
        if not pd.isnull(thresh) and units is not None:
            dict = {'threshold':thresh, 'threshold_units': units}
            if dict not in all_thresholds:
                all_thresholds.append(dict)
        
        if obj.event_lengths:
            for entry in obj.event_lengths:
                thresh = entry.threshold
                units = entry.threshold_units
                if not pd.isnull(thresh) and units is not None:
                    dict = {'threshold':thresh, 'threshold_units': units}
                    if dict not in all_thresholds:
                        all_thresholds.append(dict)
        
        if obj.fluence_spectra:
            for entry in obj.fluence_spectra:
                thresh = entry.threshold_start
                units = entry.threshold_units
                if not pd.isnull(thresh) and units is not None:
                    dict = {'threshold':thresh, 'threshold_units': units}
                    if dict not in all_thresholds:
                        all_thresholds.append(dict)
       
        if obj.threshold_crossings != []:
            for entry in obj.threshold_crossings:
                thresh = entry.threshold
                units = entry.threshold_units
                if not pd.isnull(thresh) and units is not None:
                    dict = {'threshold':thresh, 'threshold_units': units}
                    if dict not in all_thresholds:
                        all_thresholds.append(dict)


    return all_thresholds

    
def initialize_sphinx(fcast):
    """ Set up new sphinx object for a single Forecast object.
        One SPHINX object contains all matching information and
        predicted and observed values (possibly for multiple thresholds)

    """
    sphinx = cl.SPHINX(fcast.energy_channel)
    sphinx.prediction = fcast

    return sphinx



def get_threshold_crossing_time(obj, threshold):
    """ Report the threshold crossing time for a Forecast or
        Observation object for a specific threshold.
        
    Input:
        
        :obj: (Forecast or Observation object)
        :threshold: (dict) {'threshold': 10, 'threshold_units:' Unit("MeV")
        
    Output:
    
        :threshold_crossing_time: (datetime)
        
    """

    if not obj.threshold_crossings:
        return pd.NaT
    
    threshold_crossing_time = pd.NaT
    for i in range(len(obj.threshold_crossings)):
        if pd.isnull(obj.threshold_crossings[i].threshold):
            continue
        
        if obj.threshold_crossings[i].threshold == \
            threshold['threshold'] and \
            obj.threshold_crossings[i].threshold_units == \
            threshold['threshold_units']:
            threshold_crossing_time =\
                obj.threshold_crossings[i].crossing_time
    
    return threshold_crossing_time


def shortname_grouper(shortname, list_of_shortnames):
    """ Function to rewrite model shortnames when
    desired. Uses the config file (shortname_grouping) to determine the proper
    mapping of actual shortname to the shorter shortname you
    desire. Set shortname_grouping to none if you don't want to change
    any shortnames.
    The list of shortnames from the config file takes the form:
    list_of_shortnames = {
    pattern_string: replacement_shortname,
    pattern_string: replacement_shortname
    }
    Which can be as long as you desire.
    The pattern strings contain the pattern in each shortname that you are searching for,
    for example 'UMASEP-10 .*' or 'UMASEP-100 .*' will match to any UMASEP-10 or UMASEP-100
    submodule respectively. The replacement_shortname is the string you want to be the
    new shortname 
    
    """
    for patterns in range(len(list_of_shortnames)):
       
        logger.debug(list_of_shortnames[patterns][0], shortname)
        temp = re.match(list_of_shortnames[patterns][0], shortname)
        # logger.debug(str(temp)) # Most of the time this is None, only useful if there is a model name you are replacing
        if temp:
            return list_of_shortnames[patterns][1]
            break
        else:
            pass
    return shortname    
