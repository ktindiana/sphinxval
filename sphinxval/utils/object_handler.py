from . import classes as cl
from . import units_handler as vunits
import sys
import datetime
from astropy import units as u

__version__ = "0.1"
__author__ = "Katie Whitman"
__maintainer__ = "Katie Whitman"
__email__ = "kathryn.whitman@nasa.gov"

""" utils/object_handler.py contains subroutines to interact
    with class objects defined in utils/classes.py.
    
"""

def build_model_list(all_model):
    """ Identify all of the models represented in the list from the entry in
        ['sep_forecast_submission']['model']['short_name']
        
        INPUTS:
        
        :all_model: dict of array of Forecast objects order by
            energy channel
            
        OUTPUTS:
        
        :model_names: (string 1xn array) list of SEP model names
            
    """
    model_names = []
    keys = all_model.keys()
    for key in keys:
        models = all_model[key] #array
        for fcast in models:
            name = fcast.short_name
            if name == "" or name == None: continue
            if name not in model_names:
                model_names.append(name)

    print("build_model_list: All models identified are: ")
    print(model_names)
    
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
        if thresh != None and units != None:
            dict = {'threshold':thresh, 'threshold_units': units}
            if dict not in all_thresholds:
                all_thresholds.append(dict)
        
        if obj.event_lengths != []:
            for entry in obj.event_lengths:
                thresh = entry.threshold
                units = entry.threshold_units
                if thresh != None and units != None:
                    dict = {'threshold':thresh, 'threshold_units': units}
                    if dict not in all_thresholds:
                        all_thresholds.append(dict)
        
        if obj.fluence_spectra != []:
            for entry in obj.fluence_spectra:
                thresh = entry.threshold_start
                units = entry.threshold_units
                if thresh != None and units != None:
                    dict = {'threshold':thresh, 'threshold_units': units}
                    if dict not in all_thresholds:
                        all_thresholds.append(dict)
       
        if obj.threshold_crossings != []:
            for entry in obj.threshold_crossings:
                thresh = entry.threshold
                units = entry.threshold_units
                if thresh != None and units != None:
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

    if obj.threshold_crossings == []:
        return None
    
    threshold_crossing_time = None
    for i in range(len(obj.threshold_crossings)):
        if obj.threshold_crossings[i].threshold == None:
            continue
        
        if obj.threshold_crossings[i].threshold == \
            threshold['threshold'] and \
            obj.threshold_crossings[i].threshold_units == \
            threshold['threshold_units']:
            threshold_crossing_time =\
                obj.threshold_crossings[i].crossing_time
    
    return threshold_crossing_time


def get_file_path(json_fname):
    """ Extract the path information from the json filename and return as a string.
    
    """
    paths = json_fname.strip().split("/")
    if len(paths) == 1:
        paths = json_fname.strip().split("\\") #Windows
    
    #If no path, just json filename, then return an empty string
    if len(paths) == 1:
        return ""

    path = ""
    for sub in paths:
        if ("json" in sub) or ("JSON" in sub):
            continue
        path += sub + "/"

    return path
