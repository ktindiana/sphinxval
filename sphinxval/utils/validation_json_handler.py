from . import classes as cl
from . import object_handler as objh
from . import config as cfg
import json
import calendar
import datetime
from datetime import timedelta
import zulu
from . import units_handler as vunits #validation units
import os
import sys
import logging
import pandas as pd
import numpy as np

__author__ = "Katie Whitman"
__maintainer__ = "Katie Whitman"
__email__ = "kathryn.whitman@nasa.gov"

# 2023-06-22, Changes in v0.1:Building off of a previous
#   version of validation_json_handler.py, starting with
#   only the subroutines needed for the approach in SPHINX.


#Create logger
logger = logging.getLogger(__name__)

""" validation_jason_handler.py
    provides all of the information needed to read the JSON
    files in the CCMC SEP Scoreboard format or produced
    by operational_sep_quantities.py.
        
    This code helps convert json forecasts and observations
    into Forecast and Observation class objects for use
    in SPHINX validation.
  
"""

def read_in_json(filename, verbose=True):
    """Read in json file """
    if verbose:
        logger.info("Reading in " + filename)

    with open(filename) as f:
        info = json.load(f)
        if info == {}:
            logger.warning("Could not read in " + filename + "!!!!")
            return info

        info.update({'filename':filename})
         
    return info

def make_ccmc_zulu_time(dt):
    """ Make a datetime string in the format YYYY-MM-DDTHH:MM:SSZ
        
        INPUTS:
        
        :dt: (datetime)
        
        OUTPUTS:
        
        :zuludate: (string) in the format YYYY-MM-DDTHH:MM:SSZ
    
    """
    if dt is None:
        return pd.NaT
    if dt is pd.NaT:
        return pd.NaT
    if dt == 0:
        return pd.NaT

    zdt = zulu.create(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
    stzdt = str(zdt)
    stzdt = stzdt.split('+00:00')
    zuludate = stzdt[0] + "Z"
    return zuludate


def zulu_to_time(zt):
    """ Convert Zulu time to datetime
    
        INPUTS:
        
        :zt: (string) - date in the format "YYYY-MM-DDTHH:MM:SSZ"
        
        OUTPUTS:
        
        :dt: (datetime)
        
    """
    #Time e.g. 2014-01-08T05:05:00Z or 2012-07-12T22:25Z
    if zt == '':
        return pd.NaT
    if zt is None:
        return pd.NaT
    if zt is pd.NaT:
        return pd.NaT
    if zt == 0:
        return pd.NaT
        
    if 'Z' not in zt or 'T' not in zt:
        logger.warning(f"Time '{zt}' not in proper format. Returning pd.NaT.")
        return pd.NaT

    if '/' in zt:
        logger.warning(f"Time '{zt}' not in proper format. Returning pd.NaT.")
        return pd.NaT

    strzt = zt.split('T')
    strzt[1] = strzt[1].strip('Z')
    n = strzt[1].split(':')
    stdt = strzt[0] + ' ' + strzt[1]

    if len(n) == 2:
        dt = datetime.datetime.strptime(stdt, '%Y-%m-%d %H:%M')
    if len(n) == 3:
        dt = datetime.datetime.strptime(stdt, '%Y-%m-%d %H:%M:%S')
    return dt


def write_json(template, filename):
    """ Write json template to json file.
        
        INPUTS:
        
        :template: (dictionary) python dictionary to be written to
            json file
        :filename: (string) file to write to
        
        OUTPUTS:
        
        :Boolean: returns True is successful, False if not
    
    """
    with open(filename, "w") as outfile:
        json.dump(template, outfile)

    if not os.path.isfile(filename):
        return False

    logger.info("Wrote SEP values to json file --> " + filename)
    return True

###### SUBROUTINES TO READ JSONS AND CREATE CLASS OBJECTS ##########
def read_list_of_jsons(filename):
    """Read all of the json files in to a list.
    """
    json_files = []
    with open(filename) as list:
        for line in list:
            line = line.lstrip()
            if line == '': continue
            if line[0] == "#": continue
            json_fname = line.rstrip()
            json_files.append(json_fname)
    return json_files


def read_json_list(json_files, verbose=True):
    """Read all of the json files in to a list containing each json entry.
    """
    all_json = []
    for json_fname in json_files:
        json_info = read_in_json(json_fname, verbose)
        if json_info == {}:
            continue
        if verbose:
            logger.debug("read in " + json_fname)
        json_info.update({'filename':json_fname})
        all_json.append(json_info)
    return all_json


def identify_all_energy_channels_per_json(json, kind):
    """ Identify all the energy channels in a single json file.
        
        INPUTS:
        
            :json: a single json file
            :kind: (string) observation or forecast
    
        OUTPUT:
        
            :energy_channels: (list) list of energy channel dictionaries
    
    """
    if kind == 'observation':
        key1 = 'sep_observation_submission'
        key2 = 'observations'
    elif kind == 'forecast':
        key1 = 'sep_forecast_submission'
        key2 = 'forecasts'
    else:
        raise Exception(f"kind={kind} invalid.  Must be 'observation' or 'forecast'")

    json_energy_channels = []

    if key1 not in json.keys():
        return []
    
    obs = json[key1][key2]
    for block in obs:
        energy_channel = block['energy_channel']

        #Convert to floats and astropy units
        units = energy_channel['units']
        energy_channel['units'] = vunits.convert_string_to_units(units)
        energy_channel['min'] = float(energy_channel['min'])
        energy_channel['max'] = float(energy_channel['max'])
        
        json_energy_channels.append(energy_channel)
    
    return json_energy_channels


def identify_all_energy_channels(all_json, kind):
    """ Search through all OBSERVATION jsons and
        create an array of all possible energy channels.
        
        INPUTS:
        
        :all_json: (array of objects) Forecast or Observation objects
        
        OUTPUT:
        
        :all_energy_channels: (array of dictionaries) all possible
            energy channels present in all_jsons
            
    """
    
    all_energy_channels = []
    for entry in all_json:
        energy_channels = identify_all_energy_channels_per_json(entry, kind)
        for energy_channel in energy_channels:
            if energy_channel not in all_energy_channels:
                all_energy_channels.append(energy_channel)
            
    return all_energy_channels


def observation_object_from_json(obs_json, energy_channel):
    """ Create an Observation object from json file.
    """
    obs = cl.Observation(energy_channel)
    obs.add_observations_from_dict(obs_json)
    obs.check_energy_channel_format()
    
    return obs


def forecast_object_from_json(fcast_json, energy_channel):
    """ Create a Forecast object from json file.
    """
    key = objh.energy_channel_to_key(energy_channel)
    
    fcast = cl.Forecast(energy_channel)
    is_good_fcast = fcast.add_forecasts_from_dict(fcast_json)
    if not is_good_fcast:
        logger.warning(f"Note there was an issue with the forecast block. {fcast_json['filename']}, {key}")
    #If forecast not loaded because no forecast for the requested energy channel, return
    if pd.isnull(fcast.short_name):
        return fcast, is_good_fcast
    
    is_good_trig = fcast.add_triggers_from_dict(fcast_json)
    
    is_good_input = fcast.add_inputs_from_dict(fcast_json)

    fcast.check_energy_channel_format()
    
    return fcast, is_good_fcast


def energy_channel_overlap(json, short_name, all_energy_channels):
    """ Are the observed energy channels present in the forecast json?
        Including mismatch.
        
    """

    #Check if forecast json contains predictiobs for any of the
    #observed energy channels
    fcast_energy_channels = identify_all_energy_channels_per_json(json, "forecast")

    overlap = False
    for channel in all_energy_channels:
        compare_channel = channel
        if cfg.do_mismatch:
            if cfg.mm_model in short_name:
                if channel == cfg.mm_obs_energy_channel:
                    compare_channel = cfg.mm_pred_energy_channel

        if compare_channel in fcast_energy_channels:
            overlap = True
  
    return overlap, fcast_energy_channels

    


def load_objects_from_json(data_list, model_list):
    """ Read in a list of observations (data_list) and
        list of forecasts (model_list) and save them as
        Forecast or Observation objects.
        
    Input:
    
        :data_list: (string) name of file containing a list of
            observation jsons
        :model_list: (string) name of file containing a list of
            forecast jsons
            
    Output:
    
        :obs_objs: (dict) dictionary sorted by energy channel
            containing all Observation class objects created from
            the observation jsons
        :model_objs: (dict) dictionary sorted by energy channel
            containing all Forecast class objects created from
            the forecast jsons
        
    """
    list_of_obs = read_list_of_jsons(data_list)
    list_of_model = read_list_of_jsons(model_list)
    
    logger.info("STATS: Observation json files input into SPHINX: " + str(len(list_of_obs)))
    logger.info("STATS: Forecast json files input into SPHINX: " + str(len(list_of_model)))

    obs_jsons = read_json_list(list_of_obs)
    model_jsons = read_json_list(list_of_model)
    
    logger.info("STATS: Observation json files read in: " + str(len(obs_jsons)))
    logger.info("STATS: Forecast json files read in: " + str(len(model_jsons)))

    
    #Find energy channels available in the OBSERVATIONS
    #Only channels with observed values will be validated
    all_energy_channels = identify_all_energy_channels(obs_jsons, 'observation')
    
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
    if cfg.do_mismatch:
        obs_objs.update({cfg.mm_energy_key: []})
        model_objs.update({cfg.mm_energy_key: []})


    #Load observation objects
    for json in obs_jsons:
        for channel in all_energy_channels:
            key = objh.energy_channel_to_key(channel)
            obj = observation_object_from_json(json, channel)
            
            logger.debug("Created OBSERVATION object from json " + obj.source   + ", " + str(channel))
            logger.debug("Observation window start: " + str(obj.observation_window_start))
            #skip if energy block wasn't present in json
            if not pd.isnull(obj.observation_window_start):
                obs_objs[key].append(obj)
                logger.debug("Adding " + obj.source + " to dictionary under "
                    "key " + key)
        
            if cfg.do_mismatch:
                if key == cfg.mm_obs_ek:
                    obj = observation_object_from_json(json, channel)
                    if not pd.isnull(obj.observation_window_start):
                        obs_objs[cfg.mm_energy_key].append(obj)
                        logger.debug("Adding " + obj.source + " to dictionary under key " + cfg.mm_energy_key)
            

    #Load json objects
    removed_model_objs = []
    for json in model_jsons:
        short_name = json["sep_forecast_submission"]["model"]["short_name"]

        logger.debug('CHANGING SHORT NAME: Original short name: ' + str(short_name))
        if cfg.shortname_grouping:
            short_name = objh.shortname_grouper(short_name, cfg.shortname_grouping)
        logger.debug('CHANGING SHORT NAME: Renamed short name ' + str(short_name))


        channel_overlap, fcast_energy_channels = energy_channel_overlap(json, short_name, all_energy_channels)
        if not channel_overlap:
            if not cfg.do_mismatch or (cfg.mm_model not in short_name):
                logger.warning("REMOVED FROM ANALYSIS: No overlap between forecasted "
                    "and observed energy channels for "
                    f"{json['filename']}, {fcast_energy_channels}")
                obj, is_good = forecast_object_from_json(json, fcast_energy_channels[0])
                obj.valid = False
                obj.invalid_reason = f"Predicted energy channels not present in observations, {fcast_energy_channels}"
                removed_model_objs.append(obj)
                continue


        for channel in all_energy_channels:
            #Check if observed energy channel is an energy channel predicted
            #in the forecast json
            if channel not in fcast_energy_channels:
                if cfg.do_mismatch and cfg.mm_model in short_name:
                        if channel == cfg.mm_obs_energy_channel:
                            pred_channel = cfg.mm_pred_energy_channel
                            if pred_channel not in fcast_energy_channels:
                                continue
                else:
                    continue
            
            key = objh.energy_channel_to_key(channel)
            obj, is_good = forecast_object_from_json(json, channel)
            #At this point, may not be a good object if the forecast needed to use
            #a mismatched energy channel. Check that first before determine
            #outcome of object.
            #If the object is good, include here
            if not pd.isnull(obj.prediction_window_start) and not pd.isnull(obj.prediction_window_end):
                model_objs[key].append(obj)
                logger.debug("Created FORECAST object from json " + str(obj.source)  + ", " + key)
                logger.debug("Prediction window start: " + str(obj.prediction_window_start))
            else:
                if not cfg.do_mismatch or cfg.mm_model not in short_name:
                    logger.debug(f"{obj.source} is invalid. Will be removed in next step.")
                    model_objs[key].append(obj) #invalid, will be removed in next step
                    continue


            #If mismatched observation and prediction energy channels
            #enabled, then find the correct prediction energy channel
            #to load.
            if cfg.do_mismatch:
                if cfg.mm_model in short_name:
                    if channel == cfg.mm_obs_energy_channel:
                        pred_channel = cfg.mm_pred_energy_channel
                        obj, is_good = forecast_object_from_json(json, pred_channel)
 
                        if not is_good:
                            logger.warning("Note issue with creating FORECAST object from json " + str(obj.source)  + ", mismatch channel" + str(pred_channel))

                        #skip if energy block wasn't present in json
                        if not pd.isnull(obj.prediction_window_start):
                            model_objs[cfg.mm_energy_key].append(obj)
                            logger.debug("Adding " + obj.source + " to dictionary under key " + key)
                        else:
                            logger.debug(f"MISMATCHED {obj.source} is invalid. Will be removed in next step.")
                            model_objs[cfg.mm_energy_key].append(obj) #invalid, will be removed in next step
                            continue



    #Convert all_energy_channels to an array of string keys
    for i in range(len(all_energy_channels)):
        all_energy_channels[i] = objh.energy_channel_to_key(all_energy_channels[i])
        
    if cfg.do_mismatch:
        all_energy_channels.append(cfg.mm_energy_key)

    del obs_jsons
    del model_jsons

    for channel in all_energy_channels:
        logger.info("STATS: Observation objects created for : " + channel + ", " + str(len(obs_objs[channel])))
        logger.info("STATS: Forecast objects created for : " + channel + ", " + str(len(model_objs[channel])))

    return all_energy_channels, obs_objs, model_objs, removed_model_objs


######## SUBROUTINES TO AID LOADING CLASS OBJECTS ##################
#The codes below are used to create classes in classes.py
#These codes read small pieces of json files, extracted using the
#codes above.
def check_forecast_json(full_json, energy_channel):
    """ Checks that fields are present in full_json
        and returns an array containing the forecast
        for the desired energy_channel.
    
    Input:
        :full_json: (dict) complete CCMC forecast json
        
    Output:
        :dataD: (array of dict) forecast portion of the json
            with forecasts for each energy block
        :is_good: (bool) indicates if good (True) or missing
            values (False)
            
    """
    is_good = True
    dataD = {}
    if not full_json:
        logger.warning("EMPTY forecast json for  "
            + full_json['filename'] +
            ". Initializing all to None.")
        dataD = {}
        is_good = False
    
    if 'sep_forecast_submission' not in full_json:
        logger.warning("\'sep_forecast_submission\' field missing from forecast json. "
                       "Initializing all to None.")
        dataD = {}
        is_good = False
        
    if 'sep_forecast_submission' in full_json:
        if 'forecasts' in full_json['sep_forecast_submission']:
            is_good = True #forecasts are present
            jsonD = full_json['sep_forecast_submission']['forecasts']
            dataD = extract_block(jsonD, energy_channel)
            if dataD is None:
                logger.debug("Requested energy "
                    "channel " + str(energy_channel) +
                    "not found in forecast json"
                    + full_json['filename'] +
                    ". Initializing all to None.")
                dataD = {}
                #Is okay, forecasts will not have all energy channels
                #tested in this block. Tests every energy channel in
                #the prepared observations.
                #If a forecast contains ONLY an energy channel that is not
                #in the observations, then it will be initialized with all
                #null values for all tested energy channels.
                
        else:
            logger.warning("\'forecast\' block not "
                    "found in forecast json"
                    + full_json['filename'] +
                    ". Initializing all to None.")
            dataD = {}
            is_good = False
            
    return is_good, dataD


def check_observation_json(full_json, energy_channel):
    """ Checks that fields are present in full_json
        and returns an array containing the forecast
        for the desired energy_channel.
    
    Input:
        :full_json: (dict) complete OpSEP observation json
        
    Output:
        :dataD: (array of dict) observation portion of the json
            with forecasts for each energy block
        :is_good: (bool) indicates if good (True) or missing
            values (False)
            
    """
    is_good = True
    dataD = {}
    if not full_json:
        logger.debug("Empty observation json "
                + full_json['filename'] +
                ". Initializing all to None.")
        dataD = {}
        is_good = False
    
    if 'sep_observation_submission' not in full_json:
        logger.debug("Check that you have passed an "
            "observation json? "
            + full_json['filename'] +
            ". Initializing all to None.")
        dataD = {}
        is_good = False
        
    if 'sep_observation_submission' in full_json:
        if 'observations' in full_json['sep_observation_submission']:
            is_good = True #forecasts are present
            jsonD = full_json['sep_observation_submission']['observations']
            dataD = extract_block(jsonD, energy_channel)
            if dataD is None:
                logger.debug("Requested energy "
                    "channel " + str(energy_channel) +
                    "not found in observation json "
                    + full_json['filename'] +
                    ". Initializing all to None.")
                dataD = {}
        else:
            logger.debug("observation block not "
                    "found in observation json"
                    + full_json['filename'] +
                    ". Initializing all to None.")
            dataD = {}
            
    return is_good, dataD



## TRIGGERS
def dict_to_cme(cmeD):
    """ Extract values from an individual CME entry in forecast triggers.
        cmeD like:
            forecast_json['sep_forecast_submission']['triggers'][0]['cme']
        
    Input:
        :cmeD: (dictionary) single CME entry
        
    Output:
        :start_time: (datetime)
        :liftoff_time: (datetime)
        :lat: (float) latitude of CME source eruption
        :lon: (float) longitude of CME source eruption
        :pa: (float) position angle
        :half_width: (float) half width of CME cone
        :speed: (float) CME speed
        :catalog: (string) CME catalog
        :catalog_id: (string) id of CME catalog
        
    """
    start_time = pd.NaT
    liftoff_time = pd.NaT
    lat = np.nan
    lon = np.nan
    pa = np.nan
    half_width = np.nan
    speed = np.nan
    coordinates = None
    catalog = None
    catalog_id = None
    
    if 'start_time' in cmeD:
        start_time = cmeD['start_time']
        if isinstance(start_time,str):
            start_time = zulu_to_time(start_time)
            
    if 'liftoff_time' in cmeD:
        liftoff_time = cmeD['liftoff_time']
        if isinstance(liftoff_time,str):
            liftoff_time = zulu_to_time(liftoff_time)
            
    if 'lat' in cmeD:
        lat = cmeD['lat']
        if isinstance(lat,str):
            lat = float(lat)
 
    if 'lon' in cmeD:
        lon = cmeD['lon']
        if isinstance(lon,str):
            lon = float(lon)
            
    if 'pa' in cmeD:
        pa = cmeD['pa']
        if isinstance(pa,str):
            pa = float(pa)


    if 'half_width' in cmeD:
        half_width = cmeD['half_width']
        if isinstance(half_width,str):
            half_width = float(half_width)

    if 'speed' in cmeD:
        speed = cmeD['speed']
        if isinstance(speed,str):
            speed = float(speed)
            
    if 'coordinates' in cmeD:
        coordinates = cmeD['coordinates']

    if 'catalog' in cmeD:
        catalog = cmeD['catalog']
 
    if 'catalog_id' in cmeD:
        catalog = cmeD['catalog_id']


    return start_time, liftoff_time, lat, lon, pa, half_width, speed,\
        coordinates, catalog, catalog_id


def dict_to_cme_sim(cme_simD):
    """ Extract values from an individual CME entry in forecast triggers.
        cmeD like:
            forecast_json['sep_forecast_submission']['triggers'][0]['cme_simulation']
        
    Input:
        :cmeD: (dictionary) single CME simulation entry
        
    Output:
        :model: (string) model
        :sim_completion_time: (datetime) simulation completion time
        
    """
    model = None
    sim_completion_time = pd.NaT
    
    if 'model' in cme_simD:
        model = cme_simD['model']
        
    if 'simulation_completion_time' in cme_simD:
        sim_completion_time = cme_simD['simulation_completion_time']
        if isinstance(sim_completion_time,str):
            sim_completion_time = zulu_to_time(sim_completion_time)

    return model, sim_completion_time


def dict_to_flare(flareD):
    """ Extract values from an individual flare entry in forecast triggers.
        flareD like:
            forecast_json['sep_forecast_submission']['triggers'][0]['flare']
        
    Input:
        :flareD: (dictionary) single flare entry
        
    Output:
        :last_data_time: (datetime)
        :start_time: (datetime)
        :peak_time: (datetime)
        :end_time: (datetime)
        :location: (string) location of flare source eruption N00W00
        :lat: (int) latitude
        :lon: (int) longitude
        :intensity: (float) X-ray intensity of flare at last_data_time
        :integrated_intensity: (float) X-ray intensity summed from start to last
        :noaa_region: (string) identifier of NOAA active region
        
    """
    last_data_time = pd.NaT
    start_time = pd.NaT
    peak_time = pd.NaT
    end_time = pd.NaT
    location = None
    lat = np.nan
    lon = np.nan
    intensity = np.nan
    integrated_intensity = np.nan
    noaa_region = None
    is_good = True
    
    if 'last_data_time' in flareD:
        last_data_time = flareD['last_data_time']
        if isinstance(last_data_time,str):
            last_data_time = zulu_to_time(last_data_time)
            
    if 'start_time' in flareD:
        start_time = flareD['start_time']
        if isinstance(start_time,str):
            start_time = zulu_to_time(start_time)

    if 'peak_time' in flareD:
        peak_time = flareD['peak_time']
        if isinstance(peak_time,str):
            peak_time = zulu_to_time(peak_time)

    if 'end_time' in flareD:
        end_time = flareD['end_time']
        if isinstance(end_time,str):
            end_time = zulu_to_time(end_time)

    if 'location' in flareD:
        loc = flareD['location'] #N00E00, S00W00
        if isinstance(loc, str) and loc != "":
            if ('E' in loc or 'W' in loc) and ('N' in loc or 'S' in loc):
                location = loc
                if 'E' in location:
                    loc = loc.split('E')
                    lon = -(float(loc[1]))
                if 'W' in location:
                    loc = loc.split('W')
                    lon = float(loc[1])
                    
                if 'N' in loc[0]:
                    lat = float(loc[0][1:])
                if 'S' in loc[0]:
                    lat = -(float(loc[0][1:]))
                
    if 'intensity' in flareD:
        intensity = flareD['intensity']
        if isinstance(intensity,str):
            intensity = float(intensity)
            
    if 'integrated_intensity' in flareD:
        integrated_intensity = flareD['integrated_intensity']
        if isinstance(integrated_intensity,str):
            integrated_intensity = float(integrated_intensity)
    
    if 'noaa_region' in flareD:
        noaa_region = flareD['noaa_region']
        if isinstance(noaa_region, str):
            try:
                noaa_region = int(noaa_region)
            except ValueError as e:
                # Ignore invalid regions (e.g. "") with warning
                logger.warning("Invalid noaa_region in flare trigger: \'" + noaa_region
                            + "\'. Setting to None.")
                noaa_region = None
                is_good = False
                
       
    return last_data_time, start_time, peak_time, end_time, location,\
        lat, lon, intensity, integrated_intensity, noaa_region, is_good


def dict_to_particle_intensity(partD):
    """ Extract values from an individual particle intensity entry.
        partD like:
        forecast_json['sep_forecast_submission']['triggers'][0]['particle_intensity'][0]
        
    Input:
        :partD: (dictionary) single particle intensity entry
        
    Output:
        :observatory: (string)
        :instrument: (string)
        :last_data_time: (datetime)
        :ongoing_events: (array)
        
    """
    observatory = None
    instrument = None
    last_data_time = pd.NaT
    ongoing_events = None
    
    if 'observatory' in partD:
        observatory = partD['observatory']
    
    if 'instrument' in partD:
        instrument = partD['instrument']
        
    if 'last_data_time' in partD:
        last_data_time = partD['last_data_time']
        if isinstance(last_data_time,str):
            last_data_time = zulu_to_time(last_data_time)
            
    if 'ongoing_events' in partD:
        ongoing_events = partD['ongoing_events']
        
        if 'start_time' in ongoing_events:
            if isinstance(ongoing_events['start_time'],str):
                ongoing_events['start_time'] =\
                    zulu_to_time(ongoing_events['start_time'])
        
        if 'threshold' in ongoing_events:
            if isinstance(ongoing_events['threshold'],str):
                ongoing_events['threshold'] =\
                    float(ongoing_events['threshold'])
                    
        if 'energy_min' in ongoing_events:
            if isinstance(ongoing_events['energy_min'],str):
                ongoing_events['energy_min'] =\
                    float(ongoing_events['energy_min'])
                    
        if 'energy_max' in ongoing_events:
            if isinstance(ongoing_events['energy_max'],str):
                ongoing_events['energy_max'] =\
                    float(ongoing_events['energy_max'])


    return observatory, instrument, last_data_time, ongoing_events

def dict_to_human_evaluation(human_evaluation_dict): 
    """ Extract values from an individual human evaluation entry.
        human_evaluation_dict like:
        forecast_json['sep_forecast_submission']['triggers'][0]['human_evaluation'][0]
        
    Input:
        :human_evaluation_dict: (dictionary) single particle intensity entry
        
    Output:
        :last_data_time: (datetime)
        
    """
    last_data_time = pd.NaT
    if 'last_data_time' in human_evaluation_dict:
        last_data_time = human_evaluation_dict['last_data_time']
        if isinstance(last_data_time, str):
            last_data_time = zulu_to_time(last_data_time) 
    return last_data_time


####INPUTS
def dict_to_mag_connectivity(magconD):
    """ Extract values from an individual magnetic connectivity entry.
        magconD like:
        forecast_json['sep_forecast_submission']['inputs'][0]['magnetic_connectivity']
        
    Input:
        :magconD: (dictionary) single magnetic connectivity entry
        
    Output:
        :method: (string)
        :lat: (float)
        :lon: (float)
        :connection_angle: (dict)
        :solar_wind: (dict)
        
    """
    method = None
    lat = np.nan
    lon = np.nan
    connection_angle = None
    solar_wind = None
    
    if 'method' in magconD:
        method = magconD['method']
        
    if 'lat' in magconD:
        lat = magconD['lat']
        if isinstance(lat,str):
            lat = float(lat)

    if 'lon' in magconD:
        lon = magconD['lon']
        if isinstance(lon,str):
            lon = float(lon)
            
    if 'connection_angle' in magconD:
        connection_angle = magconD['connection_angle']
        
    if 'solar_wind' in magconD:
        solar_wind = magconD['solar_wind']
    
    return method, lat, lon, connection_angle, solar_wind

    
def dict_to_magnetogram(magnetoD):
    """ Extract values from an individual magnetogram entry.
        magnetoD like:
        forecast_json['sep_forecast_submission']['inputs'][0]['magnetogram']
        
    Input:
        :magconD: (dictionary) single magnetogram entry
        
    Output:
        :observatory: (string)
        :instrument: (string)
        :products: (array of dict)
        
    """
    observatory = None
    instrument = None
    products = None
    
    if 'observatory' in magnetoD:
        observatory = magnetoD['observatory']

    if 'instrument' in magnetoD:
        instrument = magnetoD['instrument']

    if 'products' in magnetoD:
        products = magnetoD['products']
        if products != []:
            for i in range(len(products)):
                time = products[i]['last_data_time']
                if isinstance(time,str):
                    time = zulu_to_time(time)
                    products[i]['last_data_time'] = time
                    
    return observatory, instrument, products


def dict_to_coronagraph(coronaD):
    """ Extract values from an individual coronagraph entry.
        coronaD like:
        forecast_json['sep_forecast_submission']['inputs'][0]['coronagraph']
        
    Input:
        :coronaD: (dictionary) single coronagraph entry
        
    Output:
        :observatory: (string)
        :instrument: (string)
        :products: (array of dict)
        
    """
    observatory = None
    instrument = None
    products = None
    
    if 'observatory' in coronaD:
        observatory = coronaD['observatory']

    if 'instrument' in coronaD:
        instrument = coronaD['instrument']

    if 'products' in coronaD:
        products = coronaD['products']
        if products != []:
            for i in range(len(products)):
                time = products[i]['last_data_time']
                if isinstance(time,str):
                    time = zulu_to_time(time)
                    products[i]['last_data_time'] = time
        
    return observatory, instrument, products


###FORECASTED VALUES OR OBSERVATIONS
def extract_block(jsonD, energy_channel):
    """ Extract a single energy block from a CCMC forecast or
        observation json dictionary.
        
    Input:
        :jsonD: (array of dict) all forecasts or observations in json
                jsonD = full_json['sep_forecast_submission']['forecasts']
                jsonD = full_json['sep_observations_submission']['observations']
        :energy_channel: (dict) energy channel dict, e.g.
            {'min': 10, 'max': -1, 'units': 'MeV'}
            
    Output:
        :dataD: (dict) only the values related to the specified
            energy channel
        e.g., dataD = jsonD[0]
            
    """
    dataD = None
    ek = objh.energy_channel_to_key(energy_channel)
    
    for block in jsonD:
        if 'energy_channel' in block:
            json_ek = objh.energy_channel_to_key(block['energy_channel'])
            if json_ek == ek:
                dataD = block
                return dataD
                
    return dataD
    


def dict_to_all_clear(dataD):
    """ Extract all_clear values from dictionary
        dataD = forecast_json['sep_forecast_submission']['forecasts'][0]
        
    Input:
        :dataD: dictionary containing a forecast or observation for
            a single energy channel
            
    Output:
        :all_clear: (boolean) all clear value
        :threshold: (float) threshold applied to get all clear value
        :threshold_units: (astropy units)
        :probability_threshold: (float) threshold applied to derive
            all clear from probabilistic model
            
    """
    all_clear = None
    threshold = np.nan
    threshold_units = None
    probability_threshold = np.nan
    
    if 'all_clear' in dataD:
        if 'all_clear_boolean' in dataD['all_clear']:
            all_clear = dataD['all_clear']['all_clear_boolean']
            if isinstance(all_clear,str):
                all_clear=boolean(all_clear)
        
        if 'threshold' in dataD['all_clear']:
            threshold = dataD['all_clear']['threshold']
            if isinstance(threshold,str):
                threshold = float(threshold)
        
        if 'threshold_units' in dataD['all_clear']:
            threshold_units = dataD['all_clear']['threshold_units']
            if isinstance(threshold_units,str):
                threshold_units =\
                    vunits.convert_string_to_units(threshold_units)
        
        if 'probability_threshold' in dataD['all_clear']:
                probability_threshold = dataD['all_clear']['probability_threshold']
                if isinstance(probability_threshold,str):
                    probability_threshold=float(probability_threshold)
            
    return all_clear, threshold, threshold_units, probability_threshold


def dict_to_flux_intensity(key, dataD):
    """ Extract peak_intensity values from dictionary
        dataD = forecast_json['sep_forecast_submission']['forecasts'][0]
        
    Input:
        :key: (str) 'peak_intensity', 'peak_intensity_max', 'peak_intensity_esp',
            'point_intensity'
        :dataD: dictionary containing a forecast or observation for
            a single energy channel
            
    Output:
            :intensity: (float) onset peak intensity value
            :units: (astropy) units
            :uncertainty: (float)
            :uncertainty_low: (float)
            :uncertainty_high: (float)
            :time: (datetime)
            
    """
    intensity = np.nan
    units = None
    uncertainty = np.nan
    uncertainty_low = np.nan
    uncertainty_high = np.nan
    time = pd.NaT
    
    if key in dataD:
        if 'intensity' in dataD[key]:
            intensity = dataD[key]['intensity']
            if isinstance(intensity, str):
                intensity = float(intensity)
        
        if 'units' in dataD[key]:
            units = dataD[key]['units']
            if isinstance(units, str):
                units = vunits.convert_string_to_units(units) #astropy
        
        if 'uncertainty' in dataD[key]:
            uncertainty = dataD[key]['uncertainty']
            if isinstance(uncertainty, str):
                uncertainty = float(uncertainty)

        if 'uncertainty_low' in dataD[key]:
            uncertainty_low = dataD[key]['uncertainty_low']
            if isinstance(uncertainty_low, str):
                uncertainty_low = float(uncertainty_low)

        if 'uncertainty_high' in dataD[key]:
            uncertainty_high = dataD[key]['uncertainty_high']
            if isinstance(uncertainty_high, str):
                uncertainty_high = float(uncertainty_high)

        if 'time' in dataD[key]:
            time = dataD[key]['time']
            if isinstance(time,str):
                time = zulu_to_time(time)
                
    return intensity, units, uncertainty, uncertainty_low,\
            uncertainty_high, time


def dict_to_event_length(event):
    """ Extract event_lengths values from dictionary
        event = forecast_json['sep_forecast_submission']['forecasts'][0]['event_lengths'][0]
        
    Input:
        :event: dictionary containing a forecast or observation for
            a single event
            
    Output:
            :start_time: (datetime) start of SEP event
            :end_time: (datetime) end of SEP event
            :threshold_start: (float) threshold to determine start of SEP event
            :threshold_end: (float) threshold to determine end of SEP event;
                if not present, then assume threshold_end = threshold_start
            :units: (astropy) units
            
    """
    start_time = pd.NaT
    end_time = pd.NaT
    threshold = np.nan
    threshold_units = None
    units = None
    
    if 'start_time' in event:
        start_time = event['start_time']
        if isinstance(start_time,str):
            start_time = zulu_to_time(start_time)

    if 'end_time' in event:
        end_time = event['end_time']
        if isinstance(end_time,str):
            end_time = zulu_to_time(end_time)

    if 'threshold_start' in event:
        threshold = event['threshold_start']
        if isinstance(threshold,str):
            threshold = float(threshold)

    if 'threshold' in event:
        threshold = event['threshold']
        if isinstance(threshold,str):
            threshold = float(threshold)

    if 'threshold_units' in event:
        threshold_units = event['threshold_units']
        if isinstance(threshold_units, str):
            threshold_units = vunits.convert_string_to_units(threshold_units) #astropy
        
    return start_time, end_time, threshold, threshold_units
    
    
def dict_to_fluence(event, fl_dict):
    """ Extract fleunce values from dictionary
        fl_dict = forecast_json['sep_forecast_submission']['forecasts'][0]['fluences'][0]
        
    Input:
        :fl_dict: dictionary containing a single fluence entry
            
    Output:
            :fluence: (float) fluence in energy_channel between event_lengths
            :units: (astropy) units
            :uncertainty_low: (float)
            :uncertainty_high: (float)
            
    """
    fluence = np.nan
    units = None
    threshold = np.nan
    threshold_units = None
    uncertainty_low = np.nan
    uncertainty_high = np.nan

    if 'fluence' in fl_dict:
        fluence = fl_dict['fluence']
        if isinstance(fluence, str):
            fluence = float(fluence)
    
    if 'units' in fl_dict:
        units = fl_dict['units']
        if isinstance(units, str):
            units = vunits.convert_string_to_units(units) #astropy
    
    if 'threshold' in event:
        threshold = event['threshold']
        if isinstance(threshold, str):
            threshold = float(threshold)

    if 'threshold_start' in event:
        threshold = event['threshold_start']
        if isinstance(threshold, str):
            threshold = float(threshold)

    if 'threshold_units' in event:
        threshold_units = event['threshold_units']
        if isinstance(threshold_units, str):
            threshold_units = vunits.convert_string_to_units(threshold_units) #astropy
    
    if 'uncertainty_low' in fl_dict:
        uncertainty_low = fl_dict['uncertainty_low']
        if isinstance(uncertainty_low, str):
            uncertainty_low = float(uncertainty_low)

    if 'uncertainty_high' in fl_dict:
        uncertainty_high = fl_dict['uncertainty_high']
        if isinstance(uncertainty_high, str):
            uncertainty_high = float(uncertainty_high)

    return fluence, units, threshold, threshold_units, uncertainty_low, uncertainty_high


def dict_to_fluence_spectrum(spectrum):
    """ Extract fluence_spectra values from dictionary
        spectrum = forecast_json['sep_forecast_submission']['forecasts'][0]['fluence_spectra'][0]
        
    Input:
        :spectrum: dictionary containing a fluence spectrum
            
    Output:
            :start_time: (datetime) start of SEP event
            :end_time: (datetime) end of SEP event
            :threshold_start: (float) threshold to determine start of SEP event
            :threshold_end: (float) threshold to determine end of SEP event; if not present, then assume
                threshold_end = threshold_start
            :threshold_units: (astropy) units
            :fluence_units: (astropy) units
            :fluence_spectrum: (array of dict)
            
    """
    start_time = pd.NaT
    end_time = pd.NaT
    threshold_start = np.nan
    threshold_end = np.nan
    threshold_units = None
    fluence_units = None
    fluence_spectrum = None
    
    if 'start_time' in spectrum:
        start_time = spectrum['start_time']
        if isinstance(start_time,str):
                start_time = zulu_to_time(start_time)

    if 'end_time' in spectrum:
        end_time = spectrum['end_time']
        if isinstance(end_time,str):
                end_time = zulu_to_time(end_time)

    if 'threshold_start' in spectrum:
        threshold_start = spectrum['threshold_start']
        if isinstance(threshold_start,str):
            threshold_start = float(threshold_start)

    if 'threshold_end' in spectrum:
        threshold_end = spectrum['threshold_end']
        if isinstance(threshold_end,str):
            threshold_end = float(threshold_end)

    if 'threshold_units' in spectrum:
        threshold_units = spectrum['threshold_units']
        if isinstance(threshold_units, str):
            threshold_units = vunits.convert_string_to_units(threshold_units) #astropy
            
    if 'fluence_units' in spectrum:
        fluence_units = spectrum['fluence_units']
        if isinstance(fluence_units, str):
            fluence_units = vunits.convert_string_to_units(fluence_units) #astropy

    if 'fluence_spectrum' in spectrum:
        fluence_spectrum = spectrum['fluence_spectrum']
        for i in range(len(fluence_spectrum)):
            entry = fluence_spectrum[i]
            if 'energy_min' in entry:
                if isinstance(entry['energy_min'],str):
                    entry['energy_min'] = float(entry['energy_min'])
            if 'energy_max' in entry:
                if isinstance(entry['energy_max'],str):
                    entry['energy_max'] = float(entry['energy_max'])
            if 'fluence' in entry:
                if isinstance(entry['fluence'],str):
                    entry['fluence'] = float(entry['fluence'])
            fluence_spectrum[i] = entry
        
        
    return start_time, end_time, threshold_start, threshold_end, \
        threshold_units, fluence_units, fluence_spectrum
        

def dict_to_threshold_crossing(cross):
    """ Extract threshold_crossing values from dictionary
        spectrum = forecast_json['sep_forecast_submission']['forecasts'][0]['threshold_crossings'][0]
        
    Input:
        :cross: dictionary containing a threshold crossing
            
    Output:
        :crossing_time: (datetime) start of SEP event
        :uncertainty: (float)
        :threshold: (float) threshold to determine start of SEP event
        :threshold_units: (astropy) units
            
    """
    crossing_time = pd.NaT
    uncertainty = np.nan
    threshold = np.nan
    threshold_units = None
    
    
    if 'crossing_time' in cross:
        crossing_time = cross['crossing_time']
        if isinstance(crossing_time,str):
            crossing_time = zulu_to_time(crossing_time)
            
    if 'uncertainty' in cross:
        uncertainty = cross['uncertainty']
        if isinstance(uncertainty, str):
            uncertainty = float(uncertainty)

    if 'threshold' in cross:
        threshold = cross['threshold']
        if isinstance(threshold,str):
            threshold = float(threshold)
            
    if 'threshold_units' in cross:
        threshold_units = cross['threshold_units']
        if isinstance(threshold_units, str):
            threshold_units =\
                vunits.convert_string_to_units(threshold_units) #astropy
            
    return crossing_time, uncertainty, threshold, threshold_units


def dict_to_probability(prob_dict):
    """ Extract probability values from dictionary
        prob_dict = forecast_json['sep_forecast_submission']['forecasts'][0]['probabilities'][0]
        
    Input:
        :cross: dictionary containing a probability of occurence
            
    Output:
        :probability_value: (float)
        :uncertainty: (float)
        :threshold: (float) flux to exceed threshold
        :threshold_units: (astropy) units
            
    """
    probability_value = np.nan
    uncertainty = np.nan
    threshold = np.nan
    threshold_units = None
    
    if 'probability_value' in prob_dict:
        probability_value = prob_dict['probability_value']
        if isinstance(probability_value, str):
            probability_value = float(probability_value)
            
    if 'uncertainty' in prob_dict:
        uncertainty = prob_dict['uncertainty']
        if isinstance(uncertainty, str):
            uncertainty = float(uncertainty)

    if 'threshold' in prob_dict:
        threshold = prob_dict['threshold']
        if isinstance(threshold,str):
            threshold = float(threshold)
            
    if 'threshold_units' in prob_dict:
        threshold_units = prob_dict['threshold_units']
        if isinstance(threshold_units, str):
            threshold_units =\
                vunits.convert_string_to_units(threshold_units) #astropy

    return probability_value, uncertainty, threshold, threshold_units


def generate_profile_dict(top):
    """ Starting at the top directory, recursively search to find
        .txt files and generate a dictionary containing the full
        path of all .txt files.
        
        This is needed for cases when the sep_profile files are not
        stored in the same directory as the json files.
        
        For models on the SEP Scoreboard and on iSWA, the .txt profiles
        and json files need to be in different directories to adhere to
        file organization rules.
        
        Find the full location of the time profile files automatically.
        
        INPUT:
        
            :top: (string) top directory
            
        OUTPUT:
        
            :profname_dict: (dictionary) filename as key and full
                filename with path as value
        
    """

    profname_dict = {}
    for path, dirnames, filenames in os.walk(top):
        for file in filenames:
            if file.endswith('.txt'):
                profname_dict.update({file: os.path.join(path,file)})
    
    bytes = sys.getsizeof(profname_dict)/1024**2 #MB
    logger.info("Filesize of generated dictionary: " + str(bytes) + " MB")
    return profname_dict
