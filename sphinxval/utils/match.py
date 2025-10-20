from . import validation_json_handler as vjson
from . import object_handler as objh
from . import config as cfg
from . import classes as cl
from . import time_profile as profile
from . import resume
import sys
import os.path
import pandas as pd
import numpy as np
import datetime
import logging

__author__ = "Katie Whitman"
__maintainer__ = "Katie Whitman"
__email__ = "kathryn.whitman@nasa.gov"

""" utils/match.py contains subroutines to match forecasts
    to observations.
    
"""

#Create logger
logger = logging.getLogger(__name__)

def create_obs_match_array(objs):
    """ Takes all of the observation objects and creates a pandas
    dataframe containing the key pieces of information needed
    to match specific observations to forecasts.
    
    Dataframe(s) must be organized by energy_channel and
    threshold. The output will be a dictionary of pandas
    dataframes with the keys as the thresholds.
    
    Values required for matching:
    observation window
    all clear boolean of SEP event in observation
    event_lengths start and end
    Onset peak time
    Max peak time
    
    Inputs:
    
        :obs_objs: (dict) dictionary containing arrays of Observation
            class objects sorted by energy channel
    
    Outputs:
    
        :obs_array: (dict of pandas dataframe) contains all values
            needed for matching to forecasts organized by threshold

    """

    dict = {} #dictionary of pandas dataframes for energy_channel
              #with thresholds at the keys
    
    #Identify all thresholds that are present within the
    #observations; [{'threshold': 10, 'threshold_units': Unit('pfu')}]
    thresholds = objh.identify_all_thresholds(objs)
    
    dict.update({'thresholds' : thresholds})
    dict.update({'dataframes': []})
    idx = [ix for ix in range(len(thresholds))]
    
    #Extract information into arrays to make a pandas dataframe
    #for each threshold and save into a dictionary with the
    #key that matches the index of the associated threshold
    for i in range(len(thresholds)):
        obs_win_st = []
        obs_win_end = []
        peak_time = []
        max_time = []
        all_clear = []
        start_time = []
        end_time = []
        thresh_cross_time = []
        for obj in objs:
            check_thresh = thresholds[i]['threshold']
            check_units = thresholds[i]['threshold_units']
        
            #observation window
            obs_win_st.append(obj.observation_window_start)
            obs_win_end.append(obj.observation_window_end)
            
            #Peak time
            peak_time.append(obj.peak_intensity.time)
            max_time.append(obj.peak_intensity_max.time)
            
            #all clear
            thresh = obj.all_clear.threshold
            units = obj.all_clear.threshold_units
            
            if thresh == check_thresh and units == check_units:
                all_clear.append(obj.all_clear.all_clear_boolean)
            else:
                all_clear.append(None)
                
            #Event Lengths
            event_lengths = obj.event_lengths
            if not event_lengths:
                start_time.append(pd.NaT)
                end_time.append(pd.NaT)
            else:
                st = pd.NaT
                et = pd.NaT
                for event in event_lengths:
                    if event.threshold == check_thresh and\
                        event.threshold_units == check_units:
                        st = event.start_time
                        et = event.end_time
                start_time.append(st)
                end_time.append(et)
                
            #Threshold Crossing Time
            threshold_crossings = obj.threshold_crossings
            if not threshold_crossings:
                thresh_cross_time.append(pd.NaT)
            else:
                st = pd.NaT
                for event in threshold_crossings:
                    if event.threshold == check_thresh and\
                        event.threshold_units == check_units:
                        st = event.crossing_time
                thresh_cross_time.append(st)
                
            
        #All arrays are now filled with values only associated
        #with the desired energy channel and threshold
        #Put into a pandas dataframe
        pd_dict = {'observation_window_start': obs_win_st,
                   'observation_window_end': obs_win_end,
                   'peak_time': peak_time,
                   'max_time' : max_time,
                   'all_clear': all_clear,
                   'start_time' : start_time,
                   'end_time': end_time,
                   'threshold_crossing_time': thresh_cross_time
        }
        df = pd.DataFrame(data=pd_dict)
        
        dict['dataframes'].append(df)
            
    return dict


         
def compile_all_obs(all_energy_channels, obs_objs):
    """ Takes all of the observation objects and creates a pandas
    dataframe containing the key pieces of information needed
    to match specific observations to forecasts.
    
    Dataframe(s) must be organized by energy_channel and
    threshold. The output will be a dictionary of pandas
    dataframes with the keys as the thresholds.
    
    Values required for matching:
    observation window
    all clear boolean of SEP event in observation
    event_lengths start and end
    Onset peak time
    Max peak time
    
    Inputs:
        
        :all_energy_channels: (array of dict keys) all energy channels
            found in the observations
        :obs_objs: (dict) dictionary containing arrays of Observation
            class objects sorted by energy channel
    
    Outputs:
    
        :obs_array: (dict of pandas dataframe) contains all values
            needed for matching to forecasts organized by
            energy_channel and threshold
            
        Format of pandas DataFrame for each energy_channel and
        threshold combo:
        Pandas DataFrame built from:
        {'observation_window_start': obs_win_st,
           'observation_window_end': obs_win_end,
           'peak_time': peak_time,
           'max_time' : max_time,
           'all_clear': all_clear,
           'start_time' : start_time,
           'end_time': end_time,
           'threshold_crossing_time': thresh_cross_time
        }
    
        For full array, format is:
        obs_array = [
            {'min.10.0.max.-1.0.units.MeV':
                {'thresholds': [{'threshold': 10, 'threshold_units':
                    Unit("1 / (cm2 s sr)")}],
                'Threshold 0': pandas DataFrame with obs info}},
                {'min.100.0.max.-1.0.units.MeV':
                {'thresholds': [{'threshold': 1, 'threshold_units':
                    Unit("1 / (cm2 s sr)")}],
                'Threshold 0': pandas DataFrame with obs info}}]
    
    """
    #Create a dictionary that contains all observed values needed for
    #matching to forecasts. a pandas dataframe containing values
    #compiled from all Observations can be accessed by a
    #specific energy channel and threshold combination.
    obs_match = {}
    for energy_key in all_energy_channels:
        logger.debug('Extracting observation arrays for matching '
            'for threshold ' + str(energy_key))
        dict = create_obs_match_array(obs_objs[energy_key])
        
        obs_match.update({energy_key: dict})
    return obs_match


def create_matched_model_array(objs, threshold):
    """ Takes all of the sphinx objects and creates a pandas
    dataframe containing the key pieces of information needed
    to check if multiple forecasts with different triggers were
    matched to the same observed SEP event. Puts into a Dataframe.
    
    The output is a pandas dataframe.
    
    This subroutine is called after the first round of forecast to
    observation matching is complete.
    
    Values required for finding multiple matches:
    time difference between last eruption and threshold crossing time
    observed SEP event threshold crossing time
    
    Inputs:
    
        :objs: (dict) dictionary containing arrays of SPHINX objects
            for a single model and energy channel
        :threshold: (dict) is the threshold of interest
    
    Outputs:
    
        :df: (pandas dataframe) contains all values
            needed for revising matched forecasts

    """
    #Extract information into arrays to make a pandas dataframe
    #for each threshold and save into a dictionary with the
    #key that matches the index of the associated threshold
    sphinx_index = [] #holds the location of each sphinx object in the objs list
                      #so can revise the values in the original object if a
                      #forecast needs to be unmatched
    matched_obs = [] #matched Observation objects
    td_eruption_thresh_cross = [] #array of all obs in prediction window
    matched_sep_source = [] #filenames of the final matched observation file
    observed_thresh_cross = [] #single final matched value
    prediction_cmes = [] #CMEs used to trigger each forecast
    prediction_flares = [] #Flares used to trigger each forecast
    thresh_key = objh.threshold_to_key(threshold)
    
    #Loop over SPHINX objects and extract observed matched threshold crossing
    #time and time difference between eruption used by model and
    #threshold crossing time
    for i, obj in enumerate(objs):
        sphinx_index.append(i)

        try:
            obs_arr = obj.prediction_observation_windows_overlap
            match_sep = obj.observed_match_sep_source[thresh_key]
            obs_thresh_cross = obj.observed_threshold_crossing[thresh_key].crossing_time
            all_td = obj.time_difference_eruptions_threshold_crossing[thresh_key] #array in float hours
            
            #Pull out the eruption time difference information for the
            #observation that was identified to contain an SEP event
            td = np.nan
            if not pd.isnull(match_sep):
                for kk in range(len(obs_arr)):
                    if obs_arr[kk].source == match_sep:
                        td = all_td[kk]
            
            cmes = obj.prediction.cmes
            flares = obj.prediction.flares
            
            #If multiple triggers, store only the one that is most
            #likely to be associated with a SEP event
            #If no best choice from parameters, take the last flare, cme
            if len(cmes) > 1:
                cme = preferred_cme(cmes) #may return multiple
                if len(cme) > 1:
                    cme = last_cme(cme) #last of preferred CMEs
                elif len(cme) == 1:
                    cme = cme[0]

                if pd.isnull(cme) or not cme: #no best option, take last of all CMEs
                    cme = last_cme(cmes)

                cmes = [cme]
                
            if len(flares) > 1:
                flare = preferred_flare(flares) #may return multiple
                if len(flare) > 1:
                    flare = last_flare(flare) #last of preferred flares
                    if len(flare) > 1:
                        flare = last_flare(flare)
                    elif len(flare) == 1:
                        flare = flare[0]

                if pd.isnull(flare): #no best option, take last
                    flare = last_flare(flares)
 
                flares = [flare]


            if not cmes: cmes = [None]
            if not flares: flares = [None]

#            logger.info(f"cmes: {cmes}, flares: {flares}")
            matched_obs.append(obs_arr)
            matched_sep_source.append(match_sep)
            observed_thresh_cross.append(obs_thresh_cross)
            td_eruption_thresh_cross.append(td)
            prediction_cmes.append(cmes[0])
            prediction_flares.append(flares[0])
            
        except: #This exception occurs if the above dictionaries don't contain the
                #queried threshold
            matched_obs.append([None])
            matched_sep_source.append(None)
            observed_thresh_cross.append(pd.NaT)
            td_eruption_thresh_cross.append(np.nan)
            prediction_cmes.append(None)
            prediction_flares.append(None)

        
    #All arrays are now filled with values only associated
    #with the desired energy channel and threshold
    #Put into a pandas dataframe
    pd_dict = {'sphinx_index': sphinx_index,
               'matched_observations': matched_obs,
               'matched_sep': matched_sep_source,
               'td_eruption_thresh_cross': td_eruption_thresh_cross,
               'observed_threshold_crossing_time': observed_thresh_cross,
               'CME Triggers': prediction_cmes,
               'Flare Triggers': prediction_flares
    }
    df = pd.DataFrame(data=pd_dict)        
    return df




def pred_and_obs_overlap(fcast, obs):
    """ Create boolean array indicating if the prediction window
        overlaps with the observation window.
        
    Input:
        
        :fcast: (Forecast object) a single forecast for a single
            energy channel
        :obs: (Pandas DataFrame) contains needed observed times for
            a single threshold (can choose any threshold as these
            values only depend on the energy channel)
    
    Output:
    
        :overlaps_bool: (array of booleans) have same indices as the
            observations in obs. True if overlap, False if no overlap
            
    """
    pred_win_st = fcast.prediction_window_start
    pred_win_end = fcast.prediction_window_end
    logger.debug("Prediction window: " + str(pred_win_st) + " to "
        + str(pred_win_end))

    overlaps_bool = ((obs['observation_window_start'] <= pred_win_end) & (obs['observation_window_end'] >= pred_win_st)).tolist()

    return overlaps_bool



def does_win_overlap(energy_key, fcast, obs_values):
    """ Find which observation windows overlap with
        the forecast prediction window.
        This is an initial rough cut of which observation
        files match with a forecast.
        
    Input:
    
        :fcast: (Forecast object) forecast for a single energy bin
        :obs_values: (dict of pandas DataFrames) for a single
            energy channel and all observed threshold.
    
    Output:
    
        :is_win_overlap: (array of bool) array indices match the order
            of the Observation objects in obs_objs[energy_key]
            where energy_key is the key made from the fcast
            energy_channel
    
    """
    
    #Return if no prediction
    if pd.isnull(fcast.prediction_window_start):
        return [None]*len(obs_values[energy_key]['dataframes'][0])
        

    #Pull out the observations needed for matching
    #We will have a table with needed information about each
    #observation
    #Extract correct energy channel and any threshold
    obs = obs_values[energy_key]['dataframes'][0]
    
    #1. Does the Prediction Window overlap with the Observation Window?
    is_win_overlap = pred_and_obs_overlap(fcast, obs)
    
    return is_win_overlap


def check_observations_coverage(sphinx):
    """ Check that observations cover 100% of forecast prediction window. 
        
        Input:
        
            :sphinx: (SPHINX object)
            
        Output:
        
            :is_covered: (bool) True if 100% coverage, False if not
    
    """
    pred_win_st = sphinx.prediction.prediction_window_start
    pred_win_end = sphinx.prediction.prediction_window_end

    #loop over matched observation objects
    obs_st = []
    obs_end = []
    for obj in sphinx.prediction_observation_windows_overlap:
        obs_st.append(obj.observation_window_start)
        obs_end.append(obj.observation_window_end)
        
    first_obs =  min(obs_st)
    last_obs = max(obs_end)
    
    is_covered = True
    if (first_obs > pred_win_st) or (last_obs < pred_win_end):
        is_covered = False
        
    return is_covered



def observed_time_in_pred_win(fcast, obs_values, obs_key, energy_key):
    """ Find whether an observed time occurred inside of the prediction
        window across all of the observations
        
        Returns boolean indicating whether the observed time
        is inside of the prediction window.
    
    Input:
    
        :fcast: (Forecast object) single forecast
        :obs_df: (dict of pandas DataFrames) for a single
            energy channel and all observed threshold.
        :obs_key: (string) identifies which time in the pandas
            Dataframe should be compared, e.g. 'threshold_crossing_time'
            
    Output:
        
        :contains_obs_time: (array of bool) array indices match the order
            of the Observation objects in obs_objs[energy_key]
            where energy_key is the key made from the fcast
            energy_channel
    
    """    
    pred_win_st = fcast.prediction_window_start
    pred_win_end = fcast.prediction_window_end

    #Return if no prediction
    if pd.isnull(fcast.prediction_window_start):
        return [None]*len(obs_values[energy_key]['dataframes'][0])
           
    #Extract pandas dataframe for correct energy and threshold
    obs_df = obs_values[energy_key]['dataframes'][0] #all obs for threshold

    #Check if a time is contained inside of the
    #prediction window; returns False if pd.NaT
    contains_obs_time = (obs_df[obs_key] >= pd.Timestamp(pred_win_st)) \
        & (obs_df[obs_key] <= pd.Timestamp(pred_win_end))
 
    return list(contains_obs_time)



def observed_time_in_pred_win_thresh(fcast, obs_values, obs_key,
        energy_key, threshold):
    """ Find whether an observed time occurred inside of the prediction
        window across all of the observations
        
        Returns boolean indicating whether the threshold was
        actively crossed inside of the prediction window.
    
    Input:
    
        :fcast: (Forecast object) single forecast
        :obs_values: (dict of pandas DataFrames) for a single
            energy channel and all observed threshold.
        :obs_key: (string) identifies which time in the pandas
            Dataframe should be compared, 'threshold_crossing_time'
        :threshold: (dict) {'threshold':10, 'units': Unit("MeV")}
            
    Output:
        
        :contains_obs_time: (array of bool) array indices match the order
            of the Observation objects in obs_objs[energy_key]
            where energy_key is the key made from the fcast
            energy_channel
    
    """
    pred_win_st = fcast.prediction_window_start
    pred_win_end = fcast.prediction_window_end

    #Return if no prediction
    if pd.isnull(fcast.prediction_window_start):
        return [None]*len(obs_values[energy_key]['dataframes'][0]), [pd.NaT]*len(obs_values[energy_key]['dataframes'][0])
       
    #Extract desired threshold
    obs_thresholds = obs_values[energy_key]['thresholds']
    try:
        ix = obs_thresholds.index(threshold)
    except:
        return [None]*len(obs_values[energy_key]['dataframes'][0]), [pd.NaT]*len(obs_values[energy_key]['dataframes'][0])
    
    #Extract pandas dataframe for correct energy and threshold
    obs = obs_values[energy_key]['dataframes'][ix] #all obs for threshold

    #Check if a threshold crossing is contained inside of the
    #prediction window; returns False if pd.NaT
    contains_obs_time = (obs[obs_key] >= pd.Timestamp(pred_win_st)) \
        & (obs[obs_key] <= pd.Timestamp(pred_win_end))
 
    return list(contains_obs_time), list(obs[obs_key])


def is_time_before(time, obs_values, obs_key, energy_key):
    """ Find whether a time is before a specific time in the
        observations.
        
        Returns boolean indicating whether time was before
        the specified observed time.
        
        ** For use with values that don't depend on threshold.
        Will choose the 0th threshold. **
        onset peak (peak_intensity)
        Max flux (peak_intensity_max)
    
    Input:
    
        :fcast: (Forecast object) single forecast
        :obs_values: (dict of pandas DataFrames) for a single
            energy channel and all observed threshold.
        :obs_key: (string) "start_time", "end_time" ,etc
            
    Output:
        
        :is_before: (array of bool) array indices match the order
            of the Observation objects in obs_objs[energy_key]
            where energy_key is the key made from the fcast
            energy_channel
    
    """
    if pd.isnull(time):
        return [None]*len(obs_values[energy_key]['dataframes'][0])

    #Check if time is before
    obs = obs_values[energy_key]['dataframes'][0]
    time_diff = (time - obs[obs_key]).dt.total_seconds()/3600.
    is_before = (time_diff <= 0)
    is_before = is_before.values.tolist()

    nat = [ix for ix in range(len(list(time_diff))) if pd.isnull(time_diff[ix])]
    for ix in nat:
        is_before[ix] = None
            
    return is_before
    

def is_time_before_thresh(time, obs_values, obs_key, energy_key, threshold):
    """ Find whether a time is before a specific time in the
        observations.
        
        Returns boolean indicating whether time was before
        the specified observed time.
    
    Input:
    
        :fcast: (Forecast object) single forecast
        :obs_values: (dict of pandas DataFrames) for a single
            energy channel and all observed threshold.
        :obs_key: (string) "start_time", "end_time" ,etc
        :threshold: (dict) {'threshold':10, 'units': Unit("MeV")}
            
    Output:
        
        :is_before: (array of bool) array indices match the order
            of the Observation objects in obs_objs[energy_key]
            where energy_key is the key made from the fcast
            energy_channel
    
    """
    if pd.isnull(time):
        return [None]*len(obs_values[energy_key]['dataframes'][0])

    #Extract desired threshold
    obs_thresholds = obs_values[energy_key]['thresholds']
    try:
        ix = obs_thresholds.index(threshold)
    except:
        return [None]*len(obs_values[energy_key]['dataframes'][0])

    #Check if time is before
    obs = obs_values[energy_key]['dataframes'][ix]

    time_diff = (time - obs[obs_key]).dt.total_seconds()/3600.
    is_before = (time_diff <= 0)
    is_before = is_before.values.tolist()

    #Check for Not a Time values
    nat = [ii for ii in range(len(list(time_diff))) if pd.isnull(time_diff[ii])]
    for ii in nat:
        is_before[ii] = None
            
    return is_before


def time_diff(time, obs_values, obs_key, energy_key):
    """ Find the time difference between a time in the forecast and
        a specific time in the observations.
        
        Returns a timedelta array where negative time means the
        forecast-associated time is before the observed time.
        
        ** For use with values that don't depend on threshold.
        Will choose the 0th threshold. **
        onset peak (peak_intensity)
        Max flux (peak_intensity_max)
    
    Input:
    
        :fcast: (Forecast object) single forecast
        :obs_values: (dict of pandas DataFrames) for a single
            energy channel and all observed threshold.
        :obs_key: (string) "start_time", "end_time" ,etc
            
    Output:
        
        :time_diff: (array of timedelta) array indices match the order
            of the Observation objects in obs_objs[energy_key]
            where energy_key is the key made from the fcast
            energy_channel
    
    """

    if pd.isnull(time):
        return [None]*len(obs_values[energy_key]['dataframes'][0])

    #Check if time is before
    obs = obs_values[energy_key]['dataframes'][0]
    time_diff = (time - obs[obs_key]).dt.total_seconds()/3600.
    time_diff = time_diff.values.tolist()

    return time_diff
    

def time_diff_thresh(time, obs_values, obs_key, energy_key, threshold):
    """ Find the time difference between a time in the forecast and
        a specific time in the observations.
        
        Returns a timedelta array where negative time means the
        forecast-associated time is before the observed time.
        
        For use with observations when the threshold is needed.
    
    Input:
    
        :fcast: (Forecast object) single forecast
        :obs_values: (dict of pandas DataFrames) for a single
            energy channel and all observed threshold.
        :obs_key: (string) "start_time", "end_time" ,etc
        :threshold: (dict) {'threshold':10, 'units': Unit("MeV")}
            
    Output:
        
        :time_diff: (array of timedelta) array indices match the order
            of the Observation objects in obs_objs[energy_key]
            where energy_key is the key made from the fcast
            energy_channel
    
    """
    if pd.isnull(time):
        return [np.nan]*len(obs_values[energy_key]['dataframes'][0])

    #Extract desired threshold
    obs_thresholds = obs_values[energy_key]['thresholds']
    try:
        ix = obs_thresholds.index(threshold)
    except:
        return [np.nan]*len(obs_values[energy_key]['dataframes'][0])

    #Check if time is before
    obs = obs_values[energy_key]['dataframes'][ix]
    time_diff = (time - obs[obs_key]).dt.total_seconds()/3600.
    time_diff = time_diff.values.tolist()

    return time_diff


############### MATCHING CRITERIA FOR SPECIFIC QUANTITIES #####
def onset_peak_criteria(sphinx, fcast, obs_values, observation_objs, energy_key):
    """ Calculate the boolean arrays that are needed to perform
        matching of onset peak (peak_intensity)
        
        Load information into sphinx object.
        
    Input:
    
        :sphinx: (SPHINX Object) initialized and contains observation
            forecast information
        :fcast: (Forecast Object) a specific forecast
        :obs_values: (pandas Dataframe) for one energy channel and
            all thresholds
        :observation_objs: (array of Observation objects) all
            observations for a specific energy channel
            
    Ouput:
        
        Output is set as values in sphinx object
        
    """
    last_trigger_time = sphinx.last_trigger_time
    last_input_time = sphinx.last_input_time
    
    ###### MATCHING: ONSET PEAK IN PREDICTION WINDOW #####
    #Is the onset peak inside of the prediction window?
    #Get the time difference - negative means trigger is before
    is_onset_peak_in_pred_win = observed_time_in_pred_win(fcast,
                                obs_values, 'peak_time', energy_key)

    if sphinx.overlapping_indices:
        for ix in sphinx.overlapping_indices:
            sphinx.peak_intensity_time_in_prediction_window.append(is_onset_peak_in_pred_win[ix])
            #logging
            logger.debug("Stored Peak Intensity Time:  " +
            str(sphinx.peak_intensity_time_in_prediction_window))
            logger.debug("  Observed peak_intensity time: "
            + str(observation_objs[ix].peak_intensity.time))


    ###### MATCHING: TRIGGERS BEFORE ONSET PEAK #####
    #Is the last trigger time before the onset peak time?
    #Get the time difference - negative means trigger is before
    td_trigger_onset_peak = time_diff(last_trigger_time, obs_values,
                                'peak_time', energy_key)
    is_trigger_before_onset_peak = is_time_before(last_trigger_time,
                                obs_values, 'peak_time', energy_key)

    if sphinx.overlapping_indices:
        for ix in sphinx.overlapping_indices:
            sphinx.triggers_before_peak_intensity.append(is_trigger_before_onset_peak[ix])
            sphinx.time_difference_triggers_peak_intensity.append(td_trigger_onset_peak[ix])
                #logging
            logger.debug("Triggers before Peak Time and time difference:  " +
                str(sphinx.triggers_before_peak_intensity[-1]))
            logger.debug("  " +
                str(sphinx.time_difference_triggers_peak_intensity[-1]))
            logger.debug("  Observed peak_intensity time: "
                + str(observation_objs[ix].peak_intensity.time))


    ###### MATCHING: INPUTS BEFORE ONSET PEAK #####
    #Is the last trigger time before the onset peak time?
    td_input_onset_peak = time_diff(last_input_time, obs_values,
                            'peak_time', energy_key)
    is_input_before_onset_peak = is_time_before(last_input_time,
                            obs_values, 'peak_time', energy_key)

    if sphinx.overlapping_indices:
        for ix in sphinx.overlapping_indices:
            sphinx.inputs_before_peak_intensity.append(is_input_before_onset_peak[ix])
            sphinx.time_difference_inputs_peak_intensity.append(td_input_onset_peak[ix])
                #logging
            logger.debug("Inputs before Peak Intensity Time and time difference  " +
                str(sphinx.inputs_before_peak_intensity[-1]))
            logger.debug("  " +
                str(sphinx.time_difference_inputs_peak_intensity[-1]))
            logger.debug("  Observed peak_intensity time: "
                + str(observation_objs[ix].peak_intensity.time))


    return



def max_flux_criteria(sphinx, fcast, obs_values, observation_objs, energy_key):
    """ Calculate the boolean arrays that are needed to perform
        matching of max flux (peak_intensity_max)
        
        Load information into sphinx object.
        
    Input:
    
        :sphinx: (SPHINX Object) initialized and contains observation
            forecast information
        :fcast: (Forecast Object) a specific forecast
        :obs_values: (pandas Dataframe) for one energy channel and
            all thresholds
        :observation_objs: (array of Observation objects) all
            observations for a specific energy channel
            
    Ouput:
        
        All output saved sphinx object

        
    """
    last_trigger_time = sphinx.last_trigger_time
    last_input_time = sphinx.last_input_time
    
    ###### MATCHING: MAX FLUX IN PREDICTION WINDOW #####
    #Is the max flux inside of the prediction window?
    #Get the time difference - negative means trigger is before
    is_max_flux_in_pred_win = observed_time_in_pred_win(fcast,
                            obs_values, 'max_time', energy_key)

    if sphinx.overlapping_indices:
        for ix in sphinx.overlapping_indices:
            sphinx.peak_intensity_max_time_in_prediction_window.append(is_max_flux_in_pred_win[ix])
                #logging
            logger.debug("Stored Max Flux Time:  " +
                str(sphinx.peak_intensity_max_time_in_prediction_window[-1]))
            logger.debug("  Observed peak_intensity_max time: " +
                str(observation_objs[ix].peak_intensity_max.time))


    ###### MATCHING: TRIGGERS BEFORE MAX FLUX #####
    #Is the last trigger time before the max flux time?
    #Get the time difference - negative means trigger is before
    td_trigger_max_time = time_diff(last_trigger_time, obs_values,
                            'max_time', energy_key)
    is_trigger_before_max_time = is_time_before(last_trigger_time,
                            obs_values, 'max_time', energy_key)

    if sphinx.overlapping_indices:
        for ix in sphinx.overlapping_indices:
            sphinx.triggers_before_peak_intensity_max.append(is_trigger_before_max_time[ix])
            sphinx.time_difference_triggers_peak_intensity_max.append(td_trigger_max_time[ix])
                #logging
            logger.debug("Triggers before Max Intensity and Time Difference:  " +
                str(sphinx.triggers_before_peak_intensity_max[-1]))
            logger.debug("  " +
                str(sphinx.time_difference_triggers_peak_intensity_max[-1]))
            logger.debug("  Observed peak_intensity_max time: "
                + str(observation_objs[ix].peak_intensity.time))


    ###### MATCHING: INPUTS BEFORE MAX FLUX TIME #####
    #Is the last trigger time before the max flux time?
    td_input_max_time = time_diff(last_input_time, obs_values,
                        'max_time', energy_key)
    is_input_before_max_time = is_time_before(last_input_time,
                        obs_values, 'max_time', energy_key)

    if sphinx.overlapping_indices:
        for ix in sphinx.overlapping_indices:
            sphinx.inputs_before_peak_intensity_max.append(is_input_before_max_time[ix])
            sphinx.time_difference_inputs_peak_intensity_max.append(td_input_max_time[ix])
                #logging
            logger.debug("Inputs before Max Intensity Time and Time Difference:  " +
                str(sphinx.inputs_before_peak_intensity_max[-1]))
            logger.debug("  " +
                str(sphinx.time_difference_inputs_peak_intensity_max[-1]))
            logger.debug("  Observed peak_intensity time: "
                + str(observation_objs[ix].peak_intensity_max.time))

    return



def threshold_cross_criteria(sphinx, fcast, obs_values, observation_objs,
        energy_key, thresh):
    """ Calculate the boolean arrays to indicate if a threshold is
        crossed inside the prediction window.
        
        Load information into sphinx object.
        
    Input:
    
        :sphinx: (SPHINX Object) initialized and contains observation
            forecast information
        :fcast: (Forecast Object) a specific forecast
        :obs_values: (pandas Dataframe) for one energy channel and
            all thresholds
        :observation_objs: (array of Observation objects) all
            observations for a specific energy channel
        :thresh: (dict) specific forecast threshold for threshold crossing
            
    Ouput:
        
        None
        
    """
    thresh_key = objh.threshold_to_key(thresh)
    
    contains_thresh_cross, crossing_times = observed_time_in_pred_win_thresh(fcast,
        obs_values, 'threshold_crossing_time', energy_key, thresh)

    if sphinx.overlapping_indices:
        for ix in sphinx.overlapping_indices:
            sphinx.threshold_crossed_in_pred_win[thresh_key].append(contains_thresh_cross[ix])
            sphinx.all_threshold_crossing_times[thresh_key].append(crossing_times[ix])
            #logging
            logger.debug("Threshold crossed in prediction window?:  " +
                str(sphinx.threshold_crossed_in_pred_win[thresh_key][-1]))


    return



def before_threshold_crossing(sphinx, fcast, obs_values, observation_objs,
        energy_key, thresh):
    """ Calculate the boolean arrays to indicate if the triggers and inputs
        occurred prior to a threshold crossing.
        
        Load information into sphinx object.
        
    Input:
    
        :sphinx: (SPHINX Object) initialized and contains observation
            forecast information
        :fcast: (Forecast Object) a specific forecast
        :obs_values: (pandas Dataframe) for one energy channel and
            all thresholds
        :observation_objs: (array of Observation objects) all
            observations for a specific energy channel
        :thresh: (dict) specific threshold for threshold crossing
            
    Ouput:
        
        None


    """
    last_trigger_time = sphinx.last_trigger_time
    last_input_time = sphinx.last_input_time
    thresh_key = objh.threshold_to_key(thresh)


    ###### MATCHING: TRIGGERS/INPUTS BEFORE SEP #####
    td_trigger_thresh_cross = time_diff_thresh(last_trigger_time, obs_values,
        'threshold_crossing_time', energy_key, thresh)
    is_trigger_before_start = is_time_before_thresh(last_trigger_time,
        obs_values, 'threshold_crossing_time', energy_key, thresh)

    if sphinx.overlapping_indices:
        for ix in sphinx.overlapping_indices:
            sphinx.triggers_before_threshold_crossing[thresh_key].append(is_trigger_before_start[ix])
            sphinx.time_difference_triggers_threshold_crossing[thresh_key].append(td_trigger_thresh_cross[ix])
            #logging
            logger.debug("Triggers before threshold crossing and time difference:  " +
                str(sphinx.triggers_before_threshold_crossing[thresh_key][-1]))
            logger.debug("  " +
                str(sphinx.time_difference_triggers_threshold_crossing[thresh_key][-1]))
            threshold_crossing_time = \
            objh.get_threshold_crossing_time(observation_objs[ix],
                thresh)
            logger.debug("  Observed threshold crossing time: "
                + str(threshold_crossing_time))


    ###### MATCHING: INPUTS BEFORE SEP #####
    #Is the last input before the SEP event start time?
    td_input_thresh_cross = time_diff_thresh(last_input_time, obs_values,
        'threshold_crossing_time', energy_key, thresh)
    is_input_before_start = is_time_before_thresh(last_input_time, obs_values,
        'threshold_crossing_time', energy_key, thresh)

    if sphinx.overlapping_indices:
        for ix in sphinx.overlapping_indices:
            sphinx.inputs_before_threshold_crossing[thresh_key].append(is_input_before_start[ix])
            sphinx.time_difference_inputs_threshold_crossing[thresh_key].append(td_input_thresh_cross[ix])
            #logging
            logger.debug("Inputs before threshold crossing time and time difference:  " \
                + str(sphinx.inputs_before_threshold_crossing[thresh_key][-1]))
            logger.debug("  " +
                str(sphinx.time_difference_inputs_threshold_crossing[thresh_key][-1]))
            threshold_crossing_time =\
                objh.get_threshold_crossing_time(observation_objs[ix],
                thresh)
            logger.debug("  Observed threshold crossing time: "
                + str(threshold_crossing_time))

    return
 
 
 
def before_sep_end(sphinx, fcast, obs_values, observation_objs,
        energy_key, thresh):
    """ Calculate the boolean arrays to indicate if the triggers and inputs
        occurred prior to the end of an SEP event.
        
        Load information into sphinx object.
        
    Input:
    
        :sphinx: (SPHINX Object) initialized and contains observation
            forecast information
        :fcast: (Forecast Object) a specific forecast
        :obs_values: (pandas Dataframe) for one energy channel and
            all thresholds
        :observation_objs: (array of Observation objects) all
            observations for a specific energy channel
        :thresh: (dict) specific threshold for threshold crossing
            
    Ouput:
        
        None
 
    """
    last_trigger_time = sphinx.last_trigger_time
    last_input_time = sphinx.last_input_time
    thresh_key = objh.threshold_to_key(thresh)


    ###### MATCHING: TRIGGERS/INPUTS BEFORE SEP #####
    td_trigger_sep_end = time_diff_thresh(last_trigger_time, obs_values,
        'end_time', energy_key, thresh)
    is_trigger_before_end = is_time_before_thresh(last_trigger_time,
        obs_values, 'end_time', energy_key, thresh)

    if sphinx.overlapping_indices:
        for ix in sphinx.overlapping_indices:
            sphinx.triggers_before_sep_end[thresh_key].append(is_trigger_before_end[ix])
            sphinx.time_difference_triggers_sep_end[thresh_key].append(td_trigger_sep_end[ix])
            #logging
            logger.debug("Triggers before SEP end and time difference:  " +
                str(sphinx.triggers_before_sep_end[thresh_key][-1]))
            logger.debug("  " +
                str(sphinx.time_difference_triggers_sep_end[thresh_key][-1]))


    ###### MATCHING: INPUTS BEFORE SEP #####
    #Is the last input before the SEP event start time?
    td_input_sep_end = time_diff_thresh(last_input_time, obs_values,
        'end_time', energy_key, thresh)
    is_input_before_end = is_time_before_thresh(last_input_time, obs_values,
        'end_time', energy_key, thresh)

    if sphinx.overlapping_indices:
        for ix in sphinx.overlapping_indices:
            sphinx.inputs_before_sep_end[thresh_key].append(is_input_before_end[ix])
            sphinx.time_difference_inputs_sep_end[thresh_key].append(td_input_sep_end[ix])
            #logging
            logger.debug("Inputs before SEP end and time difference:  " \
                + str(sphinx.inputs_before_sep_end[thresh_key][-1]))
            logger.debug("  " +
                str(sphinx.time_difference_inputs_sep_end[thresh_key][-1]))

    return
 
 


def eruption_before_threshold_crossing(sphinx, fcast, obs_values,
    observation_objs, energy_key, thresh):
    """ Calculate the boolean arrays to indicate if the eruption (flare/CME)
        occurs prior to the threshold crossing.
        
        Load information into sphinx object.
        
    Input:
    
        :sphinx: (SPHINX Object) initialized and contains observation
            forecast information
        :fcast: (Forecast Object) a specific forecast
        :obs_values: (pandas Dataframe) for one energy channel and
            all thresholds
        :observation_objs: (array of Observation objects) all
            observations for a specific energy channel
        :thresh: (dict) specific threshold for threshold crossing
            
    Ouput:
        
        None
 
    """
    last_eruption_time = sphinx.last_eruption_time
    thresh_key = objh.threshold_to_key(thresh)


    ###### MATCHING: ERUPTION BEFORE SEP #####
    td_eruption_thresh_cross = time_diff_thresh(last_eruption_time, obs_values,
        'threshold_crossing_time', energy_key, thresh)
    is_eruption_before_start = is_time_before_thresh(last_eruption_time,
        obs_values, 'threshold_crossing_time', energy_key, thresh)

    if sphinx.overlapping_indices:
        for ix in sphinx.overlapping_indices:
            sphinx.eruptions_before_threshold_crossing[thresh_key].append(is_eruption_before_start[ix])
            sphinx.time_difference_eruptions_threshold_crossing[thresh_key].append(td_eruption_thresh_cross[ix])
            #logging
            logger.debug("Eruption before SEP crossing time and time difference:  " +
                str(sphinx.eruptions_before_threshold_crossing[thresh_key][-1]))
            logger.debug("  " +
                str(sphinx.time_difference_eruptions_threshold_crossing[thresh_key][-1]))
            threshold_crossing_time = \
                objh.get_threshold_crossing_time(observation_objs[ix],
                thresh)
            logger.debug("  Observed threshold crossing time: "
                + str(threshold_crossing_time))

    return



def eruption_in_range(td_eruption_thresh_cross):
    """ Determine if the eruption (CME/flare) is in the appropriate
        range to be responsible for an observed SEP event.
        
        Require CME/flare to occur a few mins to 24 hours before an
        observed SEP event to be considered associated with that event.
    
    Input:
    
        :td_eruption_thresh_cross: (timedelta) difference in time between
            the eruption and threshold crossing for a single observation
            
    Output:
    
        :is_eruption_in_range: (bool) None if no threshold crossing,
            True if in range, False if outside of range or if there was
            a better eruption matched to the SEP event
    
    """
    is_eruption_in_range = None
    if not pd.isnull(td_eruption_thresh_cross):
        if td_eruption_thresh_cross <= -0.15\
            and td_eruption_thresh_cross > -24.:
            is_eruption_in_range = True
            
        if td_eruption_thresh_cross > -0.15\
            or td_eruption_thresh_cross <= -24.:
            is_eruption_in_range = False

    return is_eruption_in_range



def last_before_start(is_trigger_before_start, is_input_before_start):
    """ Determines whether the last trigger and/or input is before
        the start of an SEP event.
        
    Input:
    
        :is_trigger_before_start: (bool)
        :is_input_before_start: (bool)
        
    Output:
    
        :trigger_input_start: (bool) None if no SEP event, True if both
            before start, False is after start
            
    """

    trigger_input_start = None
    if is_trigger_before_start is not None:
        trigger_input_start = is_trigger_before_start
    if is_input_before_start is not None:
        if trigger_input_start is None:
            trigger_input_start = is_input_before_start
        else:
            trigger_input_start = trigger_input_start and \
                is_trigger_before_start

    return trigger_input_start



def last_before_end(is_trigger_before_end, is_input_before_end):
    """ Determines whether the last trigger and/or input is before
        the end of an SEP event.
        
    Input:
    
        :is_trigger_before_end: (bool)
        :is_input_before_end: (bool)
        
    Output:
    
        :trigger_input_end: (bool) None if no SEP event, True if both
            before end, False is after end
            
    """
    trigger_input_end = None
    if is_trigger_before_end is not None:
        trigger_input_end = is_trigger_before_end
    if is_input_before_end is not None:
        if trigger_input_end is None:
            trigger_input_end = is_input_before_end
        else:
            trigger_input_end = trigger_input_end and \
                is_trigger_before_end

    return trigger_input_end



def pred_win_sep_overlap(sphinx, fcast, obs_values, observation_objs,
        energy_key, threshold):
    """ Calculate the boolean arrays that indicate if an
        ongoing SEP event overlaps with the prediction window.
        
        Load information into sphinx object.
        
    Input:
    
        :sphinx: (SPHINX Object) initialized and contains observation
            forecast information
        :fcast: (Forecast Object) a specific forecast
        :obs_values: (pandas Dataframe) for one energy channel and
            all thresholds
        :observation_objs: (array of Observation objects) all
            observations for a specific energy channel
        :threshold: (dict) forecast {'threshold':10, 'units': Unit("MeV")}
            
    Ouput:
        
        None
            
    """
    
    pred_win_st = fcast.prediction_window_start
    pred_win_end = fcast.prediction_window_end
    thresh_key = objh.threshold_to_key(threshold)

    #Return if no prediction
    if pd.isnull(fcast.prediction_window_start):
        return [None]*len(obs_values[energy_key]['dataframes'][0])
    #Extract desired threshold
    obs_thresholds = obs_values[energy_key]['thresholds']
    try:
        ix = obs_thresholds.index(threshold)
    except:
        return [None]*len(obs_values[energy_key]['dataframes'][0])

    #Check if time is before
    obs = obs_values[energy_key]['dataframes'][ix]
    
    #Check if an observed SEP event is within any part of the prediction window
    #Prescribed logic includes prediction windows that span the observed SEP start,
    #that are fully contained within the observed SEP event, and span the
    #observed SEP end.
    overlap_start = (obs['start_time'] >= pd.Timestamp(pred_win_st))\
        & (obs['start_time'] <= pd.Timestamp(pred_win_end))
    overlap_end = (obs['end_time'] >= pd.Timestamp(pred_win_st))\
        & (obs['start_time'] <= pd.Timestamp(pred_win_end))

    is_overlap = overlap_start | overlap_end
    is_overlap = list(is_overlap)
    
    if sphinx.overlapping_indices:
        for ix in sphinx.overlapping_indices:
            sphinx.prediction_window_sep_overlap[thresh_key].append(is_overlap[ix])


    return


# observation_objs is an unnecessary input for this function!
def observed_ongoing_event(sphinx, fcast, obs_values, observation_objs,
        energy_key, threshold):
    """ Calculate the boolean arrays that indicate if there is an
        ongoing SEP event at the start of the prediction window.
        
        Load information into sphinx object.
        
    Input:
    
        :sphinx: (SPHINX Object) initialized and contains observation
            forecast information
        :fcast: (Forecast Object) a specific forecast
        :obs_values: (pandas Dataframe) for one energy channel and
            all thresholds
        :observation_objs: (array of Observation objects) all
            observations for a specific energy channel
        :threshold: (dict) {'threshold':10, 'units': Unit("MeV")}
            
    Ouput:
        
        None
            
    """
    pred_win_st = fcast.prediction_window_start
    pred_win_end = fcast.prediction_window_end
    thresh_key = objh.threshold_to_key(threshold)

    #Return if no prediction
    if pd.isnull(fcast.prediction_window_start):
        return [None]*len(obs_values[energy_key]['dataframes'][0])
    #Extract desired threshold
    obs_thresholds = obs_values[energy_key]['thresholds']
    try:
        ix = obs_thresholds.index(threshold)
    except:
        return [None]*len(obs_values[energy_key]['dataframes'][0])

    #Check if time is before
    obs = obs_values[energy_key]['dataframes'][ix]
    
    #Check if prediction window starts inside of an observed SEP event
    
    
    pred_win_begin = pd.Interval(pd.Timestamp(pred_win_st),
                        pd.Timestamp(pred_win_st))
    
    is_ongoing = []
    sep_start = pd.NaT
    sep_end = pd.NaT
    logger.debug("Ongoing observed SEP event at start of prediction window (if any):")
    for i in range(len(obs['start_time'])):
        if pd.isnull(obs['start_time'][i]):
            is_ongoing.append(None)
            continue
        else:
            sep_event = pd.Interval(pd.Timestamp(obs['start_time'][i]),
                            pd.Timestamp(obs['end_time'][i]))

        #Does the prediction window start inside of an SEP event?
        if sep_event.overlaps(pred_win_begin):
            is_ongoing.append(True)
            sep_start = obs['start_time'][i]
            sep_end = obs['end_time'][i]

            #logging
            logger.debug("  " + str(observation_objs[i].source))
            logger.debug("  Observed SEP event: "
                + str(sep_start) + " to " + str(sep_end))
            
        else:
            is_ongoing.append(False)

    if sphinx.overlapping_indices:
        for ix in sphinx.overlapping_indices:
            sphinx.observed_ongoing_events[thresh_key].append(is_ongoing[ix])

    return



###### MATCHING AND EXTRACTING OBSERVED VALUES ########
def match_observed_onset_peak(sphinx, observation_obj, is_win_overlap,
    is_eruption_in_range, is_trigger_before_onset_peak,
    is_input_before_onset_peak, is_pred_sep_overlap):
    """ Apply criteria to determine if a particular observation matches
        with the peak intensity prediction.
        If identified, save the observed peak_intensity to the SPHINX object.
       
        - Prediction window overlaps with observation
        - Last eruption within specific range before threshold crossing
        - The prediction window overlaps with an SEP event in any threshold -
            only a comparison when there is an SEP event
            NEED TO ADD IN COMPARISON WHEN NO SEP EVENT
        - The last trigger/input time is before the observed peak intensity

    Input:
        
        The ith entry for the various boolean arrays created that are the length
        of all of the observations read into the code.
        
    Output:
    
        :peak_criteria: (bool) a single boolean indicating whether a
            match was made for a specific observation and threshold
        
        The SPHINX object is updated with observed values if peak_criteria
        is found to be true.

    """
    #Already identified SEP event in a matched observation. Don't overwrite.
    if sphinx.peak_intensity_match_status == "SEP Event":
        return None
    

    #ONSET PEAK
    #Both triggers and inputs before onset peak time
    trigger_input_peak = None #None if no SEP event
    if is_trigger_before_onset_peak is not None:
        trigger_input_peak = is_trigger_before_onset_peak
    if is_input_before_onset_peak is not None:
        if trigger_input_peak is None:
            trigger_input_peak = is_input_before_onset_peak
        else:
            trigger_input_peak = trigger_input_peak and \
                is_input_before_onset_peak
    
    #MATCHING
    peak_criteria = is_win_overlap and trigger_input_peak and is_pred_sep_overlap
    if is_eruption_in_range is not None:
        peak_criteria = (peak_criteria and is_eruption_in_range)
    
    if not peak_criteria:
        if not is_win_overlap: # this code is never touched! -- LS-code-comment
            sphinx.peak_intensity_match_status = "No Matched Observation"
        if not is_pred_sep_overlap:
            sphinx.peak_intensity_match_status = "No SEP Event"
        if is_eruption_in_range is not None and not is_eruption_in_range:
            sphinx.peak_intensity_match_status = "Eruption Out of Range"
        if trigger_input_peak is not None and not trigger_input_peak: #precedence
            sphinx.peak_intensity_match_status = "Trigger/Input after Observed Phenomenon"
 
 
    if peak_criteria:
        #logging
        logger.debug("Observed peak_intensity matched:")
        logger.debug(observation_obj.source)
        logger.debug(observation_obj.peak_intensity.intensity)
        logger.debug(observation_obj.peak_intensity.time)
        
        sphinx.observed_match_peak_intensity_source = observation_obj.source
        sphinx.observed_peak_intensity = observation_obj.peak_intensity
        sphinx.peak_intensity_match_status = "SEP Event"

    return peak_criteria


def match_observed_max_flux(sphinx, observation_obj, is_win_overlap,
    is_eruption_in_range, is_trigger_before_max_time,
    is_input_before_max_time, is_pred_sep_overlap):
    """ Apply criteria to determine if a particular observation matches
        with the maximum flux (peak_intensity_max) prediction.
        If identified, save the observed peak_intensity_max to the SPHINX object.
       
        - Prediction window overlaps with observation
        - Last eruption within specified range before threshold crossing
        - The prediction window overlaps with an SEP event in any threshold -
            only a comparison when there is an SEP event
            NEED TO ADD IN COMPARISON WHEN NO SEP EVENT
        - The last trigger/input time if before the observed peak intensity

    Input:
        
        The ith entry for the various boolean arrays created that are the length
        of all of the observations read into the code.

    Output:
    
        :max_criteria: (bool) a single boolean indicating whether a
            match was made for a specific observation and threshold
        
        The SPHINX object is updated with observed values if max_criteria
        is found to be true.

    """
    
    #Already identified SEP event in a matched observation. Don't overwrite.
    if sphinx.peak_intensity_max_match_status == "SEP Event":
        return None
    
    #Both triggers and inputs before max flux time
    trigger_input_max = None #None if no SEP event
    if is_trigger_before_max_time is not None:
        trigger_input_max = is_trigger_before_max_time
    if is_input_before_max_time is not None:
        if trigger_input_max is None:
            trigger_input_max = is_input_before_max_time
        else:
            trigger_input_max = trigger_input_max and \
                is_input_before_max_time
    
    #MATCHING
    max_criteria = is_win_overlap and trigger_input_max and is_pred_sep_overlap
    if is_eruption_in_range is not None:
        max_criteria = (max_criteria and is_eruption_in_range)

    if not max_criteria:
        if not is_win_overlap: # this code is never touched! -- LS-code-comment
            sphinx.peak_intensity_max_match_status = "No Matched Observation"
        if not is_pred_sep_overlap:
            sphinx.peak_intensity_max_match_status = "No SEP Event"
        if is_eruption_in_range != None and not is_eruption_in_range:
            sphinx.peak_intensity_max_match_status = "Eruption Out of Range"
        if not trigger_input_max: #precedence
            sphinx.peak_intensity_max_match_status = "Trigger/Input after Observed Phenomenon"

    if max_criteria:
        sphinx.observed_match_peak_intensity_max_source =\
            observation_obj.source
        sphinx.observed_peak_intensity_max = observation_obj.peak_intensity_max
        sphinx.peak_intensity_max_match_status = "SEP Event"

    return max_criteria


def match_all_clear(sphinx, observation_obj, is_win_overlap,
    is_eruption_in_range, trigger_input_start, contains_thresh_cross,
    is_sep_ongoing):
    """ Apply criteria to determine the observed All Clear status for a
        particular forecast.
       
        - Prediction window overlaps with observation
        - There is no ongoing SEP event at the start of the prediction window
        - Last eruption within 24 hrs - a few mins before threshold crossing
        - The last trigger/input time if before the threshold crossing
        - Threshold crossed in prediction window = False All Clear
        - No threshold crossed in prediction window = True All Clear

    Input:
        
        The ith entry for the various boolean arrays that are the observations
            matched to the forecast.

    Output:
    
        :all_clear_status: (bool) a single boolean (or None) indicating the
            observed All Clear value
        
        The SPHINX object is updated with observed values if the value is
        found to be True or False. Otherwise, values of None will be skipped
        because it means that this particular observation doesn't match
        
    """
    #If it was already found that a threshold was crossed in a matched
    #observation and the all clear status was set to False, don't overwrite
    #with another observation later in the prediction window
    #Mainly relevant for long prediction windows > 24 - 48 hours
    if sphinx.observed_all_clear.all_clear_boolean == False: #stated explicitly
        return None
    
    #If already matched to an ongoing SEP event, ensure that a second
    #event in the prediction window doesn't overwrite the previous
    #match status
    if sphinx.all_clear_match_status == "Ongoing SEP Event":
        if is_eruption_in_range is not None:
            #If there's an eruption and it's not associated
            if not is_eruption_in_range:
                return None #nothing to be done
            #If there is an eruption in range with a second event in the prediction
            #window, then continue on with the logic
    
    #Save thresholds in All_Clear object
    ac = cl.All_Clear(None, observation_obj.all_clear.threshold,
        observation_obj.all_clear.threshold_units,
        observation_obj.all_clear.probability_threshold)
    sphinx.observed_all_clear = ac
    
    all_clear_status = None
    
    if not is_win_overlap: # this code is never touched! -- LS-code-comment
        all_clear_status = None
        sphinx.all_clear_match_status = "No Matched Observation"
        return all_clear_status
        
    #Prediction and observation windows overlap
    #If ongoing SEP event at start of prediction window, no match
    if is_sep_ongoing: #can be True, False, None
        all_clear_status = None
        sphinx.observed_match_all_clear_source = observation_obj.source
        sphinx.all_clear_match_status = "Ongoing SEP Event"
        return all_clear_status
    
    #If there is no threshold crossing in prediction window,
    #then observed all clear is True
    if not contains_thresh_cross: #can only be True or False
        sphinx.all_clear_match_status = "No SEP Event"
        all_clear_status = True
    
    #If there is a threshold crossing in the prediction window
    if contains_thresh_cross:
        #The triggers and inputs must all be before threshold crossing
        if trigger_input_start:
            #Observed all clear is False
            sphinx.all_clear_match_status = "SEP Event"
            all_clear_status = False
        else:
            all_clear_status = None
            sphinx.all_clear_match_status = "Trigger/Input after Observed Phenomenon"
            return all_clear_status

        #The eruption must occur in the right time range
        if is_eruption_in_range is not None:
            if not is_eruption_in_range:
                all_clear_status = True
                sphinx.observed_match_all_clear_source = observation_obj.source
                sphinx.all_clear_match_status = "Eruption Out of Range"
    #logging
    logger.debug("Prediction window: " + str(sphinx.prediction.prediction_window_start)
        + " to " + str(sphinx.prediction.prediction_window_end))
    #All clear status
    logger.debug("Observed all_clear matched:")
    logger.debug("  " + observation_obj.source)
    logger.debug("  " + str(all_clear_status))

    sphinx.observed_match_all_clear_source = observation_obj.source
    sphinx.observed_all_clear.all_clear_boolean = all_clear_status

    return all_clear_status



def match_sep_quantities(sphinx, observation_obj, thresh, is_win_overlap,
    is_eruption_in_range, trigger_input_start, contains_thresh_cross,
    is_sep_ongoing):
    """ Apply criteria to determine if a forecast occurs prior to SEP
        start time and extract all relevant SEP quantities.
        
        The SEP quantities returned are all related to the start time of
        the SEP event. This requires that the forecast come in prior to the
        threshold crossing.
        - start time, threshold crossing time, probability
       
       Matching:
        - Prediction window overlaps with observation
        - There is no ongoing SEP event at the start of the prediction window
        - Last eruption within specified time range before threshold crossing
        - The last trigger/input time is before the threshold crossing

    Input:
        
        :sphinx: (SPHINX object) will be updated
        :observation_obj: A single observation object
        :thresh: (dict) threshold being tested
        
        The rest are the ith entry for the various boolean arrays
        that are the length of all of the observations read into the code.

    Output:
    
        :sep_status: (bool) a single boolean (or None) indicating the
            whether SEP info was saved to the SPHINX object
            True - SEP quantities added to SPHINX
            False - no SEP event observed or an SEP is already ongoing
            None - the observation isn't associated with the forecast
        
        The SPHINX object is updated with observed values if the value is
        found to be True or False. Otherwise, values of None will be skipped
        because it means that this particular observation doesn't match
        
    """
    thresh_key = objh.threshold_to_key(thresh)
    
    #If it was already found that a threshold was crossed in a matched
    #observation and the sep match status was set to True, don't overwrite
    #with another observation later in the prediction window
    #Mainly relevant for long prediction windows > 24 - 48 hours
    if sphinx.sep_match_status[thresh_key] == 'SEP Event':
        return None

    sep_status = None
    
    #Appropriate observed probability for prediction window, etc
    prob = cl.Probability(None, 0.0, thresh['threshold'],
            thresh['threshold_units'])
    
    if not is_win_overlap: # this code is never touched! -- LS-code-comment
                           # it gets touched in the case that no observations are prepared
                           # for the forecast prediction windows, as I proved by mistake
        sep_status = None
        sphinx.sep_match_status[thresh_key] = "No Matched Observation"
        return sep_status
        
    #Prediction and observation windows overlap
    #If ongoing SEP event at start of prediction window, no match
    if is_sep_ongoing:
        sep_status = False
        prob.probability_value = np.nan #ongoing event
        sphinx.observed_probability[thresh_key] = prob
        sphinx.observed_probability_source[thresh_key] =\
            observation_obj.source
        sphinx.sep_match_status[thresh_key] = "Ongoing SEP Event"
        
        return sep_status
    
    #No threshold crossing in prediction window, no SEP event
    if not contains_thresh_cross:
        sep_status = False
        prob.probability_value = 0.0
        sphinx.observed_probability[thresh_key] = prob
        sphinx.observed_probability_source[thresh_key] =\
            observation_obj.source
        sphinx.sep_match_status[thresh_key] = "No SEP Event"
        return sep_status
    
    #If there is a threshold crossing in the prediction window
    if contains_thresh_cross:
        #The triggers and inputs must all be before threshold crossing
        if trigger_input_start:
            sep_status = True
            prob.probability_value = 1.0
            sphinx.observed_probability[thresh_key] = prob
            sphinx.observed_probability_source[thresh_key] =\
                observation_obj.source
            sphinx.sep_match_status[thresh_key] = "SEP Event"
        else:
            sep_status = None
            prob.probability_value = np.nan
            sphinx.observed_probability[thresh_key] = prob
            sphinx.observed_probability_source[thresh_key] =\
                observation_obj.source
            sphinx.sep_match_status[thresh_key] = "Trigger/Input after Observed Phenomenon"
            return sep_status
 
        #The eruption must occur in the right time range
        if is_eruption_in_range != None:
            if not is_eruption_in_range:
                sep_status = None
                prob.probability_value = 0.0
                sphinx.observed_probability[thresh_key] = prob
                sphinx.observed_probability_source[thresh_key] =\
                    observation_obj.source
                sphinx.sep_match_status[thresh_key] = "Eruption Out of Range"
                return sep_status

    
    #logging
    logger.debug("Prediction window: " + str(sphinx.prediction.prediction_window_start)
        + " to " + str(sphinx.prediction.prediction_window_end))
    #All clear status
    logger.debug("Observed SEP event matched:")
    logger.debug("  " + observation_obj.source)
    
    #FOR IDENTIFIED SEP EVENTS;  match status = "SEP Event"
    #Threshold Crossing
    threshold_crossing_time = pd.NaT
    for th in observation_obj.threshold_crossings:
        if th.threshold != thresh['threshold']:
            continue
        sphinx.observed_match_sep_source[thresh_key] = observation_obj.source
        sphinx.observed_threshold_crossing[thresh_key] = th
 
 
    #Start time
    start_time = pd.NaT
    for event in observation_obj.event_lengths:
        if event.threshold != thresh['threshold']:
            continue
        sphinx.observed_match_sep_source[thresh_key] = observation_obj.source
        sphinx.observed_start_time[thresh_key] = event.start_time

    return sep_status



def match_sep_end_time(sphinx, observation_obj, thresh, is_win_overlap,
    is_eruption_in_range, trigger_input_end, is_pred_sep_overlap):
    """ Apply criteria to determine if a forecast occurs prior to SEP
        end time and extract observed end time.
               
       Matching:
        - Prediction window overlaps with observation
        - There is a threshold crossing in the prediction window OR
            there is an ongoing SEP event at the start of the prediction window
        - Last eruption within 48 hrs - 15 mins before threshold crossing
        - The last trigger/input time if before the end time

    Input:
        
        :sphinx: (SPHINX object) will be updated
        :observation_obj: A single observation object
        :thresh: (dict) threshold being tested
        
        The rest are the ith entry for the various boolean arrays
        that are the length of all of the observations read into the code.

    Output:
    
        :sep_status: (bool) a single boolean (or None) indicating the
            whether SEP info was saved to the SPHINX object
            True - SEP quantities added to SPHINX
            False - no SEP event observed or an SEP is already ongoing
            None - the observation isn't associated with the forecast
        
        The SPHINX object is updated with observed values if the value is
        found to be True or False. Otherwise, values of None will be skipped
        because it means that this particular observation doesn't match
        
    """
    thresh_key = objh.threshold_to_key(thresh)
    
    #If it was already found that end time had a matched
    #observation and the sep match status was set to True, don't overwrite
    #with another observation later in the prediction window
    #Mainly relevant for long prediction windows > 24 - 48 hours
    if sphinx.end_time_match_status[thresh_key] == "SEP Event":
        return None

    end_status = None
    
    #Prediction and observation windows must overlap
    if not is_win_overlap: # this code is never touched! -- LS-code-comment
        end_status = None
        sphinx.end_time_match_status[thresh_key] = "No Matched Observation"
        return end_status
        
    #The prediction window must overlap with an SEP event
    if not is_pred_sep_overlap:
        end_status = False #no SEP event, no values
        sphinx.end_time_match_status[thresh_key] = "No SEP Event"
        return end_status
    
    #The triggers and inputs must all be before threshold crossing
    if trigger_input_end:
        end_status = True
        sphinx.end_time_match_status[thresh_key] = "SEP Event"
    else:
        end_status = None
        sphinx.end_time_match_status[thresh_key] = "Trigger/Input after Observed Phenomenon"
        return end_status
    
    #If there is an SEP event, the eruption must occur in the right time range
    if is_eruption_in_range is not None:
        if not is_eruption_in_range:
            sep_status = None
            sphinx.end_time_match_status[thresh_key] = "Eruption Out of Range"
            return sep_status
    
    #Matched End Time
    logger.debug("Observed SEP end time matched:")
    logger.debug("  " + observation_obj.source)
 
    #Start time and channel fluence and duraction
    end_time = pd.NaT
    for i in range(len(observation_obj.event_lengths)):
        event = observation_obj.event_lengths[i]
        if event.threshold != thresh['threshold']:
            continue
        sphinx.observed_match_sep_source[thresh_key] = observation_obj.source
        sphinx.observed_event_length[thresh_key] = event
        sphinx.observed_end_time[thresh_key] = event.end_time
        sphinx.observed_time_profile[thresh_key] = observation_obj.source
        sphinx.observed_duration[thresh_key] =\
            (event.end_time - event.start_time).total_seconds()/(60.*60.)


    fluence = np.nan
    for fl in observation_obj.fluences:
        if fl.threshold != thresh['threshold']:
            continue
        sphinx.observed_fluence[thresh_key] = fl


    #Fluence spectra
    spectrum = None
    for flsp in observation_obj.fluence_spectra:
        if flsp.threshold_start != thresh['threshold']:
            continue
        sphinx.observed_fluence_spectrum[thresh_key] = flsp

    return end_status



######## DERIVED QUANTITIES ########
def read_in_time_profiles(sphinx):
    """ Read in the time profiles matched in the sphinx object.
        Return as a dataframe.
        
    """
    obs_time_profiles = sphinx.observed_sep_profiles
    all_dates = []
    all_fluxes = []
    for tp in obs_time_profiles: #Can be multiple obs files
        tp = tp.strip().split(",")
        for fnm in tp:
            dt, flx = profile.read_single_time_profile(fnm)
            if not dt: continue
            all_dates.append(dt)
            all_fluxes.append(flx)

    #Sort all of the fluxes in time order
    first_dates = [all_dates[k][0] for k in range(len(all_dates))]
    sortidx = sorted(range(len(first_dates)),
                    key = lambda k: first_dates[k])
    dict = {"Time": [], "Flux": []}
    for idx in sortidx:
        dict["Time"].extend(all_dates[idx])
        dict["Flux"].extend(all_fluxes[idx])
        
    tpdf = pd.DataFrame(dict)
    
    return tpdf


def get_max_in_pw(tpdf, pw_st, pw_end):
    """ Check the flux inside the prediction window and pull out
        the maximum value.
        
        INPUT:
        
            :tpdf: (pandas DataFrame) dataframe containing time and flux
                columns for each observed energy channel.
            :ek: (string) energy channel key
            :pw_st: (pandas datetime) prediction window start time
            :pw_end: (pandas datetime) prediction window end time
            
        OUTPUT:
        
            :max_flux: (float) maximum flux
            :max_flux_time: (pandas datetime) time of max flux
        
    """
    sub = tpdf
    sub = sub.loc[(sub["Time"] < pw_end) & (sub["Time"] >= pw_st)]
    if sub.empty: return np.nan, pd.NaT
    
    max_flux = sub["Flux"].max()
    times = sub.loc[(sub["Flux"] == max_flux)]
    times = times["Time"].to_list()
    max_flux_time = times[0]
    
    return max_flux, max_flux_time


def get_flux_at_point(tpdf, pw_st, pw_end, time):
    """ Check the flux inside the prediction window and calculate
        the observed flux at a specific point in time. Interpolate
        if necessary.
        
        INPUT:
        
            :tpdf: (pandas DataFrame) dataframe containing time and flux
                columns for each observed energy channel.
            :ek: (string) energy channel key
            :pw_st: (pandas datetime) prediction window start time
            :pw_end: (pandas datetime) prediction window end time
            :time: (pandas datetime) time to get observed flux
            
        OUTPUT:
        
            :obs_flux: (float) observed flux at time
        
    """
    obs_flux = np.nan
    
    if pw_st == pw_end: ###HACK FOR NOW
        pw_st = pw_st - datetime.timedelta(minutes=15)
        pw_end = pw_end + datetime.timedelta(minutes=15)
        
    sub = tpdf
    sub = sub.loc[(sub["Time"] < pw_end) & (sub["Time"] >= pw_st)]
    if sub.empty: return obs_flux
    
    dates = sub["Time"].to_list()
    fluxes = sub["Flux"].to_list()
    
    f_interp = profile.interp_timeseries(dates, fluxes, "log", [time])
    if f_interp:
        obs_flux = f_interp[0]

    return obs_flux



def calculate_derived_quantities(sphinx):
    """ Calculate observed quantities that are derived for comparison
        to forecasts.
            Max flux in prediction window
            Point intensity (observed flux at a specific point in time
                specified by the forecast)
                
        **This subroutine can only be called after the SPHINX object
        has been completely filled in with all forecasts and observations.
               
       Matching:
        - Matching doesn't play a role in these values, but the
            match status will be associated with the end time
            match status to indicate what is going on in the environment

    Input:
        
        :sphinx: (SPHINX object) will be updated

    Output:
    
        :status: (bool) a single boolean (or None) indicating the
            whether info was saved to the SPHINX object
            True - quantities added to SPHINX
            None - the observation isn't associated with the forecast
        
        The SPHINX object is updated with observed values if the value is
        found to be True or False. Otherwise, values of None will be skipped
        because it means that this particular observation doesn't match
        
    """
    status = None
    
    #Check if there are forecasts associated with these derived values
    pred_point_intensity = sphinx.prediction.point_intensity.intensity
    pred_peak_intensity = sphinx.prediction.peak_intensity.intensity
    pred_peak_intensity_max = sphinx.prediction.peak_intensity_max.intensity

    if pd.isnull(pred_point_intensity) and pd.isnull(pred_peak_intensity)\
        and pd.isnull(pred_peak_intensity_max):
        return status
    
    pw_st = sphinx.prediction.prediction_window_start
    pw_end = sphinx.prediction.prediction_window_end
    if pd.isnull(pw_st) or pd.isnull(pw_end): # this code is never touched! LS-code-comment
        return status
    
    #Predicted values to compare to, so continue
    status = True

    #Read in the observed time profiles into arrays
    tpdf = read_in_time_profiles(sphinx)

    #Max flux in prediction window
    max_flux, max_flux_time = get_max_in_pw(tpdf, pw_st, pw_end)
    sphinx.observed_max_flux_in_prediction_window.intensity = max_flux
    sphinx.observed_max_flux_in_prediction_window.time = max_flux_time
    sphinx.max_flux_in_prediction_window_match_status =\
        sphinx.end_time_match_status
    if not pd.isnull(pred_peak_intensity):
        sphinx.observed_max_flux_in_prediction_window.units = sphinx.prediction.peak_intensity.units
    if not pd.isnull(pred_peak_intensity_max):
        sphinx.observed_max_flux_in_prediction_window.units = sphinx.prediction.peak_intensity_max.units

    if pd.isnull(pred_point_intensity):
        return status
        
    point_time = sphinx.prediction.point_intensity.time
    obs_point_flux = get_flux_at_point(tpdf, pw_st, pw_end, point_time)
    sphinx.observed_point_intensity.intensity = obs_point_flux
    sphinx.observed_point_intensity.time = point_time
    sphinx.observed_point_intensity.units =\
        sphinx.prediction.point_intensity.units
    
    return status



def sep_report(all_energy_channels, obs_values, model_names,
    observed_sep_events):
    """ Print out all the observed SEP events inside of the forecast
        prediction windows for each energy channel and threshold combo.
        
    Input:
    
        :all_energy_channels: (array of dict) all energy channels
            found in the observations
        :obs_values: (pandas Dataframe) contains info for all observations.
            Needed to extract the thresholds.
        :model_names: (array of strings) array of all the model short names
        :observed_sep_events: (dict) all unique SEP events the were inside
            of a prediction window organized by model, energy channel, and
            threshold
            
    Output:
    
        None, prints to screen
        
    """
    logger.debug("\n====== All observed SEP events ======")
    for model in model_names:
        logger.debug("\n")
        logger.debug("Model: " + model)
        for energy_key in all_energy_channels:
            logger.debug("Energy Channel: " + str(energy_key))
            for obs_thresh in obs_values[energy_key]['thresholds']:
                logger.debug("Threshold: " + str(obs_thresh))
                obs_thresh_key = objh.threshold_to_key(obs_thresh)
                nev =\
                len(observed_sep_events[model][energy_key][obs_thresh_key])
                logger.debug(str(nev) + " SEP Events: " +
                str(observed_sep_events[model][energy_key][obs_thresh_key]))



def preferred_flare(flares):
    """ From a list of flares, choose the most likely flare to be
        associated with an SEP event. All most likely flares within 
        2 hours of the identified best will be returned. This accounts
        for possible refits/revisions of parameters for the same flare
        or that cannot determine the source of an SEP event if multiple
        large flares occur in quick succession.
        
        Apply educated guesses until 
        known observed flare+CME+SEP associations are incorporated.
        
        flare class >= M1
        flare longitude >= 20 degrees W
        
        If given the choice, the flares with the properties described
        here are more likely to be associated with a SEP event.
        Does not rule out that a flare with different properties can
        be the correct one.
        
        INPUT:
        
            :flares: (list of Flare objects) 
            
        OUTPUT:
        
            :best: (list of Flare objects) 1 or more preferred flares
            
    """

    best = []
    
    X = 10**-4
    M = 10**-5
    west_lon = 20
 
    #Check if one of the flares has a higher likelihood than the others
    #Check first for western X-class flares.
    best_flares = []
    td_diff = 1.5 #hour
    for flare in flares:
        if pd.isnull(flare.intensity) or pd.isnull(flare.lon):
            continue
        if flare.intensity >= X and flare.lon >= west_lon:
            best_flares.append(flare)
    
    #-----flares satisfy both criteria-------
    #If only one flare found to satisfy the conditions, return
    if len(best_flares) == 1:
        return best_flares
        
    #If multiple flares satified conditions, then choose the latest
    if len(best_flares) > 1:
        last = last_flare(best_flares)
        #Check if other strong flares are close together
        for flare in best_flares:
            if not pd.isnull(flare.start_time):
                diff = (flare.start_time - last.start_time).total_seconds()/3600
            elif not pd.isnull(flare.last_data_time):
                diff = (flare.last_data_time - last.last_data_time).total_seconds()/3600

            if abs(diff) <= td_diff:
                best.append(flare)
        
                
    #If a best flare identified, return
    if best:
        return best
 
    #If no X-class western flares, then check for X-class flares
    best_flares = []
    td_diff = 1.5 #hour
    for flare in flares:
        if pd.isnull(flare.intensity):
            continue
        if flare.intensity >= X:
            best_flares.append(flare)
    
    #-----flares satisfy both criteria-------
    #If only one flare found to satisfy the conditions, return
    if len(best_flares) == 1:
        return best_flares
        
    #If multiple flares satified conditions, then choose the latest
    if len(best_flares) > 1:
        last = last_flare(best_flares)
        #Check if other strong flares are close together
        for flare in best_flares:
            if not pd.isnull(flare.start_time):
                diff = (flare.start_time - last.start_time).total_seconds()/3600
            elif not pd.isnull(flare.last_data_time):
                diff = (flare.last_data_time - last.last_data_time).total_seconds()/3600

            if abs(diff) <= td_diff:
                best.append(flare)
        
                
    #If a best flare identified, return
    if best:
        return best


    #-----No X flares, so check for M-class flares-------
    #Choose the flare with intensity > M1.0
    if not best_flares:
        for flare in flares:
            if pd.isnull(flare.intensity): continue
            if flare.intensity >= M:
                best_flares.append(flare)
 
    #If only one flare found to satisfy the conditions, return
    if len(best_flares) == 1:
        return best_flares

    #If multiple flares satified conditions, then choose the latest
    if len(best_flares) > 1:
        last = last_flare(best_flares)
        #Check if other strong flares are close together
        for flare in best_flares:
            if not pd.isnull(flare.start_time):
                diff = (flare.start_time - last.start_time).total_seconds()/3600
            elif not pd.isnull(flare.last_data_time):
                diff = (flare.last_data_time - last.last_data_time).total_seconds()/3600

            if abs(diff) <= td_diff:
                best.append(flare)

    #If haven't identified a best flare (or best flares) by now,
    #revise_eruption_matches will choose the last flare.

    return best


    
def preferred_cme(CMEs):
    """ From a list of CMEs, choose the one most likely to be
        associated with an SEP event. All most likely CMEs within 
        2 hours of the identified best will be returned. This accounts
        for possible refits/revisions of parameters for the same CME
        or that cannot determine the source of an SEP event if multiple
        fast CMEs occur in quick succession.
        
        Apply educated guesses until 
        known observed flare+CME+SEP associations are incorporated.
        
        Referencing Clayton Allison's analysis of CMEs on the
        SEP Scoreboards, find that CMEs with the following 
        characteristics are more likely to be associated with SEP events.
        
        CME speed >= 600 km/s
        CME width >= 35 degrees

        If given the choice, the CME with the properties described
        here are more likely to be associated with a SEP event.
        Does not rule out that a CME with different properties can
        be the correct one.
        
        INPUT:
        
            :CMEs: (list of CME objects) 
            
        OUTPUT:
        
            :best: (list of CME objects) 1 or more preferred CMEs
            
    """
    best = []
    target_speed = 600
    target_width = 35
    
    #-----CMEs satisfy speed and width criteria -------
    best_cmes = []
    for cme in CMEs:
        if pd.isnull(cme.speed) or pd.isnull(cme.half_width):
            continue
        if cme.speed >= target_speed and cme.half_width >= target_width:
            best_cmes.append(cme)
            
    #If only one CME identified, return
    if len(best_cmes) == 1:
        return best_cmes
    
    #If every single CME is preferred, return empty so that either
    #the last CME will be taken later on in the revision step.
    #Or, if there are flare inputs, can select based off of flare.
    if len(best_cmes) == len(CMEs):
        return []
    
    #if multiple CMEs identified, choose the latest in time (closest to SEP)
    td_diff = 1.5 #hour
    if len(best_cmes) > 1:
        last = last_cme(best_cmes)
        #Identify CMEs within an hour of best to account for refits
        for cme in best_cmes:
            if not pd.isnull(cme.start_time):
                diff = (cme.start_time - last.start_time).total_seconds()/3600
                if abs(diff) <= td_diff:
                    best.append(cme)
            elif not pd.isnull(cme.liftoff_time):
                diff = (cme.liftoff_time - last.liftoff_time).total_seconds()/3600
                if abs(diff) <= td_diff:
                    best.append(cme)


    #Return fastest CME
    if best:
        return best


    #-----CMEs satisfy only speed criteria, narrow CME -------
    best_cmes = []
    for cme in CMEs:
        if pd.isnull(cme.speed): continue
        if cme.speed >= target_speed:
            best_cmes.append(cme)
            
    #If only one CME identified, return
    if len(best_cmes) == 1:
        return best_cmes
        
    #if multiple CMEs identified, choose the fastest
    fast = None
    if len(best_cmes) > 1:
        max_speed = 0
        for cme in best_cmes:
            if cme.speed > max_speed:
                max_speed = cme.speed
                fast = cme

        #Return fastest CME
        if not pd.isnull(fast):
            #Identify CMEs close to best to account for refits
            for cme in best_cmes:
                if not pd.isnull(fast.start_time):
                    diff = (cme.start_time - fast.start_time).total_seconds()/3600
                    if abs(diff) <= td_diff:
                        best.append(cme)
                elif not pd.isnull(fast.liftoff_time):
                    diff = (cme.liftoff_time - fast.liftoff_time).total_seconds()/3600
                    if abs(diff) <= td_diff:
                        best.append(cme)


    #If haven't identified a best CME (or CMEs) by now, best to use the
    #timing criteria implemented in revise_eruption_matches.
    
    return best


def last_cme(cmes):
    """ Identify the last CME to erupt. """
    
    last_cme = None
    last = pd.NaT
    for cme in cmes:
        if not pd.isnull(cme.start_time):
            if pd.isnull(last):
                last = cme.start_time
                last_cme = cme
            elif cme.start_time > last:
                last = cme.start_time
                last_cme = cme
        
        if not pd.isnull(cme.liftoff_time):
            if pd.isnull(last):
                last = cme.liftoff_time
                last_cme = cme
            elif cme.liftoff_time > last:
                last = cme.liftoff_time
                last_cme = cme
        
    return last_cme


def last_flare(flares):
    """ Identify the last flare to erupt. """
    
    last_flare = None
    last = pd.NaT
    for flare in flares:
        if not pd.isnull(flare.start_time):
            if pd.isnull(last):
                last = flare.start_time
                last_flare = flare
            elif flare.start_time > last:
                last = flare.start_time
                last_flare = flare

        elif not pd.isnull(flare.last_data_time):
            if pd.isnull(last):
                last = flare.last_data_time
                last_flare = flare
            elif flare.last_data_time > last:
                last = flare.last_data_time
                last_flare = flare
    
    return last_flare
 


def revise_eruption_matches(evaluated_sphinx, all_energy_channels, obs_values,
        model_names, observed_sep_events):
    """ It may be that there are multiple flares or CMEs in
        a short time period that generate predictions with
        large prediction windows. These multiple forecasts with different
        triggers can match to the same SEP event. Since only one CME/flare
        combination is generally responsible for a given SEP, the code must
        pick the appropriate forecast to associate with that SEP event.
        
        In the case that there are multiple predictions with different
        CME/flare triggers, then the following selections will be applied:
        
        - The eruption must be within a few minutes to 48 hours prior to threshold
            crossing (this selection is already made in match_all_forecasts).
        - Associate only the prediction with the eruption closest in time to
            the SEP event (this subroutine). Predictions with earlier
            CME/flare triggers matched to the same SEP event will be
            unassociated, even if they originally fell within the specified
            range.
    
    Input:
    
        :evaluated_sphinx: (dictionary of SPHINX objects) contains
            sphinx objects organized by model and energy channel
        :all_energy_channels: (array of dict) all energy channels
            found in the observations
        :obs_values: (pandas Dataframe) contains info for all observations.
            Needed to extract the thresholds.
        :model_names: (array of strings) array of all the model short names
        :observed_sep_events: (dict) all unique SEP events the were inside
            of a prediction window organized by model, energy channel, and
            threshold
            
    Output:
    
        None, but the objects inside evaluated_sphinx will be updated.
        
    """
    #print("\n====== REVISING MATCHES WITH FLARE/CME TRIGGERS ======")
    for model in model_names:
        #if the model doesn't use eruptions as triggers, then this
        #doesn't apply
        if not evaluated_sphinx[model]['uses_eruptions']:
            continue

        for energy_key in all_energy_channels:
            for threshold in obs_values[energy_key]['thresholds']:
                #Pull out all the observed SEP events inside of the
                #model prediction windows for a given energy channel
                #and threshold. Want to identify if multiple predictions
                #using different eruptions as triggers were matched to the
                #same SEP event.
                thresh_key = objh.threshold_to_key(threshold)
                obs_sep = observed_sep_events[model][energy_key][thresh_key]
                if not obs_sep: continue

                #Create a dataframe containing info from all forecasts for a
                #single model, energy channel, and threshold. To allow easy
                #comparison from forecast to forecast.
                #Keys 'matched_observations', 'matched_sep',
                #'td_eruption_thresh_cross' and
                #'observed_threshold_crossing_time'
                #Dataframe indices match the indices in evaluated_sphinx                
                
                spx_df = create_matched_model_array(evaluated_sphinx[model][energy_key],
                threshold)
                #Identify the forecasts matched to the same SEP event
                for sep in obs_sep:
                    #For model, energy channel, threshold, extract all matches associated
                    #with each SEP event
                    sub = spx_df.loc[(spx_df['observed_threshold_crossing_time'] == sep)]
                    #If only one match or no forecasts associated with the SEP,
                    #nothing to do
                    if sub.empty or len(sub) == 1: continue

                    logger.info(f"----Begin {model}, {energy_key}, {thresh_key} unmatch for SEP event {sep}----")
                    logger.info(f"{len(sub)} forecasts associated with {sep}")

                    #If multiple forecasts for the same model are matched to the SEP
                    #event, check if they use different eruptions (in hours)
                    unique_td = resume.identify_unique(sub,'td_eruption_thresh_cross')
                    #Possible to have NaN values, remove
                    unique_td = [td for td in unique_td if not pd.isnull(td)]

                    #If only one value for the time difference between the last eruption
                    #and the SEP threshold crossing time, then only one trigger used
                    if len(unique_td) <= 1:
                        logger.info("All forecasts triggered by same eruption.  No unmatching.")
                        continue
 
                    #If eruption times are very close to each other, this could indicate
                    #e.g. refit of parameters. Want to keep all forecasts when parameters
                    #change just slightly.
                    td_diff = 1 #hour
                    sub_time = sub.loc[(sub['td_eruption_thresh_cross'] >= (unique_td[0]-td_diff)) &
                            (sub['td_eruption_thresh_cross'] <= (unique_td[0]+td_diff))]
                    if sub_time.empty: #No need to do unmatching
                        logger.info("Eruptions are within 1 hour of each other.  No unmatching.")
                        continue
                    
                    #Have different eruptions, so identify the preferred eruptions
                    #to be matched to this SEP event by looking at the triggers
                    cmes = sub['CME Triggers'].to_list()
                    flares = sub['Flare Triggers'].to_list()
                    sphinx_index = sub['sphinx_index'].to_list()
                    
                    cmes = [cme for cme in cmes if not pd.isnull(cme)]
                    flares = [flare for flare in flares if not pd.isnull(flare)]

                    cme_speed = [cme.speed for cme in cmes]
                    cme_width = [cme.half_width for cme in cmes]
                    td_cme = [(cme.start_time-sep).total_seconds()/3600. for cme in cmes if not pd.isnull(cme.start_time)]
                    if not td_cme:
                         td_cme = [(cme.liftoff_time-sep).total_seconds()/3600. for cme in cmes if not pd.isnull(cme.liftoff_time)]

                    cme_match_status = ['SEP Event']*len(cme_speed)
                    flare_intensity = [flare.intensity for flare in flares]
                    flare_lon = [flare.lon for flare in flares]
                    td_flare = [(flare.start_time-sep).total_seconds()/3600. for flare in flares if not pd.isnull(flare.start_time)]
                    if not td_flare:
                        td_flare = [(flare.last_data_time-sep).total_seconds()/3600. for flare in flares if not pd.isnull(flare.last_data_time)]

                    flare_match_status = ['SEP Event']*len(flare_intensity)
                    if cmes:
                        logger.info(f"{len(cmes)} CMEs associated with forecasts.")
                    if flares:
                        logger.info(f"{len(flares)} flares associated with forecasts.")
                    
                    #Choose best cme and flare
                    pref_cme = []
                    if len(cmes) > 1:
                        pref_cme = preferred_cme(cmes)
                            
                    pref_flare = []
                    if len(flares) > 1:
                        pref_flare = preferred_flare(flares)

                    logger.info(f"Found {len(pref_cme)} preferred CMEs and {len(pref_flare)} preferred flares.")
                    #Dataframe that will contain the forecasts to unmatch
                    unmatch_sub = pd.DataFrame()
 
                    #If both a preferred CME and a preferred flare are found,
                    #defer to the CME
                    if pref_cme:
                        unmatch_sub = sub[~sub['CME Triggers'].isin(pref_cme)]
                        for ii in range(len(cmes)):
                            if cmes[ii] not in pref_cme:
                                cme_match_status[ii] = 'Unmatched'
                                if flares: #Defer to CME
                                    flare_match_status[ii] = 'Defer to CME'
 
                    elif pref_flare:
                        unmatch_sub = sub[~sub['Flare Triggers'].isin(pref_flare)]
                        for ii in range(len(flares)):
                            if flares[ii] not in pref_flare:
                                flare_match_status[ii] = 'Unmatched'
                                if cmes:
                                    cme_match_status[ii] = 'Defer to Flare'


                    #IF there are NO best choices for trigger, take the last eruption
                    if not pref_flare and not pref_cme:
                        #If eruptions near an SEP event are within a few hours of each other
                        #cannot really say which is the correct one to associate with the
                        #SEP event, so keep all. e.g. March 7, 2012
                        td_diff = 3 #hour
                        sub_time = sub.loc[(sub['td_eruption_thresh_cross'] >= unique_td[0]-td_diff) &
                                (sub['td_eruption_thresh_cross'] <= unique_td[0]+td_diff)]
                        if sub_time.empty: #No need to do unmatching
                            logger.info("Eruptions within 3 hours of each other.  No unmatching.")
                            continue
                    
                        #If there are eruptions separated enough to discriminate in time
                        #Eruption time before threshold crossing are necessarily negative
                        best_eruption = max(unique_td)
                        logger.info(f"No preferred flare or CME. Best eruption from timing {best_eruption}")
                        unmatch_sub = sub.loc[(sub['td_eruption_thresh_cross'] < best_eruption-td_diff)
                                    | (sub['td_eruption_thresh_cross'] > best_eruption+td_diff)]


                        unmatch_cmes = unmatch_sub['CME Triggers'].to_list()
                        for ii in range(len(cmes)):
                            if cmes[ii] in unmatch_cmes:
                                cme_match_status[ii] = 'Unmatched'

                        unmatch_flares = unmatch_sub['Flare Triggers'].to_list()
                        for ii in range(len(flares)):
                            if flares[ii] in unmatch_flares:
                                flare_match_status[ii] = 'Unmatched'



                    if unmatch_sub.empty: continue

                    logger.info(f"Unmatching {len(unmatch_sub)} forecasts.")

                    if cme_speed:
                        logger.info("Match Status,  CME Speed,  Width,  Time to SEP Threshold Crossing,  Forecast")
                    for ii in range(len(cme_speed)):
                        logger.info(f"{cme_match_status[ii]},  {cme_speed[ii]},  {cme_width[ii]},  {td_cme[ii]},  {os.path.basename(evaluated_sphinx[model][energy_key][sphinx_index[ii]].prediction.source)}")
                    logger.info("")
                    if flare_intensity:
                        logger.info("Match Status,  Flare Intensity, Flare Longitude, Time to SEP Threshold Crossing,  Forecast")
                    for ii in range(len(flare_intensity)):
                        logger.info(f"{flare_match_status[ii]},  {flare_intensity[ii]},  {flare_lon[ii]},  {td_flare[ii]},  {os.path.basename(evaluated_sphinx[model][energy_key][sphinx_index[ii]].prediction.source)}")


                    sphinx_index = unmatch_sub['sphinx_index'].to_list()
                    for sphx_idx in sphinx_index:
                         #Identify forecast being unmatched
                        logger.info("====== UNMATCHING FORECAST FROM SEP EVENT ====")
                        logger.info("Forecast: " +
                        str(evaluated_sphinx[model][energy_key][sphx_idx].prediction.source))
                        logger.info("Last Eruption Time: " +
                        str(evaluated_sphinx[model][energy_key][sphx_idx].last_eruption_time))
                        logger.info("Initially matched to SEP Event: " + str(evaluated_sphinx[model][energy_key][sphx_idx].observed_threshold_crossing[thresh_key].crossing_time))
                        logger.info("Observation source: " + str(evaluated_sphinx[model][energy_key][sphx_idx].observed_match_sep_source[thresh_key]))

                        #Unmatch
                        evaluated_sphinx[model][energy_key][sphx_idx].unmatch(threshold)
                        logger.info("-------- UNMATCHED ----------\n")
 


#################################
####ALL MATCHING CRITERIA #######
#################################
def setup_match_all_forecasts(all_energy_channels, obs_objs, obs_values, model_objs, model_names):
    """
    Helper function to match_all_forecasts().
    Loops over all energy channels, all forecasts for each energy channel,
    all thresholds for each forecast, and all indices for which the
    forecast and observation windows overlap, and injects matching criteria
    information into a SPHINX object.
    Each SPHINX object is associated with a particular forecast.
    A dictionary of matched SPHINX objects is prepared.
    
    Input:
    
        :all_energy_channels: (array of dict) array containing energy
            channel dictionaries of all energy channels present in the
            observations
        :obs_objs: (dict) dictionary sorted by energy channel
            containing all Observation class objects created from
            the observation jsons
        :obs_values: (pandas DataFrame) Dataframe containing all observed
            values
        :model_objs: (dict) dictionary sorted by energy channel
            containing all Forecast class objects created from
            the forecast jsons
        :model_names: (array of strings) model short names used to
            organize the forecast and used as keys
            
    Output:
        
        :evaluated_sphinx: (dictionary of SPHINX objects) dictionary organized
            by model name and energy channel of SPHINX objects containing
            forecasts and matched observational values (organized in the
            SPHINX object by threshold)
        :observed_sep_events: (dictionary) contains unique observed SEP
            events that occurred within all the forecast prediction windows.
            Oganized by model, energy channel, and threshold.
    
    
    """
    #Set up dictionary of sphinx objects organized by model name and
    #energy channel.
    #Set up dictionary of observed SEP events in the prediction windows
    #organized by model, energy channel, threshold
    evaluated_sphinx = {} #forecasts that contribute to metrics
    removed_sphinx = {} #forecasts that are excluded for various reasons
    observed_sep_events = {} #list of unique observed SEP events
    observed_thresholds = {}
    for model in model_names:
        evaluated_sphinx.update({model:{'uses_eruptions':False}})
        removed_sphinx.update({model:{'uses_eruptions':False}})
        observed_sep_events.update({model:{}})
        for energy_key in all_energy_channels:
            evaluated_sphinx[model].update({energy_key:[]})
            removed_sphinx[model].update({energy_key:[]})
            observed_sep_events[model].update({energy_key:{}})
            observed_thresholds.update({energy_key:[]})
            #Save all unique observed SEP events organized by energy channel
            #and threshold
            for obs_thresh in obs_values[energy_key]['thresholds']:
                obs_thresh_key = objh.threshold_to_key(obs_thresh)
                observed_sep_events[model][energy_key].update({obs_thresh_key:[]})
                observed_thresholds[energy_key].append(obs_thresh)

    #Launch into matching of observations and forecasts
    
    for energy_key in all_energy_channels:
        logger.info("MATCHING SETUP STARTING FOR: " + energy_key)
 
        observation_objs = obs_objs[energy_key] #Observation objects
        forecasts = model_objs[energy_key] #all forecasts for channel
 
        n_tot = len(forecasts)
        ii = 0
        setup_start_time = datetime.datetime.now()
        for fcast in forecasts:
            #Report progress
            if ii % 1000 == 0 and ii != 0:
                logger.info(f"MATCH SETUP PROGRESS: Set up {ii} out of {n_tot} forecasts.")
            ii += 1

            #Check if shortname needs to be revised
            name = fcast.short_name
            # logger.debug(str(cfg.shortname_grouping))
            if cfg.shortname_grouping:
                name = objh.shortname_grouper(name, cfg.shortname_grouping)

            #Check that forecast is valid and set status
            fcast.valid_forecast(verbose=True)
            logger.debug(f"Model: {fcast.short_name}, Valid: {fcast.valid}, Reason: {fcast.invalid_reason}")

            #One SPHINX object contains all matching information and
            #predicted and observed values (and all thresholds)
            sphinx = objh.initialize_sphinx(fcast)
 
            #If the forecast is not valid, skip and save sphinx object in
            #not evaluated dataframe
            if fcast.valid == False:
                sphinx.not_evaluated = fcast.invalid_reason
                removed_sphinx[name][energy_key].append(sphinx)
                logger.warning("REMOVED FROM ANALYSIS: Forecast not valid: "
                    + fcast.source + ", " + fcast.invalid_reason)
                continue
 
 
            #If this is a set of predictions and observations that are
            #allowed to have a set of mismatched energy channels and
            #thresholds
            if cfg.do_mismatch and energy_key == cfg.mm_energy_key\
                and cfg.mm_model in fcast.short_name:
                sphinx.mismatch = True
                logger.debug("Mismatched channel allowed.")

            #Get Trigger and Input information
            sphinx.last_eruption_time, sphinx.last_trigger_time =\
                fcast.last_trigger_time()
            
            sphinx.last_input_time = fcast.last_input_time()


            #Note if the model uses eruptions as triggers for 2nd matching step
            logger.debug("\n")
            logger.debug(name)
            logger.debug(fcast.source)
            logger.debug("Issue time: " + str(fcast.issue_time))
            logger.debug("Last trigger time: " + str(sphinx.last_trigger_time))
            logger.debug("Last eruption time: " + str(sphinx.last_eruption_time))
            logger.debug("Last input time: " + str(sphinx.last_input_time))

           
            if not pd.isnull(sphinx.last_eruption_time):
                evaluated_sphinx[name]['uses_eruptions'] = True
                removed_sphinx[name]['uses_eruptions'] = True

            ###### PREDICTION AND OBSERVATION WINDOWS OVERLAP? #####
            #Do prediction and observation windows overlap?
            #Save the overlapping observations to the SPHINX object
            is_win_overlap = does_win_overlap(energy_key, fcast, obs_values)
            overlap_idx = [ix for ix in range(len(is_win_overlap)) if is_win_overlap[ix] == True]
            sphinx.overlapping_indices = overlap_idx

            #If no observations were found to overlap with the forecast, save in
            #not evaluated dataframe and continue
            if not overlap_idx:
                sphinx.not_evaluated = "No observations during forecast prediction window"
                removed_sphinx[name][energy_key].append(sphinx)
                logger.warning("REMOVED FROM ANALYSIS: " + fcast.source
                    + ", No observations overlapped with forecast prediction window.")
                continue

            logger.debug("Prediction and Observation windows overlap: ")
            if overlap_idx:
                for ix in range(len(overlap_idx)):
                    sphinx.is_win_overlap.append(is_win_overlap[overlap_idx[ix]])
                    sphinx.prediction_observation_windows_overlap.append(observation_objs[overlap_idx[ix]])
                    obs_path = os.path.dirname(observation_objs[overlap_idx[ix]].source)
                    profile_path = os.path.join(obs_path, observation_objs[overlap_idx[ix]].sep_profile)
                    sphinx.observed_sep_profiles.append(profile_path)
                    #logging
                    logger.debug("  " +
                    str(sphinx.prediction_observation_windows_overlap[ix].source))
                    logger.debug("  " +
                    str(sphinx.observed_sep_profiles[ix]))

            #Check that observations cover 100% of forecast prediction window
            is_covered = check_observations_coverage(sphinx)
            if not is_covered:
                sphinx.not_evaluated = "Observations do not fully cover prediction window"
                removed_sphinx[name][energy_key].append(sphinx)
                logger.warning("REMOVED FROM ANALYSIS: Observations do not cover full prediction window: "
                    + fcast.source)
                continue


            ###### ONSET PEAK CRITERIA #####
            onset_peak_criteria(sphinx, fcast, obs_values, observation_objs, energy_key)


            ###### MAX FLUX CRITERIA #####
            max_flux_criteria(sphinx, fcast, obs_values, observation_objs, energy_key)


            ###### THRESHOLD QUANTITIES #####
            #Identify the thresholds that are applied in the forecast
            #Checks all_clear, event_lengths, fluence_spectra, threshold_crossings,
            #probabilities
            all_fcast_thresholds = fcast.identify_all_thresholds()
            
            #If the forecast has been identified as one belonging to a model
            #that has mismatch allowed and this is the correct energy channel
            #to apply the mismatch, then check that the mismatched threshold
            #is considered. It may be that a specific forecast from the model
            #may not include the excepted threshold, but want to be consistent
            #and check it for every single forecast produced by the model.
            #If sphinx.mismatch is True, then this is the model and energy channel
            #where the excepted threshold could be applied.
            if sphinx.mismatch:
                if cfg.mm_pred_threshold not in all_fcast_thresholds:
                    all_fcast_thresholds.append(cfg.mm_pred_threshold)

            #In the case that no thresholds were present in the forecast, add
            #observed thresholds.
            #This can occur if a forecast includes only peak_intensity,
            #peak_intensity_max, and/or sep_profile. All other fields should have a
            #threshold associated with them
            if len(all_fcast_thresholds) == 0:
                #Check that forecast predicts one of the above
                if not pd.isnull(fcast.peak_intensity.intensity) or not pd.isnull(fcast.peak_intensity_max.intensity) or not pd.isnull(fcast.sep_profile):
                    logger.info(f"Peak intensity: {fcast.peak_intensity.intensity}, Peak intensity max: {fcast.peak_intensity_max.intensity}, SEP profile: {fcast.sep_profile}")
                    all_fcast_thresholds = observed_thresholds[energy_key] ###CONSIDER HERE
                    logger.warning("SUBSTITUTED THRESHOLDS: No thresholds were present within "
                        "the prediction: " + fcast.source + " for energy channel " + energy_key +
                        ". Continued analysis using thresholds applied to the prepared observations.")
                
                else:
                    sphinx.not_evaluated = "No thresholds provided in forecast"
                    #No thresholds were found to match between the forecast and the observations
                    removed_sphinx[name][energy_key].append(sphinx)
                    logger.warning("REMOVED FROM ANALYSIS: Forecast did not provide thresholds: "
                        + fcast.source)
                    continue


            observed_peak_flux = None
            observed_peak_flux_max = None            
            
            for f_thresh in all_fcast_thresholds:
                logger.debug("Checking Threshold: " + str(f_thresh))
                fcast_thresh = f_thresh
                
                #If this is a mismatch energy channel, then only want to
                #test the mismatched threshold specified in config.py
                if sphinx.mismatch:
                    if f_thresh == cfg.mm_pred_threshold:
                        #set threshold as the observed threshold
                        fcast_thresh = cfg.mm_obs_threshold
                        logger.debug("Predicted threshold associated with \'mismatched\' "
                                "observational threshold " + str(fcast_thresh))
                    else:
                        continue
                
                #Check if this threshold is present in the observations
                #Can only be compared if present in both
                if fcast_thresh not in obs_values[energy_key]['thresholds']:
                    continue

                #Add threshold so that objects saved in SPHINX object
                #in contains_thresh_cross and is_*_before* arrays
                #will be in an array in the same order as the
                #thresholds
                sphinx.thresholds.append(fcast_thresh) #threshold in observation
                sphinx.add_threshold(fcast_thresh)
                thresh_key = objh.threshold_to_key(fcast_thresh)

                #ongoing_events block in forecast
                #Check if forecast reporting an SEP event is ongoing.
                #Do not credit/penalized forecasts that simply relay
                #the information that the environment is enhanced.
                #In case of mismatch, check against the original f_thresh
                report_ongoing = False
                if sphinx.prediction.particle_intensities:
                    for pi in sphinx.prediction.particle_intensities:
                        if pi.ongoing_events:
                            for ongoing_ev in pi.ongoing_events:
                                if ongoing_ev['threshold'] == f_thresh['threshold']:
                                    report_ongoing = True


                ###### PREDICTION WINDOW OVERLAP WITH OBSERVED ####
                ############### SEP EVENT #########################
                pred_win_sep_overlap(sphinx, fcast, obs_values, observation_objs,
                    energy_key, fcast_thresh)


                ###### THRESHOLD CROSSED IN PREDICTION WINDOW #####
                #Is a threshold crossed in the prediction window?
                threshold_cross_criteria(sphinx, fcast, obs_values, observation_objs,
                    energy_key, fcast_thresh)


                ########### TRIGGERS/INPUTS BEFORE SEP ############
                #Is the last trigger/input before the threshold crossing time?
                before_threshold_crossing(sphinx, fcast, obs_values,
                observation_objs, energy_key, fcast_thresh)
                
                
                ########### TRIGGERS/INPUTS BEFORE END OF SEP #####
                #Is the last trigger/input before the threshold crossing time?
                before_sep_end(sphinx, fcast, obs_values,
                observation_objs, energy_key, fcast_thresh)


                ######### FLARE/CME BEFORE SEP START ###############
                #Is the eruption (flare/cme) before the threshold crossing?
                eruption_before_threshold_crossing(sphinx, fcast, obs_values,
                            observation_objs, energy_key, fcast_thresh)
                

                ############ ONGOING SEP EVENT AT START OF ########
                ################## PREDICTION WINDOW ##############
                observed_ongoing_event(sphinx, fcast, obs_values, observation_objs,
                    energy_key, fcast_thresh)
                

                ############ MATCHING AND EXTRACTING OBSERVED VALUES#######
                #Loop over all observations inside the prediction window
                #indices will be keys
                sphinx.is_eruption_in_range[thresh_key] = []
                sphinx.trigger_input_start[thresh_key] = []
                sphinx.trigger_input_end[thresh_key] = []
                for i in range(len(sphinx.overlapping_indices)): #index of overlapping obs
                    #Bool for eruption 24 hours to a few mins before
                    #threshold crossing.
                    #None if no SEP event
                    sphinx.is_eruption_in_range[thresh_key].append( eruption_in_range(sphinx.time_difference_eruptions_threshold_crossing[thresh_key][i]))
                          

                    #Is the last trigger or input before the threshold crossing
                    #None if no SEP event
                    sphinx.trigger_input_start[thresh_key].append( last_before_start(sphinx.triggers_before_threshold_crossing[thresh_key][i],
                        sphinx.inputs_before_threshold_crossing[thresh_key][i]))
                    
                    #Is the last trigger or input before the SEP end
                    #None if no SEP event
                    sphinx.trigger_input_end[thresh_key].append( last_before_end(sphinx.triggers_before_sep_end[thresh_key][i],
                        sphinx.inputs_before_sep_end[thresh_key][i]))


            #No thresholds matched between observations and forecast
            if not sphinx.thresholds:
                sphinx.not_evaluated = f"No matching thresholds in observations for {sphinx.prediction.all_thresholds}"
                #No thresholds were found to match between the forecast and the observations
                removed_sphinx[name][energy_key].append(sphinx)
                logger.warning("REMOVED FROM ANALYSIS: Observations did not contain forecast thresholds: "
                    + fcast.source)
                continue


            #Forecast reported an ongoing event in this energy channel
            if report_ongoing:
                sphinx.not_evaluated = ("Forecast reporting ongoing event")
                                #No thresholds were found to match between the forecast and the observations
                removed_sphinx[name][energy_key].append(sphinx)
                logger.warning(f"REMOVED FROM ANALYSIS: Forecast reporting ongoing event and is not evaluated: {fcast.source}")
                continue

            
            sphinx.not_evaluated = "Forecast is evaluated"
            evaluated_sphinx[name][energy_key].append(sphinx)
            logger.debug("Model: " + name + "(, Original name: " + sphinx.prediction.short_name + ")")
            logger.debug("Forecast energy channel: " + str(sphinx.prediction.energy_channel))
            logger.debug("Prediction Thresholds: " + str(all_fcast_thresholds))
            logger.debug("Observed Thresholds: " + str(sphinx.thresholds))
            logger.debug("Forecast index (position in evaluated_sphinx): " + str(len(evaluated_sphinx[name][energy_key]) - 1))


        setup_end_time = datetime.datetime.now()
        # MICROSECOND ADDED TO TIME DIFFERENCE TO AVOID DIVISION BY ZERO
        match_td = (setup_end_time - setup_start_time).total_seconds() + 1e-6
        rate = ii / match_td
        logger.info(f"MATCH SETUP PROGRESS: Completed {ii} matches in {match_td:0.3f} seconds at a rate of {rate:0.1f} forecasts/second.")

    total_sphinx = 0
    total_discard_sphinx = 0
    for model in model_names:
        for energy_key in all_energy_channels:
            logger.info("STATS: SPHINX evaluated for " + model + ", " + energy_key
                        + ": " + str(len(evaluated_sphinx[model][energy_key])))
            total_sphinx += len(evaluated_sphinx[model][energy_key])
            logger.info("STATS: SPHINX NOT evaluated for " + model + ", " + energy_key
                        + ": " + str(len(removed_sphinx[model][energy_key])))
            total_discard_sphinx += len(removed_sphinx[model][energy_key])
            
    logger.info("STATS: TOTAL SPHINX OBJECTS EVALUATED IN SETUP: " + str(total_sphinx))
    logger.info("STATS: TOTAL SPHINX OBJECTS DISCARDED IN SETUP: " + str(total_discard_sphinx))
 
    return evaluated_sphinx, removed_sphinx, observed_sep_events



def match_all_forecasts(all_energy_channels, model_names, obs_objs,
    model_objs):
    """ Match all forecasts to observations organized by model
        short_name, energy channel, and threshold.
        
        Identifies all observed SEP events covered by the forecast
        prediction windows.
        
    Input:
    
        :all_energy_channels: (array of dict) array containing energy
            channel dictionaries of all energy channels present in the
            observations
        :model_names: (array of strings) model short names used to
            organize the forecast and used as keys
        :obs_objs: (dict) dictionary sorted by energy channel
            containing all Observation class objects created from
            the observation jsons
        :model_objs: (dict) dictionary sorted by energy channel
            containing all Forecast class objects created from
            the forecast jsons
            
    Output:
        
        :evaluated_sphinx: (dictionary of SPHINX objects) dictionary organized
            by model name and energy channel of SPHINX objects containing
            forecasts and matched observational values (organized in the
            SPHINX object by threshold)
        :observed_sep_events: (dictionary) contains unique observed SEP
            events that occurred within all the forecast prediction windows.
            Oganized by model, energy channel, and threshold.
            
    
    """
    
    #All observed values needed for matching, organized by
    #energy channel and threshold
    #Output as a pandas dataframe containing all observed values
    logger.info("Compiling all observations into a dataframe.")

    obs_values = compile_all_obs(all_energy_channels, obs_objs)
    logger.info("Completed observations dataframe, identifying all thresholds applied to observed energy channels.")

    #Gather all thresholds applied in the observations for each energy channel
    all_obs_thresholds = {}
    for energy_key in all_energy_channels:
        all_obs_thresholds.update({energy_key: []})
        for obs_thresh in obs_values[energy_key]['thresholds']:
            all_obs_thresholds[energy_key].append(objh.threshold_to_key(obs_thresh))

    logger.info("Starting matching process. Setting up dictionary for observed SEP events to go into a list for each model.")

    evaluated_sphinx, removed_sphinx, observed_sep_events = setup_match_all_forecasts(all_energy_channels, obs_objs, obs_values, model_objs, model_names)
    
    for model in model_names:
        logger.info("MATCHING STARTING FOR: " + model)

        for energy_key in all_energy_channels:
            observation_objs = obs_objs[energy_key] #Observation objects
            logger.info("MATCHING STARTING FOR: " + energy_key)
            n_tot = len(evaluated_sphinx[model][energy_key])
            match_start_time = datetime.datetime.now()
 
            for ii in range(len(evaluated_sphinx[model][energy_key])):
                #Report progress
                if ii % 1000 == 0 and ii != 0:
                    logger.info(f"MATCH PROGRESS: Matched {ii} out of {n_tot} forecasts for {model} and {energy_key}.")
 
                sphinx = evaluated_sphinx[model][energy_key][ii]

                for fcast_thresh in sphinx.thresholds:
                    #already accounts for mismatch and whether observations
                    #have same thresholds; sorting done in setup
                    #A mismatched prediction threshold will be saved in the
                    #sphinx.thresholds array under the corresponding observed
                    #threshold
                    
                    thresh_key = objh.threshold_to_key(fcast_thresh)
                    for i in range(len(sphinx.overlapping_indices)):
                        #index of overlapping obs
                        #All input arrays are downsized to contain only the elements/values
                        #associated with the observations that overlap with the forecast
                        #prediction windows. All necessary arrays are the same length
                        #as sphinx.overlapping_indices and correspond to the observation
                        #objects at those indices in the observation dictionary.
                       
                        obs_idx = sphinx.overlapping_indices[i]
 
                        ###ONSET PEAK & MAX FLUX
                        #Prediction window overlaps with observation
                        #Last eruption within appropriate time range before
                        #threshold crossing
                        #The prediction window overlaps with an SEP event in any
                        #threshold - only a comparison when SEP event
                        #The last trigger/input time if before the observed peak
                        #intensity
                        #ONSET PEAK
                        #peak_criteria is True, False, None indicating a match
                        #with an observation or None if no SEP event observed
                        peak_criteria = match_observed_onset_peak(sphinx,
                            observation_objs[obs_idx], sphinx.is_win_overlap[i],
                            sphinx.is_eruption_in_range[thresh_key][i],
                            sphinx.triggers_before_peak_intensity[i],
                            sphinx.inputs_before_peak_intensity[i],
                            sphinx.prediction_window_sep_overlap[thresh_key][i])
                        
                       
                        #MAX FLUX
                        #max_criteria is True, False, None indicating a match
                        #with an observation or None if no SEP event observed
                        max_criteria = match_observed_max_flux(sphinx,
                            observation_objs[obs_idx], sphinx.is_win_overlap[i],
                            sphinx.is_eruption_in_range[thresh_key][i],
                            sphinx.triggers_before_peak_intensity_max[i],
                            sphinx.inputs_before_peak_intensity_max[i],
                            sphinx.prediction_window_sep_overlap[thresh_key][i])
                        
                        
                        #ALL CLEAR
                        #all_clear_status is True if no observed SEP event,
                        #False if observed SEP event that meets criteria,
                        #None - if ongoing event at start of prediction window
                        all_clear_status = match_all_clear(sphinx,
                            observation_objs[obs_idx], sphinx.is_win_overlap[i],
                            sphinx.is_eruption_in_range[thresh_key][i],
                            sphinx.trigger_input_start[thresh_key][i],
                            sphinx.threshold_crossed_in_pred_win[thresh_key][i],
                            sphinx.observed_ongoing_events[thresh_key][i])

                        #SEP QUANTITIES RELATED TO START TIME
                        sep_status = match_sep_quantities(sphinx, observation_objs[obs_idx],
                            fcast_thresh, sphinx.is_win_overlap[i],
                            sphinx.is_eruption_in_range[thresh_key][i],
                            sphinx.trigger_input_start[thresh_key][i],
                            sphinx.threshold_crossed_in_pred_win[thresh_key][i],
                            sphinx.observed_ongoing_events[thresh_key][i])

                        #Save observed SEP event
                        if sep_status == True:
                            if sphinx.observed_threshold_crossing[thresh_key].crossing_time\
                            not in observed_sep_events[model][energy_key][thresh_key]:
                                observed_sep_events[model][energy_key][thresh_key].append(sphinx.observed_threshold_crossing[thresh_key].crossing_time)

                        #SEP END TIME
                        end_status = match_sep_end_time(sphinx, observation_objs[obs_idx],
                            fcast_thresh,
                            sphinx.is_win_overlap[i],
                            sphinx.is_eruption_in_range[thresh_key][i],
                            sphinx.trigger_input_end[thresh_key][i],
                            sphinx.prediction_window_sep_overlap[thresh_key][i])

                        derived_status = calculate_derived_quantities(sphinx)

                #Save the SPHINX object with all of the forecasted and matched
                #observation values to a dictionary organized by energy channel
                msg = sphinx.match_report_streamlined()
                logger.debug(msg)
                evaluated_sphinx[model][energy_key][ii] = sphinx

                logger.debug("Model: " + sphinx.prediction.short_name)
                logger.debug("Forecast energy channel: " + str(sphinx.prediction.energy_channel))
                logger.debug("Predicted Thresholds: " + str(sphinx.prediction.identify_all_thresholds()))
                logger.debug("Observed Thresholds: " + str(sphinx.thresholds))
                logger.debug("Forecast index (position in evaluated_sphinx): " + str(ii))

            match_end_time = datetime.datetime.now()
            match_td = (match_end_time - match_start_time).total_seconds() + 1e-6
            rate = n_tot / match_td
            logger.info(f"MATCH PROGRESS: Completed {n_tot} matches for {model} and {energy_key} in {match_td:0.3f} seconds at a rate of {rate:0.1f} forecasts/second.")

    logger.info("Initial forecast-to-observation matching complete.")
    #Print uniquely identified observed SEP events
    #Redundant as info is in output files. Use for debugging.
    sep_report(all_energy_channels, obs_values, model_names,
        observed_sep_events)

    logger.info("Checking forecasts to determine if revisions should be made "
            "in matches to observed SEP events. Revising for all predictions "
            "triggered by flares and/or CMEs.")
    #In the case where the same model has forecasts derived from
    #multiple eruptions matched to the same SEP event, find the
    #best match and unmatch the other forecasts.
    revise_eruption_matches(evaluated_sphinx, all_energy_channels,
        obs_values, model_names, observed_sep_events)

    logger.info("The revision process is complete. All forecast-to-observation "
        "matches have been finalized.")

    return evaluated_sphinx, removed_sphinx, all_obs_thresholds, observed_sep_events


