from . import validation_json_handler as vjson
from . import object_handler as objh
from . import config as cfg
from . import classes as cl
import sys
import pandas as pd
import numpy as np
import datetime

__version__ = "0.1"
__author__ = "Katie Whitman"
__maintainer__ = "Katie Whitman"
__email__ = "kathryn.whitman@nasa.gov"

""" utils/match.py contains subroutines to match forecasts
    to observations.
    
"""

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
            if obj.observation_window_start == None:
                obs_win_st.append(pd.NaT)
            else:
                obs_win_st.append(obj.observation_window_start)
            if obj.observation_window_end == None:
                obs_win_end.append(pd.NaT)
            else:
                obs_win_end.append(obj.observation_window_end)
            
            #Peak time
            if obj.peak_intensity.time == None:
                peak_time.append(pd.NaT)
            else:
                peak_time.append(obj.peak_intensity.time)
                
            if obj.peak_intensity_max.time == None:
                max_time.append(pd.NaT)
            else:
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
            if event_lengths == []:
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
            if threshold_crossings == []:
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
        #print(df)
        
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
        
        :all_energy_channels: (array of dict) all energy channels
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
        print('validate: Extracting observation arrays for matching '
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
    matched_obs = [] #matched Observation objects
    td_eruption_thresh_cross = [] #array of all obs in prediction window
    matched_sep_source = [] #filenames of the final matched observation file
    observed_thresh_cross = [] #single final matched value
    thresh_key = objh.threshold_to_key(threshold)
    
    #Loop over SPHINX objects and extract observed matched threshold crossing
    #time and time difference between eruption used by model and
    #threshold crossing time
    for obj in objs:
        try:
            obs_arr = obj.prediction_observation_windows_overlap
            match_sep = obj.observed_match_sep_source[thresh_key]
            obs_thresh_cross = obj.observed_threshold_crossing[thresh_key].crossing_time
            td = obj.time_difference_eruptions_threshold_crossing[thresh_key]
        
            matched_obs.append(obs_arr)
            matched_sep_source.append(match_sep)
            observed_thresh_cross.append(obs_thresh_cross)
            td_eruption_thresh_cross.append(td)
        except:
            matched_obs.append([None])
            matched_sep_source.append(None)
            observed_thresh_cross.append(pd.NaT)
            td_eruption_thresh_cross.append([pd.NaT])

        
    #All arrays are now filled with values only associated
    #with the desired energy channel and threshold
    #Put into a pandas dataframe
    pd_dict = {'matched_observations': matched_obs,
               'matched_sep': matched_sep_source,
               'td_eruption_thresh_cross': td_eruption_thresh_cross,
               'observed_threshold_crossing_time': observed_thresh_cross
    }
    df = pd.DataFrame(data=pd_dict)
#    print(df)
        
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
    
        :decision: (array of booleans) have same indices as the
            observations in obs. True if overlap, False if no overlap
            
    """
    pred_win_st = fcast.prediction_window_start
    pred_win_end = fcast.prediction_window_end
    print("Prediction window: " + str(pred_win_st) + " to "
        + str(pred_win_end))
        
    pred_interval = pd.Interval(pd.Timestamp(pred_win_st),
        pd.Timestamp(pred_win_end))
    
    overlaps_bool = []
    for i in range(len(obs['observation_window_start'])):
        obs_interval =\
            pd.Interval(pd.Timestamp(obs['observation_window_start'][i]),
                pd.Timestamp(obs['observation_window_end'][i]))
    
        if pred_interval.overlaps(obs_interval):
            overlaps_bool.append(True)
        else:
            overlaps_bool.append(False)
            
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
    
    #Extract the forecast information needed for matching
#    energy_channel = fcast.energy_channel
#    energy_key = objh.energy_channel_to_key(energy_channel)

    #Return if no prediction
    if fcast.prediction_window_start == None:
        return [None]*len(obs_values[energy_key]['dataframes'][0])
        

    #Pull out the observations needed for matching
    #We will have a table with needed information about each
    #observation
    #Extract correct energy channel and any threshold
    obs = obs_values[energy_key]['dataframes'][0]
    
    #1. Does the Prediction Window overlap with the Observation Window?
    is_win_overlap = pred_and_obs_overlap(fcast, obs)
    
    return is_win_overlap


def observed_time_in_pred_win(fcast, obs_values, obs_key, energy_key):
    """ Find whether an observed time occurred inside of the prediction
        window across all of the observations
        
        Returns boolean indicating whether the threshold was
        actively crossed inside of the prediction window.
    
    Input:
    
        :fcast: (Forecast object) single forecast
        :obs_df: (dict of pandas DataFrames) for a single
            energy channel and all observed threshold.
        :obs_key: (string) identifies which time in the pandas
            Dataframe should be compared, 'threshold_crossing_time'
            
    Output:
        
        :contains_obs_time: (array of bool) array indices match the order
            of the Observation objects in obs_objs[energy_key]
            where energy_key is the key made from the fcast
            energy_channel
    
    """    
    pred_win_st = fcast.prediction_window_start
    pred_win_end = fcast.prediction_window_end

    #Return if no prediction
    if fcast.prediction_window_start == None:
        return [None]*len(obs_values[energy_key]['dataframes'][0])
           
    #Extract pandas dataframe for correct energy and threshold
    obs_df = obs_values[energy_key]['dataframes'][0] #all obs for threshold

    #Check if a threshold crossing is contained inside of the
    #prediction window; returns False if pd.NaT
    contains_obs_time = (obs_df[obs_key] >= pd.Timestamp(pred_win_st)) \
        & (obs_df[obs_key] < pd.Timestamp(pred_win_end))
 
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
    if fcast.prediction_window_start == None:
        return [None]*len(obs_values[energy_key]['dataframes'][0])
       
    #Extract desired threshold
    obs_thresholds = obs_values[energy_key]['thresholds']
    try:
        ix = obs_thresholds.index(threshold)
    except:
        return [None]*len(obs_values[energy_key]['dataframes'][0])
    
    #Extract pandas dataframe for correct energy and threshold
    obs = obs_values[energy_key]['dataframes'][ix] #all obs for threshold

    #Check if a threshold crossing is contained inside of the
    #prediction window; returns False if pd.NaT
    contains_obs_time = (obs[obs_key] >= pd.Timestamp(pred_win_st)) \
        & (obs[obs_key] < pd.Timestamp(pred_win_end))
 
    return list(contains_obs_time)


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
    nat = [ix for ix in range(len(list(time_diff))) if pd.isnull(time_diff[ix])]
    for ix in nat:
        is_before[ix] = None
            
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
        
        For all output, indices of arrays match with indices of
        obs_values and observation_objs
        
        :is_onset_peak_in_pred_win: (boolean array) indicates if
            onset peak is inside of the prediction window.
        :td_trigger_onset_peak: (float array) time difference in hours
            between last trigger time and peak_intensity time
        :is_trigger_before_onset_peak: (boolean array) indcates if
            the last trigger time is before the onset peak
        :td_input_onset_peak: (float array) time difference in hours
            between last input time and peak_intensity time
         :is_input_before_onset_peak: (boolean array) indcates if
            the last input time is before the onset peak

        
    """
    last_trigger_time = sphinx.last_trigger_time
    last_input_time = sphinx.last_input_time
    
    ###### MATCHING: ONSET PEAK IN PREDICTION WINDOW #####
    #Is the onset peak inside of the prediction window?
    #Get the time difference - negative means trigger is before
    is_onset_peak_in_pred_win = observed_time_in_pred_win(fcast,
                                obs_values, 'peak_time', energy_key)

    if sphinx.overlapping_indices != []:
        for ix in sphinx.overlapping_indices:
            sphinx.peak_intensity_time_in_prediction_window.append(is_onset_peak_in_pred_win[ix])
            #logging
#            print("  " +
#            str(sphinx.peak_intensity_time_in_prediction_window))
#            print("  Observed peak_intensity time: "
 #           + str(observation_objs[ix].peak_intensity.time))


    ###### MATCHING: TRIGGERS BEFORE ONSET PEAK #####
    #Is the last trigger time before the onset peak time?
    #Get the time difference - negative means trigger is before
    td_trigger_onset_peak = time_diff(last_trigger_time, obs_values,
                                'peak_time', energy_key)
    is_trigger_before_onset_peak = is_time_before(last_trigger_time,
                                obs_values, 'peak_time', energy_key)

    if sphinx.overlapping_indices != []:
        for ix in sphinx.overlapping_indices:
            sphinx.triggers_before_peak_intensity.append(is_trigger_before_onset_peak[ix])
            sphinx.time_difference_triggers_peak_intensity.append(td_trigger_onset_peak[ix])
                #logging
#                print("  " +
#                str(sphinx.triggers_before_peak_intensity[ix]))
#                print("  " +
#                str(sphinx.time_difference_triggers_peak_intensity[ix]))
#                print("  Observed peak_intensity time: "
#                + str(observation_objs[idx[ix]].peak_intensity.time))


    ###### MATCHING: INPUTS BEFORE ONSET PEAK #####
    #Is the last trigger time before the onset peak time?
    td_input_onset_peak = time_diff(last_input_time, obs_values,
                            'peak_time', energy_key)
    is_input_before_onset_peak = is_time_before(last_input_time,
                            obs_values, 'peak_time', energy_key)

    if sphinx.overlapping_indices != []:
        for ix in sphinx.overlapping_indices:
            sphinx.inputs_before_peak_intensity.append(is_input_before_onset_peak[ix])
            sphinx.time_difference_inputs_peak_intensity.append(td_input_onset_peak[ix])
                #logging
#                print("  " +
#                str(sphinx.inputs_before_peak_intensity[ix]))
#                print("  " +
#                str(sphinx.time_difference_inputs_peak_intensity[ix]))
#                print("  Observed peak_intensity time: "
#                + str(observation_objs[idx[ix]].peak_intensity.time))


    return is_onset_peak_in_pred_win, td_trigger_onset_peak, \
        is_trigger_before_onset_peak, td_input_onset_peak,\
        is_input_before_onset_peak



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
        
        For all output, indices of arrays match with indices of
        obs_values and observation_objs
        
        :is_max_flux_in_pred_win: (boolean array) indicates if
            peak_intensity_max is inside of the prediction window.
        :td_trigger_max_flux: (float array) time difference in hours
            between last trigger time and peak_intensity_max time
        :is_trigger_before_max_flux: (boolean array) indcates if
            the last trigger time is before the max flux
        :td_input_max_flux: (float array) time difference in hours
            between last input time and peak_intensity_max time
         :is_input_before_max_flux: (boolean array) indcates if
            the last input time is before the peak_intensity_max

        
    """
    last_trigger_time = sphinx.last_trigger_time
    last_input_time = sphinx.last_input_time
    
    ###### MATCHING: MAX FLUX IN PREDICTION WINDOW #####
    #Is the max flux inside of the prediction window?
    #Get the time difference - negative means trigger is before
    is_max_flux_in_pred_win = observed_time_in_pred_win(fcast,
                            obs_values, 'max_time', energy_key)

    if sphinx.overlapping_indices != []:
        for ix in sphinx.overlapping_indices:
            sphinx.peak_intensity_max_time_in_prediction_window.append(is_max_flux_in_pred_win[ix])
                #logging
#                print("  " +
#                str(sphinx.peak_intensity_max_time_in_prediction_window[ix]))
#                print("  Observed peak_intensity_max time: " +
#                str(observation_objs[idx[ix]].peak_intensity_max.time))


    ###### MATCHING: TRIGGERS BEFORE MAX FLUX #####
    #Is the last trigger time before the max flux time?
    #Get the time difference - negative means trigger is before
    td_trigger_max_time = time_diff(last_trigger_time, obs_values,
                            'max_time', energy_key)
    is_trigger_before_max_time = is_time_before(last_trigger_time,
                            obs_values, 'max_time', energy_key)

    if sphinx.overlapping_indices != []:
        for ix in sphinx.overlapping_indices:
            sphinx.triggers_before_peak_intensity_max.append(is_trigger_before_max_time[ix])
            sphinx.time_difference_triggers_peak_intensity_max.append(td_trigger_max_time[ix])
                #logging
#                print("  " +
#                str(sphinx.triggers_before_peak_intensity_max[ix]))
#                print("  " +
#                str(sphinx.time_difference_triggers_peak_intensity_max[ix]))
#                print("  Observed peak_intensity_max time: "
#                + str(observation_objs[idx[ix]].peak_intensity.time))


    ###### MATCHING: INPUTS BEFORE MAX FLUX TIME #####
    #Is the last trigger time before the max flux time?
    td_input_max_time = time_diff(last_input_time, obs_values,
                        'max_time', energy_key)
    is_input_before_max_time = is_time_before(last_input_time,
                        obs_values, 'max_time', energy_key)

    if sphinx.overlapping_indices != []:
        for ix in sphinx.overlapping_indices:
            sphinx.inputs_before_peak_intensity_max.append(is_input_before_max_time[ix])
            sphinx.time_difference_inputs_peak_intensity_max.append(td_input_max_time[ix])
                #logging
#                print("  " +
#                str(sphinx.inputs_before_peak_intensity_max[ix]))
#                print("  " +
#                str(sphinx.time_difference_inputs_peak_intensity_max[ix]))
#                print("  Observed peak_intensity time: "
#                + str(observation_objs[idx[ix]].peak_intensity_max.time))

    return is_max_flux_in_pred_win, td_trigger_max_time,\
        is_trigger_before_max_time, td_input_max_time, \
        is_input_before_max_time



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
        :thresh: (dict) specific threshold for threshold crossing
            
    Ouput:
        
        For all output, indices of arrays match with indices of
        obs_values and observation_objs
        
        :contains_thresh_cross: (boolean array) indicates if a threshold
            was crossed inside the prediction window
        
    """
    thresh_key = objh.threshold_to_key(thresh)
    
    contains_thresh_cross = observed_time_in_pred_win_thresh(fcast,
        obs_values, 'threshold_crossing_time', energy_key, thresh)

    if sphinx.overlapping_indices != []:
        for ix in sphinx.overlapping_indices:
            sphinx.threshold_crossed_in_pred_win[thresh_key].append(contains_thresh_cross[ix])
            #logging
#            print("  " +
#                str(sphinx.threshold_crossed_in_pred_win[thresh_key][ix]))


    return contains_thresh_cross



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
        
        For all output, indices of arrays match with indices of
        obs_values and observation_objs
        
        :td_trigger_thresh_cross: (float array) hours between last trigger time
            and threshold crossing time
        :is_trigger_before_start: (boolean array) indicates if trigger is
            before the threshold was crossed
        :td_input_thresh_cross: (float array) hours between last input time
            and threshold crossing time
        :is_input_before_start: (boolean array) indicates if input is
            before the threshold was crossed

 
    """
    last_trigger_time = sphinx.last_trigger_time
    last_input_time = sphinx.last_input_time
    thresh_key = objh.threshold_to_key(thresh)


    ###### MATCHING: TRIGGERS/INPUTS BEFORE SEP #####
    td_trigger_thresh_cross = time_diff_thresh(last_trigger_time, obs_values,
        'threshold_crossing_time', energy_key, thresh)
    is_trigger_before_start = is_time_before_thresh(last_trigger_time,
        obs_values, 'threshold_crossing_time', energy_key, thresh)

    if sphinx.overlapping_indices != []:
        for ix in sphinx.overlapping_indices:
            sphinx.triggers_before_threshold_crossing[thresh_key].append(is_trigger_before_start[ix])
            sphinx.time_difference_triggers_threshold_crossing[thresh_key].append(td_trigger_thresh_cross[ix])
                #logging
#                print("  " +
#                str(sphinx.triggers_before_threshold_crossing[ix]))
#                print("  " +
#                str(sphinx.time_difference_triggers_threshold_crossing[ix]))
#                threshold_crossing_time = \
#                objh.get_threshold_crossing_time(observation_objs[idx[ix]],
#                thresh)
#                print("  Observed threshold crossing time: "
#                + str(threshold_crossing_time))


    ###### MATCHING: INPUTS BEFORE SEP #####
    #Is the last input before the SEP event start time?
    td_input_thresh_cross = time_diff_thresh(last_input_time, obs_values,
        'threshold_crossing_time', energy_key, thresh)
    is_input_before_start = is_time_before_thresh(last_input_time, obs_values,
        'threshold_crossing_time', energy_key, thresh)

    if sphinx.overlapping_indices != []:
        for ix in sphinx.overlapping_indices:
            sphinx.inputs_before_threshold_crossing[thresh_key].append(is_input_before_start[ix])
            sphinx.time_difference_inputs_threshold_crossing[thresh_key].append(td_input_thresh_cross[ix])
                #logging
#                print("  " + str(sphinx.inputs_before_threshold_crossing[ix]))
#                print("  " +
#                str(sphinx.time_difference_inputs_threshold_crossing[ix]))
#                threshold_crossing_time =\
#                objh.get_threshold_crossing_time(observation_objs[idx[ix]],
#                thresh)
#                print("  Observed threshold crossing time: "
#                + str(threshold_crossing_time))

    return td_trigger_thresh_cross, is_trigger_before_start, \
        td_input_thresh_cross, is_input_before_start
 
 
 
def before_sep_end(sphinx, fcast, obs_values, observation_objs,
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
        
        For all output, indices of arrays match with indices of
        obs_values and observation_objs
        
        :td_trigger_thresh_cross: (float array) hours between last trigger time
            and threshold crossing time
        :is_trigger_before_start: (boolean array) indicates if trigger is
            before the threshold was crossed
        :td_input_thresh_cross: (float array) hours between last input time
            and threshold crossing time
        :is_input_before_start: (boolean array) indicates if input is
            before the threshold was crossed

 
    """
    last_trigger_time = sphinx.last_trigger_time
    last_input_time = sphinx.last_input_time
    thresh_key = objh.threshold_to_key(thresh)


    ###### MATCHING: TRIGGERS/INPUTS BEFORE SEP #####
    td_trigger_sep_end = time_diff_thresh(last_trigger_time, obs_values,
        'end_time', energy_key, thresh)
    is_trigger_before_end = is_time_before_thresh(last_trigger_time,
        obs_values, 'end_time', energy_key, thresh)

    if sphinx.overlapping_indices != []:
        for ix in sphinx.overlapping_indices:
            sphinx.triggers_before_sep_end[thresh_key].append(is_trigger_before_end[ix])
            sphinx.time_difference_triggers_sep_end[thresh_key].append(td_trigger_sep_end[ix])
                #logging
#                print("  " +
#                str(sphinx.triggers_before_sep_end[ix]))
#                print("  " +
#                str(sphinx.time_difference_triggers_sep_end[ix]))


    ###### MATCHING: INPUTS BEFORE SEP #####
    #Is the last input before the SEP event start time?
    td_input_sep_end = time_diff_thresh(last_input_time, obs_values,
        'end_time', energy_key, thresh)
    is_input_before_end = is_time_before_thresh(last_input_time, obs_values,
        'end_time', energy_key, thresh)

    if sphinx.overlapping_indices != []:
        for ix in sphinx.overlapping_indices:
            sphinx.inputs_before_sep_end[thresh_key].append(is_input_before_end[ix])
            sphinx.time_difference_inputs_sep_end[thresh_key].append(td_input_sep_end[ix])
                #logging
#                print("  " + str(sphinx.inputs_before_sep_end[ix]))
#                print("  " +
#                str(sphinx.time_difference_inputs_sep_end[ix]))

    return td_trigger_sep_end, is_trigger_before_end, \
        td_input_sep_end, is_input_before_end
 
 


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
        
        For all output, indices of arrays match with indices of
        obs_values and observation_objs
        
        :td_eruption_thresh_cross: (float array) hours between eruption time
            and threshold crossing time
        :is_eruption_before_start: (boolean array) indicates if eruption is
            before the threshold was crossed

 
    """
    last_eruption_time = sphinx.last_eruption_time
    thresh_key = objh.threshold_to_key(thresh)


    ###### MATCHING: ERUPTION BEFORE SEP #####
    td_eruption_thresh_cross = time_diff_thresh(last_eruption_time, obs_values,
        'threshold_crossing_time', energy_key, thresh)
    is_eruption_before_start = is_time_before_thresh(last_eruption_time,
        obs_values, 'threshold_crossing_time', energy_key, thresh)

    if sphinx.overlapping_indices != []:
        for ix in sphinx.overlapping_indices:
            sphinx.eruptions_before_threshold_crossing[thresh_key].append(is_eruption_before_start[ix])
            sphinx.time_difference_eruptions_threshold_crossing[thresh_key].append(td_eruption_thresh_cross[ix])
            #logging
#                print("  " +
#                str(sphinx.eruptions_before_threshold_crossing[ix]))
#                print("  " +
#                str(sphinx.time_difference_eruptions_threshold_crossing[ix]))
#                threshold_crossing_time = \
#                objh.get_threshold_crossing_time(observation_objs[idx[ix]],
#                thresh)
#                print("  Observed threshold crossing time: "
#                + str(threshold_crossing_time))

    return td_eruption_thresh_cross, is_eruption_before_start



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
    if is_trigger_before_start != None:
        trigger_input_start = is_trigger_before_start
    if is_input_before_start != None:
        if trigger_input_start == None:
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
    if is_trigger_before_end != None:
        trigger_input_end = is_trigger_before_end
    if is_input_before_end != None:
        if trigger_input_end == None:
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
        :threshold: (dict) {'threshold':10, 'units': Unit("MeV")}
            
    Ouput:
        
        For all output, indices of arrays match with indices of
        obs_values and observation_objs
        
        :is_ongoing_event: (boolean array) indicates if
            the prediction window starts while an event is ongoing
            
    """
    
    pred_win_st = fcast.prediction_window_start
    pred_win_end = fcast.prediction_window_end
    thresh_key = objh.threshold_to_key(threshold)

    #Return if no prediction
    if fcast.prediction_window_start == None:
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
    overlap_start = (obs['start_time'] >= pd.Timestamp(pred_win_st))\
        & (obs['start_time'] < pd.Timestamp(pred_win_end))
    overlap_end = (obs['end_time'] >= pd.Timestamp(pred_win_st))\
        & (obs['start_time'] < pd.Timestamp(pred_win_end))

    is_overlap = overlap_start | overlap_end
    is_overlap = list(is_overlap)
    
    if sphinx.overlapping_indices != []:
        for ix in sphinx.overlapping_indices:
            sphinx.prediction_window_sep_overlap[thresh_key].append(is_overlap[ix])


    return is_overlap


 
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
        
        For all output, indices of arrays match with indices of
        obs_values and observation_objs
        
        :is_ongoing_event: (boolean array) indicates if
            the prediction window starts while an event is ongoing
            
    """
    pred_win_st = fcast.prediction_window_start
    pred_win_end = fcast.prediction_window_end
    thresh_key = objh.threshold_to_key(threshold)

    #Return if no prediction
    if fcast.prediction_window_start == None:
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
    sep_start = None
    sep_end = None
#    print("Ongoing observed SEP event at start of prediction window (if any):")
    for i in range(len(obs['start_time'])):
        if pd.isnull(obs['start_time'][i]):
            is_ongoing.append(None)
            continue
        else:
            sep_event = pd.Interval(pd.Timestamp(obs['start_time'][i]),
                            pd.Timestamp(obs['end_time'][i]))

        #Is the prediction window start inside of an SEP event?
        if sep_event.overlaps(pred_win_begin):
            is_ongoing.append(True)
            sep_start = obs['start_time'][i]
            sep_end = obs['end_time'][i]

            #logging
#            print("  " + str(observation_objs[i].source))
#            print("  Observed SEP event: "
#            + str(sep_start) + " to " + str(sep_end))
            
        else:
            is_ongoing.append(False)

    if sphinx.overlapping_indices != []:
        for ix in sphinx.overlapping_indices:
            sphinx.observed_ongoing_events[thresh_key].append(is_ongoing[ix])

    return is_ongoing


#def ongoing_status(obj, energy_channel, fcast_thresh):
#    """ Check if the model itself is reporting a ongoing SEP event
#        in the particle_intensity triggers.
#
#    """




###### MATCHING AND EXTRACTING OBSERVED VALUES ########
def match_observed_onset_peak(sphinx, observation_obj, is_win_overlap,
    is_eruption_in_range, is_trigger_before_onset_peak,
    is_input_before_onset_peak, is_pred_sep_overlap):
    """ Apply criteria to determine if a particular observation matches
        with the peak intensity prediction.
        If identified, save the observed peak_intensity to the SPHINX object.
       
        - Prediction window overlaps with observation
        - Last eruption within 48 hrs - 15 mins before threshold crossing
        - The prediction window overlaps with an SEP event in any threshold -
            only a comparison when there is an SEP event
            NEED TO ADD IN COMPARISON WHEN NO SEP EVENT
        - The last trigger/input time if before the observed peak intensity

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
    if is_trigger_before_onset_peak != None:
        trigger_input_peak = is_trigger_before_onset_peak
    if is_input_before_onset_peak != None:
        if trigger_input_peak == None:
            trigger_input_peak = is_input_before_onset_peak
        else:
            trigger_input_peak = trigger_input_peak and \
                is_input_before_onset_peak
    
    #MATCHING
    peak_criteria = is_win_overlap and trigger_input_peak and is_pred_sep_overlap
    if is_eruption_in_range != None:
        peak_criteria = (peak_criteria and is_eruption_in_range)
    
    if not peak_criteria:
        if not is_win_overlap:
            sphinx.peak_intensity_match_status = "No Matched Observation"
        if not is_pred_sep_overlap:
            sphinx.peak_intensity_match_status = "No SEP Event"
        if is_eruption_in_range != None and not is_eruption_in_range:
            sphinx.peak_intensity_match_status = "Eruption Out of Range"
        if not trigger_input_peak: #precedence
            sphinx.peak_intensity_match_status = "Trigger/Input after Observed Phenomenon"
 
 
    if peak_criteria:
#        print("Observed peak_intensity matched:")
#        print(observation_obj.source)
#        print(observation_obj.peak_intensity.intensity)
#        print(observation_obj.peak_intensity.time)
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
        - Last eruption within 48 hrs - 15 mins before threshold crossing
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
    if is_trigger_before_max_time != None:
        trigger_input_max = is_trigger_before_max_time
    if is_input_before_max_time != None:
        if trigger_input_max == None:
            trigger_input_max = is_input_before_max_time
        else:
            trigger_input_max = trigger_input_max and \
                is_input_before_max_time
    
    #MATCHING
    max_criteria = is_win_overlap and trigger_input_max and is_pred_sep_overlap
    if is_eruption_in_range != None:
        max_criteria = (max_criteria and is_eruption_in_range)

    if not max_criteria:
        if not is_win_overlap:
            sphinx.peak_intensity_max_match_status = "No Matched Observation"
        if not is_pred_sep_overlap:
            sphinx.peak_intensity_max_match_status = "No SEP Event"
        if is_eruption_in_range != None and not is_eruption_in_range:
            sphinx.peak_intensity_max_match_status = "Eruption Out of Range"
        if not trigger_input_max: #precedence
            sphinx.peak_intensity_max_match_status = "Trigger/Input after Observed Phenomenon"

    if max_criteria:
#        print("Observed peak_intensity_max matched:")
#        print(observation_obj.source)
#        print(observation_obj.peak_intensity_max.intensity)
#        print(observation_obj.peak_intensity_max.time)
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
        - Last eruption within 48 hrs - 15 mins before threshold crossing
        - The last trigger/input time if before the threshold crossing
        - Threshold crossed in prediction window = False All Clear
        - No threshold crossed in prediction window = True All Clear

    Input:
        
        The ith entry for the various boolean arrays created that are the length
        of all of the observations read into the code.

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
    if sphinx.observed_all_clear.all_clear_boolean == False:
        return None
    
    #Save thresholds in All_Clear object
    ac = cl.All_Clear(None,observation_obj.all_clear.threshold,
        observation_obj.all_clear.threshold_units,
        observation_obj.all_clear.probability_threshold)
    sphinx.observed_all_clear = ac
    
    all_clear_status = None
    
    if not is_win_overlap:
        all_clear_status = None
        sphinx.all_clear_match_status = "No Matched Observation"
        return all_clear_status
        
    #Prediction and observation windows overlap
    #If ongoing SEP event at start of prediction window, no match
    if is_sep_ongoing:
        all_clear_status = None
        sphinx.observed_match_all_clear_source = observation_obj.source
        sphinx.all_clear_match_status = "Ongoing SEP Event"
        return all_clear_status
    
    #If there is no threshold crossing in prediction window,
    #then observed all clear is True
    if not contains_thresh_cross:
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
        if is_eruption_in_range != None:
            if not is_eruption_in_range:
                all_clear_status = True
                sphinx.observed_match_all_clear_source = observation_obj.source
                sphinx.all_clear_match_status = "Eruption Out of Range"

#    print("Prediction window: " + str(sphinx.prediction.prediction_window_start) + " to "
#        + str(sphinx.prediction.prediction_window_end))
#    #All clear status
#    print("Observed all_clear matched:")
#    print("  " + observation_obj.source)
#    print("  " + str(all_clear_status))
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
        - start time, threshold crossing time, fluence, fluence spectrum
       
       Matching:
        - Prediction window overlaps with observation
        - There is no ongoing SEP event at the start of the prediction window
        - Last eruption within 48 hrs - 15 mins before threshold crossing
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
    
    if not is_win_overlap:
        sep_status = None
        sphinx.sep_match_status[thresh_key] = "No Matched Observation"
        return sep_status
        
    #Prediction and observation windows overlap
    #If ongoing SEP event at start of prediction window, no match
    if is_sep_ongoing:
        sep_status = False
        prob.probability_value = None #ongoing event
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
            prob.probability_value = None
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

    
    
#    print("Prediction window: " + str(sphinx.prediction.prediction_window_start) + " to "
#        + str(sphinx.prediction.prediction_window_end))
#    #All clear status
#    print("Observed SEP event matched:")
#    print("  " + observation_obj.source)
    
    #Threshold Crossing
    threshold_crossing_time = None
    for th in observation_obj.threshold_crossings:
        if th.threshold != thresh['threshold']:
            continue
        sphinx.observed_match_sep_source[thresh_key] = observation_obj.source
        sphinx.observed_threshold_crossing[thresh_key] = th
 
 
    #Start time and channel fluence
    start_time = None
    for event in observation_obj.event_lengths:
        if event.threshold != thresh['threshold']:
            continue
        sphinx.observed_match_sep_source[thresh_key] = observation_obj.source
        sphinx.observed_start_time[thresh_key] = event.start_time

    fluence = None
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
    if not is_win_overlap:
        end_status = None
        sphinx.end_time_match_status[thresh_key] = "No Matched Observation"
        return end_status
        
    #The prediction window must overlap with an SEP event
    if not is_pred_sep_overlap:
        end_status = False #no SEP event, no values
        sphinx.end_time_match_status[thresh_key] = "No SEP Event"
        return end_status

    #If there is an SEP event, the eruption must occur in the right time range
    if is_eruption_in_range != None:
        if not is_eruption_in_range:
            sep_status = None
            sphinx.end_time_match_status[thresh_key] = "Eruption Out of Range"
            return sep_status
    #The triggers and inputs must all be before threshold crossing
    if trigger_input_end:
        end_status = True
        sphinx.end_time_match_status[thresh_key] = "SEP Event"
    else:
        end_status = None
        sphinx.end_time_match_status[thresh_key] = "Trigger/Input after Observed Phenomenon"
        return end_status
    
    #Matched End Time
#    print("Observed SEP end time matched:")
#    print("  " + observation_obj.source)
 
    #Start time and channel fluence
    end_time = None
    for i in range(len(observation_obj.event_lengths)):
        event = observation_obj.event_lengths[i]
        if event.threshold != thresh['threshold']:
            continue
        sphinx.observed_match_sep_source[thresh_key] = observation_obj.source
        sphinx.observed_event_length[thresh_key] = event
        sphinx.observed_end_time[thresh_key] = event.end_time

 #   print(sphinx.observed_end_times)

    return end_status



def match_time_profile(sphinx, observation_obj, thresh, is_win_overlap,
    is_eruption_in_range, trigger_input_end, is_pred_sep_overlap):
    """ Apply criteria to determine if a forecast occurs prior to SEP
        end time and extract the predicted time profile. If the forecast
        is before the SEP end time, then compare the time profile.
               
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
    if not is_win_overlap:
        end_status = None
        sphinx.end_time_match_status[thresh_key] = "No Matched Observation"
        return end_status
        
    #The prediction window must overlap with an SEP event
    if not is_pred_sep_overlap:
        end_status = False #no SEP event, no values
        sphinx.end_time_match_status[thresh_key] = "No SEP Event"
        return end_status

    #If there is an SEP event, the eruption must occur in the right time range
    if is_eruption_in_range != None:
        if not is_eruption_in_range:
            sep_status = None
            sphinx.end_time_match_status[thresh_key] = "Eruption Out of Range"
            return sep_status
    #The triggers and inputs must all be before threshold crossing
    if trigger_input_end:
        end_status = True
        sphinx.end_time_match_status[thresh_key] = "SEP Event"
    else:
        end_status = None
        sphinx.end_time_match_status[thresh_key] = "Trigger/Input after Observed Phenomenon"
        return end_status
    
    #Matched End Time
#    print("Observed SEP end time matched:")
#    print("  " + observation_obj.source)
 
    #Start time and channel fluence
    time_prof = None
    for i in range(len(observation_obj.event_lengths)):
        event = observation_obj.event_lengths[i]
        if event.threshold != thresh['threshold']:
            continue
        sphinx.observed_match_sep_source[thresh_key] = observation_obj.source
        sphinx.observed_time_profile[thresh_key] = observation_obj.source

 #   print(sphinx.observed_end_times)

    return end_status




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
    print("\n====== All observed SEP events ======")
    for model in model_names:
        print("\n")
        print("Model: " + model)
        for energy_key in all_energy_channels:
            print("Energy Channel: " + str(energy_key))
            for obs_thresh in obs_values[energy_key]['thresholds']:
                print("Threshold: " + str(obs_thresh))
                obs_thresh_key = objh.threshold_to_key(obs_thresh)
                nev =\
                len(observed_sep_events[model][energy_key][obs_thresh_key])
                print(str(nev) + " SEP Events: " +
                str(observed_sep_events[model][energy_key][obs_thresh_key]))



def revise_eruption_matches(matched_sphinx, all_energy_channels, obs_values,
        model_names, observed_sep_events):
    """ It may be that there are multiple flares or CMEs in
        a short time period that generate predictions with
        large prediction windows. These multiple forecasts with different
        triggers can match to the same SEP event. Since only one CME/flare
        combination is generally responsible for a given SEP, the code must
        pick the appropriate forecast to associate with that SEP event.
        
        In the case that there are multiple predictions with different
        CME/flare triggers, then the following selections will be applied:
        
        - The eruption must be within 15 minutes to 48 hours prior to threshold
            crossing (this selection is already made in match_all_forecasts).
        - Associate only the prediction with the eruption closest in time to
            the SEP event (this subroutine). Predictions with earlier
            CME/flare triggers matched to the same SEP event will be
            unassociated, even if they originally fell within the specified
            range.
    
    Input:
    
        :matched_sphinx: (dictionary of SPHINX objects) contains
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
    
        None, but the objects inside matched_sphinx will be updated.
        
    """
    print("\n====== REVISING MATCHES WITH FLARE/CME TRIGGERS ======")
    for model in model_names:
        #if the model doesn't use eruptions as triggers, then this
        #doesn't apply
        if not matched_sphinx[model]['uses_eruptions']:
            continue

        for energy_key in all_energy_channels:
            for threshold in obs_values[energy_key]['thresholds']:
                print("Checking whether to revise matching for " + model
                + ", " + str(energy_key) + ", " + str(threshold))
                #Pull out all the observed SEP events inside of the
                #model prediction windows for a given energy channel
                #and threshold. Want to identify if multiple predictions
                #using different eruptions as triggers were matched to the
                #same SEP event.
                thresh_key = objh.threshold_to_key(threshold)
                obs_sep =\
                    observed_sep_events[model][energy_key][thresh_key]
                if obs_sep == []: continue

                #Create a dataframe containing info from all forecasts for a
                #single model, energy channel, and threshold. To allow easy
                #comparison from forecast to forecast.
                #Keys 'matched_observations', 'matched_sep',
                #'td_eruption_thresh_cross' and
                #'observed_threshold_crossing_time'
                #Dataframe indices match the indices in matched_sphinx
                spx_df =\
                create_matched_model_array(matched_sphinx[model][energy_key],
                threshold)

                #Identify the forecasts matched to the same SEP event
                for sep in obs_sep:
                    obs_thresh_cross =\
                        spx_df['observed_threshold_crossing_time'].tolist()

                    idx_event = [ix for ix in range(len(obs_thresh_cross)) if obs_thresh_cross[ix] == pd.Timestamp(sep)]
                    
                    #If no or only one match, nothing to do
                    if len(idx_event) == 0 or len(idx_event) == 1: continue
                
                    #Time differences are saved in order of the observation
                    #files that fell inside the predictions windows. Identify
                    #the correct entry by comparing with filename of the
                    #matched sep observations
                    sep_source = spx_df['matched_sep'].take(idx_event).tolist()
                    matched_obs = spx_df['matched_observations'].take(idx_event).tolist()

                    #Time difference between eruptions and threshold crossing
                    #Negative values are before threshold crossings (in hours)
                    td_eruptions_array =\
                    spx_df['td_eruption_thresh_cross'].take(idx_event).tolist()

                    td_eruptions = []
                    for j in range(len(idx_event)):
                        if sep_source[j] == None:
                            td_eruptions.append(None)
                            continue
                        
                        obs_idx = [ix for ix in range(len(matched_obs[j])) if matched_obs[j][ix].source == sep_source[j]]
                        td_eruptions.append(td_eruptions_array[j][obs_idx[0]])

                    # TODO: fix the ugly hack to cast to np array to handle None values entered from above
                    td_eruptions = np.array(td_eruptions, dtype='float') # Turn None into nan
                    
                    #Need to find which eruption is the closest
                    #to the SEP event and unmatch all the other forecasts
                    #Since all the time differences are necessarily negative,
                    #the max time will be the one closest to the SEP event
                    #and the preferable match
                    best_eruption = np.nanmax(td_eruptions)

                    # If all td_eruptions  were None, then best_eruption will be a nan
                    # In this case no unmatching is necessary
                    if np.isnan(best_eruption):
                        continue

                    #If all the time differences are the same, then the
                    #same eruption was used in the forecasts and nothing
                    #to do
                    same_idx = [ix for ix in range(len(td_eruptions)) if td_eruptions[ix] == best_eruption]
                    if len(same_idx) == len(td_eruptions): continue

                    #The eruptions and forecasts to unmatch
                    #Get back to the indices associated with the sphinx objects
                    # Note that if td_eruptions[ix] is nan the index will never be added
                    #If eruptions near an SEP event are within a few hours of each other
                    #cannot really say which is the correct one to associate with the
                    #SEP event, so keep both. e.g. March 7, 2012, e.g. slight change
                    #in CME start time when CME refit
                    adj_idx = [ix for ix in range(len(td_eruptions)) if td_eruptions[ix] < (best_eruption - 3.0)]
                    sphx_idx = []
                    for ix in adj_idx:
                        sphx_idx = idx_event[ix]

                        #Identify forecast being unmatched
                        print("====== UNMATCHING FORECAST FROM SEP EVENT ====")
                        print("Forecast: " +
                        str(matched_sphinx[model][energy_key][sphx_idx].prediction.source))
                        print("Last Eruption Time: " +
                        str(matched_sphinx[model][energy_key][sphx_idx].last_eruption_time))
                        print("Initially matched to SEP Event: " + str(matched_sphinx[model][energy_key][sphx_idx].observed_threshold_crossing[thresh_key].crossing_time))
                        print("Observation source: " + str(matched_sphinx[model][energy_key][sphx_idx].observed_match_sep_source[thresh_key]))

                        #Unmatch
                        matched_sphinx[model][energy_key][sphx_idx].unmatch(threshold)
                        print("-------- UNMATCHED ----------\n")
 



#################################
####ALL MATCHING CRITERIA #######
#################################
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
        
        :matched_sphinx: (dictionary of SPHINX objects) dictionary organized
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
    print("match_all_forecasts: Compiling all observations into a dataframe: " + str(datetime.datetime.now()))

    obs_values = compile_all_obs(all_energy_channels, obs_objs)

    print("match_all_forecasts: Completed dataframe, identifying all thresholds applied to observed energy channels: " + str(datetime.datetime.now()))

    #Gather all thresholds applied in the observations for each energy channel
    all_obs_thresholds = {}
    for energy_key in all_energy_channels:
        all_obs_thresholds.update({energy_key: []})
        for obs_thresh in obs_values[energy_key]['thresholds']:
            all_obs_thresholds[energy_key].append(objh.threshold_to_key(obs_thresh))

    print("match_all_forecasts: Starting matching process. Setting up dictionary for observed SEP events to go into a list for each model: " + str(datetime.datetime.now()))

    #Set up dictionary of sphinx objects organized by model name and
    #energy channel.
    #Set up dictionary of observed SEP events in the prediction windows
    #organized by model, energy channel, threshold
    matched_sphinx = {}
    observed_sep_events = {} #list of unique observed SEP events
    for model in model_names:
        matched_sphinx.update({model:{'uses_eruptions':False}})
        observed_sep_events.update({model:{}})
        for energy_key in all_energy_channels:
            matched_sphinx[model].update({energy_key:[]})
            observed_sep_events[model].update({energy_key:{}})
            #Save all unique observed SEP events organized by energy channel
            #and threshold
            for obs_thresh in obs_values[energy_key]['thresholds']:
                obs_thresh_key = objh.threshold_to_key(obs_thresh)
                observed_sep_events[model][energy_key].update({obs_thresh_key:[]})

    print("match_all_forecasts: Starting matching process by energy channel. Setting up dictionary for observed SEP events to go into a list for each model: " + str(datetime.datetime.now()))

    #Launch into matching of observations and forecasts
    for energy_key in all_energy_channels:
        print("\n")
        print("Identifying Match Criteria for " + str(energy_key) + "," + str(datetime.datetime.now()))
        observation_objs = obs_objs[energy_key] #Observation objects

        forecasts = model_objs[energy_key] #all forecasts for channel
        for fcast in forecasts:
            #One SPHINX object contains all matching information and
            #predicted and observed values (and all thresholds)
            sphinx = objh.initialize_sphinx(fcast)
            
            #If this is a set of predictions and observations that are
            #allowed to have a set of mismatched energy channels and
            #thresholds
            if cfg.do_mismatch and energy_key == cfg.mm_energy_key:
                sphinx.mismatch = True
                print("Mismatched channel allowed.")

            #Get Trigger and Input information
            last_eruption_time, last_trigger_time =\
                fcast.last_trigger_time()
            print("\n")
            print(fcast.short_name)
            print(fcast.source)
            print("Issue time: " + str(fcast.issue_time))
#            print("Last trigger time: " + str(last_trigger_time))
#            print("Last eruption time: " + str(last_eruption_time))
            
            last_input_time = fcast.last_input_time()
#            print("Last input time: " + str(last_input_time))

            sphinx.last_eruption_time = last_eruption_time
            sphinx.last_trigger_time = last_trigger_time
            sphinx.last_input_time = last_input_time

            #Note if the model uses eruptions as triggers for 2nd matching step
            if (last_eruption_time != None) and \
                not pd.isnull(last_eruption_time):
                matched_sphinx[fcast.short_name]['uses_eruptions'] = True

            #Check that forecast prediction window is after last trigger/input
            fcast.valid_forecast(verbose=True)
            if fcast.valid == False:
                print("match_criteria_all_forecasts: Skipping invalid " + fcast.source)
                continue
                

            ###### PREDICTION AND OBSERVATION WINDOWS OVERLAP? #####
            #Do prediction and observation windows overlap?
            #Save the overlapping observations to the SPHINX object
            is_win_overlap = does_win_overlap(energy_key, fcast, obs_values)
            overlap_idx = [ix for ix in range(len(is_win_overlap)) if is_win_overlap[ix] == True]
            sphinx.overlapping_indices = overlap_idx
            print("Prediction and Observation windows overlap (if any): ")
            if overlap_idx != []:
                for ix in range(len(overlap_idx)):
                    sphinx.prediction_observation_windows_overlap.append(observation_objs[overlap_idx[ix]])
                    path = objh.get_file_path(observation_objs[overlap_idx[ix]].source)
                    sphinx.observed_sep_profiles.append(path +
                        observation_objs[overlap_idx[ix]].sep_profile)
                    #logging
                    print("  " +
                    str(sphinx.prediction_observation_windows_overlap[ix].source))
#                    print("  " +
#                    str(sphinx.observed_sep_profiles[ix]))


            ###### ONSET PEAK CRITERIA #####
            is_onset_peak_in_pred_win, td_trigger_onset_peak, \
            is_trigger_before_onset_peak, td_input_onset_peak,\
            is_input_before_onset_peak = onset_peak_criteria(sphinx,
            fcast, obs_values, observation_objs, energy_key)


            ###### MAX FLUX CRITERIA #####
            is_max_flux_in_pred_win, td_trigger_max_time,\
            is_trigger_before_max_time, td_input_max_time, \
            is_input_before_max_time = max_flux_criteria(sphinx, fcast,
            obs_values, observation_objs, energy_key)


            ###### THRESHOLD QUANTITIES #####
            #Is a threshold crossed inside the prediction window?
            #Save forecasted and observed all clear to SPHINX object
            all_fcast_thresholds = fcast.identify_all_thresholds()
            
            #Even if a mismatch is allowed for a particular model, there may
            #be some forecasts from the model that don't contain any fields
            #with the excepted threshold, e.g. all clear = True, so no
            #event_lengths or threshold_crossings fields
            if cfg.do_mismatch:
                if cfg.mm_pred_threshold not in all_fcast_thresholds:
                    all_fcast_thresholds.append(cfg.mm_pred_threshold)
            
            observed_peak_flux = None
            observed_peak_flux_max = None
            
            for f_thresh in all_fcast_thresholds:
                print("Checking Threshold: " + str(f_thresh))
                fcast_thresh = f_thresh
                
                #If this is a mismatch energy channel, then only want to
                #test the mismatched threshold specified in config.py
                if sphinx.mismatch:
                    if f_thresh == cfg.mm_pred_threshold:
                        #set threshold as the observed threshold
                        fcast_thresh = cfg.mm_obs_threshold
                        print("Predicted threshold associated with \'mismatched\' "
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

                ###### PREDICTION WINDOW OVERLAP WITH OBSERVED ####
                ############### SEP EVENT #########################
                is_pred_sep_overlap = pred_win_sep_overlap(sphinx, fcast,
                    obs_values, observation_objs, energy_key, fcast_thresh)


                ###### THRESHOLD CROSSED IN PREDICTION WINDOW #####
                #Is a threshold crossed in the prediction window?
                contains_thresh_cross = threshold_cross_criteria(sphinx, fcast,
                    obs_values, observation_objs, energy_key, fcast_thresh)


                ########### TRIGGERS/INPUTS BEFORE SEP ############
                #Is the last trigger/input before the threshold crossing time?
                td_trigger_thresh_cross, is_trigger_before_start, \
                td_input_thresh_cross, is_input_before_start =\
                before_threshold_crossing(sphinx, fcast, obs_values,
                observation_objs, energy_key, fcast_thresh)
                
                
                ########### TRIGGERS/INPUTS BEFORE END OF SEP #####
                #Is the last trigger/input before the threshold crossing time?
                td_trigger_end, is_trigger_before_end, \
                td_input_end, is_input_before_end =\
                before_sep_end(sphinx, fcast, obs_values,
                observation_objs, energy_key, fcast_thresh)


                ######### FLARE/CME BEFORE SEP START ###############
                #Is the eruption (flare/cme) before the threshold crossing?
                td_eruption_thresh_cross, is_eruption_before_start =\
                eruption_before_threshold_crossing(sphinx, fcast, obs_values,
                            observation_objs, energy_key, fcast_thresh)
                

                ############ ONGOING SEP EVENT AT START OF ########
                ################## PREDICTION WINDOW ##############
                is_sep_ongoing = observed_ongoing_event(sphinx, fcast,
                    obs_values, observation_objs, energy_key, fcast_thresh)


                ############ MATCHING AND EXTRACTING OBSERVED VALUES#######
                #Loop over all observations inside the prediction window
                for i in sphinx.overlapping_indices: #index of overlapping obs

                    #Bool for eruption 24 hours to a few mins before
                    #threshold crossing.
                    #None if no SEP event
                    is_eruption_in_range =\
                        eruption_in_range(td_eruption_thresh_cross[i])
                          

                    #Is the last trigger or input before the threshold crossing
                    #None if no SEP event
                    trigger_input_start =\
                        last_before_start(is_trigger_before_start[i],
                        is_input_before_start[i])
                    
                    
                    #Is the last trigger or input before the SEP end
                    #None if no SEP event
                    trigger_input_end =\
                        last_before_end(is_trigger_before_end[i],
                        is_input_before_end[i])

                    
                    #Check if the model reports and ongoing event
#                    reports_ongoing = ongoing_status(observation_objs[i],
#                        channel, fcast_thresh)
                    
                    ###ONSET PEAK & MAX FLUX
                    #Prediction window overlaps with observation
                    #Last eruption within 48 hrs - 15 mins before
                    #threshold crossing
                    #The prediction window overlaps with an SEP event in any
                    #threshold - only a comparison when SEP event
                    ####NEED TO ADD IN COMPARISON WHEN NO SEP EVENT
                    #The last trigger/input time if before the observed peak
                    #intensity
                    #ONSET PEAK
                    #peak_criteria is True, False, None indicating a match
                    #with an observation or None if no SEP event observed
                    peak_criteria = match_observed_onset_peak(sphinx,
                        observation_objs[i], is_win_overlap[i],
                        is_eruption_in_range,
                        is_trigger_before_onset_peak[i],
                        is_input_before_onset_peak[i], is_pred_sep_overlap[i])
                    
                   
                    #MAX FLUX
                    #max_criteria is True, False, None indicating a match
                    #with an observation or None if no SEP event observed
                    max_criteria = match_observed_max_flux(sphinx,
                        observation_objs[i], is_win_overlap[i],
                        is_eruption_in_range, is_trigger_before_max_time[i],
                        is_input_before_max_time[i], is_pred_sep_overlap[i])
                    
                    
                    #ALL CLEAR
                    #all_clear_status is True if no observed SEP event,
                    #False if observed SEP event that meets criteria,
                    #None - if ongoing event at start of prediction window
                    all_clear_status = match_all_clear(sphinx,
                        observation_objs[i], is_win_overlap[i],
                        is_eruption_in_range, trigger_input_start,
                        contains_thresh_cross[i], is_sep_ongoing[i])


                    #SEP QUANTITIES RELATED TO START TIME
                    sep_status = match_sep_quantities(sphinx, observation_objs[i], fcast_thresh, is_win_overlap[i],
                        is_eruption_in_range, trigger_input_start,
                        contains_thresh_cross[i], is_sep_ongoing[i])
                    #Save observed SEP event
                    if sep_status == True:
                        if sphinx.observed_threshold_crossing[thresh_key].crossing_time\
                        not in observed_sep_events[fcast.short_name][energy_key][thresh_key]:
                            observed_sep_events[fcast.short_name][energy_key][thresh_key].append(sphinx.observed_threshold_crossing[thresh_key].crossing_time)


                    #SEP END TIME
                    end_status = match_sep_end_time(sphinx, observation_objs[i], fcast_thresh, is_win_overlap[i],
                        is_eruption_in_range, trigger_input_end,
                        is_pred_sep_overlap[i])


            #Save the SPHINX object with all of the forecasted and matched
            #observation values to a dictionary organized by energy channel
            sphinx.match_report()
            matched_sphinx[fcast.short_name][energy_key].append(sphinx)

    #Print uniquely identified observed SEP events
    sep_report(all_energy_channels, obs_values, model_names,
        observed_sep_events)

    #In the case where the same model has forecasts derived from
    #multiple eruptions matched to the same SEP event, find the
    #best match and unmatch the other forecasts.
    revise_eruption_matches(matched_sphinx, all_energy_channels,
        obs_values, model_names, observed_sep_events)

    return matched_sphinx, all_obs_thresholds, observed_sep_events


