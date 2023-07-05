from . import validation_json_handler as vjson
from . import object_handler as objh
import sys
import pandas as pd


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
    for channel in all_energy_channels:
        print('validate: Extracting observation arrays for matching '
            'for threshold ' + str(channel))
        key = objh.energy_channel_to_key(channel)
        dict = create_obs_match_array(obs_objs[key])
        
        obs_match.update({key: dict})

    return obs_match




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



def does_win_overlap(fcast, obs_values):
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
    energy_channel = fcast.energy_channel
    energy_key = objh.energy_channel_to_key(energy_channel)

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


def observed_time_in_pred_win(fcast, obs_values, obs_key):
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
            
    Output:
        
        :contains_obs_time: (array of bool) array indices match the order
            of the Observation objects in obs_objs[energy_key]
            where energy_key is the key made from the fcast
            energy_channel
    
    """
    energy_channel = fcast.energy_channel
    energy_key = objh.energy_channel_to_key(energy_channel)
    
    pred_win_st = fcast.prediction_window_start
    pred_win_end = fcast.prediction_window_end

    #Return if no prediction
    if fcast.prediction_window_start == None:
        return [None]*len(obs_values[energy_key]['dataframes'][0])
           
    #Extract pandas dataframe for correct energy and threshold
    energy_key = objh.energy_channel_to_key(energy_channel)
    obs = obs_values[energy_key]['dataframes'][0] #all obs for threshold

    #Check if a threshold crossing is contained inside of the
    #prediction window; returns False if pd.NaT
    contains_obs_time = (obs[obs_key] >= pd.Timestamp(pred_win_st)) \
        & (obs[obs_key] < pd.Timestamp(pred_win_end))
 
    return list(contains_obs_time)



def observed_time_in_pred_win_thresh(fcast, obs_values, obs_key,
        threshold):
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
    energy_channel = fcast.energy_channel
    energy_key = objh.energy_channel_to_key(energy_channel)
    
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
    energy_key = objh.energy_channel_to_key(energy_channel)
    obs = obs_values[energy_key]['dataframes'][ix] #all obs for threshold

    #Check if a threshold crossing is contained inside of the
    #prediction window; returns False if pd.NaT
    contains_obs_time = (obs[obs_key] >= pd.Timestamp(pred_win_st)) \
        & (obs[obs_key] < pd.Timestamp(pred_win_end))
 
    return list(contains_obs_time)


def is_time_before(time, obs_values, obs_key, energy_channel):
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
    energy_key = objh.energy_channel_to_key(energy_channel)
    if pd.isnull(time):
        return [None]*len(obs_values[energy_key]['dataframes'][0])

    #Check if time is before
    obs = obs_values[energy_key]['dataframes'][0]
    time_diff = (time - obs[obs_key]).astype('timedelta64[h]')
    is_before = (time_diff <= 0)
    is_before = list(is_before)

    nat = [ix for ix in range(len(list(time_diff))) if pd.isnull(time_diff[ix])]
    for ix in nat:
        is_before[ix] = None
            
    return is_before
    

def is_time_before_thresh(time, obs_values, obs_key, energy_channel, threshold):
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
    energy_key = objh.energy_channel_to_key(energy_channel)
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

    time_diff = (time - obs[obs_key]).astype('timedelta64[h]')
    is_before = (time_diff <= 0)
    is_before = list(is_before)

    nat = [ix for ix in range(len(list(time_diff))) if pd.isnull(time_diff[ix])]
    for ix in nat:
        is_before[ix] = None
            
    return is_before


def time_diff(time, obs_values, obs_key, energy_channel):
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
    energy_key = objh.energy_channel_to_key(energy_channel)
    if pd.isnull(time):
        return [None]*len(obs_values[energy_key]['dataframes'][0])

    #Check if time is before
    obs = obs_values[energy_key]['dataframes'][0]
    time_diff = (time - obs[obs_key]).astype('timedelta64[s]')
    time_diff = time_diff/(60.*60.)

    return list(time_diff)
    

def time_diff_thresh(time, obs_values, obs_key, energy_channel, threshold):
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
    energy_key = objh.energy_channel_to_key(energy_channel)
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
    time_diff = (time - obs[obs_key]).astype('timedelta64[s]')
    time_diff = time_diff/(60.*60.)

    return list(time_diff)


############### MATCHING CRITERIA FOR SPECIFIC QUANTITIES #####
def onset_peak_criteria(sphinx, fcast, obs_values, observation_objs):
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
    channel = fcast.energy_channel
    
    ###### MATCHING: ONSET PEAK IN PREDICTION WINDOW #####
    #Is the onset peak inside of the prediction window?
    #Get the time difference - negative means trigger is before
    is_onset_peak_in_pred_win = observed_time_in_pred_win(fcast,
                                    obs_values, 'peak_time')
#    idx = [ix for ix in range(len(is_onset_peak_in_pred_win)) if is_onset_peak_in_pred_win[ix] == True]
#    print("Onset Peak is inside prediction window (if any): ")
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
                                'peak_time', channel)
    is_trigger_before_onset_peak = is_time_before(last_trigger_time,
                                obs_values, 'peak_time', channel)
#    idx = [ix for ix in range(len(is_trigger_before_onset_peak)) if is_trigger_before_onset_peak[ix] == True]
#    print("Time Difference of Triggers before Onset Peak (if any): ")
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
                            'peak_time', channel)
    is_input_before_onset_peak = is_time_before(last_input_time,
                            obs_values, 'peak_time', channel)
#    idx = [ix for ix in range(len(is_input_before_onset_peak)) if is_input_before_onset_peak[ix] == True]
#    print("Time difference of Inputs before Onset Peak (if any): ")
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



def max_flux_criteria(sphinx, fcast, obs_values, observation_objs):
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
    channel = fcast.energy_channel
    
    ###### MATCHING: MAX FLUX IN PREDICTION WINDOW #####
    #Is the max flux inside of the prediction window?
    #Get the time difference - negative means trigger is before
    is_max_flux_in_pred_win = observed_time_in_pred_win(fcast,
                            obs_values, 'max_time')
#    idx = [ix for ix in range(len(is_max_flux_in_pred_win)) if is_max_flux_in_pred_win[ix] == True]
#    print("Max Flux is inside prediction window (if any): ")
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
                            'max_time', channel)
    is_trigger_before_max_time = is_time_before(last_trigger_time,
                            obs_values, 'max_time', channel)
#    idx = [ix for ix in range(len(is_trigger_before_max_time)) if is_trigger_before_max_time[ix] == True]
#    print("Time Difference of Triggers before Max Flux time (if any): ")
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
                        'max_time', channel)
    is_input_before_max_time = is_time_before(last_input_time,
                        obs_values, 'max_time', channel)
#    idx = [ix for ix in range(len(is_input_before_max_time)) if is_input_before_max_time[ix] == True]
#    print("Time difference of Inputs before Max Time (if any): ")
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
        thresh):
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
        obs_values, 'threshold_crossing_time', thresh)
#    cross_idx = [ix for ix in range(len(contains_thresh_cross)) if contains_thresh_cross[ix] == True]
#    print("Threshold crossed in prediction window (if any): ")
    if sphinx.overlapping_indices != []:
        for ix in sphinx.overlapping_indices:
            sphinx.threshold_crossed_in_pred_win[thresh_key].append(contains_thresh_cross[ix])
            #logging
#            print("  " +
#                str(sphinx.threshold_crossed_in_pred_win[thresh_key][ix]))


    return contains_thresh_cross



def before_threshold_crossing(sphinx, fcast, obs_values, observation_objs,
        thresh):
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
    channel = fcast.energy_channel
    thresh_key = objh.threshold_to_key(thresh)


    ###### MATCHING: TRIGGERS/INPUTS BEFORE SEP #####
    td_trigger_thresh_cross = time_diff_thresh(last_trigger_time, obs_values,
        'threshold_crossing_time', channel, thresh)
    is_trigger_before_start = is_time_before_thresh(last_trigger_time,
        obs_values, 'threshold_crossing_time', channel, thresh)
#    idx = [ix for ix in range(len(is_trigger_before_start)) if is_trigger_before_start[ix] == True]
#    print("Triggers before Threshold Crossing (if any): ")
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
        'threshold_crossing_time', channel, thresh)
    is_input_before_start = is_time_before_thresh(last_input_time, obs_values,
        'threshold_crossing_time', channel, thresh)
#    idx = [ix for ix in range(len(is_input_before_start)) if is_input_before_start[ix] == True]
#    print("Inputs before Threshold Crossing (if any): ")
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
        thresh):
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
    channel = fcast.energy_channel
    thresh_key = objh.threshold_to_key(thresh)


    ###### MATCHING: TRIGGERS/INPUTS BEFORE SEP #####
    td_trigger_sep_end = time_diff_thresh(last_trigger_time, obs_values,
        'end_time', channel, thresh)
    is_trigger_before_end = is_time_before_thresh(last_trigger_time,
        obs_values, 'end_time', channel, thresh)
#    idx = [ix for ix in range(len(is_trigger_before_end)) if is_trigger_before_end[ix] == True]
 #   print("Triggers before SEP end time (if any): ")
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
        'end_time', channel, thresh)
    is_input_before_end = is_time_before_thresh(last_input_time, obs_values,
        'end_time', channel, thresh)
#    idx = [ix for ix in range(len(is_input_before_end)) if is_input_before_end[ix] == True]
#    print("Inputs before SEP end time (if any): ")
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
    observation_objs, thresh):
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
    channel = fcast.energy_channel
    thresh_key = objh.threshold_to_key(thresh)


    ###### MATCHING: ERUPTION BEFORE SEP #####
    td_eruption_thresh_cross = time_diff_thresh(last_eruption_time, obs_values,
        'threshold_crossing_time', channel, thresh)
    is_eruption_before_start = is_time_before_thresh(last_eruption_time,
        obs_values, 'threshold_crossing_time', channel, thresh)
#    idx = [ix for ix in range(len(is_eruption_before_start)) if is_eruption_before_start[ix] == True]
#    print("Eruption (flare/CME) before Threshold Crossing (if any): ")
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



def pred_win_sep_overlap(sphinx, fcast, obs_values, observation_objs,
        threshold):
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
    
    energy_channel = fcast.energy_channel
    energy_key = objh.energy_channel_to_key(energy_channel)
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
        threshold):
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
    
    energy_channel = fcast.energy_channel
    energy_key = objh.energy_channel_to_key(energy_channel)
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
    
    if peak_criteria == True:
#        print("Observed peak_intensity matched:")
#        print(observation_obj.source)
#        print(observation_obj.peak_intensity.intensity)
#        print(observation_obj.peak_intensity.time)
        sphinx.observed_peak_intensity = observation_obj.peak_intensity.intensity
        sphinx.observed_peak_intensity_units = observation_obj.peak_intensity.units
        sphinx.observed_peak_intensity_time = observation_obj.peak_intensity.time

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
    
    if max_criteria == True:
#        print("Observed peak_intensity_max matched:")
#        print(observation_obj.source)
#        print(observation_obj.peak_intensity_max.intensity)
#        print(observation_obj.peak_intensity_max.time)
        sphinx.observed_peak_intensity_max = observation_obj.peak_intensity_max.intensity
        sphinx.observed_peak_intensity_max_units = observation_obj.peak_intensity_max.units
        sphinx.observed_peak_intensity_max_time = observation_obj.peak_intensity_max.time

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
    if sphinx.observed_all_clear_boolean == False:
        return None
    
    all_clear_status = None
    
    if not is_win_overlap:
        all_clear_status = None
        return all_clear_status
        
    #Prediction and observation windows overlap
    #If ongoing SEP event at start of prediction window, no match
    if is_sep_ongoing:
        all_clear_status = None
        return all_clear_status
    
    #If there is no threshold crossing in prediction window,
    #then observed all clear is True
    if not contains_thresh_cross:
        all_clear_status = True
    
    #If there is a threshold crossing in the prediction window
    if contains_thresh_cross:
        #The eruption must occur in the right time range
        if is_eruption_in_range != None:
            if not is_eruption_in_range:
                all_clear_status = None
                return all_clear_status
        #The triggers and inputs must all be before threshold crossing
        if trigger_input_start:
            #Observed all clear is False
            all_clear_status = False
    
    
#    print("Prediction window: " + str(sphinx.prediction.prediction_window_start) + " to "
#        + str(sphinx.prediction.prediction_window_end))
#    #All clear status
#    print("Observed all_clear matched:")
#    print("  " + observation_obj.source)
#    print("  " + str(all_clear_status))
    sphinx.observed_all_clear_boolean = all_clear_status
    sphinx.observed_all_clear_threshold = observation_obj.all_clear.threshold
    sphinx.observed_all_clear_threshold_units = observation_obj.all_clear.threshold_units

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

    sep_status = None
    
    if not is_win_overlap:
        sep_status = None
        return sep_status
        
    #Prediction and observation windows overlap
    #If ongoing SEP event at start of prediction window, no match
    if is_sep_ongoing:
        sep_status = False
        return sep_status
    
    #No threshold crossing in prediction window, no SEP event
    if not contains_thresh_cross:
        sep_status = False
        return sep_status
    
    #If there is a threshold crossing in the prediction window
    if contains_thresh_cross:
        #The eruption must occur in the right time range
        if is_eruption_in_range != None:
            if not is_eruption_in_range:
                sep_status = None
                return sep_status
        #The triggers and inputs must all be before threshold crossing
        if trigger_input_start:
            sep_status = True
        else:
            sep_status = None
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
        sphinx.observed_threshold_crossing_times[thresh_key].append(th.crossing_time)
 
    #Start time and channel fluence
    start_time = None
    fluence = None
    for i in range(len(observation_obj.event_lengths)):
        event = observation_obj.event_lengths[i]
        if event.threshold != thresh['threshold']:
            continue
        sphinx.observed_start_times[thresh_key].append(event.start_time)
        sphinx.observed_fluences[thresh_key].append(observation_obj.fluences[i].fluence)

    #Fluence spectra
    spectrum = None
    for flsp in observation_obj.fluence_spectra:
        if flsp.threshold_start != thresh['threshold']:
            continue
        sphinx.observed_fluence_spectra[thresh_key].append(flsp.fluence_spectrum)

#    print(sphinx.observed_threshold_crossing_times)
#    print(sphinx.observed_start_times)
#    print(sphinx.observed_fluences)
#    print(sphinx.observed_fluence_spectra)

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

    end_status = None
    
    #Prediction and observation windows must overlap
    if not is_win_overlap:
        end_status = None
        return end_status
        
    #The prediction window must overlap with an SEP event
    if not is_pred_sep_overlap:
        end_status = False #no SEP event, no values
        return end_status

    #If there is an SEP event, the eruption must occur in the right time range
    if is_eruption_in_range != None:
        if not is_eruption_in_range:
            sep_status = None
            return sep_status
    #The triggers and inputs must all be before threshold crossing
    if trigger_input_end:
        end_status = True
    else:
        end_status = None
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
        sphinx.observed_end_times[thresh_key].append(event.end_time)

 #   print(sphinx.observed_end_times)

    return end_status




#################################
####ALL MATCHING CRITERIA #######
#################################
def match_all_forecasts(all_energy_channels, model_names, obs_objs,
    model_objs):
    """ Match all forecasts to observations organized by model
        short_name, energy channel, and threshold.
    """

    #All observed values needed for matching, organized by
    #energy channel and threshold
    obs_values = compile_all_obs(all_energy_channels, obs_objs)

    #array of sphinx objects organized by model name and energy channel
    matched_sphinx = {}
    for model in model_names:
        matched_sphinx.update({model:{}})
        for channel in all_energy_channels:
            energy_key = objh.energy_channel_to_key(channel)
            matched_sphinx[model].update({energy_key:[]})


    for channel in all_energy_channels:
        print("\n")
        print("Identifying Match Criteria for " + str(channel))
        energy_key = objh.energy_channel_to_key(channel)
        observation_objs = obs_objs[energy_key] #Observation objects

        forecasts = model_objs[energy_key] #all forecasts for channel
        for fcast in forecasts:
            #One SPHINX object contains all matching information and
            #predicted and observed values (and all thresholds)
            sphinx = objh.initialize_sphinx(fcast)

            #Get Trigger and Input information
            last_eruption_time, last_trigger_time =\
                objh.last_trigger_time(fcast)
            print("\n")
            print(fcast.short_name)
            print(fcast.source)
            print("Issue time: " + str(fcast.issue_time))
#            print("Last trigger time: " + str(last_trigger_time))
#            print("Last eruption time: " + str(last_eruption_time))
            
            last_input_time = objh.last_input_time(fcast)
#            print("Last input time: " + str(last_input_time))

            sphinx.last_eruption_time = last_eruption_time
            sphinx.last_trigger_time = last_trigger_time
            sphinx.last_input_time = last_input_time

            #Check that forecast prediction window is after last trigger/input
            objh.valid_forecast(fcast, last_trigger_time, last_input_time)
            if fcast.valid == False:
                print("match_criteria_all_forecasts: Invalid forecast. "
                    "Issue time must start after last trigger "
                    "or input time. Skipping " + fcast.source)
                continue
                

            ###### PREDICTION AND OBSERVATION WINDOWS OVERLAP? #####
            #Do prediction and observation windows overlap?
            #Save the overlapping observations to the SPHINX object
            is_win_overlap = does_win_overlap(fcast, obs_values)
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
            fcast, obs_values, observation_objs)


            ###### MAX FLUX CRITERIA #####
            is_max_flux_in_pred_win, td_trigger_max_time,\
            is_trigger_before_max_time, td_input_max_time, \
            is_input_before_max_time = max_flux_criteria(sphinx, fcast,
            obs_values, observation_objs)


            ###### THRESHOLD QUANTITIES #####
            #Is a threshold crossed inside the prediction window?
            #Save forecasted and observed all clear to SPHINX object
            all_fcast_thresholds = objh.identify_all_thresholds_one(fcast)
            
            observed_peak_flux = None
            observed_peak_flux_max = None
            
            for fcast_thresh in all_fcast_thresholds:
                print("Checking Threshold: " + str(fcast_thresh))
                
                #Check if this threshold is present in the observations
                #Can only be compared if present in both
                if fcast_thresh not in obs_values[energy_key]['thresholds']:
                    continue

                #Add threshold so that objects saved in SPHINX object
                #in contains_thresh_cross and is_*_before* arrays
                #will be in an array in the same order as the
                #thresholds
                sphinx.thresholds.append(fcast_thresh)
                sphinx.Add_Threshold(fcast_thresh)
                thresh_key = objh.threshold_to_key(fcast_thresh)

                ###### PREDICTION WINDOW OVERLAP WITH OBSERVED ####
                ############### SEP EVENT #########################
                is_pred_sep_overlap = pred_win_sep_overlap(sphinx, fcast,
                    obs_values, observation_objs, fcast_thresh)


                ###### THRESHOLD CROSSED IN PREDICTION WINDOW #####
                #Is a threshold crossed in the prediction window?
                contains_thresh_cross = threshold_cross_criteria(sphinx,
                    fcast, obs_values, observation_objs, fcast_thresh)


                ########### TRIGGERS/INPUTS BEFORE SEP ############
                #Is the last trigger/input before the threshold crossing time?
                td_trigger_thresh_cross, is_trigger_before_start, \
                td_input_thresh_cross, is_input_before_start =\
                before_threshold_crossing(sphinx, fcast, obs_values,
                observation_objs, fcast_thresh)
                
                
                ########### TRIGGERS/INPUTS BEFORE END OF SEP #####
                #Is the last trigger/input before the threshold crossing time?
                td_trigger_end, is_trigger_before_end, \
                td_input_end, is_input_before_end =\
                before_sep_end(sphinx, fcast, obs_values,
                observation_objs, fcast_thresh)


                ######### FLARE/CME BEFORE SEP START ###############
                #Is the eruption (flare/cme) before the threshold crossing?
                td_eruption_thresh_cross, is_eruption_before_start =\
                eruption_before_threshold_crossing(sphinx, fcast, obs_values,
                            observation_objs, fcast_thresh)
                

                ############ ONGOING SEP EVENT AT START OF ########
                ################## PREDICTION WINDOW ##############
                is_sep_ongoing = observed_ongoing_event(sphinx, fcast,
                    obs_values, observation_objs, fcast_thresh)


                ############ MATCHING AND EXTRACTING OBSERVED VALUES#######
                #Loop over all observations inside the prediction window
                for i in sphinx.overlapping_indices: #index of overlapping obs
                    #Bool for eruption 48 hours to 15 mins before
                    #threshold crossing.
                    #None if no SEP event
                    is_eruption_in_range = None
                    if not pd.isnull(td_eruption_thresh_cross[i]):
                        if td_eruption_thresh_cross[i] <= -0.25\
                            and td_eruption_thresh_cross[i] > -24.:
                            is_eruption_in_range = True
                        if td_eruption_thresh_cross[i] > -0.25\
                            or td_eruption_thresh_cross[i] <= -24.:
                            is_eruption_in_range = False
                
                    #Is the last trigger or input before the threshold crossing
                    #None if no SEP event
                    trigger_input_start = None
                    if is_trigger_before_start[i] != None:
                        trigger_input_start = is_trigger_before_start[i]
                    if is_input_before_start[i] != None:
                        if trigger_input_start == None:
                            trigger_input_start = is_input_before_start[i]
                        else:
                            trigger_input_start = trigger_input_start and \
                                is_trigger_before_start[i]


                    #Is the last trigger or input before the SEP end
                    #None if no SEP event
                    trigger_input_end = None
                    if is_trigger_before_end[i] != None:
                        trigger_input_end = is_trigger_before_end[i]
                    if is_input_before_end[i] != None:
                        if trigger_input_end == None:
                            trigger_input_end = is_input_before_end[i]
                        else:
                            trigger_input_end = trigger_input_end and \
                                is_trigger_before_end[i]

                
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
                    if (sep_status == None) or (sep_status == False):
                        sphinx.observed_threshold_crossing_times[thresh_key].append(pd.NaT)
                        sphinx.observed_start_times[thresh_key].append(pd.NaT)
                        sphinx.observed_fluences[thresh_key].append(None)
                        sphinx.observed_fluence_spectra[thresh_key].append(None)


                    #SEP END TIME
                    end_status = match_sep_end_time(sphinx, observation_objs[i], fcast_thresh, is_win_overlap[i],
                        is_eruption_in_range, trigger_input_end,
                        is_pred_sep_overlap[i])
                    if (end_status == None) or (end_status == False):
                        sphinx.observed_end_times[thresh_key].append(pd.NaT)


            #Save the SPHINX object with all of the forecasted and matched
            #observation values to a dictionary organized by energy channel
            sphinx.Match_Report()
            matched_sphinx[fcast.short_name][energy_key].append(sphinx)

    return matched_sphinx
