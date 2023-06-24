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
        
        dict.update({'Threshold ' + str(idx[i]): df})
            
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
                {'thresholds': [{'threshold': 10, 'threshold_units': Unit("1 / (cm2 s sr)")}],
                'Threshold 0': pandas DataFrame with obs info}},
                {'min.100.0.max.-1.0.units.MeV':
                {'thresholds': [{'threshold': 1, 'threshold_units': Unit("1 / (cm2 s sr)")}],
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
        key = vjson.energy_channel_to_key(channel)
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
    print("Checking Prediction and Observation window overlap.")
    print(fcast.source)
    
    pred_win_st = fcast.prediction_window_start
    pred_win_end = fcast.prediction_window_end
    print(pred_win_st)
    print(pred_win_end)
    pred_interval = pd.Interval(pd.Timestamp(pred_win_st),
        pd.Timestamp(pred_win_end))
    
    overlaps_bool = []
    
    for i in range(len(obs['observation_window_start'])):
        obs_interval =\
            pd.Interval(pd.Timestamp(obs['observation_window_start'][i]), pd.Timestamp(obs['observation_window_end'][i]))
    
        if pred_interval.overlaps(obs_interval):
            overlaps_bool.append(True)
        else:
            overlaps_bool.append(False)
            
    return overlaps_bool



def match_one_forecast(fcast, obs_values):
    """ Find the observations that match to a single forecast.
    """
    
    #Extract the forecast information needed for matching
    energy_channel = fcast.energy_channel
    
    #Return if no prediction
    if fcast.prediction_window_start == None:
        return None
        
    last_trigger_time = objh.last_trigger_time(fcast)
    print(fcast.short_name)
    print(fcast.source)
    print("Last trigger time: " + str(last_trigger_time))
    
    last_input_time = objh.last_input_time(fcast)
    print("Last input time: " + str(last_input_time))

    #Pull out the observations needed for matching
    #We will have a table with needed information about each
    #observation
    #Extract correct energy channel
    energy_key = vjson.energy_channel_to_key(energy_channel)
    obs = obs_values[energy_key]['Threshold 0']
    
    #1. Does the Prediction Window overlap with the Observation Window?
    overlaps_bool = pred_and_obs_overlap(fcast, obs)
    
    idx = overlaps_bool.index(True)
    print(obs_values[energy_key]['Threshold 0']['observation_window_start'][idx])
    print(obs_values[energy_key]['Threshold 0']['observation_window_end'][idx])
    
    




def match_all_forecasts(all_energy_channels, obs_objs, model_objs):
    """ Match all forecasts to observations.
    """

    #All observed values needed for matching, organized by
    #energy channel and threshold
    obs_values = compile_all_obs(all_energy_channels, obs_objs)

    match = {}
    
    for channel in all_energy_channels:
        matched_obs = []
        key = vjson.energy_channel_to_key(channel)
        forecasts = model_objs[key] #all forecasts for channel
        for fcast in forecasts:
            match_one_forecast(fcast, obs_values)
