import sys
import datetime
from . import object_handler as objh
import pandas as pd
import pickle

__version__ = "0.1"
__author__ = "Katie Whitman"
__maintainer__ = "Katie Whitman"
__email__ = "kathryn.whitman@nasa.gov"

""" utils/resume.py contains subroutines to aid in resuming
    the validation process from a starting dataframe.
    
"""


def read_in_df(filename):
    """ Read in pickle file containing SPHINX dataframe.
    
    """
    try:
        pklfile = open(filename,"rb")
        df = pickle.load(pklfile)
        return df
    except:
        sys.exit("validate: Cannot open pickle file containing "
            "input dataframe. Please check the filename.")




def identify_unique(df, value):
    """ Find all unique values in df and output list.
        Find all models, energy channels, thresholds, etc
        
        INPUT:
            :value: (string) column of df, e.g.
                'Models', 'Energy Channel Key', 'Threshold Key'
    
        OUTPUT:
            :unique: (list) list of unique values
        
    """
    unique = []
    lst = df[value].to_list()
    
    #First remove all None and pd.NaT values
    for i in range(len(lst)-1,-1,-1):
        if pd.isnull(lst[i]):
            lst.pop(i)
            continue
        if lst[i] == None:
            lst.pop(i)
            continue
    
    while len(lst) > 0:
        val = lst[0]
        unique.append(val)
        lst = [x for x in lst if x != val]
    
    return unique


def identify_thresholds_per_energy_channel(df):
    """ Identify all of the thresholds applied to a given energy
        channel. Put in the format of a dictionary:
            all_thresholds.update({energy_key: [thresh_key1, thresh_key2]})
            
        where e.g. thresh_key1 = 'threshold.10.0.units.1 / (cm2 s sr)'

        This dictionary is created to match the format output by
        match_all_forecasts in match.py. It is used as an input
        in calculate_intuitive_metrics in validation.py

    """
    all_energy_channels = identify_unique(df, 'Energy Channel Key')
    all_thresholds = {}
    for ek in all_energy_channels:
        all_thresholds.update({ek: []})

        sub = df.loc[(df['Energy Channel Key'] == ek)]
        thresh = identify_unique(df, 'Threshold Key')
        for tk in thresh:
            all_thresholds[ek].append(tk)

    return all_thresholds


def last_prediction_windows(df):
    """ Find the last prediction window assessed for a specific
        model.
        
        Organized by energy channel and threshold.
        Returns dataframe with model, energy channel, threshold,
        and latest prediction window.
        
    """
    
    models = identify_unique(df, 'Model')
    energy_channels = identify_unique(df, 'Energy Channel Key')
    df_pred_win = pd.DataFrame()
   
    for model in models:
        print(model)
        sub = df.loc[(df['Model'] == model)]
        if sub.empty: continue
                
        for ek in energy_channels:
            print(ek)
            sub_p = sub.loc[(sub['Energy Channel Key'] == ek)]
            if sub_p.empty: continue
 
            sub_p = sub_p[['Model', 'Energy Channel Key', 'Threshold Key', 'Forecast Source', 'Prediction Window Start', 'Prediction Window End']]
            sub_p = sub_p.loc[(sub_p['Prediction Window Start'] == sub_p['Prediction Window Start'].max())]

            #If multiple thresholds applied to the same energy
            #channel, might return multiple rows. Only need one.
            if len(sub_p) > 1:
                sub_p = sub_p[0:1]

            if df_pred_win.empty:
                df_pred_win = sub_p
            else:
                df_pred_win = pd.concat([df_pred_win,sub_p], ignore_index=True)
                    
    return df_pred_win


def check_fcast_for_resume(df, model_objs):
    """ Take the list of Forecast objects in the model list and
        keep only those after the last prediction window in the
        dataframe.
        
        INPUT:
        
            :df: (Pandas DataFrame) contains previously run forecast
                and observation matched values.
            :model_objs: (list of objects) Forecast objects with a set
                of new predictions
        
        OUTPUT:
        
            :select_objs: (list of objects) Forecast objects with
                prediction windows after the previously run forecasts
        
    """

    pw = last_prediction_windows(df)
    print(pw)
    select_objs = {}

    energy_keys = list(model_objs.keys())
    for ek in energy_keys:
        select_objs.update({ek: []})
    
    for ek in energy_keys:
        for obj in model_objs[ek]:
            model = obj.short_name
            if model == None: continue
                
            pred_win_st = obj.prediction_window_start
            if pred_win_st == None: continue
            
            #if energy channel not in there, then want to add?
            
            sub = pw.loc[(pw['Model'] == model)]
            #If model not already present, then new information and
            #want to add
            if sub.empty:
                #print("check_fcast_for_resume: Model " + model +
                #    " not present in previous data frame. "
                #    "Appending all forecasts.")
                select_objs[ek].append(obj)
                continue
            
            sub = sub.loc[(sub['Energy Channel Key'] == ek)]
            #if energy channel not already present, then new information
            #and want to add
            if sub.empty:
                #print("check_fcast_for_resume: Model " + model +
                #    " and energy channel " + ek +
                #    " not present in previous data frame. "
                #    "Appending all forecasts.")
                select_objs[ek].append(obj)
                continue
            
            
            #Keep if the prediction window is after the last prediction
            #window in the input dataframe for that model and energy
            #channel
            ref_pred_win_st = sub['Prediction Window Start'].iloc[0]
            #print("check_fcast_for_resume: Model " + model +
            #    " and energy channel " + ek + " last prediction "
             #   "window start " + str(ref_pred_win_st))
            #print("Current forecast prediction window "
            #        + str(pred_win_st))
            if pred_win_st > ref_pred_win_st:
                #print("Keeping forecast")
                select_objs[ek].append(obj)
    
    return select_objs
