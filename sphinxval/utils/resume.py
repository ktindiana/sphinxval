import sys
import datetime
import pandas as pd
import pickle
import logging

__author__ = "Katie Whitman"
__maintainer__ = "Katie Whitman"
__email__ = "kathryn.whitman@nasa.gov"

""" utils/resume.py contains subroutines to aid in resuming
    the validation process from a starting dataframe.
    
"""

#Create logger
logger = logging.getLogger(__name__)

def read_in_df(filename):
    """ Read in pickle file containing SPHINX dataframe.
    
    """
    try:
        pklfile = open(filename,"rb")
        df = pickle.load(pklfile)
        return df
    except:
        logger.error("Cannot open pickle file containing "
            f"input dataframe. Please check the filename: {filename}")
        sys.exit()




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


def identify_thresholds_per_energy_channel(df, ek_name='Energy Channel Key',
    tk_name='Threshold Key'):
    """ Identify all of the thresholds applied to a given energy
        channel. Put in the format of a dictionary:
            all_thresholds.update({energy_key: [thresh_key1, thresh_key2]})
            
        where e.g. thresh_key1 = 'threshold.10.0.units.1 / (cm2 s sr)'

        This dictionary is created to match the format output by
        match_all_forecasts in match.py. It is used as an input
        in calculate_intuitive_metrics in validation.py

    """
    all_energy_channels = identify_unique(df, ek_name)
    all_thresholds = {}
    for ek in all_energy_channels:
        all_thresholds.update({ek: []})

        sub = df.loc[(df[ek_name] == ek)]
        thresh = identify_unique(sub, tk_name)
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
        logger.info(model)
        sub = df.loc[(df['Model'] == model)]
        if sub.empty: continue
                
        for ek in energy_channels:
            logger.info(ek)
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


def read_in_profile_dicts(resume_obs, resume_model):
    """ Read in pickle files containing the observed profile dictionary
    and the model profile dictionary.
    
    """
    try:
        pklfile = open(resume_obs,"rb")
        obs_prof_df = pickle.load(pklfile)
        
    except:
        logger.error("Cannot open pickle file containing "
            f"input observed profile dictionary. Please check the filename: {resume_obs}")
        sys.exit()
    
    try:
        pklfile = open(resume_model,"rb")
        model_prof_df = pickle.load(pklfile)
    except:
        logger.error("Cannot open pickle file containing "
            f"input model profile dictionary. Please check the filename: {resume_model}")
        sys.exit()
    return obs_prof_df, model_prof_df