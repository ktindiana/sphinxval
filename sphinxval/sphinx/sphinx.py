from ..utils import validation_json_handler as vjson
from ..utils import config as cfg
from ..utils import object_handler as objh
from ..utils import validation as valid
from ..utils import tools
from ..utils import match
from ..utils import resume
from ..utils import report
import datetime
import pickle
import logging
import sys
import os



    
def validate(data_list, model_list, DoResume=False, df_pkl=""):
    """ Validate ingests a list of observations (data_list) and a
        list of predictions (model_list).
        
        The observations and predictions are automatically matched
        according to various criteria and then metrics are calculate.
        
        If resume is set to True, then the matched observation and
        prediction pairs will be appended to a previously existing
        Pandas DataFrame (filename df_pkl). The last prediction window
        for each model will be checked in the existing dataframe.
        Only new forecasts past that prediction window will be appended.
        
        INPUT:
        
            :data_list: (string array) filenames of observation jsons,
                preferably covering a continuous time period
            :model_list: (string array) filenames of model prediction jsons
            :resume: (boolean) set to False and only data_list and model_list
                will be used to calculate metrics; set to True and new
                predictions will be added to existing dataframe
            :df_pkl: (string) filename of pickle file containing previously
                calculated dataframe. New predictions will be appended.
                
        OUTPUT:
        
            None
            
    """
    print("sphinx: Starting SPHINX Validation and reading in files: " + str(datetime.datetime.now()))


###RESUME WILL DETERMINE WHICH OBJECTS CONTINUE ON - inside
#load_objects_from_json, check timing and if already in dataframe
    #Create Observation and Forecast objects from jsons (edge cases)
    #Unique identifier - issue time, triggers, prediction window - ignore for now
    #Use last prediction window for model or energy_channel to include new
    #obs_objs and model_objs organized by energy channel
    all_energy_channels, obs_objs, model_objs =\
        vjson.load_objects_from_json(data_list, model_list)
    print("sphinx: Loaded all JSON files into Objects: " + str(datetime.datetime.now()))

    
    #Identify the unique models represented in the forecasts
    model_names = objh.build_model_list(model_objs)
    print("sphinx: Built model list: " + str(datetime.datetime.now()))
    
    #### RESUME ####
    #If resuming, check for last prediction window times for each
    #model in the input dataframe.
    #Variables that are a starting point when resuming are labelled
    #with r_. For example, r_df is the dataframe that was already created
    #by SPHINX and was input as a starting point by the user.
    r_df = None
    if DoResume:
        print("sphinx: Resume selected. Reading in previous dataframe: "
            + df_pkl + " at time " + str(datetime.datetime.now()))
        r_df = resume.read_in_df(df_pkl)
        model_objs = resume.check_fcast_for_resume(r_df, model_objs)
        print("sphinx: Completed reading in previous dataframe and checking for new forecasts only: "
            + df_pkl + " at time " + str(datetime.datetime.now()))
    ################
    
    
    #Dictionary of SPHINX objects containing all matching criteria
    #and matched observed values for each forecast (matched_sphinx) as well as
    #a dictionary organized by model, energy channel, and threshold with the
    #unique SEP events that were present inside of the forecast prediction
    #windows (observed_sep_events).
    print("sphinx: Starting matching process: " + str(datetime.datetime.now()))
    matched_sphinx, all_observed_thresholds, observed_sep_events =\
        match.match_all_forecasts(all_energy_channels, model_names,
            obs_objs, model_objs)
    print("sphinx: Completed matching process and starting intuitive validation: " + str(datetime.datetime.now()))


    #Perform intuitive validation
    valid.intuitive_validation(matched_sphinx, model_names,
        all_energy_channels, all_observed_thresholds, observed_sep_events, DoResume, r_df)
    print("sphinx: Completed validation: " + str(datetime.datetime.now()))
